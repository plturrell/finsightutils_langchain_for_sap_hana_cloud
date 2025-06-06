"""Service for the Visual Developer environment to dynamically create and run LangChain pipelines."""

import json
import logging
import uuid
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Literal
import importlib.util
import sys
from tempfile import NamedTemporaryFile
import threading
import asyncio

import hdbcli.dbapi
from fastapi import HTTPException
from pydantic import BaseModel, Field

from langchain_hana import HanaVectorStore
from langchain_hana.embeddings import GPUAcceleratedEmbeddings, TensorRTEmbeddings, HanaInternalEmbeddings
from config import config
from database import get_db_connection
import gpu_utils
from models import (
    Flow, DebugSession, DebugBreakpoint, DebugNodeData, 
    CreateDebugSessionRequest, CreateDebugSessionResponse,
    DebugStepRequest, DebugStepResponse, SetBreakpointRequest, 
    SetBreakpointResponse, GetVariablesRequest, GetVariablesResponse
)

# Configure logging
logger = logging.getLogger(__name__)

# Flow storage - in a production app, this would be in a database
FLOWS_DIR = os.path.join(os.path.dirname(__file__), "flows")
os.makedirs(FLOWS_DIR, exist_ok=True)

# Debug session storage - in a production app, this would be in a database
DEBUG_SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "debug_sessions")
os.makedirs(DEBUG_SESSIONS_DIR, exist_ok=True)

# Active debug sessions
active_debug_sessions: Dict[str, DebugSession] = {}


class FlowNode(BaseModel):
    """Model for a node in a flow."""
    id: str
    type: str
    data: Dict[str, Any]
    position: Dict[str, float]


class FlowEdge(BaseModel):
    """Model for an edge in a flow."""
    id: str
    source: str
    target: str
    type: Optional[str] = None
    animated: Optional[bool] = None


class Flow(BaseModel):
    """Model for a complete flow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    nodes: List[FlowNode]
    edges: List[FlowEdge]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class RunFlowRequest(BaseModel):
    """Request model for running a flow."""
    flow: Flow


class RunFlowResponse(BaseModel):
    """Response model for running a flow."""
    success: bool
    results: Any
    execution_time: float = 0.0
    generated_code: str = ""


class SaveFlowRequest(BaseModel):
    """Request model for saving a flow."""
    flow: Flow


class SaveFlowResponse(BaseModel):
    """Response model for saving a flow."""
    success: bool
    flow_id: str
    message: str


class ListFlowsResponse(BaseModel):
    """Response model for listing flows."""
    flows: List[Flow]


def save_flow(flow: Flow) -> SaveFlowResponse:
    """
    Save a flow to the flows directory.
    
    Args:
        flow: The flow to save.
        
    Returns:
        SaveFlowResponse: Response with the saved flow ID.
    """
    # Ensure the flow has an ID
    if not flow.id:
        flow.id = str(uuid.uuid4())
    
    # Update timestamps
    from datetime import datetime
    now = datetime.now().isoformat()
    if not flow.created_at:
        flow.created_at = now
    flow.updated_at = now
    
    # Save the flow to a file
    flow_path = os.path.join(FLOWS_DIR, f"{flow.id}.json")
    with open(flow_path, "w") as f:
        f.write(flow.json())
    
    return SaveFlowResponse(
        success=True,
        flow_id=flow.id,
        message=f"Flow '{flow.name}' saved successfully."
    )


def list_flows() -> ListFlowsResponse:
    """
    List all saved flows.
    
    Returns:
        ListFlowsResponse: Response with all flows.
    """
    flows = []
    
    # Get all JSON files in the flows directory
    for filename in os.listdir(FLOWS_DIR):
        if filename.endswith(".json"):
            flow_path = os.path.join(FLOWS_DIR, filename)
            try:
                with open(flow_path, "r") as f:
                    flow_data = json.load(f)
                    flows.append(Flow(**flow_data))
            except Exception as e:
                logger.error(f"Error loading flow {filename}: {str(e)}")
    
    return ListFlowsResponse(flows=flows)


def get_flow(flow_id: str) -> Flow:
    """
    Get a flow by ID.
    
    Args:
        flow_id: The ID of the flow to get.
        
    Returns:
        Flow: The flow.
        
    Raises:
        HTTPException: If the flow is not found.
    """
    flow_path = os.path.join(FLOWS_DIR, f"{flow_id}.json")
    if not os.path.exists(flow_path):
        raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found")
    
    try:
        with open(flow_path, "r") as f:
            flow_data = json.load(f)
            return Flow(**flow_data)
    except Exception as e:
        logger.error(f"Error loading flow {flow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading flow: {str(e)}")


def delete_flow(flow_id: str) -> Dict[str, Any]:
    """
    Delete a flow.
    
    Args:
        flow_id: The ID of the flow to delete.
        
    Returns:
        Dict: Response indicating success.
        
    Raises:
        HTTPException: If the flow is not found.
    """
    flow_path = os.path.join(FLOWS_DIR, f"{flow_id}.json")
    if not os.path.exists(flow_path):
        raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found")
    
    try:
        os.remove(flow_path)
        return {"success": True, "message": f"Flow {flow_id} deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting flow {flow_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting flow: {str(e)}")


def generate_code_from_flow(flow: Flow) -> str:
    """
    Generate Python code from a flow.
    
    Args:
        flow: The flow to generate code for.
        
    Returns:
        str: The generated Python code.
    """
    # Extract nodes by type
    connection_nodes = [n for n in flow.nodes if n.type == 'hanaConnection']
    embedding_nodes = [n for n in flow.nodes if n.type == 'embedding']
    vectorstore_nodes = [n for n in flow.nodes if n.type == 'vectorStore']
    query_nodes = [n for n in flow.nodes if n.type == 'query']
    
    # Start building the code
    code = f'''"""
{flow.name}
{flow.description}

Auto-generated by SAP HANA LangChain Visual Developer
"""

from langchain_hana import HanaVectorStore
'''

    # Add imports based on node types
    has_gpu = any(n.data.get('params', {}).get('useGPU', False) for n in embedding_nodes)
    has_tensorrt = any(n.data.get('params', {}).get('useTensorRT', False) for n in embedding_nodes)
    
    if has_gpu and has_tensorrt:
        code += 'from langchain_hana.embeddings import TensorRTEmbeddings\n'
    elif has_gpu:
        code += 'from langchain_hana.embeddings import GPUAcceleratedEmbeddings\n'
    else:
        code += 'from langchain.embeddings import HuggingFaceEmbeddings\n'
    
    # Add database connection
    code += '''
# Connect to SAP HANA Cloud
import hdbcli.dbapi
'''
    
    if connection_nodes:
        conn = connection_nodes[0].data.get('params', {})
        code += f'''conn = hdbcli.dbapi.connect(
    address="{conn.get('host', 'localhost')}",
    port={conn.get('port', 443)},
    user="{conn.get('user', 'DBADMIN')}",
    password="********",  # Replace with your actual password
    encrypt=True,
    sslValidateCertificate=False
)
'''
    
    # Add embedding model initialization
    if embedding_nodes:
        embedding = embedding_nodes[0].data.get('params', {})
        model_name = embedding.get('model', 'all-MiniLM-L6-v2')
        
        if embedding.get('useTensorRT', False):
            code += f'''
# Initialize TensorRT-optimized embedding model
embeddings = TensorRTEmbeddings(
    model_name="{model_name}",
    device="{'cuda' if embedding.get('useGPU', True) else 'cpu'}",
    precision="fp16"
)
'''
        elif embedding.get('useGPU', False):
            code += f'''
# Initialize GPU-accelerated embedding model
embeddings = GPUAcceleratedEmbeddings(
    model_name="{model_name}",
    device="cuda",
    batch_size=32
)
'''
        else:
            code += f'''
# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="{model_name}")
'''
    
    # Add vector store initialization
    if vectorstore_nodes and connection_nodes and embedding_nodes:
        vectorstore = vectorstore_nodes[0].data.get('params', {})
        code += f'''
# Create or connect to vector store
vector_store = HanaVectorStore(
    connection=conn,
    embedding=embeddings,
    table_name="{vectorstore.get('tableName', 'LANGCHAIN_VECTORS')}",
    embedding_dimension={vectorstore.get('embeddingDimension', 384)}
)
'''
    
    # Add query execution
    if query_nodes and vectorstore_nodes:
        query = query_nodes[0].data.get('params', {})
        query_text = query.get('queryText', 'What is SAP HANA Cloud?')
        k = query.get('k', 4)
        
        code += f'''
# Perform semantic search
query = "{query_text}"
'''
        
        if query.get('useMMR', False):
            code += f'''
# Using Maximum Marginal Relevance for diverse results
results = vector_store.max_marginal_relevance_search(
    query=query, 
    k={k},
    fetch_k={k * 4},
    lambda_mult=0.5
)
'''
        else:
            code += f'''
# Standard similarity search
results = vector_store.similarity_search(
    query=query, 
    k={k}
)
'''
        
        # Add code to process results
        code += '''
# Process and display results
for i, doc in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")
'''
    
    # Add code to close the connection
    code += '''
# Close the connection
conn.close()
'''
    
    return code


def run_flow(flow: Flow) -> RunFlowResponse:
    """
    Run a flow and return the results.
    
    Args:
        flow: The flow to run.
        
    Returns:
        RunFlowResponse: The results of running the flow.
        
    Raises:
        HTTPException: If there is an error running the flow.
    """
    import time
    start_time = time.time()
    
    try:
        # Generate the code
        code = generate_code_from_flow(flow)
        
        # Find query and results nodes
        query_nodes = [n for n in flow.nodes if n.type == 'query']
        results_nodes = [n for n in flow.nodes if n.type == 'results']
        
        if not query_nodes:
            raise ValueError("Flow must contain a query node")
        
        # Extract query parameters
        query_node = query_nodes[0]
        query_params = query_node.data.get('params', {})
        query_text = query_params.get('queryText', '')
        k = query_params.get('k', 4)
        use_mmr = query_params.get('useMMR', False)
        
        # Create a connection to SAP HANA Cloud
        conn = get_db_connection()
        
        # Initialize embedding model based on flow configuration
        embedding_nodes = [n for n in flow.nodes if n.type == 'embedding']
        if embedding_nodes:
            embedding_params = embedding_nodes[0].data.get('params', {})
            model_name = embedding_params.get('model', 'all-MiniLM-L6-v2')
            use_gpu = embedding_params.get('useGPU', False)
            use_tensorrt = embedding_params.get('useTensorRT', False)
            
            if use_gpu and use_tensorrt and gpu_utils.is_gpu_available():
                embeddings = TensorRTEmbeddings(
                    model_name=model_name,
                    device="cuda",
                    precision="fp16"
                )
            elif use_gpu and gpu_utils.is_gpu_available():
                embeddings = GPUAcceleratedEmbeddings(
                    model_name=model_name,
                    device="cuda",
                    batch_size=32
                )
            else:
                # Fall back to HANA internal embeddings
                embeddings = HanaInternalEmbeddings(
                    internal_embedding_model_id=config.gpu.internal_embedding_model_id
                    if hasattr(config, 'gpu')
                    else "SAP_NEB.20240715"
                )
        else:
            # Default to HANA internal embeddings
            embeddings = HanaInternalEmbeddings(
                internal_embedding_model_id=config.gpu.internal_embedding_model_id
                if hasattr(config, 'gpu')
                else "SAP_NEB.20240715"
            )
        
        # Initialize vector store
        vectorstore_nodes = [n for n in flow.nodes if n.type == 'vectorStore']
        if vectorstore_nodes:
            vectorstore_params = vectorstore_nodes[0].data.get('params', {})
            table_name = vectorstore_params.get('tableName', 'LANGCHAIN_VECTORS')
        else:
            table_name = 'LANGCHAIN_VECTORS'
        
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embeddings,
            table_name=table_name
        )
        
        # Perform the query
        if use_mmr:
            results = vector_store.max_marginal_relevance_search(
                query=query_text,
                k=k,
                fetch_k=k * 4,
                lambda_mult=0.5
            )
        else:
            results = vector_store.similarity_search(
                query=query_text,
                k=k
            )
        
        # Convert results to a format suitable for the response
        results_data = []
        for i, doc in enumerate(results):
            results_data.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": 0.9 - (i * 0.1)  # Approximate score since similarity_search doesn't return scores
            })
        
        execution_time = time.time() - start_time
        
        return RunFlowResponse(
            success=True,
            results=results_data,
            execution_time=execution_time,
            generated_code=code
        )
        
    except Exception as e:
        logger.error(f"Error running flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running flow: {str(e)}")


def run_flow_with_dynamic_code(flow: Flow) -> RunFlowResponse:
    """
    Run a flow by dynamically executing the generated code.
    
    Args:
        flow: The flow to run.
        
    Returns:
        RunFlowResponse: The results of running the flow.
        
    Raises:
        HTTPException: If there is an error running the flow.
    """
    import time
    start_time = time.time()
    
    try:
        # Generate the code
        code = generate_code_from_flow(flow)
        
        # Create a temporary Python file
        with NamedTemporaryFile(suffix='.py', delete=False) as temp_file:
            temp_path = temp_file.name
            
            # Modify the code to capture the results
            modified_code = code + '''
# Export results for the API
import json
import os

results_json = []
for i, doc in enumerate(results):
    results_json.append({
        "content": doc.page_content,
        "metadata": doc.metadata,
        "score": 0.9 - (i * 0.1)  # Approximate score
    })

# Write results to a file
with open(os.path.join(os.path.dirname(__file__), "results.json"), "w") as f:
    json.dump(results_json, f)
'''
            temp_file.write(modified_code.encode())
        
        # Execute the temporary Python file in a separate process
        import subprocess
        result = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        # Clean up
        os.unlink(temp_path)
        
        if result.returncode != 0:
            logger.error(f"Error executing dynamic code: {result.stderr}")
            raise ValueError(f"Error executing code: {result.stderr}")
        
        # Read the results from the JSON file
        results_path = os.path.join(os.path.dirname(temp_path), "results.json")
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results_data = json.load(f)
            os.unlink(results_path)
        else:
            # If no results file was created, return an empty list
            results_data = []
        
        execution_time = time.time() - start_time
        
        return RunFlowResponse(
            success=True,
            results=results_data,
            execution_time=execution_time,
            generated_code=code
        )
        
    except Exception as e:
        logger.error(f"Error running flow with dynamic code: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running flow with dynamic code: {str(e)}"
        )


# Debugging functions

def create_debug_session(request: CreateDebugSessionRequest) -> CreateDebugSessionResponse:
    """
    Create a new debug session for a flow.
    
    Args:
        request: The request containing the flow and optional breakpoints.
        
    Returns:
        CreateDebugSessionResponse: Response with the session ID.
    """
    flow = request.flow
    
    # Save the flow if it doesn't have an ID
    if not flow.id:
        save_result = save_flow(flow)
        flow.id = save_result.flow_id
    
    # Create a new debug session
    now = datetime.now().isoformat()
    session = DebugSession(
        flow_id=flow.id,
        breakpoints=request.breakpoints or [],
        created_at=now,
        updated_at=now
    )
    
    # Initialize node data for each node in the flow
    for node in flow.nodes:
        session.node_data[node.id] = DebugNodeData(
            node_id=node.id,
            status="not_executed"
        )
    
    # Save the session
    session_path = os.path.join(DEBUG_SESSIONS_DIR, f"{session.session_id}.json")
    with open(session_path, "w") as f:
        f.write(session.json())
    
    # Add to active sessions
    active_debug_sessions[session.session_id] = session
    
    return CreateDebugSessionResponse(
        session_id=session.session_id,
        status="ready",
        message="Debug session created successfully."
    )


def get_debug_session(session_id: str) -> DebugSession:
    """
    Get a debug session by ID.
    
    Args:
        session_id: The ID of the debug session.
        
    Returns:
        DebugSession: The debug session.
        
    Raises:
        HTTPException: If the session is not found.
    """
    # Check if the session is in active sessions
    if session_id in active_debug_sessions:
        return active_debug_sessions[session_id]
    
    # Otherwise, try to load it from disk
    session_path = os.path.join(DEBUG_SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Debug session {session_id} not found")
    
    try:
        with open(session_path, "r") as f:
            session_data = json.load(f)
            session = DebugSession(**session_data)
            
            # Add to active sessions
            active_debug_sessions[session_id] = session
            
            return session
    except Exception as e:
        logger.error(f"Error loading debug session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading debug session: {str(e)}")


def save_debug_session(session: DebugSession) -> None:
    """
    Save a debug session to disk.
    
    Args:
        session: The debug session to save.
    """
    session_path = os.path.join(DEBUG_SESSIONS_DIR, f"{session.session_id}.json")
    
    # Update the timestamp
    session.updated_at = datetime.now().isoformat()
    
    try:
        with open(session_path, "w") as f:
            f.write(session.json())
    except Exception as e:
        logger.error(f"Error saving debug session {session.session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving debug session: {str(e)}")


def delete_debug_session(session_id: str) -> Dict[str, Any]:
    """
    Delete a debug session.
    
    Args:
        session_id: The ID of the debug session to delete.
        
    Returns:
        Dict[str, Any]: Response indicating success.
        
    Raises:
        HTTPException: If the session is not found.
    """
    session_path = os.path.join(DEBUG_SESSIONS_DIR, f"{session_id}.json")
    if not os.path.exists(session_path):
        raise HTTPException(status_code=404, detail=f"Debug session {session_id} not found")
    
    try:
        # Remove from active sessions
        if session_id in active_debug_sessions:
            del active_debug_sessions[session_id]
        
        # Delete the file
        os.remove(session_path)
        
        return {"success": True, "message": f"Debug session {session_id} deleted successfully."}
    except Exception as e:
        logger.error(f"Error deleting debug session {session_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting debug session: {str(e)}")


def set_breakpoint(request: SetBreakpointRequest) -> SetBreakpointResponse:
    """
    Set a breakpoint in a debug session.
    
    Args:
        request: The request containing the session ID and breakpoint.
        
    Returns:
        SetBreakpointResponse: Response indicating success.
    """
    session = get_debug_session(request.session_id)
    
    # Find any existing breakpoint for this node
    existing_bp = next((bp for bp in session.breakpoints if bp.node_id == request.breakpoint.node_id), None)
    
    if existing_bp:
        # Update existing breakpoint
        session.breakpoints.remove(existing_bp)
        session.breakpoints.append(request.breakpoint)
    else:
        # Add new breakpoint
        session.breakpoints.append(request.breakpoint)
    
    # Save the session
    save_debug_session(session)
    
    return SetBreakpointResponse(
        success=True,
        message=f"Breakpoint set on node {request.breakpoint.node_id}."
    )


def get_node_execution_order(flow: Flow) -> List[str]:
    """
    Determine the execution order of nodes in a flow.
    
    Args:
        flow: The flow to analyze.
        
    Returns:
        List[str]: The ordered list of node IDs.
    """
    # Create a dependency graph
    dependencies = {node.id: [] for node in flow.nodes}
    
    for edge in flow.edges:
        if edge.target in dependencies:
            dependencies[edge.target].append(edge.source)
    
    # Find nodes with no dependencies
    execution_order = []
    visited = set()
    
    def visit(node_id):
        if node_id in visited:
            return
        
        visited.add(node_id)
        
        for dep in dependencies[node_id]:
            visit(dep)
        
        execution_order.append(node_id)
    
    # Visit all nodes
    for node_id in dependencies:
        if node_id not in visited:
            visit(node_id)
    
    return execution_order


def execute_node(flow: Flow, node_id: str, variables: Dict[str, Any]) -> Tuple[Any, Dict[str, Any], float]:
    """
    Execute a single node in a flow.
    
    Args:
        flow: The flow containing the node.
        node_id: The ID of the node to execute.
        variables: Current variables for the debug session.
        
    Returns:
        Tuple[Any, Dict[str, Any], float]: The node output, updated variables, and execution time.
    """
    node = next((n for n in flow.nodes if n.id == node_id), None)
    if not node:
        raise ValueError(f"Node {node_id} not found in flow")
    
    start_time = time.time()
    
    try:
        # Execute the node based on its type
        result = None
        
        if node.type == 'hanaConnection':
            # Create a database connection
            params = node.data.params
            conn = hdbcli.dbapi.connect(
                address=params.get('host', 'localhost'),
                port=params.get('port', 443),
                user=params.get('user', 'DBADMIN'),
                password="password",  # In a real app, use secure password handling
                encrypt=True,
                sslValidateCertificate=False
            )
            result = conn
            variables['connection'] = conn
            
        elif node.type == 'embedding':
            # Initialize embedding model
            params = node.data.params
            model_name = params.get('model', 'all-MiniLM-L6-v2')
            use_gpu = params.get('useGPU', False)
            use_tensorrt = params.get('useTensorRT', False)
            
            if use_gpu and use_tensorrt and gpu_utils.is_gpu_available():
                embeddings = TensorRTEmbeddings(
                    model_name=model_name,
                    device="cuda",
                    precision="fp16"
                )
            elif use_gpu and gpu_utils.is_gpu_available():
                embeddings = GPUAcceleratedEmbeddings(
                    model_name=model_name,
                    device="cuda",
                    batch_size=32
                )
            else:
                # Fall back to HANA internal embeddings
                embeddings = HanaInternalEmbeddings(
                    internal_embedding_model_id=config.gpu.internal_embedding_model_id
                    if hasattr(config, 'gpu')
                    else "SAP_NEB.20240715"
                )
            
            result = embeddings
            variables['embeddings'] = embeddings
            
        elif node.type == 'vectorStore':
            # Initialize vector store
            if 'connection' not in variables or 'embeddings' not in variables:
                raise ValueError("Vector store requires connection and embeddings")
            
            params = node.data.params
            vector_store = HanaVectorStore(
                connection=variables['connection'],
                embedding=variables['embeddings'],
                table_name=params.get('tableName', 'LANGCHAIN_VECTORS')
            )
            
            result = vector_store
            variables['vector_store'] = vector_store
            
        elif node.type == 'query':
            # Execute query
            if 'vector_store' not in variables:
                raise ValueError("Query requires vector store")
            
            params = node.data.params
            query_text = params.get('queryText', '')
            k = params.get('k', 4)
            use_mmr = params.get('useMMR', False)
            
            if use_mmr:
                results = variables['vector_store'].max_marginal_relevance_search(
                    query=query_text,
                    k=k,
                    fetch_k=k * 4,
                    lambda_mult=0.5
                )
            else:
                results = variables['vector_store'].similarity_search(
                    query=query_text,
                    k=k
                )
            
            result = results
            variables['query_results'] = results
            
        elif node.type == 'results':
            # Process results
            if 'query_results' not in variables:
                raise ValueError("Results node requires query results")
            
            results_data = []
            for i, doc in enumerate(variables['query_results']):
                results_data.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": 0.9 - (i * 0.1)  # Approximate score
                })
            
            result = results_data
        
        execution_time = time.time() - start_time
        
        return result, variables, execution_time
    
    except Exception as e:
        logger.error(f"Error executing node {node_id}: {str(e)}")
        raise ValueError(f"Error executing node {node_id}: {str(e)}")


def step_debug_session(request: DebugStepRequest) -> DebugStepResponse:
    """
    Step through a debug session.
    
    Args:
        request: The request containing the session ID and step type.
        
    Returns:
        DebugStepResponse: Response with updated session and node output.
    """
    session = get_debug_session(request.session_id)
    
    # Get the flow
    flow = get_flow(session.flow_id)
    
    # Get the execution order
    execution_order = get_node_execution_order(flow)
    
    try:
        if request.step_type == "reset":
            # Reset the session
            for node_id in session.node_data:
                session.node_data[node_id].status = "not_executed"
                session.node_data[node_id].input_data = None
                session.node_data[node_id].output_data = None
                session.node_data[node_id].execution_time = None
                session.node_data[node_id].error = None
            
            session.current_node_id = None
            session.status = "ready"
            session.variables = {}
            
            save_debug_session(session)
            
            return DebugStepResponse(
                session=session,
                message="Debug session reset."
            )
        
        # Determine the next node to execute
        next_node_id = None
        
        if session.status == "ready":
            # Start execution with the first node
            next_node_id = execution_order[0]
            session.status = "running"
        elif session.status == "paused":
            # Continue from the current node
            if request.step_type == "step":
                # Get the index of the current node
                current_idx = execution_order.index(session.current_node_id)
                if current_idx < len(execution_order) - 1:
                    next_node_id = execution_order[current_idx + 1]
                else:
                    session.status = "completed"
            elif request.step_type == "continue":
                # Run until the next breakpoint or completion
                current_idx = execution_order.index(session.current_node_id)
                for i in range(current_idx + 1, len(execution_order)):
                    next_node_id = execution_order[i]
                    # Check if this node has a breakpoint
                    if any(bp.node_id == next_node_id and bp.enabled for bp in session.breakpoints):
                        break
        
        # Execute the next node if available
        if next_node_id:
            # Mark the node as executing
            session.node_data[next_node_id].status = "executing"
            session.current_node_id = next_node_id
            save_debug_session(session)
            
            # Execute the node
            try:
                output, updated_vars, exec_time = execute_node(flow, next_node_id, session.variables)
                
                # Update the session
                session.variables = updated_vars
                session.node_data[next_node_id].status = "completed"
                session.node_data[next_node_id].output_data = output
                session.node_data[next_node_id].execution_time = exec_time
                
                # Check if this node has a breakpoint
                if any(bp.node_id == next_node_id and bp.enabled for bp in session.breakpoints):
                    session.status = "paused"
                else:
                    # Check if we've reached the end
                    if execution_order.index(next_node_id) == len(execution_order) - 1:
                        session.status = "completed"
            except Exception as e:
                session.node_data[next_node_id].status = "error"
                session.node_data[next_node_id].error = str(e)
                session.status = "error"
            
            save_debug_session(session)
            
            return DebugStepResponse(
                session=session,
                node_output=session.node_data[next_node_id].output_data,
                execution_time=session.node_data[next_node_id].execution_time or 0.0,
                message=f"Executed node {next_node_id}."
            )
        else:
            return DebugStepResponse(
                session=session,
                message="No more nodes to execute."
            )
    
    except Exception as e:
        logger.error(f"Error stepping debug session {request.session_id}: {str(e)}")
        session.status = "error"
        save_debug_session(session)
        
        return DebugStepResponse(
            session=session,
            message=f"Error stepping debug session: {str(e)}"
        )


def get_variables(request: GetVariablesRequest) -> GetVariablesResponse:
    """
    Get variables from a debug session.
    
    Args:
        request: The request containing the session ID and optional variable names.
        
    Returns:
        GetVariablesResponse: Response with variables.
    """
    session = get_debug_session(request.session_id)
    
    if request.variable_names:
        # Return only the requested variables
        variables = {name: session.variables.get(name) for name in request.variable_names if name in session.variables}
    else:
        # Return all variables
        variables = session.variables
    
    return GetVariablesResponse(variables=variables)