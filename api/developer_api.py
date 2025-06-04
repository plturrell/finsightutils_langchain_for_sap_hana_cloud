"""API endpoints for the Visual Developer environment."""

import logging
from typing import Dict, Any, List
import os
import numpy as np
from sklearn.manifold import TSNE
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends

from models import (
    Flow, 
    RunFlowRequest, 
    RunFlowResponse,
    SaveFlowRequest,
    SaveFlowResponse,
    ListFlowsResponse,
    GetVectorsRequest,
    GetVectorsResponse,
    VectorDataPoint,
    # Debug models
    DebugSession,
    DebugBreakpoint,
    CreateDebugSessionRequest,
    CreateDebugSessionResponse,
    DebugStepRequest,
    DebugStepResponse,
    SetBreakpointRequest,
    SetBreakpointResponse,
    GetVariablesRequest,
    GetVariablesResponse
)

from developer_service import (
    run_flow,
    generate_code_from_flow,
    save_flow,
    list_flows,
    get_flow,
    delete_flow,
    # Debug functions
    create_debug_session,
    get_debug_session,
    step_debug_session,
    set_breakpoint,
    get_variables,
    delete_debug_session
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/developer", tags=["developer"])


@router.post("/run", response_model=RunFlowResponse)
async def run_flow_endpoint(request: RunFlowRequest) -> RunFlowResponse:
    """
    Run a flow and return the results.
    
    Args:
        request: The request containing the flow to run.
        
    Returns:
        RunFlowResponse: The results of running the flow.
    """
    try:
        return run_flow(request.flow)
    except Exception as e:
        logger.error(f"Error running flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error running flow: {str(e)}")


@router.post("/generate-code", response_model=Dict[str, str])
async def generate_code_endpoint(flow: Flow) -> Dict[str, str]:
    """
    Generate Python code from a flow.
    
    Args:
        flow: The flow to generate code for.
        
    Returns:
        Dict[str, str]: The generated Python code.
    """
    try:
        code = generate_code_from_flow(flow)
        return {"code": code}
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating code: {str(e)}")


@router.post("/flows", response_model=SaveFlowResponse)
async def save_flow_endpoint(request: SaveFlowRequest) -> SaveFlowResponse:
    """
    Save a flow.
    
    Args:
        request: The request containing the flow to save.
        
    Returns:
        SaveFlowResponse: Response with the saved flow ID.
    """
    try:
        return save_flow(request.flow)
    except Exception as e:
        logger.error(f"Error saving flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving flow: {str(e)}")


@router.get("/flows", response_model=ListFlowsResponse)
async def list_flows_endpoint() -> ListFlowsResponse:
    """
    List all saved flows.
    
    Returns:
        ListFlowsResponse: Response with all flows.
    """
    try:
        return list_flows()
    except Exception as e:
        logger.error(f"Error listing flows: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing flows: {str(e)}")


@router.get("/flows/{flow_id}", response_model=Flow)
async def get_flow_endpoint(flow_id: str) -> Flow:
    """
    Get a flow by ID.
    
    Args:
        flow_id: The ID of the flow to get.
        
    Returns:
        Flow: The flow.
    """
    try:
        return get_flow(flow_id)
    except Exception as e:
        logger.error(f"Error getting flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting flow: {str(e)}")


@router.delete("/flows/{flow_id}", response_model=Dict[str, Any])
async def delete_flow_endpoint(flow_id: str) -> Dict[str, Any]:
    """
    Delete a flow.
    
    Args:
        flow_id: The ID of the flow to delete.
        
    Returns:
        Dict[str, Any]: Response indicating success.
    """
    try:
        return delete_flow(flow_id)
    except Exception as e:
        logger.error(f"Error deleting flow: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting flow: {str(e)}")


@router.post("/vectors", response_model=GetVectorsResponse)
async def get_vectors_endpoint(request: GetVectorsRequest) -> GetVectorsResponse:
    """
    Get vector embeddings with reduced dimensionality for visualization.
    
    Args:
        request: The request containing parameters for getting vectors.
        
    Returns:
        GetVectorsResponse: Response with vector data for visualization.
    """
    try:
        from database import get_db_connection
        from sklearn.manifold import TSNE
        from sklearn.cluster import KMeans
        import numpy as np
        import hashlib
        import os
        import pickle
        
        # Get the database connection
        conn = get_db_connection()
        
        # Parameters
        table_name = request.tableName
        max_points = request.maxPoints or 500
        filter_dict = request.filter or {}
        page = request.page or 1
        page_size = request.pageSize or max_points
        offset = (page - 1) * page_size
        
        # Build the filter SQL if needed
        filter_sql = ""
        filter_params = {}
        
        if filter_dict:
            filter_parts = []
            for i, (key, value) in enumerate(filter_dict.items()):
                param_name = f"param_{i}"
                if key == 'content':
                    filter_parts.append(f"DOCUMENT LIKE :{param_name}")
                    filter_params[param_name] = f"%{value}%"
                elif isinstance(value, list):
                    placeholder_list = ','.join([f":{param_name}_{j}" for j in range(len(value))])
                    filter_parts.append(f"METADATA.{key} IN ({placeholder_list})")
                    for j, v in enumerate(value):
                        filter_params[f"{param_name}_{j}"] = v
                else:
                    filter_parts.append(f"METADATA.{key} = :{param_name}")
                    filter_params[param_name] = value
            
            if filter_parts:
                filter_sql = "WHERE " + " AND ".join(filter_parts)
        
        # Query to get the vectors
        query = f"""
        SELECT ID, DOCUMENT, METADATA, VECTOR 
        FROM {table_name}
        {filter_sql}
        LIMIT {page_size} OFFSET {offset}
        """
        
        # Count total number of vectors
        count_query = f"""
        SELECT COUNT(*) FROM {table_name}
        {filter_sql}
        """
        
        # Execute the count query
        cursor = conn.cursor()
        cursor.execute(count_query, filter_params)
        total_count = cursor.fetchone()[0]
        
        # Execute the main query
        cursor.execute(query, filter_params)
        rows = cursor.fetchall()
        
        # Create cache directory if it doesn't exist
        cache_dir = os.path.join(os.path.dirname(__file__), "vector_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate a cache key based on the query parameters
        cache_key = hashlib.md5(
            f"{table_name}_{str(filter_dict)}_{max_points}_{page}_{page_size}".encode()
        ).hexdigest()
        cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
        
        # Check if we have cached results
        vectors_with_reduced = []
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    vectors_with_reduced = pickle.load(f)
                    logger.info(f"Using cached vector reduction for {cache_key}")
            except Exception as e:
                logger.warning(f"Could not load cache: {str(e)}")
                vectors_with_reduced = []
        
        # If not in cache, process vectors
        if not vectors_with_reduced:
            # Extract vectors for dimensionality reduction
            vectors = []
            vector_data = []
            
            for row in rows:
                vector_id = row[0]
                document = row[1]
                metadata = row[2] if row[2] else {}
                vector = row[3]
                
                # Convert vector to numpy array
                if isinstance(vector, str):
                    # Parse vector from string if needed
                    vector = np.array([float(x) for x in vector.strip('[]').split(',')])
                elif isinstance(vector, bytes):
                    # Parse vector from binary if needed
                    import struct
                    fmt = f"{len(vector)//4}f"  # Assuming 4 bytes per float
                    vector = np.array(struct.unpack(fmt, vector))
                
                vectors.append(vector)
                vector_data.append({
                    "id": str(vector_id),
                    "content": document,
                    "metadata": metadata,
                    "vector": vector.tolist()
                })
            
            if vectors:
                # Convert to numpy array
                vectors_array = np.array(vectors)
                
                # Perform dimensionality reduction
                if len(vectors_array) > 1:
                    # Use T-SNE for dimensionality reduction
                    tsne = TSNE(n_components=3, perplexity=min(30, len(vectors_array)-1), random_state=42)
                    reduced_vectors = tsne.fit_transform(vectors_array)
                    
                    # Perform K-means clustering
                    n_clusters = min(5, len(vectors_array))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(vectors_array)
                    
                    # Add cluster information and reduced vectors
                    for i, data in enumerate(vector_data):
                        data["reduced_vector"] = reduced_vectors[i].tolist()
                        data["metadata"]["cluster"] = int(cluster_labels[i])
                        vectors_with_reduced.append(VectorDataPoint(**data))
                else:
                    # If only one vector, use a default position
                    for i, data in enumerate(vector_data):
                        data["reduced_vector"] = [0.0, 0.0, 0.0]
                        data["metadata"]["cluster"] = 0
                        vectors_with_reduced.append(VectorDataPoint(**data))
                
                # Cache the results
                try:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(vectors_with_reduced, f)
                    logger.info(f"Cached vector reduction as {cache_key}")
                except Exception as e:
                    logger.warning(f"Could not save cache: {str(e)}")
        
        # Return the response with pagination information
        return GetVectorsResponse(
            vectors=vectors_with_reduced,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=(total_count + page_size - 1) // page_size
        )
    except Exception as e:
        logger.error(f"Error getting vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting vectors: {str(e)}")


# Debug Endpoints
@router.post("/debug/sessions", response_model=CreateDebugSessionResponse)
async def create_debug_session_endpoint(request: CreateDebugSessionRequest) -> CreateDebugSessionResponse:
    """
    Create a new debug session for a flow.
    
    Args:
        request: The request containing the flow and optional breakpoints.
        
    Returns:
        CreateDebugSessionResponse: Response with the session ID.
    """
    try:
        return create_debug_session(request)
    except Exception as e:
        logger.error(f"Error creating debug session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating debug session: {str(e)}")


@router.get("/debug/sessions/{session_id}", response_model=DebugSession)
async def get_debug_session_endpoint(session_id: str) -> DebugSession:
    """
    Get a debug session by ID.
    
    Args:
        session_id: The ID of the debug session.
        
    Returns:
        DebugSession: The debug session.
    """
    try:
        return get_debug_session(session_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting debug session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting debug session: {str(e)}")


@router.delete("/debug/sessions/{session_id}", response_model=Dict[str, Any])
async def delete_debug_session_endpoint(session_id: str) -> Dict[str, Any]:
    """
    Delete a debug session.
    
    Args:
        session_id: The ID of the debug session to delete.
        
    Returns:
        Dict[str, Any]: Response indicating success.
    """
    try:
        return delete_debug_session(session_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting debug session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting debug session: {str(e)}")


@router.post("/debug/step", response_model=DebugStepResponse)
async def step_debug_session_endpoint(request: DebugStepRequest) -> DebugStepResponse:
    """
    Step through a debug session.
    
    Args:
        request: The request containing the session ID and step type.
        
    Returns:
        DebugStepResponse: Response with updated session and node output.
    """
    try:
        return step_debug_session(request)
    except Exception as e:
        logger.error(f"Error stepping debug session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error stepping debug session: {str(e)}")


@router.post("/debug/breakpoints", response_model=SetBreakpointResponse)
async def set_breakpoint_endpoint(request: SetBreakpointRequest) -> SetBreakpointResponse:
    """
    Set a breakpoint in a debug session.
    
    Args:
        request: The request containing the session ID and breakpoint.
        
    Returns:
        SetBreakpointResponse: Response indicating success.
    """
    try:
        return set_breakpoint(request)
    except Exception as e:
        logger.error(f"Error setting breakpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting breakpoint: {str(e)}")


@router.post("/debug/variables", response_model=GetVariablesResponse)
async def get_variables_endpoint(request: GetVariablesRequest) -> GetVariablesResponse:
    """
    Get variables from a debug session.
    
    Args:
        request: The request containing the session ID and optional variable names.
        
    Returns:
        GetVariablesResponse: Response with variables.
    """
    try:
        return get_variables(request)
    except Exception as e:
        logger.error(f"Error getting variables: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting variables: {str(e)}")