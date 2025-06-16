"""
Developer tools routes for version 1 of the API.

This module provides API routes for developer tools, including:
- Flow execution and management
- Code generation
- Vector visualization
"""

import os
import json
import uuid
import logging
import pickle
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np

from fastapi import Depends, HTTPException, Path, Body
from pydantic import BaseModel, Field
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

# Import models and database helpers
from ...db import get_db_connection
from ...models import (
    RunFlowRequest,
    RunFlowResponse,
    GenerateCodeRequest,
    GenerateCodeResponse,
    SaveFlowRequest,
    SaveFlowResponse,
    ListFlowsResponse,
    GetFlowResponse,
    GetVectorsRequest,
    GetVectorsResponse,
    VectorDataPoint
)

# Import error utilities
from ...utils.error_utils import (
    handle_flow_execution_error,
    handle_vector_search_error,
    handle_vector_visualization_error,
    create_context_aware_error
)

# Import base router
from ..base import BaseRouter

# Set up logging
logger = logging.getLogger(__name__)

# Storage directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
FLOWS_DIR = os.path.join(ROOT_DIR, "flows")
os.makedirs(FLOWS_DIR, exist_ok=True)

VECTOR_CACHE_DIR = os.path.join(ROOT_DIR, "vector_cache")
os.makedirs(VECTOR_CACHE_DIR, exist_ok=True)

# Create router
router = BaseRouter(
    prefix="/developer",
    tags=["Developer Tools"]
)


@router.post("/run", response_model=RunFlowResponse)
async def run_flow(request: RunFlowRequest = Body(...)):
    """
    Run a LangChain flow against SAP HANA Cloud.
    
    This endpoint:
    - Executes a LangChain flow against SAP HANA Cloud
    - Returns results and generated code
    - Supports similarity search and MMR search
    
    Returns:
        RunFlowResponse: The flow execution results
    """
    try:
        import time
        start_time = time.time()
        
        # Get flow components
        connection_node = next((node for node in request.flow.nodes if node.type == "hanaConnection"), None)
        embedding_node = next((node for node in request.flow.nodes if node.type == "embedding"), None)
        vectorstore_node = next((node for node in request.flow.nodes if node.type == "vectorStore"), None)
        query_node = next((node for node in request.flow.nodes if node.type == "query"), None)
        
        if not query_node:
            raise HTTPException(status_code=400, detail="No query node found in the flow")
        
        # Get database connection
        conn = get_db_connection()
        
        # Initialize embedding model
        if embedding_node and embedding_node.data.params.get("useGPU", False):
            # For GPU embeddings, we'll use HanaInternalEmbeddings in this implementation
            # This could be extended to use GPU embeddings in a real implementation
            from langchain_hana.embeddings import HanaInternalEmbeddings
            embedding_model = HanaInternalEmbeddings(
                internal_embedding_model_id="SAP_NEB.20240715"
            )
        else:
            # Use SAP HANA's internal embeddings
            from langchain_hana.embeddings import HanaInternalEmbeddings
            embedding_model = HanaInternalEmbeddings(
                internal_embedding_model_id="SAP_NEB.20240715"
            )
        
        # Initialize vector store
        from langchain_hana import HanaVectorStore
        table_name = vectorstore_node.data.params.get("tableName", "EMBEDDINGS") if vectorstore_node else "EMBEDDINGS"
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=table_name
        )
        
        # Execute query
        query_text = query_node.data.params.get("queryText", "")
        k = query_node.data.params.get("k", 4)
        use_mmr = query_node.data.params.get("useMMR", False)
        
        # Get results with actual similarity scores
        if use_mmr:
            # Maximal Marginal Relevance search with scores
            mmr_results = vector_store.max_marginal_relevance_search(
                query=query_text,
                k=k,
                fetch_k=k * 4,
                lambda_mult=0.5
            )
            
            # For MMR, we need to compute similarity scores since they're not returned directly
            if isinstance(vector_store.embedding, HanaInternalEmbeddings):
                query_embedding = vector_store._embed_query_hana_internal(query_text)
            else:
                query_embedding = vector_store.embedding.embed_query(query_text)
                
            # Get vectors for all results for scoring
            result_docs_with_scores = vector_store.similarity_search_with_score_and_vector_by_vector(
                embedding=query_embedding,
                k=len(mmr_results) + 5  # Get a few extra to ensure we have all MMR results
            )
            
            # Create a mapping of document content to score
            content_to_score = {doc.page_content: score for doc, score, _ in result_docs_with_scores}
            
            # Process MMR results with actual scores when available
            results = mmr_results
            result_data = []
            for doc in results:
                # Use the actual score if available, otherwise calculate approximate similarity
                if doc.page_content in content_to_score:
                    score = content_to_score[doc.page_content]
                else:
                    # If we can't find the exact content match, find the most similar document
                    # from our scored results to get a more accurate approximation
                    best_match_score = 0.0
                    for scored_doc, scored_score, _ in result_docs_with_scores:
                        # Simple similarity heuristic based on character overlap
                        similarity = len(set(doc.page_content[:100]).intersection(scored_doc.page_content[:100])) / 100
                        if similarity > 0.7 and scored_score > best_match_score:  # Reasonable similarity threshold
                            best_match_score = scored_score
                    
                    score = best_match_score if best_match_score > 0 else 0.7  # Final fallback
                result_data.append({
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                })
        else:
            # Standard similarity search with scores
            results_with_scores = vector_store.similarity_search_with_score(
                query=query_text,
                k=k
            )
            
            # Process results for API response with actual scores
            result_data = []
            for doc, score in results_with_scores:
                result_data.append({
                    "content": doc.page_content,
                    "score": score,
                    "metadata": doc.metadata
                })
        
        # Generate code from the flow
        generated_code = generate_python_code(request.flow)
        
        execution_time = time.time() - start_time
        
        return RunFlowResponse(
            success=True,
            results=result_data,
            execution_time=execution_time,
            generated_code=generated_code
        )
    except HTTPException as e:
        # Pass through already formatted HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error running flow: {str(e)}")
        flow_info = {
            "flow_name": request.flow.name,
            "nodes": [node.type for node in request.flow.nodes],
            "contains_query_node": any(node.type == "query" for node in request.flow.nodes),
            "contains_connection_node": any(node.type == "hanaConnection" for node in request.flow.nodes),
            "contains_embedding_node": any(node.type == "embedding" for node in request.flow.nodes),
            "contains_vectorstore_node": any(node.type == "vectorStore" for node in request.flow.nodes)
        }
        raise handle_flow_execution_error(e, flow_info)


@router.post("/generate-code", response_model=GenerateCodeResponse)
async def generate_code(request: GenerateCodeRequest = Body(...)):
    """
    Generate Python code from a LangChain flow.
    
    This endpoint converts a flow definition into executable Python code.
    
    Returns:
        GenerateCodeResponse: The generated Python code
    """
    try:
        code = generate_python_code(request)
        return GenerateCodeResponse(code=code)
    except Exception as e:
        logger.error(f"Error generating code: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "flow_execution",
            status_code=500,
            additional_context={
                "message": "Error generating Python code from flow definition.",
                "suggestions": [
                    "Check that your flow contains all required nodes.",
                    "Verify that node parameters are correctly set.",
                    "Ensure the flow structure is valid."
                ]
            }
        )


@router.post("/flows", response_model=SaveFlowResponse)
async def save_flow(request: SaveFlowRequest = Body(...)):
    """
    Save a flow definition to the file system.
    
    This endpoint:
    - Generates a unique ID if not provided
    - Adds timestamps for creation and update
    - Saves the flow as JSON
    
    Returns:
        SaveFlowResponse: The result of the save operation
    """
    try:
        # Generate a unique ID if it doesn't exist
        if not request.flow.id:
            request.flow.id = str(uuid.uuid4())
        
        # Add timestamp
        request.flow.updated_at = datetime.now().isoformat()
        if not request.flow.created_at:
            request.flow.created_at = request.flow.updated_at
        
        # Store the flow to file
        flow_path = os.path.join(FLOWS_DIR, f"{request.flow.id}.json")
        with open(flow_path, "w") as f:
            f.write(request.flow.json())
        
        return SaveFlowResponse(
            success=True,
            flow_id=request.flow.id,
            message="Flow saved successfully"
        )
    except Exception as e:
        logger.error(f"Error saving flow: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "flow_execution",
            status_code=500,
            additional_context={
                "message": "Error saving flow definition.",
                "suggestions": [
                    "Check if the flows directory is writable.",
                    "Verify that the flow data is valid and serializable.",
                    "Ensure there's enough disk space available."
                ]
            }
        )


@router.get("/flows", response_model=ListFlowsResponse)
async def list_flows():
    """
    List all saved flows.
    
    This endpoint retrieves all flows saved in the flows directory.
    
    Returns:
        ListFlowsResponse: List of all saved flows
    """
    try:
        flows = []
        
        # Get all JSON files in the flows directory
        if os.path.exists(FLOWS_DIR):
            for filename in os.listdir(FLOWS_DIR):
                if filename.endswith(".json"):
                    flow_path = os.path.join(FLOWS_DIR, filename)
                    try:
                        with open(flow_path, "r") as f:
                            flow_data = json.load(f)
                            flows.append(flow_data)
                    except Exception as e:
                        logger.error(f"Error loading flow {filename}: {str(e)}")
        
        return ListFlowsResponse(flows=flows)
    except Exception as e:
        logger.error(f"Error listing flows: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "flow_execution",
            status_code=500,
            additional_context={
                "message": "Error retrieving flow list.",
                "suggestions": [
                    "Check if the flows directory exists and is readable.",
                    "Verify that you have permission to access the flow files.",
                    "Ensure the flow files are valid JSON files."
                ]
            }
        )


@router.get("/flows/{flow_id}", response_model=GetFlowResponse)
async def get_flow(flow_id: str = Path(..., description="The ID of the flow to retrieve")):
    """
    Get a specific flow by ID.
    
    This endpoint retrieves a specific flow by its ID.
    
    Args:
        flow_id: The ID of the flow to retrieve
        
    Returns:
        GetFlowResponse: The requested flow
        
    Raises:
        HTTPException: If the flow is not found
    """
    try:
        flow_path = os.path.join(FLOWS_DIR, f"{flow_id}.json")
        if not os.path.exists(flow_path):
            raise HTTPException(status_code=404, detail="Flow not found")
        
        with open(flow_path, "r") as f:
            flow_data = json.load(f)
            return GetFlowResponse(**flow_data)
            
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error getting flow: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "flow_execution",
            status_code=500,
            additional_context={
                "message": f"Error retrieving flow with ID {flow_id}.",
                "suggestions": [
                    "Check if the flow file exists and is readable.",
                    "Verify that you have permission to access the flow file.",
                    "Ensure the flow ID is correct.",
                    "Check if the flow file contains valid JSON data."
                ]
            }
        )


@router.delete("/flows/{flow_id}", response_model=Dict[str, bool])
async def delete_flow(flow_id: str = Path(..., description="The ID of the flow to delete")):
    """
    Delete a specific flow by ID.
    
    This endpoint deletes a specific flow by its ID.
    
    Args:
        flow_id: The ID of the flow to delete
        
    Returns:
        Dict[str, bool]: Success indicator
        
    Raises:
        HTTPException: If the flow is not found
    """
    try:
        flow_path = os.path.join(FLOWS_DIR, f"{flow_id}.json")
        if not os.path.exists(flow_path):
            raise HTTPException(status_code=404, detail="Flow not found")
        
        os.remove(flow_path)
        
        return {"success": True}
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error deleting flow: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "flow_execution",
            status_code=500,
            additional_context={
                "message": f"Error deleting flow with ID {flow_id}.",
                "suggestions": [
                    "Check if the flow file exists and is writable.",
                    "Verify that you have permission to delete files in the flows directory.",
                    "Ensure the flow ID is correct.",
                    "Check if another process might be using the flow file."
                ]
            }
        )


@router.post("/vectors", response_model=GetVectorsResponse)
async def get_vectors(request: GetVectorsRequest = Body(...)):
    """
    Get vector embeddings from HANA and reduce their dimensionality for visualization.
    
    This endpoint:
    - Retrieves vector embeddings from SAP HANA
    - Applies dimensionality reduction (t-SNE, PCA, or UMAP)
    - Performs clustering (K-means, DBSCAN, or HDBSCAN)
    - Returns reduced vectors for visualization
    
    Returns:
        GetVectorsResponse: The vector data with reduced dimensions
    """
    try:
        # Get the database connection
        conn = get_db_connection()
        
        # Parameters
        table_name = request.tableName
        max_points = request.maxPoints or 500
        filter_dict = request.filter or {}
        page = request.page or 1
        page_size = request.pageSize or max_points
        offset = (page - 1) * page_size
        clustering_algorithm = request.clusteringAlgorithm or "kmeans"
        dimensionality_reduction = request.dimensionalityReduction or "tsne"
        
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
        cache_dir = VECTOR_CACHE_DIR
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate a cache key based on the query parameters
        cache_key = hashlib.md5(
            f"{table_name}_{str(filter_dict)}_{max_points}_{page}_{page_size}_{clustering_algorithm}_{dimensionality_reduction}".encode()
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
                    # Choose dimensionality reduction technique
                    if dimensionality_reduction == "tsne":
                        # Use t-SNE for dimensionality reduction
                        reducer = TSNE(
                            n_components=3, 
                            perplexity=min(30, len(vectors_array)-1), 
                            random_state=42
                        )
                    elif dimensionality_reduction == "pca":
                        # Use PCA for dimensionality reduction
                        reducer = PCA(n_components=3, random_state=42)
                    elif dimensionality_reduction == "umap":
                        # Use UMAP for dimensionality reduction if available
                        try:
                            import umap
                            reducer = umap.UMAP(
                                n_components=3,
                                n_neighbors=min(15, len(vectors_array)-1),
                                min_dist=0.1,
                                random_state=42
                            )
                        except ImportError:
                            # Fall back to t-SNE if UMAP is not available
                            logger.warning("UMAP not available, falling back to t-SNE")
                            reducer = TSNE(
                                n_components=3, 
                                perplexity=min(30, len(vectors_array)-1), 
                                random_state=42
                            )
                    else:
                        # Default to t-SNE
                        reducer = TSNE(
                            n_components=3, 
                            perplexity=min(30, len(vectors_array)-1), 
                            random_state=42
                        )
                    
                    # Perform dimensionality reduction
                    reduced_vectors = reducer.fit_transform(vectors_array)
                    
                    # Perform clustering
                    if clustering_algorithm == "kmeans":
                        # K-means clustering
                        n_clusters = min(5, len(vectors_array))
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = clusterer.fit_predict(vectors_array)
                    elif clustering_algorithm == "dbscan":
                        # DBSCAN clustering
                        clusterer = DBSCAN(eps=0.5, min_samples=5)
                        cluster_labels = clusterer.fit_predict(vectors_array)
                    elif clustering_algorithm == "hdbscan":
                        # HDBSCAN clustering if available
                        try:
                            import hdbscan
                            clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=2)
                            cluster_labels = clusterer.fit_predict(vectors_array)
                        except ImportError:
                            # Fall back to K-means if HDBSCAN is not available
                            logger.warning("HDBSCAN not available, falling back to K-means")
                            n_clusters = min(5, len(vectors_array))
                            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                            cluster_labels = clusterer.fit_predict(vectors_array)
                    else:
                        # Default to K-means
                        n_clusters = min(5, len(vectors_array))
                        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                        cluster_labels = clusterer.fit_predict(vectors_array)
                    
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
    except HTTPException as e:
        # Pass through already formatted HTTP exceptions
        raise e
    except Exception as e:
        logger.error(f"Error getting vectors: {str(e)}")
        viz_params = {
            "table_name": request.tableName,
            "max_points": request.maxPoints,
            "page": request.page,
            "page_size": request.pageSize,
            "clustering_algorithm": request.clusteringAlgorithm,
            "dimensionality_reduction": request.dimensionalityReduction,
            "filter_applied": bool(request.filter),
        }
        raise handle_vector_visualization_error(e, viz_params)


def generate_python_code(flow):
    """
    Generate Python code from a flow definition.
    
    Args:
        flow: The flow definition
        
    Returns:
        str: The generated Python code
    """
    # Find the required nodes
    connection_node = next((node for node in flow.nodes if node.type == "hanaConnection"), None)
    embedding_node = next((node for node in flow.nodes if node.type == "embedding"), None)
    vectorstore_node = next((node for node in flow.nodes if node.type == "vectorStore"), None)
    query_node = next((node for node in flow.nodes if node.type == "query"), None)
    
    # Generate the code
    code = f"""'''
{flow.name}
{flow.description}

Auto-generated by SAP HANA LangChain Visual Developer
'''

from langchain_hana import HanaVectorStore
from langchain_hana.embeddings import HanaInternalEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import HuggingFaceEmbeddings
from hdbcli import dbapi
import os

# Create connection to SAP HANA Cloud
connection = dbapi.connect(
    address="{connection_node.data.params.host if connection_node else 'your-hana-host.hanacloud.ondemand.com'}",
    port={connection_node.data.params.port if connection_node else 443},
    user="{connection_node.data.params.user if connection_node else 'DBADMIN'}",
    password=os.environ.get("HANA_PASSWORD", "YourPasswordHere"),
    encrypt=True,
    sslValidateCertificate=True
)

# Initialize embedding model
"""
    
    if embedding_node and embedding_node.data.params.get("useGPU", False):
        code += f"""# GPU-accelerated embeddings
model_name = "{embedding_node.data.params.get('model', 'all-MiniLM-L6-v2')}"
model_kwargs = {{"device": "cuda"}}

embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs
)
"""
    elif embedding_node:
        code += f"""# CPU embeddings
model_name = "{embedding_node.data.params.get('model', 'all-MiniLM-L6-v2')}"
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name
)
"""
    else:
        code += """# Use SAP HANA's internal embeddings
embedding_model = HanaInternalEmbeddings(
    internal_embedding_model_id="SAP_NEB.20240715"
)
"""
    
    code += f"""
# Initialize HANA Vector Store
vectorstore = HanaVectorStore(
    connection=connection,
    embedding=embedding_model,
    table_name="{vectorstore_node.data.params.get('tableName', 'EMBEDDINGS') if vectorstore_node else 'EMBEDDINGS'}"
)

# Execute query
query_text = "{query_node.data.params.get('queryText', 'What is SAP HANA Cloud?') if query_node else 'What is SAP HANA Cloud?'}"
"""
    
    if query_node and query_node.data.params.get("useMMR", False):
        code += f"""# Using Maximal Marginal Relevance for diversity
results = vectorstore.max_marginal_relevance_search(
    query=query_text,
    k={query_node.data.params.get('k', 4) if query_node else 4},
    fetch_k={query_node.data.params.get('k', 4) * 5 if query_node else 20},
    lambda_mult=0.5  # Balance between relevance and diversity
)
"""
    else:
        code += f"""# Standard similarity search
results = vectorstore.similarity_search(
    query=query_text,
    k={query_node.data.params.get('k', 4) if query_node else 4}
)
"""
    
    code += """
# Process results
print(f"Query: {query_text}")
print(f"Found {len(results)} results\\n")

for i, result in enumerate(results):
    print(f"Result {i+1}:")
    print(f"Content: {result.page_content[:150]}..." if len(result.page_content) > 150 else f"Content: {result.page_content}")
    print(f"Metadata: {result.metadata}")
    print()
"""
    
    return code