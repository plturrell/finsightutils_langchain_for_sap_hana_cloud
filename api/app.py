"""FastAPI application for SAP HANA Cloud vector store integration.

This module sets up the FastAPI application and includes all routes
for different deployment platforms. It supports multiple deployment options:
- NVIDIA LaunchPad (GPU-accelerated)
- Together.ai (managed AI platform)
- SAP BTP (enterprise platform)
- Vercel (serverless)
"""

import os
import logging
import time
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from hdbcli import dbapi
from langchain_core.embeddings import Embeddings
from langchain_hana import HanaInternalEmbeddings

from config import config
from database import get_db_connection, DatabaseConnectionError
from models import (
    APIResponse,
    AddTextsRequest,
    DeleteRequest,
    MMRQueryRequest,
    MMRVectorQueryRequest,
    QueryRequest,
    QueryResponse,
    VectorQueryRequest,
)
from services import VectorStoreService
from embeddings import GPUAcceleratedEmbeddings, GPUHybridEmbeddings
from embeddings_multi_gpu import MultiGPUEmbeddings, MultiGPUHybridEmbeddings
from embeddings_tensorrt import TensorRTEmbeddings, TensorRTHybridEmbeddings
import gpu_utils
from tensorrt_utils import TENSORRT_AVAILABLE
import benchmark_api
import developer_api
from health import router as health_router
from error_utils import (
    create_context_aware_error,
    handle_vector_search_error,
    handle_data_insertion_error,
)

# Import version information
from version import VERSION, get_version_info

# Detect platform
PLATFORM = os.environ.get("PLATFORM", "unknown")
PLATFORM_SUPPORTS_GPU = os.environ.get("PLATFORM_SUPPORTS_GPU", "false").lower() == "true"

# Log version information
version_info = get_version_info()
logger.info(f"Starting API version {VERSION} (Build: {version_info.get('build_id', 'development')})")

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.api.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="SAP HANA Cloud Vector Store API",
    description="""
    API for SAP HANA Cloud vector store operations with NVIDIA GPU acceleration.
    
    This API provides endpoints for:
    
    * **Embedding Generation**: Create embeddings using GPU-accelerated models
    * **Vector Storage**: Store and retrieve vector embeddings in SAP HANA Cloud
    * **Similarity Search**: Perform semantic search using embeddings
    * **MMR Search**: Maximal Marginal Relevance search for diverse results
    * **Health Monitoring**: Check system health and GPU status
    
    The API supports TensorRT optimization and multi-GPU acceleration for high-performance embedding generation.
    """,
    version=VERSION,
    contact={
        "name": "SAP LangChain Integration Team",
        "url": "https://github.com/sap/langchain-integration-for-sap-hana-cloud",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
    openapi_tags=[
        {"name": "Vector Store", "description": "Operations for managing vector embeddings"},
        {"name": "Query", "description": "Vector similarity search operations"},
        {"name": "Health", "description": "Health check and monitoring endpoints"},
        {"name": "GPU", "description": "GPU information and acceleration settings"},
        {"name": "Benchmarks", "description": "Performance benchmarking endpoints"},
        {"name": "Developer", "description": "Developer-specific operations and debugging"},
    ],
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware with secure configuration
if config.cors.enable_cors:
    # Log CORS configuration
    if "*" in config.cors.allowed_origins:
        logger.warning(
            "CORS is configured to allow all origins (*). "
            "This is not recommended for production environments. "
            "Set CORS_ORIGINS environment variable to restrict allowed origins."
        )
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allowed_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allowed_methods,
        allow_headers=config.cors.allowed_headers,
    )
    logger.info(f"CORS middleware enabled with {len(config.cors.allowed_origins)} allowed origins")

# Request processing time middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Platform"] = PLATFORM
        response.headers["X-API-Version"] = VERSION
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        process_time = time.time() - start_time
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "context": {
                    "operation": "api_request",
                    "suggestion": "Please check your input parameters and try again",
                    "process_time": process_time
                }
            },
            headers={
                "X-Process-Time": str(process_time),
                "X-Platform": PLATFORM,
                "X-API-Version": VERSION
            }
        )

# Include routers
app.include_router(benchmark_api.router)
app.include_router(developer_api.router)
app.include_router(health_router)


# Get embeddings model
def get_embeddings() -> Embeddings:
    """
    Get embeddings model.
    
    Returns:
        Embeddings: Embeddings model.
    """
    if hasattr(config, 'gpu') and config.gpu.enabled:
        # Check if we have multiple GPUs
        multi_gpu = len(gpu_utils.get_gpu_info().get("devices", [])) > 1
        # Check if TensorRT is enabled in config
        use_tensorrt = getattr(config.gpu, 'use_tensorrt', True) and TENSORRT_AVAILABLE
        
        # Use GPU-accelerated embeddings
        if config.gpu.use_internal_embeddings:
            # Use hybrid embeddings (GPU-accelerated with HANA internal fallback)
            if use_tensorrt:
                logger.info("Using TensorRT hybrid embeddings with internal HANA embeddings")
                return TensorRTHybridEmbeddings(
                    internal_embedding_model_id=config.gpu.internal_embedding_model_id,
                    external_model_name=config.gpu.embedding_model,
                    use_internal=True,
                    device="cuda" if gpu_utils.is_torch_available() else "cpu",
                    batch_size=config.gpu.batch_size,
                    precision=getattr(config.gpu, 'tensorrt_precision', 'fp16'),
                )
            elif multi_gpu:
                logger.info("Using Multi-GPU hybrid embeddings with internal HANA embeddings")
                return MultiGPUHybridEmbeddings(
                    internal_embedding_model_id=config.gpu.internal_embedding_model_id,
                    external_model_name=config.gpu.embedding_model,
                    use_internal=True,
                    batch_size=config.gpu.batch_size,
                )
            else:
                logger.info("Using GPU hybrid embeddings with internal HANA embeddings")
                return GPUHybridEmbeddings(
                    internal_embedding_model_id=config.gpu.internal_embedding_model_id,
                    external_model_name=config.gpu.embedding_model,
                    use_internal=True,
                    device="cuda" if gpu_utils.is_torch_available() else "cpu",
                    batch_size=config.gpu.batch_size,
                )
        else:
            # Use GPU-accelerated embeddings only
            if use_tensorrt:
                logger.info("Using TensorRT-optimized embeddings")
                return TensorRTEmbeddings(
                    model_name=config.gpu.embedding_model,
                    device="cuda" if gpu_utils.is_torch_available() else "cpu",
                    batch_size=config.gpu.batch_size,
                    precision=getattr(config.gpu, 'tensorrt_precision', 'fp16'),
                )
            elif multi_gpu:
                logger.info("Using Multi-GPU accelerated embeddings")
                return MultiGPUEmbeddings(
                    model_name=config.gpu.embedding_model,
                    batch_size=config.gpu.batch_size,
                )
            else:
                logger.info("Using GPU-accelerated embeddings")
                return GPUAcceleratedEmbeddings(
                    model_name=config.gpu.embedding_model,
                    device="cuda" if gpu_utils.is_torch_available() else "cpu",
                    batch_size=config.gpu.batch_size,
                )
    else:
        # Use standard HANA internal embeddings
        logger.info("Using standard HANA internal embeddings")
        return HanaInternalEmbeddings(
            internal_embedding_model_id=config.gpu.internal_embedding_model_id
            if hasattr(config, 'gpu')
            else "SAP_NEB.20240715"
        )


# Dependency to get vector store service
def get_vectorstore_service(
    connection: dbapi.Connection = Depends(get_db_connection),
    embeddings: Embeddings = Depends(get_embeddings),
    table_name: Optional[str] = None,
) -> VectorStoreService:
    """
    Get vector store service.
    
    Args:
        connection: Database connection.
        embeddings: Embeddings model.
        table_name: Optional table name.
        
    Returns:
        VectorStoreService: Vector store service.
    """
    return VectorStoreService(
        connection=connection,
        embedding=embeddings,
        table_name=table_name,
    )


# Exception handler for database connection errors
@app.exception_handler(DatabaseConnectionError)
async def database_connection_exception_handler(
    request: Request, exc: DatabaseConnectionError
):
    """Handle database connection errors."""
    return JSONResponse(
        status_code=500,
        content={"message": str(exc)},
    )


@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint providing API information.
    
    Returns:
        dict: Basic API information
    """
    return {
        "message": "SAP HANA Cloud Vector Store API with NVIDIA GPU Acceleration",
        "version": VERSION,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }


# Legacy health check endpoint - redirects to new health router
@app.get("/health", tags=["Health"])
async def legacy_health_check():
    """
    Legacy health check endpoint that redirects to the new comprehensive health check.
    
    Returns:
        dict: Health status and information about available health endpoints
    """
    return {
        "status": "ok",
        "message": "For more detailed health information, use the new health endpoints:",
        "endpoints": {
            "basic_health": "/api/health",
            "system_info": "/api/health/system",
            "database_status": "/api/health/database",
            "gpu_status": "/api/health/gpu",
            "comprehensive": "/api/health/complete",
            "prometheus_metrics": "/api/health/metrics"
        },
        "platform": PLATFORM,
        "version": VERSION
    }


@app.post("/texts", response_model=APIResponse, tags=["Vector Store"])
async def add_texts(
    request: AddTextsRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Add texts to the vector store.
    
    This endpoint converts the provided texts to embeddings using the configured embedding model
    and stores them in the SAP HANA Cloud vector store. If GPU acceleration is enabled,
    the embedding generation will be accelerated using available NVIDIA GPUs.
    
    Parameters:
    - **texts**: List of text strings to convert to embeddings and store
    - **metadatas**: Optional list of metadata dictionaries for each text
    - **table_name**: Optional custom table name for storing the embeddings
    
    Returns:
        APIResponse: Success status and message
    
    Raises:
        HTTPException: If there's an error during text processing or database insertion
    """
    try:
        service.add_texts(request.texts, request.metadatas)
        return APIResponse(
            success=True,
            message=f"Successfully added {len(request.texts)} texts to the vector store",
        )
    except Exception as e:
        logger.error(f"Failed to add texts: {str(e)}")
        insertion_info = {
            "text_count": len(request.texts),
            "has_metadata": request.metadatas is not None,
            "table_name": request.table_name or config.vectorstore.table_name,
        }
        raise handle_data_insertion_error(e, insertion_info)


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Query the vector store for similar texts.
    
    This endpoint performs similarity search based on the input query. It converts
    the query to an embedding vector using the configured model and finds the most
    similar documents in the vector store.
    
    Parameters:
    - **query**: Text query to search for
    - **k**: Number of results to return (default: 4)
    - **filter**: Optional metadata filter to narrow down search results
    - **table_name**: Optional custom table name for the query
    
    Returns:
        QueryResponse: Object containing search results with document content and metadata
    
    Raises:
        HTTPException: If there's an error during query processing or search execution
    """
    try:
        results = service.similarity_search(
            query=request.query,
            k=request.k,
            filter=request.filter,
        )
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        query_info = {
            "query": request.query,
            "k": request.k,
            "has_filter": request.filter is not None,
            "table_name": request.table_name or config.vectorstore.table_name,
        }
        raise handle_vector_search_error(e, query_info)


@app.post("/query/vector", response_model=QueryResponse, tags=["Query"])
async def query_by_vector(
    request: VectorQueryRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Query the vector store using a pre-computed embedding vector.
    
    This endpoint allows direct similarity search using a pre-computed embedding vector.
    It's useful when you've already generated embeddings elsewhere and want to skip
    the embedding generation step.
    
    Parameters:
    - **embedding**: Pre-computed embedding vector (list of floats)
    - **k**: Number of results to return (default: 4)
    - **filter**: Optional metadata filter to narrow down search results
    - **table_name**: Optional custom table name for the query
    
    Returns:
        QueryResponse: Object containing search results with document content and metadata
    
    Raises:
        HTTPException: If there's an error during the vector search operation
    """
    try:
        results = service.similarity_search_by_vector(
            embedding=request.embedding,
            k=request.k,
            filter=request.filter,
        )
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(f"Vector query failed: {str(e)}")
        query_info = {
            "embedding_dimensions": len(request.embedding),
            "k": request.k,
            "has_filter": request.filter is not None,
            "table_name": request.table_name or config.vectorstore.table_name,
        }
        raise handle_vector_search_error(e, query_info)


@app.post("/query/mmr", response_model=QueryResponse, tags=["Query"])
async def mmr_query(
    request: MMRQueryRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Perform max marginal relevance (MMR) search for diverse results.
    
    This endpoint performs MMR search to return semantically similar but diverse results.
    MMR balances between relevance to the query and diversity among the results.
    The GPU-accelerated implementation provides significantly faster processing.
    
    Parameters:
    - **query**: Text query to search for
    - **k**: Number of results to return (default: 4)
    - **fetch_k**: Number of documents to consider before reranking (default: 20)
    - **lambda_mult**: Balance factor between relevance and diversity (0-1, default: 0.5)
    - **filter**: Optional metadata filter to narrow down search results
    - **table_name**: Optional custom table name for the query
    
    Returns:
        QueryResponse: Object containing diverse search results with document content and metadata
    
    Raises:
        HTTPException: If there's an error during query processing or search execution
    """
    try:
        results = service.max_marginal_relevance_search(
            query=request.query,
            k=request.k,
            fetch_k=request.fetch_k,
            lambda_mult=request.lambda_mult,
            filter=request.filter,
        )
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(f"MMR query failed: {str(e)}")
        query_info = {
            "query": request.query,
            "k": request.k,
            "fetch_k": request.fetch_k,
            "lambda_mult": request.lambda_mult,
            "has_filter": request.filter is not None,
            "table_name": request.table_name or config.vectorstore.table_name,
            "mmr_enabled": True,
        }
        raise handle_vector_search_error(e, query_info)


@app.post("/query/mmr/vector", response_model=QueryResponse, tags=["Query"])
async def mmr_query_by_vector(
    request: MMRVectorQueryRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Perform max marginal relevance (MMR) search using a pre-computed embedding vector.
    
    This endpoint performs MMR search using a pre-computed embedding vector to find
    semantically similar but diverse results. It's useful when you've already generated
    the embedding elsewhere and want to skip the embedding generation step.
    The GPU-accelerated implementation provides significantly faster processing.
    
    Parameters:
    - **embedding**: Pre-computed embedding vector (list of floats)
    - **k**: Number of results to return (default: 4)
    - **fetch_k**: Number of documents to consider before reranking (default: 20)
    - **lambda_mult**: Balance factor between relevance and diversity (0-1, default: 0.5)
    - **filter**: Optional metadata filter to narrow down search results
    - **table_name**: Optional custom table name for the query
    
    Returns:
        QueryResponse: Object containing diverse search results with document content and metadata
    
    Raises:
        HTTPException: If there's an error during query processing or search execution
    """
    try:
        results = service.max_marginal_relevance_search_by_vector(
            embedding=request.embedding,
            k=request.k,
            fetch_k=request.fetch_k,
            lambda_mult=request.lambda_mult,
            filter=request.filter,
        )
        return QueryResponse(results=results)
    except Exception as e:
        logger.error(f"MMR vector query failed: {str(e)}")
        query_info = {
            "embedding_dimensions": len(request.embedding),
            "k": request.k,
            "fetch_k": request.fetch_k,
            "lambda_mult": request.lambda_mult,
            "has_filter": request.filter is not None,
            "table_name": request.table_name or config.vectorstore.table_name,
            "mmr_enabled": True,
        }
        raise handle_vector_search_error(e, query_info)


@app.post("/delete", response_model=APIResponse, tags=["Vector Store"])
async def delete(
    request: DeleteRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Delete documents from the vector store based on metadata filters.
    
    This endpoint allows selective deletion of documents from the vector store
    by specifying metadata filters. For example, you can delete all documents
    with a specific source or category.
    
    Parameters:
    - **filter**: Metadata filter dictionary to select documents for deletion
    - **table_name**: Optional custom table name to delete from
    
    Returns:
        APIResponse: Success status and message
    
    Raises:
        HTTPException: If there's an error during the deletion process
    
    Example:
        ```json
        {
          "filter": {"source": "database-migration-docs"}
        }
        ```
    """
    try:
        result = service.delete(filter=request.filter)
        return APIResponse(
            success=result,
            message="Successfully deleted documents from the vector store",
        )
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        additional_context = {
            "filter": request.filter,
            "table_name": request.table_name or config.vectorstore.table_name,
            "operation": "delete",
        }
        raise create_context_aware_error(
            str(e), 
            "data_insertion", 
            additional_context=additional_context
        )


# Embeddings generation endpoint
@app.post("/embeddings", tags=["Vector Store"])
async def generate_embeddings(
    request: dict,
    embeddings: Embeddings = Depends(get_embeddings),
):
    """
    Generate embeddings for the provided texts without storing them.
    
    This endpoint converts the provided texts to embeddings using the configured model
    with GPU acceleration if available. The embeddings are returned directly without
    storing them in the database, which is useful for client-side operations or testing.
    
    Parameters:
    - **texts**: List of text strings to convert to embeddings
    - **model**: Optional model name to use for embedding generation
    
    Returns:
        dict: Dictionary containing the generated embeddings
    
    Raises:
        HTTPException: If there's an error during the embedding generation process
    
    Example request:
        ```json
        {
          "texts": ["This is a sample text", "Another example sentence"],
          "model": "all-MiniLM-L6-v2"
        }
        ```
    """
    try:
        if not request.get("texts"):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "validation_error",
                    "message": "No texts provided for embedding generation",
                    "context": {
                        "operation": "embedding_generation",
                        "suggestion": "Please provide at least one text in the 'texts' field",
                    }
                }
            )
            
        texts = request.get("texts", [])
        
        # Log the request size for performance monitoring
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings using the configured model with GPU acceleration
        start_time = time.time()
        result = embeddings.embed_documents(texts)
        process_time = time.time() - start_time
        
        # Return the embeddings with timing information
        return {
            "embeddings": result,
            "count": len(result),
            "dimensions": len(result[0]) if result else 0,
            "process_time_seconds": process_time,
            "texts_per_second": len(texts) / process_time if process_time > 0 else 0,
        }
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        context = {
            "text_count": len(request.get("texts", [])),
            "model_requested": request.get("model", "default"),
            "operation": "embedding_generation"
        }
        if isinstance(e, HTTPException):
            raise e
        raise create_context_aware_error(str(e), "embedding_generation", context)

# GPU Information endpoint
@app.get("/gpu/info", response_model=Dict, tags=["GPU"])
async def gpu_info():
    """
    Get detailed GPU information and acceleration capabilities.
    
    This endpoint provides information about the available NVIDIA GPUs, including:
    - GPU availability status
    - Number of GPUs
    - GPU models and specs
    - Memory information
    - CUDA version
    - TensorRT availability
    - Library support (CuPy, PyTorch)
    
    Returns:
        Dict: Comprehensive GPU information
    
    Example response:
        ```json
        {
          "gpu_available": true,
          "cupy_available": true,
          "torch_available": true,
          "device_count": 1,
          "devices": [
            {
              "name": "NVIDIA T4",
              "memory_total": 16384,
              "memory_free": 15360,
              "compute_capability": "7.5"
            }
          ],
          "cuda_version": "11.7"
        }
        ```
    """
    is_available = gpu_utils.is_gpu_available()
    info = {
        "gpu_available": is_available,
        "cupy_available": gpu_utils.is_cupy_available(),
        "torch_available": gpu_utils.is_torch_available(),
    }
    
    if is_available:
        info.update(gpu_utils.get_gpu_info())
    
    return info


# OpenAPI schema endpoint
@app.get("/openapi.json", tags=["Developer"])
async def get_openapi_schema():
    """
    Get the complete OpenAPI schema for the API.
    
    This endpoint returns the complete OpenAPI specification in JSON format.
    This can be used for generating client libraries or documentation.
    
    Returns:
        dict: The complete OpenAPI specification
    """
    return app.openapi()

# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler.
    
    Cleans up resources when the application shuts down, including:
    - Closing all database connections
    - Releasing GPU resources if applicable
    """
    from database import connection_pool
    
    # Close all database connections
    logger.info("Shutting down application, cleaning up resources...")
    
    try:
        connection_pool.close_all()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connections: {str(e)}")
    
    # Release GPU resources if applicable
    try:
        if gpu_utils.is_torch_available():
            import torch
            torch.cuda.empty_cache()
            logger.info("GPU memory cache cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU resources: {str(e)}")
    
    logger.info("Application shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Log GPU information
    if gpu_utils.is_gpu_available():
        gpu_info = gpu_utils.get_gpu_info()
        logger.info(f"NVIDIA GPU acceleration is available with {gpu_info.get('device_count', 0)} device(s)")
        for i, device in enumerate(gpu_info.get("devices", [])):
            logger.info(f"GPU {i}: {device.get('name')} with {device.get('memory_total') / (1024**2):.2f} MB memory")
    else:
        logger.info("NVIDIA GPU acceleration is not available, using CPU")
    
    uvicorn.run(
        "app:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
    )