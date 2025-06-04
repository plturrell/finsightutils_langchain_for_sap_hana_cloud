"""FastAPI application for SAP HANA Cloud vector store integration."""

import logging
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

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.api.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="SAP HANA Cloud Vector Store API",
    description="API for SAP HANA Cloud vector store operations with GPU acceleration",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(benchmark_api.router)
app.include_router(developer_api.router)


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


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "SAP HANA Cloud Vector Store API with NVIDIA GPU Acceleration"}


@app.get("/health")
async def health_check(connection: dbapi.Connection = Depends(get_db_connection)):
    """Health check endpoint."""
    try:
        # Simple query to check database connection
        cursor = connection.cursor()
        cursor.execute("SELECT 1 FROM DUMMY")
        cursor.close()
        
        # Check GPU status
        gpu_status = "available" if gpu_utils.is_gpu_available() else "unavailable"
        gpu_info = gpu_utils.get_gpu_info()
        gpu_count = gpu_info.get("device_count", 0)
        
        return {
            "status": "healthy",
            "database": "connected",
            "gpu_acceleration": gpu_status,
            "gpu_count": gpu_count,
            "gpu_info": gpu_info.get("devices", []),
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@app.post("/texts", response_model=APIResponse)
async def add_texts(
    request: AddTextsRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Add texts to the vector store.
    
    Args:
        request: Request model with texts and optional metadata.
        service: Vector store service.
        
    Returns:
        APIResponse: API response.
    """
    try:
        service.add_texts(request.texts, request.metadatas)
        return APIResponse(
            success=True,
            message=f"Successfully added {len(request.texts)} texts to the vector store",
        )
    except Exception as e:
        logger.error(f"Failed to add texts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to add texts: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Query the vector store.
    
    Args:
        request: Query request.
        service: Vector store service.
        
    Returns:
        QueryResponse: Query response.
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
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/query/vector", response_model=QueryResponse)
async def query_by_vector(
    request: VectorQueryRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Query the vector store by vector.
    
    Args:
        request: Vector query request.
        service: Vector store service.
        
    Returns:
        QueryResponse: Query response.
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
        raise HTTPException(status_code=500, detail=f"Vector query failed: {str(e)}")


@app.post("/query/mmr", response_model=QueryResponse)
async def mmr_query(
    request: MMRQueryRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Perform max marginal relevance search using GPU acceleration if available.
    
    Args:
        request: MMR query request.
        service: Vector store service.
        
    Returns:
        QueryResponse: Query response.
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
        raise HTTPException(status_code=500, detail=f"MMR query failed: {str(e)}")


@app.post("/query/mmr/vector", response_model=QueryResponse)
async def mmr_query_by_vector(
    request: MMRVectorQueryRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Perform max marginal relevance search by vector using GPU acceleration if available.
    
    Args:
        request: MMR vector query request.
        service: Vector store service.
        
    Returns:
        QueryResponse: Query response.
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
        raise HTTPException(status_code=500, detail=f"MMR vector query failed: {str(e)}")


@app.post("/delete", response_model=APIResponse)
async def delete(
    request: DeleteRequest,
    service: VectorStoreService = Depends(get_vectorstore_service),
):
    """
    Delete documents from the vector store.
    
    Args:
        request: Delete request.
        service: Vector store service.
        
    Returns:
        APIResponse: API response.
    """
    try:
        result = service.delete(filter=request.filter)
        return APIResponse(
            success=result,
            message="Successfully deleted documents from the vector store",
        )
    except Exception as e:
        logger.error(f"Delete failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


# GPU Information endpoint
@app.get("/gpu/info", response_model=Dict)
async def gpu_info():
    """
    Get GPU information.
    
    Returns:
        Dict: GPU information.
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