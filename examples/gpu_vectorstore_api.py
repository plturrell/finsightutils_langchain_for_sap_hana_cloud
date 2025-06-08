#!/usr/bin/env python
"""
GPU-Accelerated Vector Store API Example

This example demonstrates how to create a simple FastAPI service 
that uses the GPU-accelerated vector store for SAP HANA Cloud.

Usage:
    uvicorn gpu_vectorstore_api:app --host 0.0.0.0 --port 8000

Requirements:
    - SAP HANA Cloud instance with credentials
    - NVIDIA GPU with CUDA support (for GPU acceleration)
    - Python packages: langchain, langchain_hana, sentence-transformers, fastapi, uvicorn
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from pydantic import BaseModel

from hdbcli import dbapi
from langchain_hana.gpu.hana_gpu_vectorstore import HanaGPUVectorStore
from langchain_hana.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="GPU-Accelerated Vector Store API")

# Global variables
vectorstore = None
embedding_model = None
connection = None

# Environment variables for database connection
HANA_HOST = os.environ.get("HANA_HOST")
HANA_PORT = int(os.environ.get("HANA_PORT", "443"))
HANA_USER = os.environ.get("HANA_USER")
HANA_PASSWORD = os.environ.get("HANA_PASSWORD")
TABLE_NAME = os.environ.get("HANA_TABLE", "VECTORSTORE_API_DEMO")


# Pydantic models for API
class Document(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}


class SearchQuery(BaseModel):
    query: str
    k: int = 4
    filter: Optional[Dict[str, Any]] = None
    fetch_k: Optional[int] = None
    lambda_mult: Optional[float] = None
    use_mmr: bool = False


class PerformanceStats(BaseModel):
    enable: bool = True


# Helper function to initialize components
def initialize():
    """Initialize the database connection, embedding model, and vector store."""
    global connection, embedding_model, vectorstore

    # Check if already initialized
    if vectorstore is not None:
        return

    # Validate required environment variables
    if not all([HANA_HOST, HANA_USER, HANA_PASSWORD]):
        raise ValueError(
            "Missing required environment variables. Please set HANA_HOST, HANA_USER, and HANA_PASSWORD."
        )

    # Connect to database
    connection = dbapi.connect(
        address=HANA_HOST,
        port=HANA_PORT,
        user=HANA_USER,
        password=HANA_PASSWORD,
        encrypt=True,
        sslValidateCertificate=False,
    )

    # Initialize embedding model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Initialize vectorstore with GPU acceleration
    vectorstore = HanaGPUVectorStore(
        connection=connection,
        embedding=embedding_model,
        table_name=TABLE_NAME,
        distance_strategy=DistanceStrategy.COSINE,
        gpu_acceleration_config={
            "use_gpu_batching": True,
            "embedding_batch_size": 32,
            "db_batch_size": 500,
            "build_index": True,
            "index_type": "hnsw",
            "rebuild_index_on_add": False,
            "prefetch_size": 10000,
        }
    )

    # Enable performance profiling
    vectorstore.enable_profiling(True)

    logger.info("Initialized database connection, embedding model, and vector store")


# Dependency to get vector store
def get_vectorstore():
    """Get initialized vector store as a dependency."""
    if vectorstore is None:
        initialize()
    return vectorstore


# API routes
@app.post("/documents", status_code=201)
async def add_documents(
    documents: List[Document], 
    background_tasks: BackgroundTasks,
    vectorstore: HanaGPUVectorStore = Depends(get_vectorstore)
):
    """Add documents to the vector store."""
    # Extract texts and metadata
    texts = [doc.text for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # For smaller document sets, add synchronously
    if len(texts) <= 10:
        vectorstore.add_texts(texts, metadatas)
        return {"message": f"Added {len(texts)} documents to the vector store"}
    
    # For larger document sets, add asynchronously in the background
    background_tasks.add_task(vectorstore.aadd_texts, texts, metadatas)
    return {"message": f"Adding {len(texts)} documents to the vector store in the background"}


@app.post("/search")
async def search(
    query: SearchQuery,
    vectorstore: HanaGPUVectorStore = Depends(get_vectorstore)
):
    """Search the vector store for documents similar to the query."""
    try:
        # Choose search method based on parameters
        if query.use_mmr:
            # Use MMR search for diverse results
            fetch_k = query.fetch_k or min(50, max(query.k * 4, 20))
            lambda_mult = query.lambda_mult or 0.5
            
            results = await vectorstore.amax_marginal_relevance_search(
                query=query.query,
                k=query.k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=query.filter
            )
        else:
            # Use standard similarity search
            results = await vectorstore.asimilarity_search(
                query=query.query,
                k=query.k,
                filter=query.filter
            )
        
        # Convert results to API format
        documents = []
        for doc in results:
            documents.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {"documents": documents}
        
    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


@app.post("/upsert")
async def upsert_documents(
    documents: List[Document],
    filter: Dict[str, Any],
    vectorstore: HanaGPUVectorStore = Depends(get_vectorstore)
):
    """Update or insert documents in the vector store."""
    # Extract texts and metadata
    texts = [doc.text for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    try:
        await vectorstore.aupsert_texts(texts, metadatas, filter=filter)
        return {"message": f"Upserted {len(texts)} documents with filter {filter}"}
    except Exception as e:
        logger.error(f"Error during upsert: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upsert error: {str(e)}")


@app.delete("/documents")
async def delete_documents(
    filter: Dict[str, Any],
    vectorstore: HanaGPUVectorStore = Depends(get_vectorstore)
):
    """Delete documents from the vector store."""
    try:
        await vectorstore.adelete(filter=filter)
        return {"message": f"Deleted documents with filter {filter}"}
    except Exception as e:
        logger.error(f"Error during delete: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Delete error: {str(e)}")


@app.get("/performance")
def get_performance_stats(vectorstore: HanaGPUVectorStore = Depends(get_vectorstore)):
    """Get performance statistics for the vector store."""
    try:
        stats = vectorstore.get_performance_stats()
        gpu_info = vectorstore.get_gpu_info()
        return {
            "performance_stats": stats,
            "gpu_info": gpu_info
        }
    except Exception as e:
        logger.error(f"Error getting performance stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting performance stats: {str(e)}")


@app.post("/performance")
def set_performance_profiling(
    settings: PerformanceStats,
    vectorstore: HanaGPUVectorStore = Depends(get_vectorstore)
):
    """Enable or disable performance profiling."""
    try:
        vectorstore.enable_profiling(settings.enable)
        return {"message": f"Performance profiling {'enabled' if settings.enable else 'disabled'}"}
    except Exception as e:
        logger.error(f"Error setting performance profiling: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error setting performance profiling: {str(e)}")


@app.post("/reset-stats")
def reset_performance_stats(vectorstore: HanaGPUVectorStore = Depends(get_vectorstore)):
    """Reset performance statistics."""
    try:
        vectorstore.reset_performance_stats()
        return {"message": "Performance statistics reset"}
    except Exception as e:
        logger.error(f"Error resetting performance stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error resetting performance stats: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "gpu_available": vectorstore is not None and vectorstore.vector_engine.gpu_available}


# Startup and shutdown events
@app.on_event("startup")
def startup_event():
    """Initialize components on startup."""
    try:
        initialize()
        logger.info("API started successfully")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")


@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on shutdown."""
    global connection, vectorstore
    
    try:
        if vectorstore:
            vectorstore.release_resources()
        
        if connection:
            connection.close()
            
        logger.info("Resources released successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)