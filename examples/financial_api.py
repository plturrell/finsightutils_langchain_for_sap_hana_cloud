#!/usr/bin/env python3
"""
Financial Embeddings API for SAP HANA Cloud

This module provides a FastAPI-based service for integrating financial domain-specific 
embeddings with SAP HANA Cloud. It offers endpoints for:

1. Embedding generation with financial domain models
2. Document storage and retrieval
3. Question answering with RAG
4. Metrics collection and visualization

Designed to be deployed as a microservice with Docker for production environments.

Usage:
    uvicorn financial_api:app --host 0.0.0.0 --port 8000
"""

import os
import json
import time
import logging
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Body, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import LangChain and SAP HANA integration components
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# Import our financial embeddings components
from langchain_hana.connection import create_connection, test_connection, close_connection
from langchain_hana.vectorstore import HanaVectorStore
from langchain_hana.financial import (
    FinE5Embeddings,
    FinE5TensorRTEmbeddings,
    create_financial_embeddings,
    FINANCIAL_EMBEDDING_MODELS
)

# Import OpenAI for completion (if available)
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import visualization component
from financial_metrics_visualization import FinancialMetricsVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("financial_api")

# Initialize FastAPI app
app = FastAPI(
    title="Financial Embeddings API for SAP HANA Cloud",
    description="API for integrating financial domain-specific embeddings with SAP HANA Cloud",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Pydantic models for request/response validation
# =============================================================================

class ConnectionConfig(BaseModel):
    """SAP HANA Cloud connection configuration."""
    host: str = Field(..., description="SAP HANA Cloud host")
    port: int = Field(443, description="SAP HANA Cloud port")
    user: str = Field(..., description="SAP HANA Cloud username")
    password: str = Field(..., description="SAP HANA Cloud password")
    encrypt: bool = Field(True, description="Whether to use encryption")
    ssl_validate_certificate: bool = Field(False, description="Whether to validate SSL certificate")


class EmbeddingConfig(BaseModel):
    """Configuration for financial embeddings."""
    model_type: str = Field("default", description="Financial embedding model type")
    use_gpu: bool = Field(True, description="Whether to use GPU acceleration if available")
    use_tensorrt: bool = Field(False, description="Whether to use TensorRT acceleration")
    add_financial_prefix: bool = Field(True, description="Whether to add financial context prefix")
    financial_prefix_type: str = Field("general", description="Type of financial prefix")
    cache_enabled: bool = Field(True, description="Whether to enable embedding cache")


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    table_name: str = Field(..., description="Table name for vector store")
    create_table: bool = Field(True, description="Whether to create the table if it doesn't exist")
    create_index: bool = Field(True, description="Whether to create HNSW index")


class DocumentItem(BaseModel):
    """Document for storage in the vector store."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")


class QueryRequest(BaseModel):
    """Request for querying the vector store."""
    query: str = Field(..., description="Query text")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter criteria")
    k: int = Field(3, description="Number of results to return")
    fetch_k: Optional[int] = Field(None, description="Number of results to fetch before applying MMR")
    lambda_mult: Optional[float] = Field(None, description="Diversity factor for MMR (0.0 to 1.0)")


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""
    texts: List[str] = Field(..., description="List of texts to embed")
    model_type: Optional[str] = Field("default", description="Financial embedding model type")


class QARequest(BaseModel):
    """Request for question answering."""
    question: str = Field(..., description="Question to answer")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter criteria")
    temperature: float = Field(0.1, description="Temperature for LLM")
    model: str = Field("gpt-3.5-turbo", description="LLM model to use")


class MetricsRequest(BaseModel):
    """Request for metrics visualization."""
    metrics_data: Dict[str, Any] = Field(..., description="Metrics data for visualization")
    model_name: str = Field("Financial Embeddings Model", description="Model name for visualization")
    output_format: str = Field("png", description="Output format (png, html, json)")

# =============================================================================
# Application state and dependencies
# =============================================================================

# Application state
class AppState:
    """Application state singleton."""
    connection = None
    embeddings_model = None
    vector_store = None
    llm = None
    rag_chain = None
    visualizer = None
    metrics = {
        "embeddings_generated": 0,
        "documents_stored": 0,
        "queries_processed": 0,
        "qa_requests": 0,
        "avg_query_time": 0,
        "total_query_time": 0,
    }
    config = {
        "connection": None,
        "embeddings": None,
        "vector_store": None,
    }


app_state = AppState()


# Dependencies
async def get_connection():
    """Get or create SAP HANA Cloud connection."""
    if app_state.connection is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection not initialized. Call /setup endpoint first."
        )
    return app_state.connection


async def get_embeddings():
    """Get embeddings model."""
    if app_state.embeddings_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings model not initialized. Call /setup endpoint first."
        )
    return app_state.embeddings_model


async def get_vector_store():
    """Get vector store."""
    if app_state.vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized. Call /setup endpoint first."
        )
    return app_state.vector_store


async def get_rag_chain():
    """Get RAG chain."""
    if app_state.rag_chain is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG chain not initialized. Call /setup/rag endpoint first."
        )
    return app_state.rag_chain


async def get_visualizer():
    """Get metrics visualizer."""
    if app_state.visualizer is None:
        # Initialize visualizer with temp directory
        output_dir = os.environ.get("METRICS_OUTPUT_DIR", "/tmp/financial_metrics")
        os.makedirs(output_dir, exist_ok=True)
        app_state.visualizer = FinancialMetricsVisualizer(output_dir=output_dir)
    return app_state.visualizer

# =============================================================================
# API Routes
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "Financial Embeddings API for SAP HANA Cloud",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            {"path": "/setup", "method": "POST", "description": "Initialize the API with SAP HANA connection and embeddings"},
            {"path": "/health", "method": "GET", "description": "Health check endpoint"},
            {"path": "/metrics", "method": "GET", "description": "Get usage metrics"},
            {"path": "/embeddings", "method": "POST", "description": "Generate embeddings for texts"},
            {"path": "/documents", "method": "POST", "description": "Add documents to vector store"},
            {"path": "/search", "method": "POST", "description": "Search for similar documents"},
            {"path": "/qa", "method": "POST", "description": "Answer questions using RAG"},
        ],
        "models_available": list(FINANCIAL_EMBEDDING_MODELS.keys()),
        "gpu_available": torch.cuda.is_available(),
    }


@app.get("/health", tags=["General"])
async def health_check():
    """Health check endpoint."""
    status_info = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connection_initialized": app_state.connection is not None,
        "embeddings_initialized": app_state.embeddings_model is not None,
        "vector_store_initialized": app_state.vector_store is not None,
        "rag_initialized": app_state.rag_chain is not None,
        "gpu_available": torch.cuda.is_available()
    }
    
    # Check connection if initialized
    if app_state.connection is not None:
        try:
            is_valid, info = test_connection(app_state.connection)
            status_info["connection_valid"] = is_valid
            if is_valid:
                status_info["connection_info"] = {
                    "version": info.get("version", "Unknown"),
                    "current_schema": info.get("current_schema", "Unknown")
                }
        except Exception as e:
            status_info["status"] = "degraded"
            status_info["connection_error"] = str(e)
    
    return status_info


@app.get("/metrics", tags=["General"])
async def get_metrics():
    """Get usage metrics."""
    # Add embedding model info if available
    embedding_info = {}
    if app_state.embeddings_model is not None:
        if hasattr(app_state.embeddings_model, "get_embedding_dimension"):
            embedding_info["dimension"] = app_state.embeddings_model.get_embedding_dimension()
        
        if hasattr(app_state.embeddings_model, "model_name"):
            embedding_info["model_name"] = app_state.embeddings_model.model_name
        
        if hasattr(app_state.embeddings_model, "get_performance_stats"):
            try:
                embedding_info["performance"] = app_state.embeddings_model.get_performance_stats()
            except:
                pass
    
    # Return combined metrics
    return {
        "usage": app_state.metrics,
        "embeddings": embedding_info,
        "config": {
            "connection": app_state.config.get("connection"),
            "embeddings": app_state.config.get("embeddings"),
            "vector_store": app_state.config.get("vector_store"),
        }
    }


@app.post("/setup", tags=["Configuration"])
async def setup(
    connection_config: ConnectionConfig,
    embedding_config: EmbeddingConfig,
    vector_store_config: VectorStoreConfig,
    background_tasks: BackgroundTasks
):
    """Initialize the API with SAP HANA connection and embeddings."""
    try:
        # Create connection
        logger.info(f"Creating connection to {connection_config.host}:{connection_config.port}")
        connection = create_connection(
            host=connection_config.host,
            port=connection_config.port,
            user=connection_config.user,
            password=connection_config.password,
            encrypt=connection_config.encrypt,
            sslValidateCertificate=connection_config.ssl_validate_certificate
        )
        
        # Test connection
        is_valid, info = test_connection(connection)
        if not is_valid:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to connect to SAP HANA: {info.get('error', 'Unknown error')}"
            )
        
        # Create embeddings model
        logger.info(f"Creating embeddings model with type: {embedding_config.model_type}")
        embeddings = create_financial_embeddings(
            model_type=embedding_config.model_type,
            use_gpu=embedding_config.use_gpu,
            use_tensorrt=embedding_config.use_tensorrt,
            add_financial_prefix=embedding_config.add_financial_prefix,
            financial_prefix_type=embedding_config.financial_prefix_type,
            enable_caching=embedding_config.cache_enabled
        )
        
        # Create vector store
        logger.info(f"Creating vector store with table: {vector_store_config.table_name}")
        vector_store = HanaVectorStore(
            connection=connection,
            embedding=embeddings,
            table_name=vector_store_config.table_name,
            create_table=vector_store_config.create_table
        )
        
        # Create HNSW index if requested
        if vector_store_config.create_index:
            background_tasks.add_task(create_hnsw_index, vector_store)
        
        # Update application state
        app_state.connection = connection
        app_state.embeddings_model = embeddings
        app_state.vector_store = vector_store
        
        # Store configuration (excluding sensitive data)
        app_state.config["connection"] = {
            "host": connection_config.host,
            "port": connection_config.port,
            "user": connection_config.user,
            "encrypt": connection_config.encrypt,
            "ssl_validate_certificate": connection_config.ssl_validate_certificate
        }
        app_state.config["embeddings"] = embedding_config.dict()
        app_state.config["vector_store"] = vector_store_config.dict()
        
        return {
            "status": "success",
            "message": "Setup completed successfully",
            "connection_info": {
                "version": info.get("version", "Unknown"),
                "current_schema": info.get("current_schema", "Unknown")
            },
            "embeddings_info": {
                "model_type": embedding_config.model_type,
                "model_name": FINANCIAL_EMBEDDING_MODELS.get(embedding_config.model_type, "Unknown"),
                "dimension": getattr(embeddings, "embedding_dim", "Unknown")
            },
            "vector_store_info": {
                "table_name": vector_store_config.table_name,
                "creating_index": vector_store_config.create_index
            }
        }
    except Exception as e:
        logger.error(f"Setup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Setup failed: {str(e)}"
        )


@app.post("/setup/rag", tags=["Configuration"])
async def setup_rag(
    model: str = Body("gpt-3.5-turbo", description="LLM model to use"),
    temperature: float = Body(0.1, description="Temperature for LLM"),
    k: int = Body(3, description="Number of documents to retrieve")
):
    """Initialize the RAG chain."""
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI integration not available. Install langchain-openai package."
        )
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenAI API key not found. Set the OPENAI_API_KEY environment variable."
        )
    
    try:
        # Get vector store
        vector_store = await get_vector_store()
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        
        # Create a retriever from the vector store
        retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Create a financial analyst prompt template
        template = """
        You are a specialized financial analyst assistant with expertise in interpreting financial data, 
        market reports, earnings announcements, and SEC filings. Answer the user's question based on 
        the retrieved financial information.

        Guidelines:
        - Base your answer strictly on the information provided in the context
        - Be precise with numbers, percentages, and financial metrics
        - Maintain professional financial terminology
        - If the information is not in the context, acknowledge the limitation
        - Provide balanced analysis without investment advice
        - For financial metrics, explain their significance briefly

        Context:
        {context}

        Question: {question}

        Financial Analysis:
        """
        
        prompt = PromptTemplate.from_template(template)
        
        # Define the processing function for formatting context
        def format_docs(docs):
            return "\n\n".join([f"Document ({doc.metadata.get('type', 'Unknown')} - {doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}" for doc in docs])
        
        # Create the RAG chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Update application state
        app_state.llm = llm
        app_state.rag_chain = rag_chain
        
        return {
            "status": "success",
            "message": "RAG chain initialized successfully",
            "config": {
                "model": model,
                "temperature": temperature,
                "k": k
            }
        }
    except Exception as e:
        logger.error(f"RAG setup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG setup failed: {str(e)}"
        )


@app.post("/embeddings", tags=["Embeddings"])
async def generate_embeddings(
    request: EmbeddingRequest,
    embeddings: Embeddings = Depends(get_embeddings)
):
    """Generate embeddings for the provided texts."""
    try:
        start_time = time.time()
        
        # Generate embeddings
        if len(request.texts) == 1:
            embedding = embeddings.embed_query(request.texts[0])
            result = [embedding]
        else:
            result = embeddings.embed_documents(request.texts)
        
        # Update metrics
        processing_time = time.time() - start_time
        app_state.metrics["embeddings_generated"] += len(request.texts)
        
        return {
            "embeddings": result,
            "count": len(result),
            "dimensions": len(result[0]) if result else 0,
            "model": getattr(embeddings, "model_name", "unknown"),
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Embedding generation failed: {str(e)}"
        )


@app.post("/documents", tags=["Documents"])
async def add_documents(
    documents: List[DocumentItem],
    vector_store: HanaVectorStore = Depends(get_vector_store)
):
    """Add documents to the vector store."""
    try:
        # Prepare documents
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to vector store
        start_time = time.time()
        vector_store.add_texts(texts, metadatas=metadatas)
        processing_time = time.time() - start_time
        
        # Update metrics
        app_state.metrics["documents_stored"] += len(documents)
        
        return {
            "status": "success",
            "message": f"Added {len(documents)} documents to vector store",
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        logger.error(f"Document storage error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document storage failed: {str(e)}"
        )


@app.post("/search", tags=["Search"])
async def search_documents(
    request: QueryRequest,
    vector_store: HanaVectorStore = Depends(get_vector_store)
):
    """Search for similar documents in the vector store."""
    try:
        start_time = time.time()
        
        # Perform search
        if request.fetch_k is not None and request.lambda_mult is not None:
            # Use MMR search
            results = vector_store.max_marginal_relevance_search(
                request.query,
                k=request.k,
                fetch_k=request.fetch_k,
                lambda_mult=request.lambda_mult,
                filter=request.filter
            )
        else:
            # Use standard similarity search
            results = vector_store.similarity_search(
                request.query,
                k=request.k,
                filter=request.filter
            )
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        # Update metrics
        processing_time = time.time() - start_time
        app_state.metrics["queries_processed"] += 1
        app_state.metrics["total_query_time"] += processing_time
        app_state.metrics["avg_query_time"] = (
            app_state.metrics["total_query_time"] / app_state.metrics["queries_processed"]
        )
        
        return {
            "results": formatted_results,
            "count": len(formatted_results),
            "query": request.query,
            "filter": request.filter,
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@app.post("/qa", tags=["Question Answering"])
async def answer_question(
    request: QARequest,
    rag_chain = Depends(get_rag_chain)
):
    """Answer a question using RAG."""
    try:
        start_time = time.time()
        
        # Run RAG chain
        response = rag_chain.invoke(request.question)
        
        # Update metrics
        processing_time = time.time() - start_time
        app_state.metrics["qa_requests"] += 1
        
        return {
            "question": request.question,
            "answer": response,
            "processing_time_seconds": processing_time
        }
    except Exception as e:
        logger.error(f"QA error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Question answering failed: {str(e)}"
        )


@app.post("/visualization/metrics", tags=["Visualization"])
async def visualize_metrics(
    request: MetricsRequest,
    visualizer = Depends(get_visualizer)
):
    """Generate visualization for retrieval metrics."""
    try:
        # Generate visualization
        output_path = visualizer.visualize_retrieval_metrics(
            request.metrics_data,
            model_name=request.model_name,
            filename=f"retrieval_metrics_{int(time.time())}.{request.output_format}"
        )
        
        # Return file response if format is not JSON
        if request.output_format != "json":
            return FileResponse(
                output_path,
                media_type=f"image/{request.output_format}",
                filename=os.path.basename(output_path)
            )
        
        # Return metrics data for JSON format
        return {
            "metrics": request.metrics_data,
            "visualization_path": output_path,
            "model_name": request.model_name
        }
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics visualization failed: {str(e)}"
        )


@app.post("/visualization/dashboard", tags=["Visualization"])
async def generate_dashboard(
    metrics_data: Dict[str, Any] = Body(..., description="Metrics data for the dashboard"),
    model_name: str = Body("Financial Embeddings Model", description="Model name for visualization"),
    visualizer = Depends(get_visualizer)
):
    """Generate a comprehensive dashboard of visualizations."""
    try:
        # Create dashboard
        dashboard_path = visualizer.create_summary_dashboard(
            metrics_data,
            model_name,
            filename=f"financial_dashboard_{int(time.time())}.html"
        )
        
        return FileResponse(
            dashboard_path,
            media_type="text/html",
            filename=os.path.basename(dashboard_path)
        )
    except Exception as e:
        logger.error(f"Dashboard generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard generation failed: {str(e)}"
        )


@app.delete("/cleanup", tags=["Configuration"])
async def cleanup():
    """Clean up resources and connections."""
    try:
        # Close connection if it exists
        if app_state.connection is not None:
            close_connection(app_state.connection)
            app_state.connection = None
        
        # Clear GPU memory if using GPU
        if app_state.embeddings_model is not None:
            if hasattr(app_state.embeddings_model, "clear_gpu_memory"):
                app_state.embeddings_model.clear_gpu_memory()
            app_state.embeddings_model = None
        
        # Clear other resources
        app_state.vector_store = None
        app_state.rag_chain = None
        app_state.llm = None
        
        # Preserve metrics for debugging
        metrics_copy = app_state.metrics.copy()
        
        return {
            "status": "success",
            "message": "Resources cleaned up successfully",
            "metrics": metrics_copy
        }
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cleanup failed: {str(e)}"
        )

# =============================================================================
# Background Tasks
# =============================================================================

def create_hnsw_index(vector_store: HanaVectorStore):
    """Background task to create HNSW index."""
    try:
        logger.info("Creating HNSW index for vector store")
        vector_store.create_hnsw_index()
        logger.info("HNSW index created successfully")
    except Exception as e:
        logger.error(f"Failed to create HNSW index: {str(e)}")

# =============================================================================
# Main Entrypoint
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("financial_api:app", host="0.0.0.0", port=8000, reload=True)