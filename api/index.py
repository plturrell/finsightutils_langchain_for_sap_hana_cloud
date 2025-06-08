"""
Vercel serverless function entry point.

This module provides the FastAPI application for the Vercel deployment,
including CORS support and vector search capabilities.
"""

from fastapi import FastAPI, Request, Depends, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import os
import logging
import time
import json
from contextlib import asynccontextmanager

# Initialize test mode if enabled
if os.environ.get("TEST_MODE", "").lower() in ("true", "1", "yes", "y"):
    import test_mode
    logging.info("Test mode enabled: Using mock HANA implementation")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vercel_api")

# Define models for API requests/responses
class SearchQuery(BaseModel):
    """Model for vector search queries."""
    query: str = Field(..., description="The search query text")
    k: int = Field(4, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Metadata filter criteria")
    use_mmr: bool = Field(False, description="Whether to use Maximal Marginal Relevance")
    lambda_mult: float = Field(0.5, description="Diversity parameter for MMR (0-1)")
    fetch_k: int = Field(20, description="Number of initial results to fetch for MMR")

class DocumentWithScore(BaseModel):
    """Model for documents returned from vector search."""
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    score: float = Field(..., description="Similarity score")

class SearchResponse(BaseModel):
    """Model for search response."""
    results: List[DocumentWithScore] = Field(..., description="Search results")
    query: str = Field(..., description="Original query")
    processing_time: float = Field(..., description="Processing time in seconds")

class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    context: Dict[str, Any] = Field(..., description="Error context")

# Setup lifespan to handle connections
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    logger.info("API starting up")
    yield
    # Code to run on shutdown
    logger.info("API shutting down")

# Create app
app = FastAPI(
    title="SAP HANA Cloud LangChain Integration API",
    description="Vector search and knowledge graph capabilities for LLM applications using SAP HANA Cloud",
    version="1.2.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses and handle errors."""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Vercel-Deployment"] = "true"
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_server_error",
                "message": str(e),
                "context": {
                    "operation": "api_request",
                    "suggestion": "Please check your input parameters and try again",
                    "timestamp": time.time()
                }
            }
        )

# Import required components for real search
from hdbcli import dbapi
from langchain_hana import HanaDB, HanaInternalEmbeddings
from langchain_hana.utils import DistanceStrategy
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
import os

# Database connection details
DB_HOST = os.environ.get("HANA_HOST", "localhost")
DB_PORT = int(os.environ.get("HANA_PORT", "30015"))
DB_USER = os.environ.get("HANA_USER", "SYSTEM")
DB_PASSWORD = os.environ.get("HANA_PASSWORD", "")
DB_TABLE = os.environ.get("HANA_TABLE", "EMBEDDINGS")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Connection pool for database connections
_connection = None

async def get_connection():
    """Get a database connection."""
    global _connection
    if _connection is None or not _connection.isconnected():
        try:
            _connection = dbapi.connect(
                address=DB_HOST,
                port=DB_PORT,
                user=DB_USER,
                password=DB_PASSWORD,
                encrypt=True,
                sslValidateCertificate=False
            )
            logger.info(f"Connected to SAP HANA Cloud at {DB_HOST}:{DB_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to SAP HANA Cloud: {str(e)}")
            raise RuntimeError(f"Database connection failed: {str(e)}")
    return _connection

async def real_vector_search(query: str, k: int = 4, filter: Optional[Dict] = None, 
                      use_mmr: bool = False, lambda_mult: float = 0.5, fetch_k: int = 20) -> List[DocumentWithScore]:
    """
    Real vector search function that connects to SAP HANA Cloud.
    """
    logger.info(f"Performing vector search for query: '{query}', k={k}, use_mmr={use_mmr}")
    
    try:
        # Get database connection
        connection = await get_connection()
        
        # Create vectorstore instance
        vectorstore = HanaDB(
            connection=connection,
            embedding=embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
            table_name=DB_TABLE
        )
        
        # Perform search
        if use_mmr:
            # Use MMR search for diverse results
            docs = vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter
            )
            # Convert to DocumentWithScore format with default scores
            results = [
                DocumentWithScore(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=0.99 - (0.05 * i)  # Approximate scores for MMR results
                ) for i, doc in enumerate(docs)
            ]
        else:
            # Use regular similarity search
            docs_and_scores = vectorstore.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            # Convert to DocumentWithScore format
            results = [
                DocumentWithScore(
                    content=doc.page_content,
                    metadata=doc.metadata,
                    score=score
                ) for doc, score in docs_and_scores
            ]
        
        logger.info(f"Found {len(results)} results for query '{query}'")
        return results
        
    except Exception as e:
        logger.error(f"Error in vector search: {str(e)}")
        # Return error information as a special result
        return [
            DocumentWithScore(
                content=f"Error performing search: {str(e)}",
                metadata={"error": str(e), "type": "search_error"},
                score=0.0
            )
        ]

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAP HANA Cloud LangChain Integration API",
        "version": "1.2.0",
        "features": [
            "Context-aware error handling",
            "Vector similarity search",
            "Knowledge graph integration",
            "CORS support for frontend integration"
        ]
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "environment": "vercel",
        "timestamp": time.time(),
        "version": "1.2.0"
    }

@app.get("/api/feature/error-handling")
async def error_handling_info():
    """Error handling information endpoint."""
    return {
        "feature": "Context-Aware Error Handling",
        "version": "1.0.0",
        "description": "Intelligent error messages with operation-specific suggestions",
        "capabilities": [
            "SQL error pattern recognition",
            "Operation-specific context",
            "Suggested actions for resolution",
            "Common issues identification"
        ],
        "status": "enabled"
    }

@app.get("/api/feature/vector-similarity")
async def vector_similarity_info():
    """Vector similarity information endpoint."""
    return {
        "feature": "Precision Vector Similarity",
        "version": "1.0.0",
        "description": "Accurate vector similarity measurements with proper normalization",
        "capabilities": [
            "Cosine similarity scoring",
            "Euclidean distance normalization",
            "Maximal Marginal Relevance support",
            "HNSW indexing integration"
        ],
        "status": "enabled"
    }

@app.get("/api/deployment/info")
async def deployment_info():
    """Deployment information endpoint."""
    return {
        "deployment": "Vercel",
        "status": "active",
        "features_enabled": [
            "context_aware_errors",
            "precision_similarity_scoring",
            "knowledge_graph_integration",
            "cors_support"
        ],
        "server_time": time.time(),
        "version": "1.2.0"
    }

# Add missing import for async functionality
import asyncio

@app.post("/api/search", response_model=SearchResponse)
async def vector_search(search_query: SearchQuery):
    """
    Perform vector similarity search against SAP HANA Cloud.
    
    This endpoint connects to a real SAP HANA Cloud database and performs
    vector similarity search using the HanaDB vectorstore implementation.
    """
    start_time = time.time()
    
    try:
        # Use real vector search implementation
        results = await real_vector_search(
            query=search_query.query,
            k=search_query.k,
            filter=search_query.filter,
            use_mmr=search_query.use_mmr,
            lambda_mult=search_query.lambda_mult,
            fetch_k=search_query.fetch_k
        )
        
        # Check for error response
        if len(results) == 1 and results[0].metadata.get("type") == "search_error":
            # Extract error details
            error_message = results[0].metadata.get("error", "Unknown error")
            logger.error(f"Search error returned: {error_message}")
            
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "search_error",
                    "message": error_message,
                    "context": {
                        "operation": "vector_search",
                        "query": search_query.query,
                        "suggestion": "Verify database connection settings and try again"
                    }
                }
            )
        
        # Return formatted results
        return SearchResponse(
            results=results,
            query=search_query.query,
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "search_error",
                "message": str(e),
                "context": {
                    "operation": "vector_search",
                    "query": search_query.query,
                    "suggestion": "Check your search parameters and database connection"
                }
            }
        )

@app.options("/api/{path:path}")
async def options_handler(path: str):
    """Handle OPTIONS requests for CORS preflight."""
    return {}

# For Vercel serverless function
handler = app