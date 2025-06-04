"""
Vercel serverless function entry point.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vercel_api")

# Create app
app = FastAPI(
    title="SAP HANA Cloud LangChain Integration API",
    description="Vector search and knowledge graph capabilities for LLM applications using SAP HANA Cloud",
    version="1.2.0",
)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": str(e),
                "context": {
                    "operation": "api_request",
                    "suggestion": "Please check your input parameters and try again"
                }
            }
        )

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAP HANA Cloud LangChain Integration API",
        "version": "1.2.0",
        "features": [
            "Context-aware error handling",
            "Vector similarity search",
            "Knowledge graph integration"
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
            "knowledge_graph_integration"
        ],
        "server_time": time.time(),
        "version": "1.2.0"
    }

# For Vercel serverless function
handler = app