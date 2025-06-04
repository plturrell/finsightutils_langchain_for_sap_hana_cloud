"""
Simplified version of the application for Vercel deployment.
"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vercel_app")

# Create the FastAPI app
app = FastAPI(
    title="SAP HANA Cloud LangChain Integration",
    description="Vector search and knowledge graph capabilities for LLM applications using SAP HANA Cloud",
    version="1.2.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
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
                    "operation": "request_processing",
                    "suggestion": "Please try again later or contact support if the issue persists."
                }
            }
        )

@app.get("/")
async def root():
    return {
        "message": "SAP HANA Cloud LangChain Integration API",
        "version": "1.2.0",
        "features": [
            "Context-aware error handling",
            "Vector similarity search",
            "Knowledge graph integration",
            "GPU acceleration (when available)"
        ],
        "documentation": "/docs"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "environment": "vercel",
        "timestamp": time.time(),
        "version": "1.2.0"
    }

@app.get("/api/feature/error-handling")
async def error_handling_info():
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