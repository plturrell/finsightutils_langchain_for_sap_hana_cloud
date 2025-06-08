"""
Simple test API for validating the API endpoints.
"""

import os
import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

# Initialize logging
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("test_api")

# Create a FastAPI app
app = FastAPI(
    title="SAP HANA Cloud LangChain Integration Test API",
    description="Test API for SAP HANA Cloud LangChain Integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize test mode if enabled
if os.environ.get("TEST_MODE", "").lower() in ("true", "1", "yes", "y"):
    import test_mode
    logger.info("Test mode enabled: Using mock HANA implementation")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "SAP HANA Cloud LangChain Integration Test API",
        "version": "1.0.0",
        "test_mode": os.environ.get("TEST_MODE", "false")
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "environment": "test",
        "version": "1.0.0"
    }

@app.get("/health/ping")
async def ping():
    """Simple ping endpoint."""
    return "pong"

@app.get("/health/status")
async def status():
    """Health status endpoint."""
    return {
        "status": "ok",
        "components": [
            {
                "name": "api",
                "status": "ok",
                "details": {
                    "test_mode": os.environ.get("TEST_MODE", "false"),
                    "python_version": os.environ.get("PYTHON_VERSION", "unknown")
                }
            }
        ]
    }

@app.get("/api/feature/error-handling")
async def error_handling_info():
    """Error handling information."""
    return {
        "feature": "Context-Aware Error Handling",
        "version": "1.0.0",
        "description": "Intelligent error messages with operation-specific suggestions"
    }

@app.get("/api/feature/vector-similarity")
async def vector_similarity_info():
    """Vector similarity information."""
    return {
        "feature": "Vector Similarity",
        "version": "1.0.0",
        "description": "Vector similarity search with SAP HANA Cloud"
    }

@app.get("/api/deployment/info")
async def deployment_info():
    """Deployment information."""
    return {
        "deployment": "test",
        "status": "active",
        "environment": "test"
    }

@app.get("/gpu/info")
async def gpu_info():
    """GPU information."""
    return {
        "gpu_available": False,
        "message": "Running in test mode without GPU acceleration"
    }

@app.post("/api/search")
async def search(query: dict):
    """Simple search endpoint."""
    return {
        "results": [
            {
                "content": "This is a test document",
                "metadata": {"source": "test.txt"},
                "score": 0.95
            }
        ],
        "query": query.get("query", ""),
        "processing_time": 0.01
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)