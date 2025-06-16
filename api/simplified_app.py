"""
Simplified SAP HANA Cloud LangChain Integration API with Arrow Flight.

This is a minimal version of the API that only includes the Arrow Flight
functionality and basic health check endpoints for testing.
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
logger = logging.getLogger("api")

# Create a FastAPI app
app = FastAPI(
    title="SAP HANA Cloud LangChain Integration API - Arrow Flight",
    description="Simplified API for SAP HANA Cloud LangChain Integration with Arrow Flight support",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        "message": "SAP HANA Cloud LangChain Integration with Arrow Flight API",
        "version": "1.1.0",
        "test_mode": os.environ.get("TEST_MODE", "true")
    }

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": time.time(),
        "environment": "test",
        "version": "1.1.0",
        "arrow_flight": True
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
                    "test_mode": os.environ.get("TEST_MODE", "true"),
                    "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
                    "arrow_flight": True
                }
            }
        ]
    }

@app.get("/flight/info")
async def flight_info():
    """Get information about Arrow Flight service."""
    return {
        "status": "available",
        "host": os.environ.get("FLIGHT_HOST", "0.0.0.0"),
        "port": int(os.environ.get("FLIGHT_PORT", "8815")),
        "auto_start": os.environ.get("FLIGHT_AUTO_START", "true") == "true"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)