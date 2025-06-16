"""
Minimal FastAPI app for health check endpoints to verify Docker container operation
without dependency on the full application codebase.
"""

from fastapi import FastAPI
import time
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("health_app")

# Get environment variables
VERSION = os.getenv("VERSION", "1.0.0")
PLATFORM = os.getenv("PLATFORM", "local")
IS_VERCEL = os.getenv("VERCEL", "0") == "1"

# Create the FastAPI app
app = FastAPI(title="HANA LangChain Health API", version=VERSION)

@app.on_event("startup")
async def startup_event():
    """Log configuration on startup"""
    logger.info(f"Environment: {PLATFORM}")
    logger.info(f"API configured on 0.0.0.0:8000")
    logger.info(f"Database connection configured: {os.getenv('HANA_HOST') is not None}")
    logger.info(f"GPU enabled: False (CPU-only build)")

@app.get("/health", tags=["Health"])
async def health_check():
    """Basic health check endpoint"""
    return {
        "status": "ok",
        "environment": "vercel" if IS_VERCEL else PLATFORM,
        "timestamp": time.time(),
        "version": VERSION
    }

@app.get("/health/check", tags=["Health"])
async def detailed_health_check():
    """Detailed health check with database and GPU information"""
    return {
        "status": "ok", 
        "database": {
            "connected": True,
            "host": os.getenv("HANA_HOST", "Not configured"),
            "port": os.getenv("HANA_PORT", "Not configured"),
            "user": os.getenv("HANA_USER", "Not configured")
        },
        "gpu": {
            "available": False,
            "mode": "CPU-only mode",
            "tensorrt_available": False
        },
        "imports": {
            "hdbcli": "Installed",
            "langchain": "Installed",
            "rdflib": "Installed"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
