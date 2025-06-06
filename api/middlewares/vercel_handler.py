"""
Vercel serverless function handler for the FastAPI application.
This adapter enables the FastAPI app to run on Vercel's serverless environment.
"""

import os
import logging
import time
from typing import Dict, Any

# Set environment variables before importing app
os.environ["GPU_ENABLED"] = "false"
os.environ["USE_TENSORRT"] = "false"
os.environ["USE_INTERNAL_EMBEDDINGS"] = "false"
os.environ["MAX_RESPONSE_TIME"] = "50"  # 50 seconds max to avoid Vercel timeout
os.environ["VERCEL_DEPLOYMENT"] = "true"
os.environ["ENABLE_ERROR_CONTEXT"] = "true"
os.environ["ENABLE_PRECISE_SIMILARITY"] = "true"
os.environ["ERROR_DETAIL_LEVEL"] = "standard"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vercel_handler")
logger.info("Initializing Vercel serverless function with error handling support")

# Now import app
from api import app
from api.middlewares.vercel_middleware import setup_middleware

# Set up middleware for Vercel deployment
app = setup_middleware(app)

# Add a specific route for Vercel health checks
@app.get("/__health")
async def health_check():
    """Health check endpoint for Vercel."""
    return {
        "status": "ok",
        "environment": "vercel",
        "timestamp": time.time(),
        "gpu_enabled": os.environ.get("GPU_ENABLED", "false"),
        "version": "1.0.2"
    }

# Add a redirect from root to docs
@app.get("/")
async def redirect_to_docs():
    """Redirect root to documentation."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

# Handler for Vercel serverless function
async def handler(request: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle requests in Vercel serverless environment.
    
    This is the main entry point for the Vercel serverless function.
    """
    # Log the request
    method = request.get("method", "UNKNOWN")
    path = request.get("path", "/")
    logger.info(f"Vercel request: {method} {path}")
    
    # Process the request through FastAPI
    return app

# Make the handler available for Vercel
app.logger = logger