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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vercel_handler")
logger.info("Initializing Vercel serverless function")

# Now import app
from app import app
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure CORS is enabled
if not any(isinstance(m, CORSMiddleware) for m in app.user_middleware):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Add middleware for Vercel-specific headers and request timing
@app.middleware("http")
async def add_vercel_headers(request: Request, call_next):
    """Add Vercel-specific headers and track request timing."""
    start_time = time.time()
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Add Vercel-specific headers
        response.headers["x-vercel-deployment"] = "1"
        response.headers["x-response-time"] = str(round((time.time() - start_time) * 1000))
        
        return response
    except Exception as e:
        # Log any errors
        logger.error(f"Error processing request: {str(e)}")
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "message": str(e)},
        )

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