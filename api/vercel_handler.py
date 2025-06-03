"""
Vercel serverless function handler for the FastAPI application.
This adapter enables the FastAPI app to run on Vercel's serverless environment.
"""

from app import app
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("vercel_handler")

# Disable GPU acceleration in Vercel environment
os.environ["GPU_ENABLED"] = "false"
os.environ["USE_TENSORRT"] = "false"

# Add middleware for Vercel-specific headers
@app.middleware("http")
async def add_vercel_headers(request: Request, call_next):
    """Add Vercel-specific headers to responses."""
    response = await call_next(request)
    response.headers["x-vercel-deployment"] = "1"
    return response

# Add a specific route for Vercel health checks
@app.get("/__health")
async def health_check():
    """Health check endpoint for Vercel."""
    return {"status": "ok", "environment": "vercel"}

# Handler for Vercel serverless function
def handler(request, response):
    """
    Handle requests in Vercel serverless environment.
    """
    # Log the request
    logger.info(f"Vercel request: {request.method} {request.url}")
    
    # Process the request through FastAPI
    return app(request, response)

# Make the handler available for Vercel
handler = app