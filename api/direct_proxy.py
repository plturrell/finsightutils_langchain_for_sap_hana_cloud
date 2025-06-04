"""
Direct proxy to T4 GPU backend

This module provides a simplified proxy that directly forwards requests to the T4 GPU backend
without additional processing, making it easier to diagnose connection issues.
"""

import os
import json
import logging
import requests
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
T4_GPU_BACKEND_URL = os.getenv("T4_GPU_BACKEND_URL", "https://jupyter0-513syzm60.brevlab.com")
VERCEL_URL = os.getenv("VERCEL_URL", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# Create FastAPI app
app = FastAPI(
    title="SAP HANA Cloud LangChain T4 GPU Direct Proxy",
    description="Direct proxy to T4 GPU backend for debugging",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SAP HANA Cloud LangChain T4 GPU Direct Proxy",
        "backend_url": T4_GPU_BACKEND_URL,
        "status": "active"
    }

@app.get("/proxy-info")
async def proxy_info():
    """Get information about the proxy setup"""
    return {
        "backend_url": T4_GPU_BACKEND_URL,
        "environment": ENVIRONMENT,
        "vercel_url": VERCEL_URL,
        "python_version": os.getenv("PYTHON_VERSION", "unknown"),
        "runtime": "Vercel",
        "timestamp": time.time()
    }

@app.get("/proxy-health")
async def proxy_health():
    """Check if the proxy can connect to the backend"""
    try:
        # Test connection to backend
        start_time = time.time()
        response = requests.get(
            f"{T4_GPU_BACKEND_URL}/api/health",
            timeout=10
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "proxy_status": "healthy",
            "backend_reachable": response.status_code == 200,
            "backend_status_code": response.status_code,
            "backend_response_time_ms": elapsed_time,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Backend connection error: {str(e)}")
        return {
            "proxy_status": "healthy",
            "backend_reachable": False,
            "error": str(e),
            "timestamp": time.time()
        }

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(path: str, request: Request):
    """Proxy all requests to the backend"""
    # Get the request method
    method = request.method.lower()
    
    # Get full URL to the backend
    url = f"{T4_GPU_BACKEND_URL}/{path}"
    
    # Log the request
    logger.debug(f"Proxying {method.upper()} request to {url}")
    
    # Get request headers
    headers = dict(request.headers)
    headers.pop("host", None)  # Remove the host header
    
    # Get request body for non-GET requests
    body = None
    if method != "get":
        body = await request.body()
        if body:
            try:
                # Try to parse as JSON for logging
                body_json = json.loads(body)
                logger.debug(f"Request body: {json.dumps(body_json, indent=2)}")
            except:
                # Not JSON
                logger.debug(f"Request body (raw): {body}")
    
    try:
        # Forward the request to the backend
        response = getattr(requests, method)(
            url,
            headers=headers,
            data=body,
            params=request.query_params,
            timeout=30,
            allow_redirects=False,
        )
        
        # Create FastAPI response
        content = response.content
        status_code = response.status_code
        
        # Try to parse response content as JSON for logging
        try:
            content_json = response.json()
            logger.debug(f"Response: {status_code} - {json.dumps(content_json, indent=2)}")
        except:
            # Not JSON
            logger.debug(f"Response: {status_code} - {content[:200]}...")
        
        # Create response with the same headers
        return Response(
            content=content,
            status_code=status_code,
            headers=dict(response.headers),
        )
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        return {
            "error": "Proxy error",
            "message": str(e),
            "url": url,
            "method": method.upper(),
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)