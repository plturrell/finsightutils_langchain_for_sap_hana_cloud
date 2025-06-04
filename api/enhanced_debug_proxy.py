"""
Enhanced Debug Proxy for SAP HANA Cloud LangChain T4 GPU Integration

This module provides a highly detailed debugging proxy specifically designed to
diagnose 500 errors and connectivity issues between the Vercel frontend and 
T4 GPU backend on Brev Cloud.

Key features:
- Detailed request/response logging with headers and body contents
- Verbose error reporting with rich context information
- Connectivity validation with multiple test methods
- Configurable timeouts for different operations
- Exception traceback capture and reporting
- Memory usage tracking for serverless function optimization
"""

import os
import sys
import json
import time
import logging
import traceback
import requests
import uuid
import psutil
from typing import Dict, Any, Optional, List, Union
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger("enhanced-debug-proxy")

# Load environment variables with defaults
T4_GPU_BACKEND_URL = os.getenv("T4_GPU_BACKEND_URL", "https://jupyter0-513syzm60.brevlab.com")
VERCEL_URL = os.getenv("VERCEL_URL", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))
CONNECTION_TEST_TIMEOUT = int(os.getenv("CONNECTION_TEST_TIMEOUT", "5"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG")
ENABLE_MEMORY_TRACKING = os.getenv("ENABLE_MEMORY_TRACKING", "true").lower() == "true"

# Set log level from environment
if LOG_LEVEL.upper() == "DEBUG":
    logger.setLevel(logging.DEBUG)
elif LOG_LEVEL.upper() == "INFO":
    logger.setLevel(logging.INFO)
elif LOG_LEVEL.upper() == "WARNING":
    logger.setLevel(logging.WARNING)
elif LOG_LEVEL.upper() == "ERROR":
    logger.setLevel(logging.ERROR)

# Create FastAPI app
app = FastAPI(
    title="SAP HANA Cloud LangChain T4 GPU Enhanced Debug Proxy",
    description="Advanced diagnostic proxy for debugging connection issues with T4 GPU backend",
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

# Track memory usage for diagnostics
def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics for the process
    """
    if not ENABLE_MEMORY_TRACKING:
        return {"memory_tracking_enabled": False}
    
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # RSS in MB
            "vms_mb": memory_info.vms / (1024 * 1024),  # VMS in MB
            "memory_percent": memory_percent,
            "memory_tracking_enabled": True
        }
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {str(e)}")
        return {"memory_tracking_enabled": False, "error": str(e)}

# Request ID middleware for tracking
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Add request ID to response headers and to the request state
    """
    request_id = f"debug_{uuid.uuid4()}"
    request.state.request_id = request_id
    
    # Log memory usage before handling request
    if ENABLE_MEMORY_TRACKING:
        logger.debug(f"Memory before request {request_id}: {json.dumps(get_memory_usage())}")
    
    # Process the request
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log memory usage after handling request
    if ENABLE_MEMORY_TRACKING:
        logger.debug(f"Memory after request {request_id}: {json.dumps(get_memory_usage())}")
    
    return response

@app.get("/")
async def root():
    """Root endpoint providing basic information"""
    return {
        "message": "SAP HANA Cloud LangChain T4 GPU Enhanced Debug Proxy",
        "backend_url": T4_GPU_BACKEND_URL,
        "status": "active",
        "environment": ENVIRONMENT,
        "request_timeout": REQUEST_TIMEOUT,
        "health_check_timeout": HEALTH_CHECK_TIMEOUT,
        "memory_usage": get_memory_usage()
    }

@app.get("/debug-info")
async def debug_info():
    """Get detailed debug information about the proxy setup"""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    return {
        "backend_url": T4_GPU_BACKEND_URL,
        "environment": ENVIRONMENT,
        "vercel_url": VERCEL_URL,
        "python_version": python_version,
        "request_timeout": REQUEST_TIMEOUT,
        "health_check_timeout": HEALTH_CHECK_TIMEOUT,
        "connection_test_timeout": CONNECTION_TEST_TIMEOUT,
        "runtime": "Vercel",
        "log_level": LOG_LEVEL,
        "memory_usage": get_memory_usage(),
        "system_info": {
            "platform": sys.platform,
            "python_path": sys.executable,
            "sys_path": sys.path,
            "env_vars": {k: v for k, v in os.environ.items() if not k.lower().contains("password") and not k.lower().contains("secret")},
        },
        "timestamp": time.time()
    }

@app.get("/connection-test")
async def connection_test():
    """
    Comprehensive test of the connection to the backend, using multiple test methods
    """
    results = {
        "tests": {},
        "summary": {
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0
        },
        "timestamp": time.time()
    }
    
    # Test 1: Basic health check
    try:
        start_time = time.time()
        response = requests.get(
            f"{T4_GPU_BACKEND_URL}/api/health",
            timeout=CONNECTION_TEST_TIMEOUT
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        results["tests"]["health_check"] = {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time_ms": elapsed_time,
            "response_body": response.text if len(response.text) < 500 else f"{response.text[:500]}... (truncated)"
        }
        
        if response.status_code == 200:
            results["summary"]["passed_tests"] += 1
        else:
            results["summary"]["failed_tests"] += 1
        
        results["summary"]["total_tests"] += 1
    except Exception as e:
        results["tests"]["health_check"] = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        results["summary"]["failed_tests"] += 1
        results["summary"]["total_tests"] += 1
    
    # Test 2: TCP connection test (using requests low-level API)
    try:
        start_time = time.time()
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=0)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        
        # Parse URL to get host and port
        from urllib.parse import urlparse
        parsed_url = urlparse(T4_GPU_BACKEND_URL)
        host = parsed_url.hostname
        port = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
        
        # Try to establish connection without sending a request
        with session.get(
            T4_GPU_BACKEND_URL,
            timeout=CONNECTION_TEST_TIMEOUT,
            stream=True
        ) as response:
            # Just connect, don't download content
            pass
        
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        results["tests"]["tcp_connection"] = {
            "success": True,
            "host": host,
            "port": port,
            "connection_time_ms": elapsed_time
        }
        results["summary"]["passed_tests"] += 1
        results["summary"]["total_tests"] += 1
    except Exception as e:
        results["tests"]["tcp_connection"] = {
            "success": False,
            "host": host if 'host' in locals() else None,
            "port": port if 'port' in locals() else None,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        results["summary"]["failed_tests"] += 1
        results["summary"]["total_tests"] += 1
    
    # Test 3: DNS resolution test
    try:
        import socket
        start_time = time.time()
        
        from urllib.parse import urlparse
        parsed_url = urlparse(T4_GPU_BACKEND_URL)
        host = parsed_url.hostname
        
        # Resolve DNS
        ip_address = socket.gethostbyname(host)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        results["tests"]["dns_resolution"] = {
            "success": True,
            "host": host,
            "ip_address": ip_address,
            "resolution_time_ms": elapsed_time
        }
        results["summary"]["passed_tests"] += 1
        results["summary"]["total_tests"] += 1
    except Exception as e:
        results["tests"]["dns_resolution"] = {
            "success": False,
            "host": host if 'host' in locals() else None,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        results["summary"]["failed_tests"] += 1
        results["summary"]["total_tests"] += 1
    
    # Test 4: Test a simple POST request (embeddings endpoint)
    try:
        test_data = {
            "texts": ["Simple test sentence for embedding generation"],
            "model_name": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{T4_GPU_BACKEND_URL}/api/embeddings",
            json=test_data,
            timeout=CONNECTION_TEST_TIMEOUT * 2  # Allow more time for this test
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        results["tests"]["embeddings_api"] = {
            "success": response.status_code == 200,
            "status_code": response.status_code,
            "response_time_ms": elapsed_time,
        }
        
        # Add response details if successful
        if response.status_code == 200:
            try:
                response_json = response.json()
                # Don't include the actual embedding vectors to keep response small
                if "embeddings" in response_json:
                    response_json["embeddings"] = f"[{len(response_json['embeddings'])} embeddings, dimensions: {len(response_json['embeddings'][0]) if response_json['embeddings'] else 0}]"
                results["tests"]["embeddings_api"]["response"] = response_json
            except:
                results["tests"]["embeddings_api"]["response"] = "Unable to parse JSON response"
        else:
            results["tests"]["embeddings_api"]["response_body"] = response.text if len(response.text) < 500 else f"{response.text[:500]}... (truncated)"
        
        if response.status_code == 200:
            results["summary"]["passed_tests"] += 1
        else:
            results["summary"]["failed_tests"] += 1
            
        results["summary"]["total_tests"] += 1
    except Exception as e:
        results["tests"]["embeddings_api"] = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        results["summary"]["failed_tests"] += 1
        results["summary"]["total_tests"] += 1
    
    # Set overall success status
    results["success"] = results["summary"]["failed_tests"] == 0
    results["message"] = f"Passed {results['summary']['passed_tests']} of {results['summary']['total_tests']} tests"
    
    return results

@app.get("/proxy-health")
async def proxy_health():
    """Check if the proxy can connect to the backend"""
    try:
        # Test connection to backend
        start_time = time.time()
        response = requests.get(
            f"{T4_GPU_BACKEND_URL}/api/health",
            timeout=HEALTH_CHECK_TIMEOUT
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Try to parse response as JSON
        response_body = None
        try:
            response_body = response.json()
        except:
            response_body = response.text if len(response.text) < 500 else f"{response.text[:500]}... (truncated)"
        
        return {
            "proxy_status": "healthy",
            "backend_reachable": response.status_code == 200,
            "backend_status_code": response.status_code,
            "backend_response_time_ms": elapsed_time,
            "backend_response": response_body,
            "memory_usage": get_memory_usage(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Backend connection error: {str(e)}")
        return {
            "proxy_status": "healthy",
            "backend_reachable": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "memory_usage": get_memory_usage(),
            "timestamp": time.time()
        }

@app.get("/debug-headers")
async def debug_headers(request: Request):
    """
    Debug endpoint that returns all request headers
    Useful for debugging authentication and other header-related issues
    """
    headers = {k: v for k, v in request.headers.items()}
    return {
        "request_headers": headers,
        "request_id": request.state.request_id,
        "timestamp": time.time()
    }

@app.post("/debug-request")
async def debug_request(request: Request):
    """
    Debug endpoint that echoes back the request details
    Useful for debugging request body and parameter issues
    """
    body = await request.body()
    body_str = body.decode("utf-8") if body else None
    
    # Try to parse as JSON
    body_json = None
    if body_str:
        try:
            body_json = json.loads(body_str)
        except:
            body_json = None
    
    return {
        "method": request.method,
        "url": str(request.url),
        "headers": {k: v for k, v in request.headers.items()},
        "query_params": {k: v for k, v in request.query_params.items()},
        "body_text": body_str,
        "body_json": body_json,
        "request_id": request.state.request_id,
        "timestamp": time.time()
    }

async def log_request_details(request: Request) -> Dict[str, Any]:
    """
    Log detailed information about the request
    Returns a dictionary with request details
    """
    # Get request details
    method = request.method
    url = str(request.url)
    path = request.url.path
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ['authorization']}
    query_params = {k: v for k, v in request.query_params.items()}
    
    # Get request body
    body = await request.body()
    body_str = None
    body_json = None
    
    if body:
        try:
            body_str = body.decode("utf-8")
            try:
                body_json = json.loads(body_str)
                logger.debug(f"Request body (JSON): {json.dumps(body_json, indent=2)}")
            except:
                # Not JSON
                if len(body_str) > 1000:
                    logger.debug(f"Request body (text, truncated): {body_str[:1000]}...")
                else:
                    logger.debug(f"Request body (text): {body_str}")
        except:
            logger.debug(f"Request body (binary, length: {len(body)})")
    
    request_details = {
        "method": method,
        "url": url,
        "path": path,
        "headers": headers,
        "query_params": query_params,
        "body": body_json or body_str,
    }
    
    logger.debug(f"Incoming request: {method} {path}")
    logger.debug(f"Request details: {json.dumps(request_details, default=str, indent=2)}")
    
    return request_details, body

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])
async def proxy(path: str, request: Request):
    """Proxy all requests to the backend with enhanced logging and error handling"""
    # Get the request method
    method = request.method.lower()
    
    # Get full URL to the backend
    url = f"{T4_GPU_BACKEND_URL}/{path}"
    
    # Log detailed request information
    request_details, body = await log_request_details(request)
    
    # Get request headers
    headers = dict(request.headers)
    headers.pop("host", None)  # Remove the host header
    
    # Set up response metadata
    response_metadata = {
        "proxy_request_id": request.state.request_id,
        "backend_url": url,
        "method": method.upper(),
        "start_time": time.time()
    }
    
    try:
        # Forward the request to the backend
        logger.info(f"Forwarding {method.upper()} request to {url}")
        
        # Determine timeout based on endpoint
        custom_timeout = REQUEST_TIMEOUT
        if "health" in path:
            custom_timeout = HEALTH_CHECK_TIMEOUT
        elif "embeddings" in path:
            # Embedding generation can take longer
            custom_timeout = REQUEST_TIMEOUT * 2
        
        # Make the request
        response = getattr(requests, method)(
            url,
            headers=headers,
            data=body,
            params=request.query_params,
            timeout=custom_timeout,
            allow_redirects=False,
        )
        
        # Update response metadata
        response_metadata["end_time"] = time.time()
        response_metadata["duration_ms"] = (response_metadata["end_time"] - response_metadata["start_time"]) * 1000
        response_metadata["status_code"] = response.status_code
        
        # Create FastAPI response
        content = response.content
        status_code = response.status_code
        
        # Try to parse response content as JSON for logging
        try:
            content_json = response.json()
            # For large responses, don't log the full content
            if "embeddings" in content_json and isinstance(content_json["embeddings"], list):
                embedding_count = len(content_json["embeddings"])
                embedding_dim = len(content_json["embeddings"][0]) if embedding_count > 0 else 0
                logger.debug(f"Response: {status_code} - Embeddings response with {embedding_count} embeddings of dimension {embedding_dim}")
            else:
                logger.debug(f"Response: {status_code} - {json.dumps(content_json, indent=2)}")
        except:
            # Not JSON
            logger.debug(f"Response: {status_code} - {content[:200]}...")
        
        # Add debugging headers to the response
        response_headers = dict(response.headers)
        response_headers["X-Debug-Proxy-Request-ID"] = request.state.request_id
        response_headers["X-Debug-Proxy-Duration-MS"] = str(int(response_metadata["duration_ms"]))
        
        # Create response with the same headers
        return Response(
            content=content,
            status_code=status_code,
            headers=response_headers,
        )
    except requests.Timeout as e:
        logger.error(f"Timeout error connecting to backend: {str(e)}")
        response_metadata["error"] = str(e)
        response_metadata["error_type"] = "timeout"
        response_metadata["traceback"] = traceback.format_exc()
        
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": "Backend timeout",
                "message": f"The backend service at {T4_GPU_BACKEND_URL} did not respond within {custom_timeout} seconds",
                "timeout_setting": custom_timeout,
                "backend_url": T4_GPU_BACKEND_URL,
                "endpoint": path,
                "method": method.upper(),
                "request_id": request.state.request_id,
                "error_details": str(e),
                "metadata": response_metadata
            }
        )
    except requests.ConnectionError as e:
        logger.error(f"Connection error to backend: {str(e)}")
        response_metadata["error"] = str(e)
        response_metadata["error_type"] = "connection_error"
        response_metadata["traceback"] = traceback.format_exc()
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Backend connection error",
                "message": f"Cannot connect to the backend service at {T4_GPU_BACKEND_URL}. Service may be down or unreachable.",
                "backend_url": T4_GPU_BACKEND_URL,
                "endpoint": path,
                "method": method.upper(),
                "request_id": request.state.request_id,
                "error_details": str(e),
                "suggestion": "Check if the T4 GPU backend is running and the URL is correct. You can verify by accessing its health endpoint directly.",
                "metadata": response_metadata
            }
        )
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        response_metadata["error"] = str(e)
        response_metadata["error_type"] = type(e).__name__
        response_metadata["traceback"] = traceback.format_exc()
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Proxy error",
                "message": f"An error occurred while proxying the request to {T4_GPU_BACKEND_URL}",
                "backend_url": T4_GPU_BACKEND_URL,
                "endpoint": path,
                "method": method.upper(),
                "request_id": request.state.request_id,
                "error_details": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc() if ENVIRONMENT == "development" else None,
                "metadata": response_metadata
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)