"""
Direct proxy to T4 GPU backend

This module provides a simplified proxy that directly forwards requests to the T4 GPU backend
without additional processing, making it easier to diagnose connection issues.
"""

import os
import json
import logging
import requests
import socket
import sys
import traceback
import urllib.parse
from fastapi import FastAPI, Request, Response, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import ssl

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
T4_GPU_BACKEND_URL = os.getenv("T4_GPU_BACKEND_URL", "https://jupyter0-513syzm60.brevlab.com")
VERCEL_URL = os.getenv("VERCEL_URL", "")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
CONNECTION_TEST_TIMEOUT = int(os.getenv("CONNECTION_TEST_TIMEOUT", "5"))
DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30"))
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "10"))

# Parse backend URL for connection validation
parsed_url = urllib.parse.urlparse(T4_GPU_BACKEND_URL)
BACKEND_HOSTNAME = parsed_url.hostname
BACKEND_PORT = parsed_url.port or (443 if parsed_url.scheme == 'https' else 80)
BACKEND_USE_SSL = parsed_url.scheme == 'https'

# Connection validation functions
def test_dns_resolution(hostname):
    """Test DNS resolution for the backend hostname"""
    try:
        start_time = time.time()
        ip_address = socket.gethostbyname(hostname)
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return {
            "success": True,
            "hostname": hostname,
            "ip_address": ip_address,
            "resolution_time_ms": elapsed_time
        }
    except socket.gaierror as e:
        return {
            "success": False,
            "hostname": hostname,
            "error": str(e),
            "error_code": e.errno,
            "error_type": "socket.gaierror"
        }
    except Exception as e:
        return {
            "success": False,
            "hostname": hostname,
            "error": str(e),
            "error_type": type(e).__name__
        }

def test_tcp_connection(hostname, port, use_ssl=False, timeout=5):
    """Test raw TCP connection to the backend"""
    try:
        start_time = time.time()
        
        # Create socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        
        # Wrap with SSL if needed
        if use_ssl:
            context = ssl.create_default_context()
            sock = context.wrap_socket(sock, server_hostname=hostname)
        
        # Connect
        sock.connect((hostname, port))
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Close connection
        sock.close()
        
        return {
            "success": True,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "connection_time_ms": elapsed_time
        }
    except socket.timeout:
        return {
            "success": False,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "error": "Connection timed out",
            "error_type": "socket.timeout"
        }
    except ssl.SSLError as e:
        return {
            "success": False,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "error": str(e),
            "error_type": "ssl.SSLError"
        }
    except ConnectionRefusedError:
        return {
            "success": False,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "error": "Connection refused",
            "error_type": "ConnectionRefusedError"
        }
    except Exception as e:
        return {
            "success": False,
            "hostname": hostname,
            "port": port,
            "ssl": use_ssl,
            "error": str(e),
            "error_type": type(e).__name__
        }

def test_http_request(url, timeout=5):
    """Test HTTP/HTTPS request to the backend"""
    try:
        start_time = time.time()
        response = requests.get(
            url,
            timeout=timeout,
            allow_redirects=False
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Try to parse response as JSON
        response_data = None
        try:
            response_data = response.json()
        except:
            response_data = response.text[:200] + "..." if len(response.text) > 200 else response.text
        
        return {
            "success": 200 <= response.status_code < 300,
            "url": url,
            "status_code": response.status_code,
            "response_time_ms": elapsed_time,
            "headers": dict(response.headers),
            "response": response_data
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "url": url,
            "error": "Request timed out",
            "error_type": "requests.exceptions.Timeout"
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "success": False,
            "url": url,
            "error": str(e),
            "error_type": "requests.exceptions.ConnectionError"
        }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": str(e),
            "error_type": type(e).__name__
        }

def perform_connection_diagnostics(url=None, timeout=5):
    """Perform comprehensive connection diagnostics"""
    if not url:
        url = T4_GPU_BACKEND_URL
    
    # Parse URL
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == 'https' else 80)
    use_ssl = parsed.scheme == 'https'
    
    # Results container
    results = {
        "url": url,
        "hostname": hostname,
        "port": port,
        "ssl": use_ssl,
        "tests": {},
        "summary": {
            "success": False,
            "timestamp": time.time()
        }
    }
    
    # Test 1: DNS resolution
    results["tests"]["dns_resolution"] = test_dns_resolution(hostname)
    
    # Test 2: TCP connection (without SSL)
    results["tests"]["tcp_connection"] = test_tcp_connection(hostname, port, False, timeout)
    
    # Test 3: TCP connection with SSL (if applicable)
    if use_ssl:
        results["tests"]["ssl_connection"] = test_tcp_connection(hostname, port, True, timeout)
    
    # Test 4: HTTP request
    results["tests"]["http_request"] = test_http_request(url, timeout)
    
    # Test 5: API health endpoint
    health_url = f"{url}/api/health"
    results["tests"]["api_health"] = test_http_request(health_url, timeout)
    
    # Determine overall success
    dns_success = results["tests"]["dns_resolution"]["success"]
    tcp_success = results["tests"]["tcp_connection"]["success"]
    ssl_success = results["tests"]["ssl_connection"]["success"] if use_ssl else True
    http_success = results["tests"]["http_request"]["success"]
    
    results["summary"]["success"] = dns_success and tcp_success and ssl_success and http_success
    
    # Add diagnostic information
    if not results["summary"]["success"]:
        if not dns_success:
            results["summary"]["primary_issue"] = "DNS resolution failed"
            results["summary"]["recommendation"] = "Check if the hostname is correct and DNS is working properly"
        elif not tcp_success:
            results["summary"]["primary_issue"] = "TCP connection failed"
            results["summary"]["recommendation"] = "Check if the host is reachable and the port is open"
        elif not ssl_success:
            results["summary"]["primary_issue"] = "SSL/TLS connection failed"
            results["summary"]["recommendation"] = "Check SSL/TLS configuration or try connecting without SSL"
        elif not http_success:
            results["summary"]["primary_issue"] = "HTTP request failed"
            results["summary"]["recommendation"] = "Check if the server is accepting HTTP requests"
        else:
            results["summary"]["primary_issue"] = "Unknown issue"
            results["summary"]["recommendation"] = "Review individual test results for details"
    
    return results

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
        "status": "active",
        "backend_hostname": BACKEND_HOSTNAME,
        "backend_port": BACKEND_PORT,
        "backend_use_ssl": BACKEND_USE_SSL
    }

@app.get("/proxy-info")
async def proxy_info():
    """Get information about the proxy setup"""
    return {
        "backend_url": T4_GPU_BACKEND_URL,
        "backend_hostname": BACKEND_HOSTNAME,
        "backend_port": BACKEND_PORT,
        "backend_use_ssl": BACKEND_USE_SSL,
        "environment": ENVIRONMENT,
        "vercel_url": VERCEL_URL,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "runtime": "Vercel",
        "timeouts": {
            "default": DEFAULT_TIMEOUT,
            "health_check": HEALTH_CHECK_TIMEOUT,
            "connection_test": CONNECTION_TEST_TIMEOUT
        },
        "timestamp": time.time()
    }

@app.get("/connection-diagnostics")
async def connection_diagnostics(custom_url: str = None):
    """
    Run comprehensive connection diagnostics to the backend
    
    This endpoint tests multiple aspects of the connection:
    - DNS resolution
    - TCP connectivity
    - SSL/TLS handshake
    - HTTP/HTTPS request
    - API health endpoint
    """
    try:
        # Use the specified URL or default to backend URL
        url = custom_url or T4_GPU_BACKEND_URL
        
        # Run diagnostics
        results = perform_connection_diagnostics(url, CONNECTION_TEST_TIMEOUT)
        
        return results
    except Exception as e:
        logger.error(f"Error during connection diagnostics: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
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
            timeout=HEALTH_CHECK_TIMEOUT
        )
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Try to parse response as JSON
        response_data = None
        try:
            response_data = response.json()
        except:
            response_data = response.text[:200] + "..." if len(response.text) > 200 else response.text
        
        return {
            "proxy_status": "healthy",
            "backend_reachable": response.status_code == 200,
            "backend_status_code": response.status_code,
            "backend_response_time_ms": elapsed_time,
            "backend_response": response_data,
            "backend_url": T4_GPU_BACKEND_URL,
            "backend_hostname": BACKEND_HOSTNAME,
            "backend_port": BACKEND_PORT,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Backend connection error: {str(e)}")
        
        # Run quick diagnostics
        dns_result = test_dns_resolution(BACKEND_HOSTNAME)
        
        return {
            "proxy_status": "healthy",
            "backend_reachable": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "dns_resolution": dns_result,
            "backend_url": T4_GPU_BACKEND_URL,
            "backend_hostname": BACKEND_HOSTNAME,
            "backend_port": BACKEND_PORT,
            "recommendation": "Run /connection-diagnostics for detailed troubleshooting",
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
    
    # Set appropriate timeout based on endpoint
    timeout = DEFAULT_TIMEOUT
    if "health" in path:
        timeout = HEALTH_CHECK_TIMEOUT
    elif "embeddings" in path:
        # Embedding generation can take longer
        timeout = DEFAULT_TIMEOUT * 2
    
    try:
        # Record request start time
        start_time = time.time()
        
        # Forward the request to the backend
        response = getattr(requests, method)(
            url,
            headers=headers,
            data=body,
            params=request.query_params,
            timeout=timeout,
            allow_redirects=False,
        )
        
        # Calculate elapsed time
        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create FastAPI response
        content = response.content
        status_code = response.status_code
        
        # Create response headers
        response_headers = dict(response.headers)
        response_headers["X-Proxy-Backend-Time-Ms"] = str(int(elapsed_time))
        response_headers["X-Proxy-Backend-Url"] = T4_GPU_BACKEND_URL
        
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
            headers=response_headers,
        )
    except requests.Timeout as e:
        logger.error(f"Proxy timeout error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={
                "error": "Proxy timeout error",
                "message": f"Backend request timed out after {timeout} seconds",
                "url": url,
                "method": method.upper(),
                "endpoint": path,
                "timeout_value": timeout,
                "backend_url": T4_GPU_BACKEND_URL,
                "error_details": str(e),
                "error_type": "requests.Timeout",
                "recommendation": "Try again later or check if the backend is running properly",
                "timestamp": time.time()
            }
        )
    except requests.ConnectionError as e:
        logger.error(f"Proxy connection error: {str(e)}")
        
        # Run quick DNS check
        dns_result = test_dns_resolution(BACKEND_HOSTNAME)
        
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "error": "Proxy connection error",
                "message": "Cannot connect to the backend server",
                "url": url,
                "method": method.upper(),
                "endpoint": path,
                "backend_url": T4_GPU_BACKEND_URL,
                "dns_resolution": dns_result,
                "error_details": str(e),
                "error_type": "requests.ConnectionError",
                "recommendation": "Check if the backend is running and accessible",
                "timestamp": time.time()
            }
        )
    except Exception as e:
        logger.error(f"Proxy error: {str(e)}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Proxy error",
                "message": str(e),
                "url": url,
                "method": method.upper(),
                "endpoint": path,
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc() if ENVIRONMENT == "development" else None,
                "timestamp": time.time()
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)