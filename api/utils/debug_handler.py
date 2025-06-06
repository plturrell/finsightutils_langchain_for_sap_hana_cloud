"""
Debug handler for Vercel deployment

This module provides enhanced error handling and logging specifically 
for Vercel deployments to help diagnose server-side issues.
"""

import os
import sys
import traceback
import logging
import json
from typing import Dict, Any, Optional
from fastapi import Request, Response
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Function to get detailed error information
def get_error_details(exc: Exception) -> Dict[str, Any]:
    """
    Get detailed information about an exception.
    
    Args:
        exc: The exception to analyze
        
    Returns:
        Dict with error details
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_list = traceback.extract_tb(exc_traceback)
    
    # Get the last frame of the traceback (where the error occurred)
    if tb_list:
        last_frame = tb_list[-1]
        filename = last_frame.filename
        lineno = last_frame.lineno
        line = last_frame.line
    else:
        filename = "unknown"
        lineno = 0
        line = "unknown"
    
    # Get the exception chain
    exception_chain = []
    current_exc = exc
    while current_exc:
        exception_chain.append({
            "type": current_exc.__class__.__name__,
            "message": str(current_exc),
        })
        current_exc = current_exc.__cause__
    
    return {
        "exception_type": exc_type.__name__ if exc_type else "Unknown",
        "exception_message": str(exc_value),
        "filename": filename,
        "line_number": lineno,
        "line": line,
        "traceback": traceback.format_exc(),
        "exception_chain": exception_chain,
        "sys_path": sys.path,
        "python_version": sys.version,
        "environment": os.environ.get("ENVIRONMENT", "unknown"),
    }

async def debug_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle exceptions with detailed debug information.
    
    Args:
        request: The FastAPI request
        exc: The exception to handle
        
    Returns:
        JSONResponse with error details
    """
    error_details = get_error_details(exc)
    
    # Log the error with full details
    logger.error(f"Unhandled exception: {error_details['exception_message']}")
    logger.error(f"Details: {json.dumps(error_details, indent=2)}")
    
    # In production, only return minimal information
    if os.environ.get("ENVIRONMENT", "").lower() == "production":
        return JSONResponse(
            status_code=500,
            content={
                "error": "A server error has occurred",
                "request_id": getattr(request.state, "request_id", "unknown"),
                "error_type": error_details["exception_type"],
                "message": "Error logs have been recorded. Contact support with this request_id for assistance."
            }
        )
    else:
        # In development, return full error details
        return JSONResponse(
            status_code=500,
            content={
                "error": "A server error has occurred",
                "request_id": getattr(request.state, "request_id", "unknown"),
                "details": error_details
            }
        )

# Function to verify requests reaching the server
async def log_request(request: Request) -> None:
    """
    Log detailed information about incoming requests
    
    Args:
        request: The FastAPI request
    """
    # Generate request ID if not already present
    if not hasattr(request.state, "request_id"):
        import uuid
        request.state.request_id = f"req_{uuid.uuid4()}"
    
    # Log basic request info
    logger.info(f"Request {request.state.request_id}: {request.method} {request.url.path}")
    
    # Log headers
    headers = {k: v for k, v in request.headers.items()}
    logger.debug(f"Request headers: {json.dumps(headers, indent=2)}")
    
    # Log query params
    params = {k: v for k, v in request.query_params.items()}
    if params:
        logger.debug(f"Query params: {json.dumps(params, indent=2)}")
    
    # Try to log body for non-GET requests
    if request.method != "GET":
        try:
            body = await request.body()
            if body:
                try:
                    # Try to parse as JSON
                    body_str = body.decode("utf-8")
                    body_json = json.loads(body_str)
                    logger.debug(f"Request body (JSON): {json.dumps(body_json, indent=2)}")
                except (UnicodeDecodeError, json.JSONDecodeError):
                    # Log as raw if not JSON
                    logger.debug(f"Request body (raw): {body}")
        except Exception as e:
            logger.warning(f"Could not log request body: {str(e)}")

# Response logging middleware
async def log_response(response: Response, response_body: bytes) -> None:
    """
    Log detailed information about outgoing responses
    
    Args:
        response: The FastAPI response
        response_body: The response body
    """
    # Log basic response info
    logger.info(f"Response status: {response.status_code}")
    
    # Log headers
    headers = {k: v for k, v in response.headers.items()}
    logger.debug(f"Response headers: {json.dumps(headers, indent=2)}")
    
    # Log body
    try:
        body_str = response_body.decode("utf-8")
        try:
            # Try to parse as JSON
            body_json = json.loads(body_str)
            logger.debug(f"Response body (JSON): {json.dumps(body_json, indent=2)}")
        except json.JSONDecodeError:
            # Log as text if not JSON
            if len(body_str) > 1000:
                logger.debug(f"Response body (text, truncated): {body_str[:1000]}...")
            else:
                logger.debug(f"Response body (text): {body_str}")
    except UnicodeDecodeError:
        # Log as raw if not text
        logger.debug(f"Response body (raw): {response_body[:100]}...")