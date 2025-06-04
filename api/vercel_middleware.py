"""
Middleware specifically for Vercel deployment.
This middleware adds CORS and error handling for Vercel serverless functions.
"""

import time
import logging
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI

logger = logging.getLogger("vercel_middleware")

class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware that adds timing information to responses."""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Vercel-Deployment"] = "1"
            return response
        except Exception as e:
            logger.exception(f"Error processing request: {str(e)}")
            process_time = time.time() - start_time
            
            # Return JSON error response with context-aware information
            error_message = str(e)
            status_code = 500
            
            # Try to extract operation type from URL path
            path = request.url.path
            operation_type = "api_request"
            
            if "query" in path:
                operation_type = "similarity_search"
            elif "texts" in path:
                operation_type = "add_texts"
            elif "delete" in path:
                operation_type = "delete"
            
            # Create context-aware error response
            error_response = {
                "error": "Internal server error",
                "message": error_message,
                "context": {
                    "operation": operation_type,
                    "processing_time": process_time,
                    "suggestion": "Please check your input parameters and try again"
                }
            }
            
            response = JSONResponse(
                status_code=status_code,
                content=error_response,
            )
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Vercel-Deployment"] = "1"
            return response

def setup_middleware(app: FastAPI):
    """Set up middleware for Vercel deployment."""
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add timing middleware
    app.add_middleware(TimingMiddleware)
    
    return app