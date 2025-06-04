"""
Middleware specifically for Vercel deployment.

This middleware adds CORS, error handling, and context-aware error messages
for Vercel serverless functions, ensuring robust deployment even when
SAP HANA Cloud connection is not available.
"""

import time
import logging
import traceback
import json
from typing import Dict, Any, Optional, Callable, List, Union

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import ValidationError

try:
    from langchain_hana.error_utils import create_context_aware_error
    HAS_ERROR_UTILS = True
except ImportError:
    HAS_ERROR_UTILS = False

logger = logging.getLogger("vercel_middleware")

# Operation type mapping based on URL path patterns
PATH_TO_OPERATION_TYPE = {
    "/api/search": "similarity_search",
    "/api/docs": "documentation",
    "/api/vectorstore": "vector_store_management",
    "/api/texts": "add_texts",
    "/api/delete": "delete",
    "/api/health": "health_check",
    "/api/feature": "feature_info",
    "/api/deployment": "deployment_info",
    "/api/knowledgegraph": "knowledge_graph",
}

class VercelErrorMiddleware(BaseHTTPMiddleware):
    """
    Context-aware error handling middleware for Vercel deployment.
    
    This middleware provides:
    1. Comprehensive error handling with context-aware suggestions
    2. Request timing information
    3. Integration with SAP HANA error handling utilities when available
    4. Fallback handling when database connection is not available
    5. Detailed logging for debugging
    """
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        request_id = f"{time.time()}-{id(request)}"
        
        # Add request ID to request state for logging
        request.state.request_id = request_id
        
        # Log request information (excluding sensitive data)
        logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
        
        try:
            # Process the request
            response = await call_next(request)
            
            # Add timing information
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Vercel-Deployment"] = "1"
            
            logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.4f}s")
            return response
            
        except HTTPException as http_exc:
            # Handle HTTP exceptions (raised explicitly by route handlers)
            process_time = time.time() - start_time
            logger.warning(f"HTTP Exception in request {request_id}: {http_exc.status_code} - {http_exc.detail}")
            
            # Extract operation type from path
            operation_type = self._get_operation_type(request.url.path)
            
            # Create error response
            error_response = self._create_error_response(
                error=http_exc.detail if isinstance(http_exc.detail, str) else str(http_exc.detail),
                status_code=http_exc.status_code,
                operation_type=operation_type,
                request_id=request_id,
                process_time=process_time,
                additional_context=http_exc.detail if isinstance(http_exc.detail, dict) else None
            )
            
            # Create response with headers
            response = JSONResponse(
                status_code=http_exc.status_code,
                content=error_response,
            )
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Vercel-Deployment"] = "1"
            return response
            
        except ValidationError as val_exc:
            # Handle Pydantic validation errors
            process_time = time.time() - start_time
            logger.warning(f"Validation error in request {request_id}: {str(val_exc)}")
            
            # Extract field information from validation error
            fields = []
            for error in val_exc.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                fields.append(f"{field_path}: {error['msg']}")
            
            # Create error response
            error_response = self._create_error_response(
                error=str(val_exc),
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                operation_type=self._get_operation_type(request.url.path),
                request_id=request_id,
                process_time=process_time,
                additional_context={
                    "field_errors": fields,
                    "suggestion": "Please check the request body format and field types"
                }
            )
            
            # Create response with headers
            response = JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                content=error_response,
            )
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Vercel-Deployment"] = "1"
            return response
            
        except Exception as exc:
            # Handle unexpected exceptions
            process_time = time.time() - start_time
            logger.exception(f"Unhandled exception in request {request_id}: {str(exc)}")
            
            # Extract operation type from path
            operation_type = self._get_operation_type(request.url.path)
            
            # Get exception traceback for debugging
            tb = traceback.format_exc()
            
            # Create context-aware error response
            error_response = self._create_error_response(
                error=str(exc),
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                operation_type=operation_type,
                request_id=request_id,
                process_time=process_time,
                traceback=tb
            )
            
            # Create response with headers
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=error_response,
            )
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Vercel-Deployment"] = "1"
            return response
    
    def _get_operation_type(self, path: str) -> str:
        """
        Determine the operation type from the request path.
        
        Args:
            path: The request URL path
            
        Returns:
            str: The operation type identifier
        """
        # Try to match path with known patterns
        for pattern, op_type in PATH_TO_OPERATION_TYPE.items():
            if pattern in path:
                return op_type
        
        # Default operation type
        return "api_request"
    
    def _create_error_response(
        self,
        error: str,
        status_code: int,
        operation_type: str,
        request_id: str,
        process_time: float,
        traceback: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a context-aware error response.
        
        This method uses langchain_hana.error_utils if available, otherwise
        it falls back to a generic error response.
        
        Args:
            error: The error message
            status_code: The HTTP status code
            operation_type: The type of operation being performed
            request_id: The request identifier
            process_time: The request processing time
            traceback: Optional exception traceback for debugging
            additional_context: Optional additional context for the error
            
        Returns:
            Dict[str, Any]: The error response dictionary
        """
        # Use context-aware error handling if available
        if HAS_ERROR_UTILS:
            try:
                # Augment additional context with request information
                context = additional_context or {}
                context.update({
                    "request_id": request_id,
                    "processing_time": process_time,
                    "status_code": status_code
                })
                
                # Create context-aware error with intelligent suggestions
                error_info = create_context_aware_error(
                    error=error,
                    operation_type=operation_type,
                    additional_context=context
                )
                
                # Add traceback for 500 errors in non-production environments
                if status_code >= 500 and traceback:
                    error_info["traceback"] = traceback
                
                return error_info
                
            except Exception as e:
                # If error_utils fails, fall back to generic error
                logger.warning(f"Error using context-aware error handling: {str(e)}")
        
        # Generate a generic error response
        error_type = "internal_server_error"
        if status_code == 400:
            error_type = "bad_request"
        elif status_code == 401:
            error_type = "unauthorized"
        elif status_code == 403:
            error_type = "forbidden"
        elif status_code == 404:
            error_type = "not_found"
        elif status_code == 422:
            error_type = "validation_error"
            
        # Create generic response
        response = {
            "error": error_type,
            "message": error,
            "context": {
                "operation": operation_type,
                "request_id": request_id,
                "processing_time": process_time,
                "suggestion": self._get_suggestion_for_operation(operation_type, status_code)
            }
        }
        
        # Add additional context if provided
        if additional_context:
            response["context"].update(additional_context)
            
        # Add traceback for 500 errors in non-production environments
        if status_code >= 500 and traceback:
            response["traceback"] = traceback
            
        return response
    
    def _get_suggestion_for_operation(self, operation_type: str, status_code: int) -> str:
        """
        Get a helpful suggestion based on the operation type and status code.
        
        Args:
            operation_type: The type of operation being performed
            status_code: The HTTP status code
            
        Returns:
            str: A helpful suggestion for resolving the error
        """
        # Default suggestion
        default_suggestion = "Please check your request parameters and try again"
        
        # Operation-specific suggestions
        operation_suggestions = {
            "similarity_search": {
                400: "Check your search query parameters and metadata filter format",
                500: "The search operation failed. Please verify your database connection and vector store configuration"
            },
            "add_texts": {
                400: "Verify your document format and metadata structure",
                500: "Adding documents failed. Check your database connection and vector table configuration"
            },
            "delete": {
                400: "Check your delete filter criteria format",
                500: "Delete operation failed. Verify your database connection and permissions"
            },
            "vector_store_management": {
                400: "Verify your vector store configuration parameters",
                500: "Vector store management operation failed. Check database connection and permissions"
            }
        }
        
        # Get suggestion for specific operation and status code
        if operation_type in operation_suggestions and status_code in operation_suggestions[operation_type]:
            return operation_suggestions[operation_type][status_code]
            
        # Generic suggestions by status code
        code_suggestions = {
            400: "Please check your request parameters for errors",
            401: "Authentication required. Please provide valid credentials",
            403: "You don't have permission to perform this operation",
            404: "The requested resource was not found",
            422: "Validation error in request data. Please check field types and formats",
            500: "Internal server error occurred. Please try again later or contact support"
        }
        
        # Return suggestion by status code or default
        return code_suggestions.get(status_code, default_suggestion)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware for Vercel deployment.
    
    This middleware provides:
    1. Basic rate limiting based on client IP address
    2. Sliding window rate limiting with adjustable window size
    3. Different limits for different endpoint types
    4. Protection against excessive load in serverless environments
    """
    
    def __init__(
        self,
        app: FastAPI,
        requests_per_minute: int = 60,
        search_requests_per_minute: int = 20,
        window_size: int = 60
    ):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: The FastAPI application
            requests_per_minute: Maximum requests per minute for general endpoints
            search_requests_per_minute: Maximum requests per minute for search endpoints
            window_size: The sliding window size in seconds
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.search_requests_per_minute = search_requests_per_minute
        self.window_size = window_size
        self.request_history = {}  # IP -> list of timestamps
    
    async def dispatch(self, request: Request, call_next):
        # Get client IP address
        client_ip = self._get_client_ip(request)
        
        # Clean up old requests
        self._cleanup_old_requests()
        
        # Get current time
        current_time = time.time()
        
        # Add request to history
        if client_ip not in self.request_history:
            self.request_history[client_ip] = []
        
        self.request_history[client_ip].append(current_time)
        
        # Determine endpoint type and rate limit
        path = request.url.path
        is_search_endpoint = "/search" in path
        
        # Calculate rate limit
        rate_limit = self.search_requests_per_minute if is_search_endpoint else self.requests_per_minute
        
        # Check if rate limit exceeded
        request_count = len(self.request_history[client_ip])
        if request_count > rate_limit:
            # Remove oldest request to maintain window
            self.request_history[client_ip].pop(0)
            
            # Return rate limit exceeded response
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "context": {
                        "limit": rate_limit,
                        "window_size": self.window_size,
                        "retry_after": 60,  # Suggest retry after 60 seconds
                        "endpoint_type": "search" if is_search_endpoint else "general"
                    }
                }
            )
            response.headers["Retry-After"] = "60"
            return response
        
        # Process the request
        return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """
        Get the client IP address from the request.
        
        Args:
            request: The FastAPI request
            
        Returns:
            str: The client IP address
        """
        # Check for forwarded IP (for clients behind proxies)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP in the list
            return forwarded_for.split(",")[0]
        
        # Get client host from request
        return request.client.host if request.client else "unknown"
    
    def _cleanup_old_requests(self):
        """Clean up old requests outside the sliding window."""
        current_time = time.time()
        window_start = current_time - self.window_size
        
        # For each client IP
        for ip in list(self.request_history.keys()):
            # Filter out requests older than window start
            self.request_history[ip] = [
                ts for ts in self.request_history[ip] if ts >= window_start
            ]
            
            # Remove empty lists
            if not self.request_history[ip]:
                del self.request_history[ip]


def setup_middleware(app: FastAPI, enable_rate_limiting: bool = False):
    """
    Set up all middleware for Vercel deployment.
    
    Args:
        app: The FastAPI application
        enable_rate_limiting: Whether to enable rate limiting
        
    Returns:
        FastAPI: The FastAPI application with middleware added
    """
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add error handling middleware
    app.add_middleware(VercelErrorMiddleware)
    
    # Add rate limiting middleware if enabled
    if enable_rate_limiting:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=60,
            search_requests_per_minute=20,
            window_size=60
        )
    
    return app