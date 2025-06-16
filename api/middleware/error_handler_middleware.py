"""
ErrorHandlerMiddleware for the SAP HANA LangChain Integration API.

This middleware catches exceptions and converts them to standardized API responses,
ensuring consistent error handling across the application.
"""

import traceback
import logging
from typing import Dict, Any, Callable, Optional, Type

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_429_TOO_MANY_REQUESTS,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE
)

from ..config_standardized import get_standardized_settings
from ..models.base_standardized import ErrorResponse
from ..utils.standardized_exceptions import BaseAPIException

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware to handle exceptions and convert them to standardized responses."""
    
    def __init__(
        self,
        app: FastAPI,
        debug: bool = None,
        include_exception_details: bool = None,
        error_handlers: Dict[Any, Callable] = None
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            debug: Whether to include debug information in error responses
            include_exception_details: Whether to include exception details in error responses
            error_handlers: Dictionary mapping exception types to handler functions
        """
        super().__init__(app)
        self.debug = debug if debug is not None else settings.environment.debug
        self.include_exception_details = (
            include_exception_details
            if include_exception_details is not None
            else settings.api.include_exception_details
        )
        
        # Default error handlers
        self.error_handlers = {
            ValidationError: self.handle_validation_error,
            BaseAPIException: self.handle_api_exception,
            **error_handlers or {}
        }
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and handle any exceptions.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response from the next middleware or route handler or error response
        """
        # Get request ID from state if available
        request_id = getattr(request.state, "request_id", None)
        
        try:
            # Process the request
            return await call_next(request)
        except Exception as e:
            # Create a context for logging
            log_context = {
                "request_id": request_id,
                "path": request.url.path,
                "method": request.method,
                "exception": str(e),
                "exception_type": e.__class__.__name__
            }
            
            # Log the exception
            logger.error(f"Error processing request: {e}", exc_info=True, extra=log_context)
            
            # Find an appropriate handler for the exception
            for exc_type, handler in self.error_handlers.items():
                if isinstance(e, exc_type):
                    return handler(request, e)
            
            # If no specific handler found, use the default handler
            return self.handle_default_exception(request, e)
    
    def handle_validation_error(self, request: Request, exc: ValidationError) -> JSONResponse:
        """
        Handle Pydantic validation errors.
        
        Args:
            request: FastAPI request object
            exc: Validation error
            
        Returns:
            JSONResponse with validation error details
        """
        # Get request ID from state if available
        request_id = getattr(request.state, "request_id", None)
        
        # Extract error details for better readability
        error_details = []
        for error in exc.errors():
            # Make the error location more human-readable
            loc_str = " -> ".join([str(loc) for loc in error.get("loc", [])])
            error_details.append({
                "location": loc_str,
                "message": error.get("msg"),
                "type": error.get("type")
            })
        
        # Create error response
        error_response = ErrorResponse(
            status_code=HTTP_400_BAD_REQUEST,
            error_code="validation_error",
            message="Validation error in request data",
            details={"errors": error_details},
            request_id=request_id,
            suggestion="Please check the request format and try again"
        )
        
        return JSONResponse(
            status_code=HTTP_400_BAD_REQUEST,
            content=error_response.dict()
        )
    
    def handle_api_exception(self, request: Request, exc: BaseAPIException) -> JSONResponse:
        """
        Handle standardized API exceptions.
        
        Args:
            request: FastAPI request object
            exc: API exception
            
        Returns:
            JSONResponse with error details
        """
        # Get request ID from state if available
        request_id = getattr(request.state, "request_id", None)
        
        # Special handling for Arrow Flight endpoints
        path = request.url.path
        if path.startswith("/api/flight"):
            # Arrow Flight has special error handling requirements
            # We need to return a specific format for compatibility with Arrow Flight clients
            # This is specific to the PyArrow Flight implementation
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "code": exc.error_code,
                    "message": exc.detail,
                    "details": exc.details,
                },
                headers=exc.headers
            )
        
        # Create error response
        error_response = ErrorResponse(
            status_code=exc.status_code,
            error_code=exc.error_code,
            message=exc.detail,
            details=exc.details,
            suggestion=exc.suggestion,
            docs_url=exc.docs_url,
            request_id=request_id
        )
        
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response.dict(),
            headers=exc.headers
        )
    
    def handle_default_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """
        Handle all other exceptions.
        
        Args:
            request: FastAPI request object
            exc: Exception
            
        Returns:
            JSONResponse with error details
        """
        # Get request ID from state if available
        request_id = getattr(request.state, "request_id", None)
        
        # Special handling for Arrow Flight endpoints
        path = request.url.path
        if path.startswith("/api/flight"):
            # Arrow Flight has special error handling requirements
            return JSONResponse(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "code": "internal_error",
                    "message": "An unexpected error occurred",
                    "details": str(exc) if self.debug else None,
                }
            )
        
        # Create details dictionary
        details = None
        if self.debug or self.include_exception_details:
            details = {
                "exception_type": exc.__class__.__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc()
            }
        
        # Create error response
        error_response = ErrorResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="internal_server_error",
            message="An unexpected error occurred",
            details=details,
            request_id=request_id,
            suggestion="Please try again later or contact support"
        )
        
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_response.dict()
        )


def setup_error_handler_middleware(
    app: FastAPI,
    debug: bool = None,
    include_exception_details: bool = None,
    error_handlers: Dict[Any, Callable] = None
) -> None:
    """
    Configure and add the error handler middleware to the application.
    
    Args:
        app: FastAPI application
        debug: Whether to include debug information in error responses
        include_exception_details: Whether to include exception details in error responses
        error_handlers: Dictionary mapping exception types to handler functions
    """
    app.add_middleware(
        ErrorHandlerMiddleware,
        debug=debug,
        include_exception_details=include_exception_details,
        error_handlers=error_handlers
    )