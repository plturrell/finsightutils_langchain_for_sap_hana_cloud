"""
RequestIdMiddleware for the SAP HANA LangChain Integration API.

This middleware adds a unique request ID to each request, making it possible to trace
requests across the application and in logs.
"""

import uuid
import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..config_standardized import get_standardized_settings

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add a unique request ID to each request and response."""
    
    def __init__(self, app: FastAPI, header_name: str = None):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            header_name: Name of the header to use for the request ID
        """
        super().__init__(app)
        self.header_name = header_name or settings.api.request_id_header or "X-Request-ID"
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and add a request ID.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response with request ID header
        """
        # Check if a request ID is already in the headers
        request_id = request.headers.get(self.header_name)
        
        # If not, generate a unique request ID
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add request ID to request state for access in route handlers
        request.state.request_id = request_id
        
        # Log the request ID
        logger.debug(f"Request {request_id} started")
        
        # Process the request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        # Log the request ID
        logger.debug(f"Request {request_id} completed")
        
        return response


def setup_request_id_middleware(app: FastAPI, header_name: str = None) -> None:
    """
    Configure and add the request ID middleware to the application.
    
    Args:
        app: FastAPI application
        header_name: Name of the header to use for the request ID
    """
    app.add_middleware(RequestIdMiddleware, header_name=header_name)