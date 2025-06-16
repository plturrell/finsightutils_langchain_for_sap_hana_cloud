"""
ResponseWrapperMiddleware for the SAP HANA LangChain Integration API.

This middleware wraps responses in a standardized format, ensuring consistency
across the API and providing additional metadata with each response.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union, Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

from ..config_standardized import get_standardized_settings
from ..models.base_standardized import APIResponse

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


class ResponseWrapperMiddleware(BaseHTTPMiddleware):
    """Middleware to wrap responses in a standardized format."""
    
    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = None,
        include_metadata: bool = None,
        exclude_paths: List[str] = None,
        exclude_status_codes: List[int] = None,
        processing_time_header: str = None,
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            enabled: Whether to enable response wrapping
            include_metadata: Whether to include metadata in responses
            exclude_paths: List of paths to exclude from wrapping
            exclude_status_codes: List of status codes to exclude from wrapping
            processing_time_header: Name of the header for processing time
        """
        super().__init__(app)
        
        # Set defaults from settings if not provided
        self.enabled = enabled if enabled is not None else settings.api.wrap_responses
        self.include_metadata = include_metadata if include_metadata is not None else settings.api.include_metadata
        self.exclude_paths = exclude_paths or settings.api.exclude_wrap_paths
        self.exclude_status_codes = exclude_status_codes or [204]  # No Content
        self.processing_time_header = processing_time_header or "X-Process-Time"
        
        # Add standard excluded paths
        self.exclude_paths.extend([
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static",
            "/public",
        ])
        
        logger.info(f"Response wrapper middleware initialized (enabled: {self.enabled})")
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and wrap the response.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Wrapped response
        """
        # Skip wrapping if disabled
        if not self.enabled:
            return await call_next(request)
        
        # Check if this path should be excluded
        path = request.url.path
        for excluded_path in self.exclude_paths:
            if path.startswith(excluded_path):
                return await call_next(request)
        
        # Special handling for Arrow Flight endpoints
        if path.startswith("/api/flight"):
            # Arrow Flight has specific response format requirements
            # We should not wrap these responses
            return await call_next(request)
        
        # Start timing
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Add processing time header
        response.headers[self.processing_time_header] = str(processing_time)
        
        # Skip wrapping for excluded status codes
        if response.status_code in self.exclude_status_codes:
            return response
        
        # Skip wrapping for non-JSON responses
        content_type = response.headers.get("content-type", "").lower()
        if not content_type.startswith("application/json"):
            return response
        
        # Skip wrapping for streaming responses
        if isinstance(response, StreamingResponse):
            return response
        
        # Get request ID from state if available
        request_id = getattr(request.state, "request_id", None)
        
        try:
            # Get response body
            body = getattr(response, "body", None)
            if not body:
                return response
            
            # Parse response body as JSON
            try:
                data = json.loads(body.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Not valid JSON, return as is
                return response
            
            # Skip if already wrapped
            if isinstance(data, dict) and "data" in data and "metadata" in data:
                return response
            
            # Create metadata
            metadata = {}
            if self.include_metadata:
                metadata = {
                    "request_id": request_id,
                    "processing_time": processing_time,
                    "timestamp": time.time(),
                    "api_version": settings.api.version,
                }
            
            # Create wrapped response
            wrapped_data = APIResponse(
                data=data,
                metadata=metadata,
            ).dict()
            
            # Create new response
            return JSONResponse(
                content=wrapped_data,
                status_code=response.status_code,
                headers=dict(response.headers),
            )
        except Exception as e:
            # Log the error
            logger.error(f"Error wrapping response: {str(e)}", exc_info=True)
            
            # Return original response
            return response


def configure_response_wrapper(
    app: FastAPI,
    enabled: bool = None,
    include_metadata: bool = None,
    exclude_paths: List[str] = None,
    exclude_status_codes: List[int] = None,
    processing_time_header: str = None,
) -> None:
    """
    Configure response wrapper for the application.
    
    This can be used to apply the wrapper only to specific routes instead of using
    the middleware for all routes.
    
    Args:
        app: FastAPI application
        enabled: Whether to enable response wrapping
        include_metadata: Whether to include metadata in responses
        exclude_paths: List of paths to exclude from wrapping
        exclude_status_codes: List of status codes to exclude from wrapping
        processing_time_header: Name of the header for processing time
    """
    # Set defaults from settings if not provided
    enabled = enabled if enabled is not None else settings.api.wrap_responses
    include_metadata = include_metadata if include_metadata is not None else settings.api.include_metadata
    exclude_paths = exclude_paths or settings.api.exclude_wrap_paths
    exclude_status_codes = exclude_status_codes or [204]  # No Content
    processing_time_header = processing_time_header or "X-Process-Time"
    
    # Add standard excluded paths
    exclude_paths.extend([
        "/docs",
        "/redoc",
        "/openapi.json",
        "/static",
        "/public",
    ])
    
    # Skip if wrapping is disabled
    if not enabled:
        return
    
    # Log configuration
    logger.info(f"Configuring response wrapper (enabled: {enabled})")
    
    # Add response wrapper middleware
    app.add_middleware(
        ResponseWrapperMiddleware,
        enabled=enabled,
        include_metadata=include_metadata,
        exclude_paths=exclude_paths,
        exclude_status_codes=exclude_status_codes,
        processing_time_header=processing_time_header,
    )