"""
CORS middleware for the SAP HANA LangChain Integration API.

This module configures Cross-Origin Resource Sharing (CORS) for the FastAPI application,
allowing controlled access from different origins.
"""

import logging
from typing import List, Dict, Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..config_standardized import get_standardized_settings

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


def configure_cors(
    app: FastAPI,
    allow_origins: List[str] = None,
    allow_methods: List[str] = None,
    allow_headers: List[str] = None,
    allow_credentials: bool = None,
    expose_headers: List[str] = None,
    max_age: int = None
) -> None:
    """
    Configure CORS for the FastAPI application.
    
    Args:
        app: FastAPI application
        allow_origins: List of allowed origins
        allow_methods: List of allowed HTTP methods
        allow_headers: List of allowed HTTP headers
        allow_credentials: Whether to allow credentials
        expose_headers: List of headers to expose
        max_age: Max age for CORS preflight requests in seconds
    """
    # Use default values from settings if not provided
    allow_origins = allow_origins or settings.cors.origins
    allow_methods = allow_methods or settings.cors.allow_methods
    allow_headers = allow_headers or settings.cors.allow_headers
    allow_credentials = allow_credentials if allow_credentials is not None else settings.cors.allow_credentials
    expose_headers = expose_headers or settings.cors.expose_headers
    max_age = max_age or settings.cors.max_age
    
    # If running in development mode, allow localhost origins
    if settings.environment.development_mode and allow_origins != ["*"]:
        # Add common development origins if not using wildcard
        dev_origins = [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8000",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
            "http://127.0.0.1:8080",
        ]
        
        # Add to allowed origins if not already included
        for origin in dev_origins:
            if origin not in allow_origins:
                allow_origins.append(origin)
    
    # Log CORS configuration
    logger.info(f"Configuring CORS with allowed origins: {allow_origins}")
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=expose_headers,
        max_age=max_age,
    )


def setup_cors_for_arrow_flight(
    app: FastAPI,
    allow_origins: List[str] = None,
    arrow_flight_path_prefix: str = "/api/flight"
) -> None:
    """
    Configure special CORS settings for Arrow Flight endpoints.
    
    Arrow Flight may require different CORS settings than the rest of the API.
    This function adds specific settings for Arrow Flight endpoints.
    
    Args:
        app: FastAPI application
        allow_origins: List of allowed origins for Arrow Flight
        arrow_flight_path_prefix: Path prefix for Arrow Flight endpoints
    """
    # Use default values from settings if not provided
    allow_origins = allow_origins or settings.arrow_flight.cors_origins or ["*"]
    
    # Arrow Flight specific CORS settings
    arrow_flight_headers = [
        "Content-Type",
        "Authorization",
        "X-Api-Key",
        "X-Flight-Token",
        "X-Arrow-Flight-Protocol",
        "X-Arrow-Flight-SQL-Protocol",
        "Arrow-Flight-Protocol",
        "Arrow-Flight-SQL-Protocol",
    ]
    
    logger.info(f"Configuring special CORS for Arrow Flight with allowed origins: {allow_origins}")
    
    # Log that we're using Arrow Flight specific CORS
    logger.info(f"Using Arrow Flight specific CORS for paths starting with: {arrow_flight_path_prefix}")
    
    # This is a workaround for FastAPI's limitation with path-specific CORS
    # We add a middleware that will apply special CORS headers for Arrow Flight paths
    @app.middleware("http")
    async def arrow_flight_cors_middleware(request, call_next):
        response = await call_next(request)
        
        # Check if this is an Arrow Flight path
        if request.url.path.startswith(arrow_flight_path_prefix):
            # Set Arrow Flight specific CORS headers
            origin = request.headers.get("origin")
            if origin and (origin in allow_origins or "*" in allow_origins):
                response.headers["Access-Control-Allow-Origin"] = origin
                response.headers["Access-Control-Allow-Credentials"] = "true"
                response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
                response.headers["Access-Control-Allow-Headers"] = ", ".join(arrow_flight_headers)
                response.headers["Access-Control-Expose-Headers"] = "X-Flight-Token, X-Arrow-Flight-Protocol"
                response.headers["Access-Control-Max-Age"] = "3600"
        
        return response