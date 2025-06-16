"""
Middleware components for the SAP HANA LangChain Integration API.

This package contains all middleware components used by the FastAPI application,
providing cross-cutting concerns such as authentication, logging, error handling,
and more.
"""

from .auth_middleware import AuthMiddleware
from .cors_middleware import configure_cors
from .error_handler_middleware import ErrorHandlerMiddleware
from .logging_middleware import LoggingMiddleware
from .rate_limit_middleware import RateLimitMiddleware
from .request_id_middleware import RequestIdMiddleware
from .response_wrapper_middleware import ResponseWrapperMiddleware
from .security_headers_middleware import SecurityHeadersMiddleware
from .telemetry_middleware import TelemetryMiddleware
from .gpu_middleware import GPUMiddleware
from .arrow_flight_middleware import ArrowFlightMiddleware

# Register all middlewares in order of execution (first to last)
# Note: FastAPI executes middleware in reverse order (last to first)
MIDDLEWARE_REGISTRY = [
    SecurityHeadersMiddleware,  # Executed last (add security headers to response)
    ResponseWrapperMiddleware,  # Wrap responses in standardized format
    ErrorHandlerMiddleware,     # Handle exceptions
    RateLimitMiddleware,        # Rate limiting
    TelemetryMiddleware,        # Collect telemetry
    GPUMiddleware,              # GPU state management
    ArrowFlightMiddleware,      # Arrow Flight handling
    LoggingMiddleware,          # Log requests and responses
    AuthMiddleware,             # Authentication
    RequestIdMiddleware,        # Generate request ID
    configure_cors,             # CORS configuration (function, not middleware class)
]

__all__ = [
    'AuthMiddleware',
    'configure_cors',
    'ErrorHandlerMiddleware',
    'LoggingMiddleware',
    'RateLimitMiddleware',
    'RequestIdMiddleware',
    'ResponseWrapperMiddleware',
    'SecurityHeadersMiddleware',
    'TelemetryMiddleware',
    'GPUMiddleware',
    'ArrowFlightMiddleware',
    'MIDDLEWARE_REGISTRY',
]