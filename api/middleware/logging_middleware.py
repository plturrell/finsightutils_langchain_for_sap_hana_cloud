"""
LoggingMiddleware for the SAP HANA LangChain Integration API.

This middleware logs request and response information, providing insights into
API usage, performance, and potential issues.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..config_standardized import get_standardized_settings

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""
    
    def __init__(
        self,
        app: FastAPI,
        level: int = None,
        exclude_paths: List[str] = None,
        exclude_methods: List[str] = None,
        log_request_body: bool = None,
        log_response_body: bool = None,
        log_headers: bool = None,
        max_body_length: int = None,
        sensitive_headers: Set[str] = None,
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            level: Logging level (default from settings)
            exclude_paths: List of paths to exclude from logging
            exclude_methods: List of HTTP methods to exclude from logging
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            log_headers: Whether to log request and response headers
            max_body_length: Maximum length of logged bodies
            sensitive_headers: Set of headers to redact
        """
        super().__init__(app)
        
        # Set defaults from settings if not provided
        self.level = level or getattr(logging, settings.logging.level.upper(), logging.INFO)
        self.exclude_paths = exclude_paths or settings.logging.exclude_paths
        self.exclude_methods = exclude_methods or settings.logging.exclude_methods
        self.log_request_body = log_request_body if log_request_body is not None else settings.logging.log_request_body
        self.log_response_body = log_response_body if log_response_body is not None else settings.logging.log_response_body
        self.log_headers = log_headers if log_headers is not None else settings.logging.log_headers
        self.max_body_length = max_body_length or settings.logging.max_body_length
        
        # Sensitive headers to redact
        self.sensitive_headers = {h.lower() for h in (sensitive_headers or {
            "authorization", "x-api-key", "cookie", "set-cookie"
        })}
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and log details.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response from the next middleware or route handler
        """
        # Skip logging for excluded paths and methods
        if self._should_skip_logging(request):
            return await call_next(request)
        
        # Get request info
        request_id = getattr(request.state, "request_id", None)
        path = request.url.path
        method = request.method
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request)
        
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        self._log_response(request, response, duration_ms)
        
        return response
    
    def _should_skip_logging(self, request: Request) -> bool:
        """
        Check if logging should be skipped for this request.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if logging should be skipped, False otherwise
        """
        # Check path exclusions
        path = request.url.path
        for excluded_path in self.exclude_paths:
            if path.startswith(excluded_path):
                return True
        
        # Check method exclusions
        if request.method.upper() in [m.upper() for m in self.exclude_methods]:
            return True
        
        return False
    
    async def _log_request(self, request: Request) -> None:
        """
        Log details about the request.
        
        Args:
            request: FastAPI request object
        """
        request_id = getattr(request.state, "request_id", None)
        path = request.url.path
        method = request.method
        client_host = request.client.host if request.client else None
        query_params = dict(request.query_params)
        
        # Prepare log data
        log_data = {
            "request_id": request_id,
            "client_ip": client_host,
            "method": method,
            "path": path,
            "query_params": query_params or None,
        }
        
        # Log headers if enabled
        if self.log_headers:
            headers = dict(request.headers)
            # Redact sensitive headers
            for header in self.sensitive_headers:
                if header in headers:
                    headers[header] = "[REDACTED]"
            log_data["headers"] = headers
        
        # Log request body if enabled
        if self.log_request_body:
            try:
                body = await request.body()
                if body:
                    body_str = body.decode("utf-8")
                    if len(body_str) > self.max_body_length:
                        body_str = f"{body_str[:self.max_body_length]}... (truncated)"
                    
                    # Try to parse as JSON
                    try:
                        body_json = json.loads(body_str)
                        log_data["body"] = body_json
                    except:
                        log_data["body"] = body_str
            except Exception as e:
                log_data["body_error"] = f"Error reading request body: {str(e)}"
        
        # Log the request
        logger.log(self.level, f"Request: {method} {path}", extra={"request_data": log_data})
    
    def _log_response(self, request: Request, response: Response, duration_ms: float) -> None:
        """
        Log details about the response.
        
        Args:
            request: FastAPI request object
            response: Response object
            duration_ms: Request duration in milliseconds
        """
        request_id = getattr(request.state, "request_id", None)
        path = request.url.path
        method = request.method
        status_code = response.status_code
        
        # Prepare log data
        log_data = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }
        
        # Log headers if enabled
        if self.log_headers:
            headers = dict(response.headers)
            # Redact sensitive headers
            for header in self.sensitive_headers:
                if header in headers:
                    headers[header] = "[REDACTED]"
            log_data["headers"] = headers
        
        # Log response body if enabled
        if self.log_response_body:
            try:
                body = getattr(response, "body", None)
                if body:
                    try:
                        body_str = body.decode("utf-8") if isinstance(body, bytes) else str(body)
                        if len(body_str) > self.max_body_length:
                            body_str = f"{body_str[:self.max_body_length]}... (truncated)"
                        
                        # Try to parse as JSON
                        try:
                            body_json = json.loads(body_str)
                            
                            # For vector responses, don't log the full vectors
                            if isinstance(body_json, dict) and "embeddings" in body_json:
                                if isinstance(body_json["embeddings"], list):
                                    # Only include the count and dimensions of embeddings
                                    dims = len(body_json["embeddings"][0]) if body_json["embeddings"] else 0
                                    log_data["body"] = {
                                        **{k: v for k, v in body_json.items() if k != "embeddings"},
                                        "embeddings": f"[{len(body_json['embeddings'])} embeddings, dimensions: {dims}]"
                                    }
                                else:
                                    log_data["body"] = body_json
                            else:
                                log_data["body"] = body_json
                        except:
                            log_data["body"] = body_str
                    except:
                        log_data["body"] = "[non-string body]"
            except Exception as e:
                log_data["body_error"] = f"Error reading response body: {str(e)}"
        
        # Determine log level based on status code
        level = self.level
        if status_code >= 500:
            level = logging.ERROR
        elif status_code >= 400:
            level = logging.WARNING
        
        # Log the response
        logger.log(level, f"Response: {status_code} {method} {path} ({duration_ms:.2f}ms)", extra={"response_data": log_data})


def setup_logging_middleware(
    app: FastAPI,
    level: int = None,
    exclude_paths: List[str] = None,
    exclude_methods: List[str] = None,
    log_request_body: bool = None,
    log_response_body: bool = None,
    log_headers: bool = None,
    max_body_length: int = None,
    sensitive_headers: Set[str] = None,
) -> None:
    """
    Configure and add the logging middleware to the application.
    
    Args:
        app: FastAPI application
        level: Logging level
        exclude_paths: List of paths to exclude from logging
        exclude_methods: List of HTTP methods to exclude from logging
        log_request_body: Whether to log request bodies
        log_response_body: Whether to log response bodies
        log_headers: Whether to log request and response headers
        max_body_length: Maximum length of logged bodies
        sensitive_headers: Set of headers to redact
    """
    app.add_middleware(
        LoggingMiddleware,
        level=level,
        exclude_paths=exclude_paths,
        exclude_methods=exclude_methods,
        log_request_body=log_request_body,
        log_response_body=log_response_body,
        log_headers=log_headers,
        max_body_length=max_body_length,
        sensitive_headers=sensitive_headers,
    )