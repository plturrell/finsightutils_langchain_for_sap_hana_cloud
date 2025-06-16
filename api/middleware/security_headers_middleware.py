"""
SecurityHeadersMiddleware for the SAP HANA LangChain Integration API.

This middleware adds security headers to responses to protect against common web vulnerabilities.
"""

import logging
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..config_standardized import get_standardized_settings

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers to responses."""
    
    def __init__(
        self,
        app: FastAPI,
        headers: Dict[str, str] = None,
        hsts: bool = None,
        hsts_max_age: int = None,
        hsts_include_subdomains: bool = None,
        hsts_preload: bool = None,
        content_security_policy: str = None,
        exclude_paths: List[str] = None,
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            headers: Custom security headers to add
            hsts: Whether to add HSTS header
            hsts_max_age: HSTS max age in seconds
            hsts_include_subdomains: Whether to include subdomains in HSTS
            hsts_preload: Whether to enable HSTS preload
            content_security_policy: Content Security Policy
            exclude_paths: List of paths to exclude from adding security headers
        """
        super().__init__(app)
        
        # Set defaults from settings if not provided
        self.hsts = hsts if hsts is not None else settings.security.hsts_enabled
        self.hsts_max_age = hsts_max_age or settings.security.hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains if hsts_include_subdomains is not None else settings.security.hsts_include_subdomains
        self.hsts_preload = hsts_preload if hsts_preload is not None else settings.security.hsts_preload
        self.content_security_policy = content_security_policy or settings.security.content_security_policy
        self.exclude_paths = exclude_paths or settings.security.exclude_paths
        
        # Default security headers
        self.default_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-XSS-Protection": "1; mode=block",
            "X-Frame-Options": "SAMEORIGIN",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        
        # Add Content-Security-Policy if provided
        if self.content_security_policy:
            self.default_headers["Content-Security-Policy"] = self.content_security_policy
        
        # Add HSTS header if enabled
        if self.hsts:
            hsts_header = f"max-age={self.hsts_max_age}"
            if self.hsts_include_subdomains:
                hsts_header += "; includeSubDomains"
            if self.hsts_preload:
                hsts_header += "; preload"
            self.default_headers["Strict-Transport-Security"] = hsts_header
        
        # Merge with custom headers
        self.headers = {**self.default_headers, **(headers or {})}
        
        logger.info("Security headers middleware initialized")
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and add security headers to the response.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response with security headers
        """
        # Process the request
        response = await call_next(request)
        
        # Check if this path should be excluded
        path = request.url.path
        for excluded_path in self.exclude_paths:
            if path.startswith(excluded_path):
                return response
        
        # Special handling for Arrow Flight endpoints
        if path.startswith("/api/flight"):
            # Only add minimal security headers for Arrow Flight
            # to avoid conflicts with Arrow Flight protocol
            response.headers["X-Content-Type-Options"] = "nosniff"
            return response
        
        # Add security headers
        for name, value in self.headers.items():
            # Don't override existing headers
            if name not in response.headers:
                response.headers[name] = value
        
        return response


def setup_security_headers_middleware(
    app: FastAPI,
    headers: Dict[str, str] = None,
    hsts: bool = None,
    hsts_max_age: int = None,
    hsts_include_subdomains: bool = None,
    hsts_preload: bool = None,
    content_security_policy: str = None,
    exclude_paths: List[str] = None,
) -> None:
    """
    Configure and add the security headers middleware to the application.
    
    Args:
        app: FastAPI application
        headers: Custom security headers to add
        hsts: Whether to add HSTS header
        hsts_max_age: HSTS max age in seconds
        hsts_include_subdomains: Whether to include subdomains in HSTS
        hsts_preload: Whether to enable HSTS preload
        content_security_policy: Content Security Policy
        exclude_paths: List of paths to exclude from adding security headers
    """
    app.add_middleware(
        SecurityHeadersMiddleware,
        headers=headers,
        hsts=hsts,
        hsts_max_age=hsts_max_age,
        hsts_include_subdomains=hsts_include_subdomains,
        hsts_preload=hsts_preload,
        content_security_policy=content_security_policy,
        exclude_paths=exclude_paths,
    )