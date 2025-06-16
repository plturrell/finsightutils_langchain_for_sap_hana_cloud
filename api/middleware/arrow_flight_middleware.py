"""
ArrowFlightMiddleware for the SAP HANA LangChain Integration API.

This middleware provides special handling for Arrow Flight protocol integration,
including server setup, authentication, and performance optimization.
"""

import logging
import os
import json
import time
import threading
from typing import Dict, List, Optional, Union, Callable, Any

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_401_UNAUTHORIZED, HTTP_500_INTERNAL_SERVER_ERROR

from ..config_standardized import get_standardized_settings
from ..utils.standardized_exceptions import (
    ArrowFlightException,
    ArrowFlightServerException,
    ArrowFlightAuthException,
)

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Try to import PyArrow Flight
try:
    import pyarrow as pa
    import pyarrow.flight as flight
    ARROW_FLIGHT_AVAILABLE = True
except ImportError:
    ARROW_FLIGHT_AVAILABLE = False
    logger.warning("PyArrow Flight not available, Arrow Flight features will be disabled")


class ArrowFlightMiddleware(BaseHTTPMiddleware):
    """Middleware for handling Arrow Flight protocol integration."""
    
    def __init__(
        self,
        app: FastAPI,
        enabled: bool = None,
        host: str = None,
        port: int = None,
        auth_enabled: bool = None,
        tls_enabled: bool = None,
        location: str = None,
        path_prefix: str = "/api/flight",
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            enabled: Whether Arrow Flight is enabled
            host: Arrow Flight server host
            port: Arrow Flight server port
            auth_enabled: Whether authentication is enabled for Arrow Flight
            tls_enabled: Whether TLS is enabled for Arrow Flight
            location: Arrow Flight server location URI
            path_prefix: Path prefix for Arrow Flight endpoints
        """
        super().__init__(app)
        
        # Set defaults from settings if not provided
        self.enabled = enabled if enabled is not None else settings.arrow_flight.enabled
        self.host = host or settings.arrow_flight.host
        self.port = port or settings.arrow_flight.port
        self.auth_enabled = auth_enabled if auth_enabled is not None else settings.arrow_flight.auth_enabled
        self.tls_enabled = tls_enabled if tls_enabled is not None else settings.arrow_flight.tls_enabled
        self.location = location or settings.arrow_flight.location
        self.path_prefix = path_prefix
        
        # Skip if Arrow Flight is not available or not enabled
        if not ARROW_FLIGHT_AVAILABLE:
            logger.warning("Arrow Flight middleware initialized but PyArrow Flight is not available")
            return
        
        if not self.enabled:
            logger.info("Arrow Flight middleware initialized but disabled")
            return
        
        # Flight server state
        self.flight_server = None
        self.flight_server_thread = None
        self.flight_server_running = False
        
        # Initialize the server if enabled
        if self.enabled:
            self._init_flight_server()
        
        logger.info(f"Arrow Flight middleware initialized (enabled: {self.enabled}, host: {self.host}, port: {self.port})")
    
    def _init_flight_server(self):
        """Initialize the Arrow Flight server."""
        if not ARROW_FLIGHT_AVAILABLE:
            logger.warning("Cannot initialize Arrow Flight server, PyArrow Flight is not available")
            return
        
        try:
            # Import the server implementation from the routes package
            # This avoids circular imports
            from ..routes.flight import FlightServer
            
            # Create the server
            self.flight_server = FlightServer(
                host=self.host,
                port=self.port,
                auth_enabled=self.auth_enabled,
                tls_enabled=self.tls_enabled,
                location=self.location,
            )
            
            # Start the server in a separate thread
            def run_server():
                logger.info(f"Starting Arrow Flight server on {self.host}:{self.port}")
                try:
                    self.flight_server_running = True
                    self.flight_server.serve()
                except Exception as e:
                    logger.error(f"Error running Arrow Flight server: {str(e)}")
                    self.flight_server_running = False
            
            self.flight_server_thread = threading.Thread(target=run_server, daemon=True)
            self.flight_server_thread.start()
            
            # Wait for the server to start
            retries = 0
            while not self.flight_server_running and retries < 5:
                time.sleep(1)
                retries += 1
            
            if not self.flight_server_running:
                logger.error("Failed to start Arrow Flight server")
            else:
                logger.info(f"Arrow Flight server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize Arrow Flight server: {str(e)}")
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and handle Arrow Flight protocol integration.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response from the next middleware or route handler
        """
        # Skip if Arrow Flight is not available or not enabled
        if not ARROW_FLIGHT_AVAILABLE or not self.enabled:
            return await call_next(request)
        
        # Check if this is an Arrow Flight path
        path = request.url.path
        if not path.startswith(self.path_prefix):
            return await call_next(request)
        
        # Set request state for Arrow Flight
        request.state.arrow_flight_enabled = True
        request.state.arrow_flight_server = self.flight_server
        request.state.arrow_flight_location = self.location
        
        # Check if Arrow Flight server is running
        if not self.flight_server_running:
            # Try to restart the server if it's not running
            if self.flight_server:
                self._init_flight_server()
                
                # If it's still not running, return an error
                if not self.flight_server_running:
                    raise ArrowFlightServerException(
                        detail="Arrow Flight server is not running",
                        suggestion="Please check the server logs for more information"
                    )
            else:
                raise ArrowFlightServerException(
                    detail="Arrow Flight server is not initialized",
                    suggestion="Please check the server configuration"
                )
        
        # Process the request
        try:
            return await call_next(request)
        except Exception as e:
            # Check if this is an Arrow Flight exception
            if isinstance(e, ArrowFlightException):
                raise
            
            # Otherwise, wrap it in an Arrow Flight exception
            logger.error(f"Error handling Arrow Flight request: {str(e)}")
            raise ArrowFlightException(
                detail="Error handling Arrow Flight request",
                suggestion="Please check the request parameters",
                details={"error": str(e)}
            )
    
    def get_server_info(self) -> Dict[str, Any]:
        """
        Get Arrow Flight server information.
        
        Returns:
            Dictionary with server information
        """
        info = {
            "enabled": self.enabled,
            "host": self.host,
            "port": self.port,
            "location": self.location,
            "running": self.flight_server_running,
            "auth_enabled": self.auth_enabled,
            "tls_enabled": self.tls_enabled,
        }
        
        # Add additional information if server is running
        if self.flight_server_running and self.flight_server:
            info.update({
                "start_time": getattr(self.flight_server, "start_time", None),
                "uptime": time.time() - getattr(self.flight_server, "start_time", time.time()),
                "endpoints": [
                    "DoGet", "DoPut", "DoExchange", "ListActions", "ListFlights"
                ],
            })
        
        return info
    
    def stop_server(self) -> None:
        """Stop the Arrow Flight server."""
        if self.flight_server and self.flight_server_running:
            logger.info("Stopping Arrow Flight server")
            try:
                self.flight_server.shutdown()
                self.flight_server_running = False
            except Exception as e:
                logger.error(f"Error stopping Arrow Flight server: {str(e)}")
    
    def restart_server(self) -> None:
        """Restart the Arrow Flight server."""
        self.stop_server()
        self._init_flight_server()


class ArrowFlightTokenManager:
    """Manager for Arrow Flight authentication tokens."""
    
    def __init__(self, secret_key: str = None, algorithm: str = None):
        """
        Initialize the token manager.
        
        Args:
            secret_key: Secret key for JWT token generation
            algorithm: Algorithm for JWT token generation
        """
        self.secret_key = secret_key or settings.auth.secret_key
        self.algorithm = algorithm or settings.auth.algorithm
        self.tokens = {}
    
    def generate_token(self, username: str, roles: List[str] = None, expires_in: int = 3600) -> str:
        """
        Generate a token for Arrow Flight authentication.
        
        Args:
            username: Username
            roles: List of roles
            expires_in: Token expiration time in seconds
            
        Returns:
            JWT token
        """
        if not ARROW_FLIGHT_AVAILABLE:
            raise ArrowFlightException(
                detail="PyArrow Flight is not available",
                suggestion="Please install PyArrow Flight"
            )
        
        try:
            from jose import jwt
            import datetime
            
            # Create token payload
            payload = {
                "sub": username,
                "roles": roles or ["user"],
                "flight_permissions": ["read", "write"],
                "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=expires_in),
                "iat": datetime.datetime.utcnow(),
                "aud": "arrow-flight",
            }
            
            # Generate token
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            
            # Store token
            self.tokens[token] = payload
            
            return token
        except Exception as e:
            logger.error(f"Error generating Arrow Flight token: {str(e)}")
            raise ArrowFlightAuthException(
                detail="Error generating Arrow Flight token",
                suggestion="Please check the server logs for more information"
            )
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an Arrow Flight token.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload
            
        Raises:
            ArrowFlightAuthException: If the token is invalid
        """
        if not ARROW_FLIGHT_AVAILABLE:
            raise ArrowFlightException(
                detail="PyArrow Flight is not available",
                suggestion="Please install PyArrow Flight"
            )
        
        try:
            from jose import jwt, JWTError
            
            # Decode token
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm],
                audience="arrow-flight"
            )
            
            # Check if token has expired
            if "exp" in payload and payload["exp"] < time.time():
                raise ArrowFlightAuthException(
                    detail="Arrow Flight token has expired",
                    suggestion="Please get a new token"
                )
            
            return payload
        except JWTError as e:
            logger.error(f"Error validating Arrow Flight token: {str(e)}")
            raise ArrowFlightAuthException(
                detail="Invalid Arrow Flight token",
                suggestion="Please provide a valid token"
            )
    
    def revoke_token(self, token: str) -> None:
        """
        Revoke an Arrow Flight token.
        
        Args:
            token: JWT token
        """
        if token in self.tokens:
            del self.tokens[token]


def setup_arrow_flight_middleware(
    app: FastAPI,
    enabled: bool = None,
    host: str = None,
    port: int = None,
    auth_enabled: bool = None,
    tls_enabled: bool = None,
    location: str = None,
    path_prefix: str = "/api/flight",
) -> ArrowFlightMiddleware:
    """
    Configure and add the Arrow Flight middleware to the application.
    
    Args:
        app: FastAPI application
        enabled: Whether Arrow Flight is enabled
        host: Arrow Flight server host
        port: Arrow Flight server port
        auth_enabled: Whether authentication is enabled for Arrow Flight
        tls_enabled: Whether TLS is enabled for Arrow Flight
        location: Arrow Flight server location URI
        path_prefix: Path prefix for Arrow Flight endpoints
        
    Returns:
        ArrowFlightMiddleware instance
    """
    # Check if Arrow Flight is available
    if not ARROW_FLIGHT_AVAILABLE:
        logger.warning("PyArrow Flight is not available, Arrow Flight features will be disabled")
        enabled = False
    
    # Create the middleware
    middleware = ArrowFlightMiddleware(
        app,
        enabled=enabled,
        host=host,
        port=port,
        auth_enabled=auth_enabled,
        tls_enabled=tls_enabled,
        location=location,
        path_prefix=path_prefix,
    )
    
    # Add endpoints for Arrow Flight server management
    @app.get("/api/flight/info", tags=["Arrow Flight"])
    def flight_info():
        """Get Arrow Flight server information."""
        return middleware.get_server_info()
    
    @app.post("/api/flight/restart", tags=["Arrow Flight"])
    def restart_flight_server():
        """Restart the Arrow Flight server."""
        middleware.restart_server()
        return {"status": "success", "message": "Arrow Flight server restarted"}
    
    return middleware