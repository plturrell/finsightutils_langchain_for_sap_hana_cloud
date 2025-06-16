"""
AuthMiddleware for the SAP HANA LangChain Integration API.

This middleware handles authentication for API requests, supporting API keys,
JWT tokens, and Arrow Flight authentication.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Callable

from fastapi import FastAPI, Request, Response
from fastapi.security import APIKeyHeader
from jose import jwt, JWTError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.status import (
    HTTP_401_UNAUTHORIZED,
    HTTP_403_FORBIDDEN,
)

from ..config_standardized import get_standardized_settings
from ..models.auth_standardized import User
from ..utils.standardized_exceptions import (
    AuthenticationException,
    AuthorizationException,
    InvalidTokenException,
    ExpiredTokenException,
    MissingAPIKeyException,
    InvalidAPIKeyException,
)

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to handle authentication for API requests."""
    
    def __init__(
        self,
        app: FastAPI,
        secret_key: str = None,
        algorithm: str = None,
        api_key_name: str = None,
        token_header_name: str = None,
        public_paths: List[str] = None,
        require_auth: bool = None,
    ):
        """
        Initialize the middleware.
        
        Args:
            app: FastAPI application
            secret_key: Secret key for JWT token verification
            algorithm: Algorithm for JWT token verification
            api_key_name: Name of the API key header
            token_header_name: Name of the JWT token header
            public_paths: List of paths that don't require authentication
            require_auth: Whether to require authentication
        """
        super().__init__(app)
        
        # Set defaults from settings if not provided
        self.secret_key = secret_key or settings.auth.secret_key
        self.algorithm = algorithm or settings.auth.algorithm
        self.api_key_name = api_key_name or settings.auth.api_key_name
        self.token_header_name = token_header_name or settings.auth.token_header_name
        self.public_paths = public_paths or settings.auth.public_paths
        self.require_auth = require_auth if require_auth is not None else settings.auth.require_auth
        
        # Add standard public paths
        self.public_paths.extend([
            "/docs",
            "/redoc",
            "/openapi.json",
            "/",
            "/health",
            "/api/health",
            "/api/v1/health",
            "/api/v2/health",
        ])
        
        # Setup API key handler
        self.api_key_header = APIKeyHeader(name=self.api_key_name, auto_error=False)
        
        logger.info("Auth middleware initialized")
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """
        Process the request and handle authentication.
        
        Args:
            request: FastAPI request object
            call_next: Next middleware or route handler
            
        Returns:
            Response from the next middleware or route handler
        """
        # Check if this path is public
        path = request.url.path
        if self._is_public_path(path):
            return await call_next(request)
        
        # If auth is disabled, skip authentication
        if not self.require_auth:
            return await call_next(request)
        
        # Check for API key in header
        api_key = request.headers.get(self.api_key_name)
        
        # Check for JWT token in header
        token = request.headers.get(self.token_header_name)
        
        # Check for Authorization header (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
        
        # Special handling for Arrow Flight authentication
        if path.startswith("/api/flight"):
            # Check for X-Flight-Token header
            flight_token = request.headers.get("X-Flight-Token")
            if flight_token:
                try:
                    # Validate the flight token
                    user = self._validate_flight_token(flight_token)
                    # Add user to request state
                    request.state.user = user
                    return await call_next(request)
                except Exception as e:
                    # Log the error
                    logger.warning(f"Invalid Flight token: {str(e)}")
        
        # If no API key or token is provided, and auth is required, raise an exception
        if not api_key and not token and self.require_auth:
            raise MissingAPIKeyException(
                detail="Authentication required",
                suggestion="Provide a valid API key or JWT token"
            )
        
        # Validate API key if provided
        if api_key:
            try:
                # Validate the API key
                user = self._validate_api_key(api_key)
                # Add user to request state
                request.state.user = user
                return await call_next(request)
            except Exception as e:
                # Log the error
                logger.warning(f"Invalid API key: {str(e)}")
                # If token is also provided, try that instead
                if not token:
                    raise InvalidAPIKeyException(
                        detail="Invalid API key",
                        suggestion="Provide a valid API key"
                    )
        
        # Validate JWT token if provided
        if token:
            try:
                # Validate the token
                user = self._validate_token(token)
                # Add user to request state
                request.state.user = user
                return await call_next(request)
            except ExpiredTokenException:
                # Token has expired
                raise ExpiredTokenException(
                    detail="Token has expired",
                    suggestion="Please login again to get a new token"
                )
            except Exception as e:
                # Log the error
                logger.warning(f"Invalid token: {str(e)}")
                raise InvalidTokenException(
                    detail="Invalid token",
                    suggestion="Provide a valid JWT token"
                )
        
        # If execution reaches here, authentication failed
        raise AuthenticationException(
            detail="Authentication failed",
            suggestion="Provide a valid API key or JWT token"
        )
    
    def _is_public_path(self, path: str) -> bool:
        """
        Check if a path is public (doesn't require authentication).
        
        Args:
            path: Request path
            
        Returns:
            True if the path is public, False otherwise
        """
        # Check if path is in public paths
        for public_path in self.public_paths:
            if path == public_path or path.startswith(public_path):
                return True
        
        return False
    
    def _validate_api_key(self, api_key: str) -> User:
        """
        Validate an API key.
        
        Args:
            api_key: API key to validate
            
        Returns:
            User object if the API key is valid
            
        Raises:
            InvalidAPIKeyException: If the API key is invalid
        """
        # This is where you would implement API key validation
        # For example, check against a database of valid API keys
        
        # For now, check against a list of valid API keys from settings
        if api_key in settings.auth.valid_api_keys:
            # Get API key details from settings (or database in a real implementation)
            # In a real implementation, you would load these from a secure database
            api_key_details = settings.auth.api_key_details.get(api_key, {})
            
            # Create a user object with the API key details
            user = User(
                username=api_key_details.get("username", "api_user"),
                email=api_key_details.get("email", ""),
                roles=api_key_details.get("roles", ["user"]),
                permissions=api_key_details.get("permissions", []),
                is_active=True,
                auth_method="api_key",
            )
            
            return user
        
        # Invalid API key
        raise InvalidAPIKeyException(
            detail="Invalid API key",
            suggestion="Provide a valid API key"
        )
    
    def _validate_token(self, token: str) -> User:
        """
        Validate a JWT token.
        
        Args:
            token: JWT token to validate
            
        Returns:
            User object if the token is valid
            
        Raises:
            InvalidTokenException: If the token is invalid
            ExpiredTokenException: If the token has expired
        """
        try:
            # Decode the token
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm]
            )
            
            # Check if token has expired
            if "exp" in payload and payload["exp"] < time.time():
                raise ExpiredTokenException(
                    detail="Token has expired",
                    suggestion="Please login again to get a new token"
                )
            
            # Get user information from token
            username = payload.get("sub")
            email = payload.get("email", "")
            roles = payload.get("roles", ["user"])
            permissions = payload.get("permissions", [])
            
            # Create a user object with the token details
            user = User(
                username=username,
                email=email,
                roles=roles,
                permissions=permissions,
                is_active=True,
                auth_method="jwt",
            )
            
            return user
        except JWTError as e:
            # Log the error
            logger.warning(f"JWT decode error: {str(e)}")
            raise InvalidTokenException(
                detail="Invalid token",
                suggestion="Provide a valid JWT token"
            )
    
    def _validate_flight_token(self, token: str) -> User:
        """
        Validate a Flight token.
        
        Args:
            token: Flight token to validate
            
        Returns:
            User object if the token is valid
            
        Raises:
            InvalidTokenException: If the token is invalid
            ExpiredTokenException: If the token has expired
        """
        try:
            # Flight tokens are just JWT tokens with a specific audience
            # Decode the token
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm],
                audience="arrow-flight"
            )
            
            # Check if token has expired
            if "exp" in payload and payload["exp"] < time.time():
                raise ExpiredTokenException(
                    detail="Flight token has expired",
                    suggestion="Please get a new Flight token"
                )
            
            # Get user information from token
            username = payload.get("sub")
            email = payload.get("email", "")
            roles = payload.get("roles", ["user"])
            permissions = payload.get("permissions", [])
            
            # Flight-specific permissions
            flight_permissions = payload.get("flight_permissions", [])
            
            # Create a user object with the token details
            user = User(
                username=username,
                email=email,
                roles=roles,
                permissions=permissions + flight_permissions,
                is_active=True,
                auth_method="flight_token",
            )
            
            return user
        except JWTError as e:
            # Log the error
            logger.warning(f"Flight token decode error: {str(e)}")
            raise InvalidTokenException(
                detail="Invalid Flight token",
                suggestion="Provide a valid Flight token"
            )


def setup_auth_middleware(
    app: FastAPI,
    secret_key: str = None,
    algorithm: str = None,
    api_key_name: str = None,
    token_header_name: str = None,
    public_paths: List[str] = None,
    require_auth: bool = None,
) -> None:
    """
    Configure and add the authentication middleware to the application.
    
    Args:
        app: FastAPI application
        secret_key: Secret key for JWT token verification
        algorithm: Algorithm for JWT token verification
        api_key_name: Name of the API key header
        token_header_name: Name of the JWT token header
        public_paths: List of paths that don't require authentication
        require_auth: Whether to require authentication
    """
    app.add_middleware(
        AuthMiddleware,
        secret_key=secret_key,
        algorithm=algorithm,
        api_key_name=api_key_name,
        token_header_name=token_header_name,
        public_paths=public_paths,
        require_auth=require_auth,
    )