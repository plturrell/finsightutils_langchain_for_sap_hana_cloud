"""
Common dependencies for API routers.

This module provides shared dependencies used across different API routers,
including authentication, validation, and context management.
"""

import logging
from typing import Optional, List, Dict, Any, Union

from fastapi import Depends, HTTPException, Header, Query, Request, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer

from ..config_standardized import get_standardized_settings
from ..models.auth_standardized import User
from ..utils.standardized_exceptions import (
    AuthenticationException,
    AuthorizationException,
    InvalidAPIKeyException,
    GPUNotAvailableException,
    TensorRTNotAvailableException,
    ArrowFlightException,
)

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Setup API key authentication
api_key_header = APIKeyHeader(name=settings.auth.api_key_name, auto_error=False)

# Setup OAuth2 authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)


async def get_current_user(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
    token: Optional[str] = Depends(oauth2_scheme),
) -> User:
    """
    Get the current authenticated user.
    
    Args:
        request: FastAPI request object
        api_key: API key from header
        token: OAuth2 token
        
    Returns:
        User object
        
    Raises:
        AuthenticationException: If authentication fails
    """
    # Check if user is already in request state (set by middleware)
    if hasattr(request.state, "user") and request.state.user:
        return request.state.user
    
    # If authentication is not required, return a default user
    if not settings.auth.require_auth:
        return User(
            username="anonymous",
            email="",
            roles=["user"],
            permissions=[],
            is_active=True,
            auth_method="none",
        )
    
    # If authentication is required but no credentials provided, raise an exception
    if not api_key and not token:
        raise AuthenticationException(
            detail="Authentication required",
            suggestion="Provide a valid API key or JWT token"
        )
    
    # Try to authenticate with API key
    if api_key:
        # In a real implementation, you would validate the API key against a database
        # For now, check against a list of valid API keys from settings
        if api_key in settings.auth.valid_api_keys:
            # Get API key details
            api_key_details = settings.auth.api_key_details.get(api_key, {})
            
            # Create and return a user object
            user = User(
                username=api_key_details.get("username", "api_user"),
                email=api_key_details.get("email", ""),
                roles=api_key_details.get("roles", ["user"]),
                permissions=api_key_details.get("permissions", []),
                is_active=True,
                auth_method="api_key",
            )
            
            # Store the user in request state for future use
            request.state.user = user
            
            return user
        
        # Invalid API key
        raise InvalidAPIKeyException(
            detail="Invalid API key",
            suggestion="Provide a valid API key"
        )
    
    # Try to authenticate with OAuth2 token
    if token:
        # In a real implementation, you would validate the token against your auth provider
        # For now, this is a placeholder
        try:
            from jose import jwt, JWTError
            
            # Decode the token
            payload = jwt.decode(
                token, settings.auth.secret_key, algorithms=[settings.auth.algorithm]
            )
            
            # Extract user information
            username = payload.get("sub")
            if not username:
                raise AuthenticationException(
                    detail="Invalid token payload",
                    suggestion="Token does not contain a valid username"
                )
            
            # Create and return a user object
            user = User(
                username=username,
                email=payload.get("email", ""),
                roles=payload.get("roles", ["user"]),
                permissions=payload.get("permissions", []),
                is_active=True,
                auth_method="jwt",
            )
            
            # Store the user in request state for future use
            request.state.user = user
            
            return user
        except JWTError:
            raise AuthenticationException(
                detail="Invalid authentication token",
                suggestion="Token has expired or is invalid"
            )
    
    # If we get here, authentication failed
    raise AuthenticationException(
        detail="Authentication failed",
        suggestion="Provide valid authentication credentials"
    )


async def get_admin_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Get the current user and verify they have admin role.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User object with admin role
        
    Raises:
        AuthorizationException: If the user does not have admin role
    """
    if "admin" not in current_user.roles:
        raise AuthorizationException(
            detail="Admin privileges required",
            suggestion="This operation requires admin privileges"
        )
    
    return current_user


def check_permission(permission: str):
    """
    Create a dependency to check if a user has a specific permission.
    
    Args:
        permission: Permission to check
        
    Returns:
        Dependency function
    """
    
    async def has_permission(
        current_user: User = Depends(get_current_user),
    ) -> User:
        """
        Check if the current user has the required permission.
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User object with the required permission
            
        Raises:
            AuthorizationException: If the user does not have the required permission
        """
        # Admin users have all permissions
        if "admin" in current_user.roles:
            return current_user
        
        # Check if the user has the specific permission
        if permission not in current_user.permissions:
            raise AuthorizationException(
                detail=f"Permission '{permission}' required",
                suggestion="You do not have the required permission for this operation"
            )
        
        return current_user
    
    return has_permission


async def get_gpu_info(
    request: Request,
    use_gpu: bool = Query(False, description="Whether to use GPU acceleration"),
    use_tensorrt: bool = Query(False, description="Whether to use TensorRT acceleration"),
    x_use_gpu: Optional[str] = Header(None, description="Whether to use GPU acceleration"),
    x_use_tensorrt: Optional[str] = Header(None, description="Whether to use TensorRT acceleration"),
) -> Dict[str, Any]:
    """
    Get GPU information and check if GPU and TensorRT are available.
    
    Args:
        request: FastAPI request object
        use_gpu: Whether to use GPU acceleration (from query parameter)
        use_tensorrt: Whether to use TensorRT acceleration (from query parameter)
        x_use_gpu: Whether to use GPU acceleration (from header)
        x_use_tensorrt: Whether to use TensorRT acceleration (from header)
        
    Returns:
        GPU information
        
    Raises:
        GPUNotAvailableException: If GPU is requested but not available
        TensorRTNotAvailableException: If TensorRT is requested but not available
    """
    # Check if GPU is enabled in settings
    if not settings.gpu.enabled:
        if use_gpu or (x_use_gpu and x_use_gpu.lower() in ("true", "1", "yes")):
            raise GPUNotAvailableException(
                detail="GPU acceleration is disabled in the server configuration",
                suggestion="Contact the administrator to enable GPU acceleration"
            )
        
        return {"available": False, "enabled": False}
    
    # Check if GPU info is in request state (set by middleware)
    if hasattr(request.state, "gpu_info") and request.state.gpu_info:
        gpu_info = request.state.gpu_info
    else:
        # No GPU info available
        if use_gpu or (x_use_gpu and x_use_gpu.lower() in ("true", "1", "yes")):
            raise GPUNotAvailableException(
                detail="GPU information not available",
                suggestion="GPU middleware may not be configured correctly"
            )
        
        return {"available": False, "enabled": False}
    
    # Check if GPU is requested by header (overrides query parameter)
    requires_gpu = use_gpu
    if x_use_gpu:
        requires_gpu = x_use_gpu.lower() in ("true", "1", "yes")
    
    # Check if TensorRT is requested by header (overrides query parameter)
    requires_tensorrt = use_tensorrt
    if x_use_tensorrt:
        requires_tensorrt = x_use_tensorrt.lower() in ("true", "1", "yes")
    
    # Check if GPU is available if requested
    if requires_gpu and not gpu_info.get("available", False):
        raise GPUNotAvailableException(
            detail="GPU acceleration is requested but not available",
            details={"gpu_info": gpu_info}
        )
    
    # Check if TensorRT is available if requested
    if requires_tensorrt and not gpu_info.get("tensorrt_available", False):
        raise TensorRTNotAvailableException(
            detail="TensorRT acceleration is requested but not available",
            details={"gpu_info": gpu_info}
        )
    
    # Add request flags to GPU info
    gpu_info["requested"] = requires_gpu
    gpu_info["tensorrt_requested"] = requires_tensorrt
    
    return gpu_info


async def get_arrow_flight_info(
    request: Request,
    use_arrow_flight: bool = Query(False, description="Whether to use Arrow Flight protocol"),
    x_use_arrow_flight: Optional[str] = Header(None, description="Whether to use Arrow Flight protocol"),
) -> Dict[str, Any]:
    """
    Get Arrow Flight information and check if it's available.
    
    Args:
        request: FastAPI request object
        use_arrow_flight: Whether to use Arrow Flight protocol (from query parameter)
        x_use_arrow_flight: Whether to use Arrow Flight protocol (from header)
        
    Returns:
        Arrow Flight information
        
    Raises:
        ArrowFlightException: If Arrow Flight is requested but not available
    """
    # Check if Arrow Flight is enabled in settings
    if not settings.arrow_flight.enabled:
        if use_arrow_flight or (x_use_arrow_flight and x_use_arrow_flight.lower() in ("true", "1", "yes")):
            raise ArrowFlightException(
                detail="Arrow Flight protocol is disabled in the server configuration",
                suggestion="Contact the administrator to enable Arrow Flight protocol"
            )
        
        return {"available": False, "enabled": False}
    
    # Check if Arrow Flight middleware is available
    if hasattr(request.app.state, "arrow_flight_middleware"):
        flight_info = request.app.state.arrow_flight_middleware.get_server_info()
    else:
        # No Arrow Flight middleware available
        if use_arrow_flight or (x_use_arrow_flight and x_use_arrow_flight.lower() in ("true", "1", "yes")):
            raise ArrowFlightException(
                detail="Arrow Flight information not available",
                suggestion="Arrow Flight middleware may not be configured correctly"
            )
        
        return {"available": False, "enabled": False}
    
    # Check if Arrow Flight is requested by header (overrides query parameter)
    requires_arrow_flight = use_arrow_flight
    if x_use_arrow_flight:
        requires_arrow_flight = x_use_arrow_flight.lower() in ("true", "1", "yes")
    
    # Check if Arrow Flight is available if requested
    if requires_arrow_flight and not flight_info.get("running", False):
        raise ArrowFlightException(
            detail="Arrow Flight protocol is requested but the server is not running",
            details={"flight_info": flight_info}
        )
    
    # Add request flag to Flight info
    flight_info["requested"] = requires_arrow_flight
    
    return flight_info