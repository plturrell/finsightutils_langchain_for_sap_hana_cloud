"""
Base router configuration for the SAP HANA LangChain Integration API.

This module provides base router classes and utilities for creating consistent API routes.
"""

import logging
from typing import Optional, List, Dict, Any, Union, Type, Callable, TypeVar

from fastapi import APIRouter, Depends, Query, Path, Body
from pydantic import BaseModel

from ..config_standardized import get_standardized_settings
from ..models.base_standardized import APIResponse
from .dependencies import get_current_user, get_admin_user, check_permission

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Type variable for router classes
T = TypeVar("T", bound=BaseModel)


class BaseRouter(APIRouter):
    """Base router class with common configuration."""
    
    def __init__(
        self,
        *args,
        prefix: str = "",
        tags: List[str] = None,
        dependencies: List[Depends] = None,
        responses: Dict[int, Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the router with common configuration.
        
        Args:
            prefix: URL path prefix
            tags: OpenAPI tags
            dependencies: Route dependencies
            responses: Common responses
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Set default tags if not provided
        tags = tags or []
        
        # Set default dependencies if not provided
        dependencies = dependencies or []
        
        # Set common responses if not provided
        if responses is None:
            responses = {
                400: {"description": "Bad Request"},
                401: {"description": "Unauthorized"},
                403: {"description": "Forbidden"},
                404: {"description": "Not Found"},
                500: {"description": "Internal Server Error"},
            }
        
        super().__init__(
            *args,
            prefix=prefix,
            tags=tags,
            dependencies=dependencies,
            responses=responses,
            **kwargs
        )
    
    def add_api_route(
        self,
        path: str,
        endpoint: Callable,
        *,
        response_model: Type[BaseModel] = None,
        status_code: int = 200,
        tags: List[str] = None,
        dependencies: List[Depends] = None,
        summary: Optional[str] = None,
        description: Optional[str] = None,
        response_description: str = "Successful Response",
        responses: Dict[int, Dict[str, Any]] = None,
        deprecated: bool = False,
        methods: List[str] = None,
        operation_id: Optional[str] = None,
        response_model_include: Union[set, dict, None] = None,
        response_model_exclude: Union[set, dict, None] = None,
        response_model_by_alias: bool = True,
        response_model_exclude_unset: bool = False,
        response_model_exclude_defaults: bool = False,
        response_model_exclude_none: bool = False,
        include_in_schema: bool = True,
        **kwargs
    ):
        """
        Add an API route with standardized response wrapping.
        
        This method wraps the endpoint response in a standardized APIResponse format
        if response wrapping is enabled in settings.
        
        Args:
            path: URL path
            endpoint: Route handler function
            response_model: Response model
            status_code: HTTP status code
            tags: OpenAPI tags
            dependencies: Route dependencies
            summary: Route summary
            description: Route description
            response_description: Response description
            responses: Additional responses
            deprecated: Whether the route is deprecated
            methods: HTTP methods
            operation_id: Operation ID
            response_model_include: Fields to include in response
            response_model_exclude: Fields to exclude from response
            response_model_by_alias: Whether to serialize by alias
            response_model_exclude_unset: Whether to exclude unset fields
            response_model_exclude_defaults: Whether to exclude default fields
            response_model_exclude_none: Whether to exclude None fields
            include_in_schema: Whether to include in OpenAPI schema
            **kwargs: Additional keyword arguments
        """
        # Only wrap responses if enabled in settings
        if settings.api.wrap_responses and response_model:
            # Create a wrapped response model
            wrapped_response_model = APIResponse[response_model]
            
            # Override the response model
            return super().add_api_route(
                path=path,
                endpoint=endpoint,
                response_model=wrapped_response_model,
                status_code=status_code,
                tags=tags,
                dependencies=dependencies,
                summary=summary,
                description=description,
                response_description=response_description,
                responses=responses,
                deprecated=deprecated,
                methods=methods,
                operation_id=operation_id,
                response_model_include=response_model_include,
                response_model_exclude=response_model_exclude,
                response_model_by_alias=response_model_by_alias,
                response_model_exclude_unset=response_model_exclude_unset,
                response_model_exclude_defaults=response_model_exclude_defaults,
                response_model_exclude_none=response_model_exclude_none,
                include_in_schema=include_in_schema,
                **kwargs
            )
        
        # Otherwise, use the original response model
        return super().add_api_route(
            path=path,
            endpoint=endpoint,
            response_model=response_model,
            status_code=status_code,
            tags=tags,
            dependencies=dependencies,
            summary=summary,
            description=description,
            response_description=response_description,
            responses=responses,
            deprecated=deprecated,
            methods=methods,
            operation_id=operation_id,
            response_model_include=response_model_include,
            response_model_exclude=response_model_exclude,
            response_model_by_alias=response_model_by_alias,
            response_model_exclude_unset=response_model_exclude_unset,
            response_model_exclude_defaults=response_model_exclude_defaults,
            response_model_exclude_none=response_model_exclude_none,
            include_in_schema=include_in_schema,
            **kwargs
        )


class AuthenticatedRouter(BaseRouter):
    """Router that requires authentication for all routes."""
    
    def __init__(
        self,
        *args,
        prefix: str = "",
        tags: List[str] = None,
        dependencies: List[Depends] = None,
        responses: Dict[int, Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the router with authentication dependency.
        
        Args:
            prefix: URL path prefix
            tags: OpenAPI tags
            dependencies: Route dependencies
            responses: Common responses
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Add authentication dependency
        dependencies = dependencies or []
        dependencies.append(Depends(get_current_user))
        
        super().__init__(
            *args,
            prefix=prefix,
            tags=tags,
            dependencies=dependencies,
            responses=responses,
            **kwargs
        )


class AdminRouter(BaseRouter):
    """Router that requires admin privileges for all routes."""
    
    def __init__(
        self,
        *args,
        prefix: str = "",
        tags: List[str] = None,
        dependencies: List[Depends] = None,
        responses: Dict[int, Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the router with admin authentication dependency.
        
        Args:
            prefix: URL path prefix
            tags: OpenAPI tags
            dependencies: Route dependencies
            responses: Common responses
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Add admin authentication dependency
        dependencies = dependencies or []
        dependencies.append(Depends(get_admin_user))
        
        super().__init__(
            *args,
            prefix=prefix,
            tags=tags,
            dependencies=dependencies,
            responses=responses,
            **kwargs
        )


def permission_required(permission: str):
    """
    Create a router that requires a specific permission for all routes.
    
    Args:
        permission: Required permission
        
    Returns:
        Router class with permission dependency
    """
    
    class PermissionRouter(BaseRouter):
        """Router that requires a specific permission for all routes."""
        
        def __init__(
            self,
            *args,
            prefix: str = "",
            tags: List[str] = None,
            dependencies: List[Depends] = None,
            responses: Dict[int, Dict[str, Any]] = None,
            **kwargs
        ):
            """
            Initialize the router with permission dependency.
            
            Args:
                prefix: URL path prefix
                tags: OpenAPI tags
                dependencies: Route dependencies
                responses: Common responses
                *args: Additional positional arguments
                **kwargs: Additional keyword arguments
            """
            # Add permission dependency
            dependencies = dependencies or []
            dependencies.append(Depends(check_permission(permission)))
            
            super().__init__(
                *args,
                prefix=prefix,
                tags=tags,
                dependencies=dependencies,
                responses=responses,
                **kwargs
            )
    
    return PermissionRouter