"""
Standardized base models for the SAP HANA Cloud LangChain Integration API.

This module provides a consistent set of base models for request and response
handling, ensuring uniform data structures across all API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union

from fastapi import status
from pydantic import BaseModel, Field, root_validator, validator
from pydantic.generics import GenericModel

# Define a generic type variable for response data
T = TypeVar("T")


class BaseAPIModel(BaseModel):
    """Base model for all API models with common configuration."""
    
    class Config:
        """Pydantic model configuration."""
        # Allow extra attributes during model creation
        extra = "ignore"
        
        # Use field names in schema
        use_enum_values = True
        
        # Allow arbitrary types for field values
        arbitrary_types_allowed = True
        
        # Populate models with field name as attribute name
        populate_by_name = True


class ErrorDetail(BaseAPIModel):
    """Error details for API responses."""
    
    status_code: int = Field(..., description="HTTP status code")
    error_code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    suggestion: Optional[str] = Field(
        None, description="Suggested action to resolve the error"
    )
    docs_url: Optional[str] = Field(
        None, description="URL to documentation for this error"
    )
    request_id: Optional[str] = Field(
        None, description="Request ID for debugging"
    )
    
    @classmethod
    def from_exception(cls, exc: Exception) -> "ErrorDetail":
        """Create an error detail from an exception."""
        from ..utils.standardized_exceptions import BaseAPIException
        
        if isinstance(exc, BaseAPIException):
            return cls(
                status_code=exc.status_code,
                error_code=exc.error_code,
                message=exc.detail,
                details=exc.details,
                suggestion=exc.suggestion,
                docs_url=exc.docs_url,
            )
        else:
            return cls(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_code="internal_server_error",
                message=str(exc),
            )


class ResponseMeta(BaseAPIModel):
    """Metadata for API responses."""
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp in ISO 8601 format"
    )
    request_id: Optional[str] = Field(
        None, description="Request ID for debugging"
    )
    version: Optional[str] = Field(
        None, description="API version"
    )
    environment: Optional[str] = Field(
        None, description="Deployment environment"
    )
    gpu_info: Optional[Dict[str, Any]] = Field(
        None, description="GPU information"
    )
    arrow_flight_enabled: Optional[bool] = Field(
        None, description="Whether Arrow Flight is enabled"
    )


class APIResponse(GenericModel, Generic[T], BaseAPIModel):
    """Standard response envelope for all API responses."""
    
    data: T = Field(
        ...,
        description="Response data"
    )
    meta: ResponseMeta = Field(
        default_factory=ResponseMeta,
        description="Response metadata"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional contextual information"
    )
    error: Optional[ErrorDetail] = Field(
        None,
        description="Error information if there was an error"
    )


class ErrorResponse(BaseAPIModel):
    """Error response for API errors."""
    
    status_code: int = Field(..., description="HTTP status code")
    error_code: str = Field(..., description="Error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional error details"
    )
    suggestion: Optional[str] = Field(
        None, description="Suggested action to resolve the error"
    )
    docs_url: Optional[str] = Field(
        None, description="URL to documentation for this error"
    )
    request_id: Optional[str] = Field(
        None, description="Request ID for debugging"
    )
    
    @classmethod
    def from_exception(cls, exc: Exception) -> "ErrorResponse":
        """Create an error response from an exception."""
        return cls(**ErrorDetail.from_exception(exc).dict())


class PaginationParams(BaseAPIModel):
    """Pagination parameters for list endpoints."""
    
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(20, ge=1, le=100, description="Number of items per page")


class PaginationMeta(BaseAPIModel):
    """Pagination metadata for list responses."""
    
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_items: int = Field(..., description="Total number of items")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether there is a next page")
    has_prev: bool = Field(..., description="Whether there is a previous page")
    
    @root_validator
    def calculate_has_next_prev(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate has_next and has_prev based on other values."""
        page = values.get("page")
        total_pages = values.get("total_pages")
        
        if page is not None and total_pages is not None:
            values["has_next"] = page < total_pages
            values["has_prev"] = page > 1
            
        return values


class PaginatedResponse(GenericModel, Generic[T], BaseAPIModel):
    """Paginated response for list endpoints."""
    
    items: List[T] = Field(..., description="List of items")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    meta: ResponseMeta = Field(
        default_factory=ResponseMeta,
        description="Response metadata"
    )
    
    @classmethod
    def create(
        cls,
        items: List[T],
        total_items: int,
        page: int,
        page_size: int,
        request_id: Optional[str] = None,
        gpu_info: Optional[Dict[str, Any]] = None,
        arrow_flight_enabled: Optional[bool] = None,
    ) -> "PaginatedResponse[T]":
        """Create a paginated response."""
        total_pages = max(1, (total_items + page_size - 1) // page_size)
        
        pagination = PaginationMeta(
            page=page,
            page_size=page_size,
            total_items=total_items,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )
        
        meta = ResponseMeta(
            request_id=request_id,
            gpu_info=gpu_info,
            arrow_flight_enabled=arrow_flight_enabled
        )
        
        return cls(items=items, pagination=pagination, meta=meta)


# Vector-specific models
class EmbeddingFormat(BaseAPIModel):
    """Format information for vector embeddings."""
    
    dimension: int = Field(..., description="Dimension of the embedding vector")
    model: str = Field(..., description="Model used to generate the embedding")
    content_type: str = Field("float32", description="Data type of the embedding values")
    format: str = Field("dense", description="Format of the embedding (dense or sparse)")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "dimension": 768,
                "model": "all-MiniLM-L6-v2",
                "content_type": "float32",
                "format": "dense"
            }
        }


class VectorSearchResult(BaseAPIModel):
    """Result item from vector search."""
    
    id: str = Field(..., description="Document ID")
    score: float = Field(..., description="Similarity score")
    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "id": "doc123",
                "score": 0.92,
                "content": "This is a document about SAP HANA Cloud.",
                "metadata": {
                    "source": "knowledge_base",
                    "created_at": "2023-01-01T12:00:00Z"
                }
            }
        }


class ArrowFlightInfo(BaseAPIModel):
    """Arrow Flight service information."""
    
    enabled: bool = Field(..., description="Whether Arrow Flight is enabled")
    endpoint: Optional[str] = Field(None, description="Arrow Flight endpoint URL")
    port: Optional[int] = Field(None, description="Arrow Flight port")
    
    class Config:
        """Model configuration."""
        schema_extra = {
            "example": {
                "enabled": True,
                "endpoint": "grpc://localhost:8815",
                "port": 8815
            }
        }