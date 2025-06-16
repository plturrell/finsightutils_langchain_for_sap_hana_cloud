"""
Data models for SAP HANA Cloud integration API.

This module provides Pydantic models for API requests and responses.
"""

from api.models.models import (
    EmbeddingRequest, 
    EmbeddingResponse, 
    VectorStoreRequest, 
    VectorStoreResponse,
    HealthResponse,
    ErrorResponse,
    ConfigurationResponse,
    QueryResultDocumentResponse,
    DocumentResponse
)

from api.models.flight_models import (
    FlightQueryRequest,
    FlightQueryResponse,
    FlightUploadRequest,
    FlightUploadResponse,
    FlightListResponse,
    FlightInfoResponse,
    FlightCollection
)

__all__ = [
    # Core models
    "EmbeddingRequest",
    "EmbeddingResponse",
    "VectorStoreRequest",
    "VectorStoreResponse",
    "HealthResponse",
    "ErrorResponse",
    "ConfigurationResponse",
    "QueryResultDocumentResponse",
    "DocumentResponse",
    
    # Arrow Flight models
    "FlightQueryRequest",
    "FlightQueryResponse",
    "FlightUploadRequest",
    "FlightUploadResponse",
    "FlightListResponse",
    "FlightInfoResponse",
    "FlightCollection"
]