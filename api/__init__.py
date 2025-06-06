"""
SAP HANA Cloud LangChain Integration API.

This package provides a comprehensive API for integrating LangChain with SAP HANA Cloud,
optimized for GPU acceleration and enterprise deployments.
"""

from api.version import __version__

# Expose key components at the package level
from api.embeddings import (
    EmbeddingProvider,
    TensorRTEmbedding,
    MultiGPUEmbedding,
    EnhancedTensorRTEmbedding
)

from api.services import (
    APIService,
    VectorService,
    EmbeddingService,
    DeveloperService
)

from api.models import (
    EmbeddingRequest,
    EmbeddingResponse,
    VectorStoreRequest,
    VectorStoreResponse,
    HealthResponse
)

from api.utils import (
    handle_error,
    format_error,
    ErrorContext,
    TimeoutManager
)

__all__ = [
    "__version__",
    
    # Embeddings
    "EmbeddingProvider",
    "TensorRTEmbedding",
    "MultiGPUEmbedding",
    "EnhancedTensorRTEmbedding",
    
    # Services
    "APIService",
    "VectorService",
    "EmbeddingService",
    "DeveloperService",
    
    # Models
    "EmbeddingRequest",
    "EmbeddingResponse",
    "VectorStoreRequest",
    "VectorStoreResponse",
    "HealthResponse",
    
    # Utils
    "handle_error",
    "format_error",
    "ErrorContext",
    "TimeoutManager"
]