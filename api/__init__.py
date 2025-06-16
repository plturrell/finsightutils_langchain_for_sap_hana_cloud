"""
SAP HANA Cloud LangChain Integration API.

This package provides a comprehensive API for integrating LangChain with SAP HANA Cloud,
optimized for GPU acceleration and enterprise deployments.
"""

from api.version import __version__

# For the simplified Arrow Flight app, we skip the regular imports
# to avoid dependency issues in the container
try:
    # Only import if not in Arrow Flight mode
    if not __name__ == "api.simplified_app":
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

        # Import main application and Vercel handler
        from api.core import app, handler
except ImportError:
    # In case of import errors (for the simplified app), we continue without these imports
    pass

__all__ = [
    "__version__",
    
    # Main FastAPI application and Vercel handler
    "app",
    "handler",
    
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