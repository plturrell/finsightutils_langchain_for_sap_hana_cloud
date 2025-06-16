"""
Embeddings module for SAP HANA Cloud integration.

This module provides various embedding implementations, including:
- Standard LangChain embeddings
- TensorRT accelerated embeddings
- Multi-GPU embeddings
"""

# Import the provider registry and interface
from api.embeddings.embedding_providers import (
    EmbeddingProvider, 
    BaseEmbeddingProvider, 
    EmbeddingProviderRegistry
)

# Import the main embedding classes
from api.embeddings.embeddings import (
    DefaultEmbeddingProvider, 
    GPUAcceleratedEmbeddings, 
    GPUHybridEmbeddings
)

# Try to import TensorRT optimized embeddings if available
try:
    from api.embeddings.embeddings_tensorrt import TensorRTEmbedding
    # Register with the provider registry
    EmbeddingProviderRegistry.register("tensorrt", TensorRTEmbedding)
except ImportError:
    # TensorRT might not be available
    pass

# Try to import multi-GPU embeddings if available
try:
    from api.embeddings.embeddings_multi_gpu import MultiGPUEmbedding
    # Register with the provider registry
    EmbeddingProviderRegistry.register("multi_gpu", MultiGPUEmbedding)
except ImportError:
    # Multi-GPU might not be available
    pass

# Try to import enhanced TensorRT embeddings if available
try:
    from api.embeddings.embeddings_tensorrt_enhanced import EnhancedTensorRTEmbedding
    # Register with the provider registry
    EmbeddingProviderRegistry.register("tensorrt_enhanced", EnhancedTensorRTEmbedding)
except ImportError:
    # Enhanced TensorRT might not be available
    pass

# Create a factory function to get embedding providers
def get_embedding_provider(provider_type: str = "hybrid", **kwargs):
    """
    Factory function to get an embedding provider.
    
    Args:
        provider_type: Type of provider to use ('gpu', 'hybrid', 'tensorrt', 'multi_gpu', 'tensorrt_enhanced')
        **kwargs: Arguments to pass to the provider constructor
        
    Returns:
        An instance of the requested provider
    """
    return EmbeddingProviderRegistry.get_provider(provider_type, **kwargs)

__all__ = [
    "DefaultEmbeddingProvider", 
    "GPUAcceleratedEmbeddings", 
    "GPUHybridEmbeddings",
    "EmbeddingProvider",
    "BaseEmbeddingProvider", 
    "EmbeddingProviderRegistry",
    "get_embedding_provider",
    "TensorRTEmbedding",
    "MultiGPUEmbedding",
    "EnhancedTensorRTEmbedding"
]