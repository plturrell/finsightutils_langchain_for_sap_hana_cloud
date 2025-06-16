"""
GPU acceleration utilities for SAP HANA Cloud LangChain integration.

This module provides utilities for leveraging GPU acceleration to improve 
performance of embedding generation and vector operations.

Key components:
- TensorRTEmbeddings: High-performance embedding generation using NVIDIA TensorRT
- HanaTensorRTEmbeddings: SAP HANA Cloud-optimized TensorRT embeddings
- HanaTensorRTVectorStore: GPU-accelerated vectorstore for SAP HANA Cloud
- GPUAccelerator: Unified interface for GPU acceleration across different frameworks
- DynamicBatchProcessor: Memory-aware dynamic batch sizing for optimal throughput
- EmbeddingBatchProcessor: Specialized batch processor for embedding generation
- TensorCoreOptimizer: T4 GPU Tensor Core optimizations
- MultiGPUManager: Distributed processing across multiple GPUs
- VectorSerialization: Memory-efficient vector serialization
- ArrowFlightClient: High-performance data transfer using Apache Arrow Flight
- ArrowFlightServer: Arrow Flight server for SAP HANA Cloud
- ArrowGpuMemoryManager: GPU-aware memory management for Arrow data
- HanaArrowFlightVectorStore: Vector store using Arrow Flight for efficient data transfer
"""

# Base GPU acceleration utilities
from langchain_hana.gpu.tensorrt_embeddings import (
    TensorRTEmbeddings, 
    GPUInfo, 
    get_available_gpus
)
from langchain_hana.gpu.accelerator import (
    GPUAccelerator, 
    AcceleratorType,
    MemoryStrategy
)
from langchain_hana.gpu.tensor_core_optimizer import (
    TensorCoreOptimizer,
    optimize_model_for_t4
)
from langchain_hana.gpu.utils import (
    detect_gpu_capabilities,
    get_optimal_batch_size,
    gpu_maximal_marginal_relevance
)
from langchain_hana.gpu.batch_processor import (
    ModelMemoryProfile,
    BatchProcessingStats,
    DynamicBatchProcessor,
    EmbeddingBatchProcessor
)

# SAP HANA Cloud specific GPU-accelerated components
from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings
from langchain_hana.gpu.hana_tensorrt_vectorstore import HanaTensorRTVectorStore
from langchain_hana.gpu.vector_serialization import (
    serialize_vector,
    deserialize_vector,
    serialize_batch,
    deserialize_batch,
    get_vector_memory_usage,
    vector_to_arrow_array,
    vectors_to_arrow_batch,
    arrow_batch_to_vectors,
    arrow_batch_to_documents,
    serialize_arrow_batch,
    deserialize_arrow_batch
)
from langchain_hana.gpu.tensor_core_optimizer import (
    TensorCoreOptimizer,
    optimize_model_for_t4
)
from langchain_hana.gpu.multi_gpu_manager import (
    EnhancedMultiGPUManager,
    get_multi_gpu_manager
)
from langchain_hana.gpu.calibration_datasets import (
    create_enhanced_calibration_dataset,
    get_domain_calibration_texts,
    get_mixed_calibration_texts
)

# Arrow Flight integration components
try:
    from langchain_hana.gpu.arrow_flight_client import ArrowFlightClient
    from langchain_hana.gpu.arrow_flight_server import (
        HanaArrowFlightServer,
        start_arrow_flight_server
    )
    from langchain_hana.gpu.arrow_gpu_memory_manager import ArrowGpuMemoryManager
    from langchain_hana.gpu.arrow_flight_vectorstore import HanaArrowFlightVectorStore
    from langchain_hana.gpu.arrow_flight_multi_gpu import ArrowFlightMultiGPUManager
    HAS_ARROW_FLIGHT = True
except ImportError:
    HAS_ARROW_FLIGHT = False

__all__ = [
    # Core GPU utilities
    'detect_gpu_capabilities',
    'get_optimal_batch_size',
    'gpu_maximal_marginal_relevance',
    
    # Memory management
    'ModelMemoryProfile',
    'BatchProcessingStats',
    'MemoryStrategy',
    
    # Multi-GPU management
    'EnhancedMultiGPUManager',
    'get_multi_gpu_manager',
    
    # Batch processing
    'DynamicBatchProcessor',
    'EmbeddingBatchProcessor',
    
    # Vector serialization
    'serialize_batch',
    'deserialize_batch',
    'serialize_vector',
    'deserialize_vector',
    
    # Calibration datasets
    'create_enhanced_calibration_dataset',
    'get_domain_calibration_texts',
    'get_mixed_calibration_texts',
    
    # Model optimization
    'optimize_model_for_t4',
]

# Add Arrow serialization components to __all__
__all__ += [
    # Arrow serialization
    'vector_to_arrow_array',
    'vectors_to_arrow_batch',
    'arrow_batch_to_vectors',
    'arrow_batch_to_documents',
    'serialize_arrow_batch',
    'deserialize_arrow_batch',
]

# Add Arrow Flight components to __all__ if available
if HAS_ARROW_FLIGHT:
    __all__ += [
        # Arrow Flight integration
        'ArrowFlightClient',
        'HanaArrowFlightServer',
        'start_arrow_flight_server',
        'ArrowGpuMemoryManager',
        'HanaArrowFlightVectorStore',
        'ArrowFlightMultiGPUManager',
        'HAS_ARROW_FLIGHT',
    ]