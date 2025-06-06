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
"""

# Base GPU acceleration utilities
from langchain_hana.gpu.tensorrt_embeddings import TensorRTEmbeddings, GPUInfo, get_available_gpus
from langchain_hana.gpu.accelerator import GPUAccelerator, AcceleratorType, MemoryStrategy
from langchain_hana.gpu.batch_processor import (
    DynamicBatchProcessor,
    EmbeddingBatchProcessor,
    ModelMemoryProfile,
    BatchProcessingStats,
    GPUMemoryInfo
)

# SAP HANA Cloud specific GPU-accelerated components
from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings
from langchain_hana.gpu.hana_tensorrt_vectorstore import HanaTensorRTVectorStore
from langchain_hana.gpu.vector_serialization import (
    serialize_vector,
    deserialize_vector,
    serialize_vectors_batch,
    deserialize_vectors_batch,
    get_vector_memory_usage
)
from langchain_hana.gpu.tensor_core_optimizer import (
    TensorCoreOptimizer,
    optimize_tensor_for_tensor_cores,
    check_tensor_core_support
)
from langchain_hana.gpu.multi_gpu_manager import (
    MultiGPUManager,
    get_multi_gpu_manager,
    MultiGPUStrategy
)
from langchain_hana.gpu.calibration_datasets import (
    create_calibration_dataset,
    create_enhanced_calibration_dataset
)

__all__ = [
    # Base TensorRT components
    "TensorRTEmbeddings",
    "GPUInfo",
    "get_available_gpus",
    "GPUAccelerator",
    "AcceleratorType",
    "MemoryStrategy",
    "DynamicBatchProcessor",
    "EmbeddingBatchProcessor",
    "ModelMemoryProfile",
    "BatchProcessingStats",
    "GPUMemoryInfo",
    
    # SAP HANA Cloud specific components
    "HanaTensorRTEmbeddings",
    "HanaTensorRTVectorStore",
    
    # Vector serialization
    "serialize_vector",
    "deserialize_vector",
    "serialize_vectors_batch",
    "deserialize_vectors_batch",
    "get_vector_memory_usage",
    
    # Tensor core optimization
    "TensorCoreOptimizer",
    "optimize_tensor_for_tensor_cores",
    "check_tensor_core_support",
    
    # Multi-GPU support
    "MultiGPUManager",
    "get_multi_gpu_manager",
    "MultiGPUStrategy",
    
    # Calibration datasets
    "create_calibration_dataset",
    "create_enhanced_calibration_dataset"
]