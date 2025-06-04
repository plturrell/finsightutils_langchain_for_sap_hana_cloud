"""
GPU acceleration utilities for SAP HANA Cloud LangChain integration.

This module provides utilities for leveraging GPU acceleration to improve 
performance of embedding generation and vector operations.

Key components:
- TensorRTEmbeddings: High-performance embedding generation using NVIDIA TensorRT
- GPUAccelerator: Unified interface for GPU acceleration across different frameworks
- DynamicBatchProcessor: Memory-aware dynamic batch sizing for optimal throughput
- EmbeddingBatchProcessor: Specialized batch processor for embedding generation
"""

# Make the most important classes available at the package level
from langchain_hana.gpu.tensorrt_embeddings import TensorRTEmbeddings, GPUInfo, get_available_gpus
from langchain_hana.gpu.accelerator import GPUAccelerator, AcceleratorType, MemoryStrategy
from langchain_hana.gpu.batch_processor import (
    DynamicBatchProcessor,
    EmbeddingBatchProcessor,
    ModelMemoryProfile,
    BatchProcessingStats,
    GPUMemoryInfo
)

__all__ = [
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
    "GPUMemoryInfo"
]