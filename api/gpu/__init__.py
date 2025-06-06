"""
GPU acceleration utilities for SAP HANA Cloud integration.

This module provides GPU acceleration utilities, including:
- TensorRT utilities
- Multi-GPU management
- Dynamic batch processing
- Memory optimization
"""

from api.gpu.gpu_utils import detect_gpus, get_gpu_info, get_available_memory
from api.gpu.tensorrt_utils import create_tensorrt_engine, optimize_with_tensorrt
from api.gpu.multi_gpu import distribute_workload, setup_multi_gpu
from api.gpu.dynamic_batching import DynamicBatcher, calculate_optimal_batch_size
from api.gpu.memory_optimization import optimize_memory_usage, MemoryOptimizer

__all__ = [
    # GPU utils
    "detect_gpus",
    "get_gpu_info",
    "get_available_memory",
    
    # TensorRT
    "create_tensorrt_engine",
    "optimize_with_tensorrt",
    
    # Multi-GPU
    "distribute_workload",
    "setup_multi_gpu",
    
    # Batching
    "DynamicBatcher",
    "calculate_optimal_batch_size",
    
    # Memory
    "optimize_memory_usage",
    "MemoryOptimizer"
]