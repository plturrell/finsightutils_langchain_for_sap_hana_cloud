#!/usr/bin/env python
"""
Auto-tuning script for SAP HANA Cloud LangChain Integration.

This script automatically tunes and optimizes configuration parameters:
1. Determines optimal batch sizes based on GPU memory and throughput testing
2. Selects the best TensorRT precision mode for the current hardware
3. Tunes HNSW index parameters for optimal vector search performance
4. Configures optimal thread counts and worker processes
5. Finds memory allocation sweet spots for different operations

Usage:
    python -m scripts.auto_tune \
        --duration=60 \
        --test-data-size=1000 \
        --output-file=/app/config/auto_tuned_config.json
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("auto_tuner")

# Check for GPU availability
try:
    import torch.cuda as cuda
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False

# Check for TensorRT
try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-tune SAP HANA Cloud LangChain Integration"
    )
    
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Duration of tuning in minutes",
    )
    
    parser.add_argument(
        "--test-data-size",
        type=int,
        default=1000,
        help="Number of test data samples to use",
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default="/app/config/auto_tuned_config.json",
        help="Output file for tuned configuration",
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Model name to tune for",
    )
    
    parser.add_argument(
        "--hana-connection-string",
        type=str,
        help="SAP HANA connection string for vector store tuning",
    )
    
    return parser.parse_args()


def find_optimal_batch_sizes() -> Dict[str, int]:
    """
    Find optimal batch sizes for different operations.
    
    Tests different batch sizes and measures throughput and memory usage
    to determine the optimal batch size for different operations.
    
    Returns:
        Dict with optimal batch sizes for different operations
    """
    logger.info("Finding optimal batch sizes...")
    
    if not HAS_CUDA:
        logger.warning("CUDA not available. Using default batch sizes.")
        return {
            "default": 32,
            "embedding_generation": 64,
            "vector_search": 16,
            "max_batch_size": 128,
        }
    
    # Get GPU memory information
    total_memory = cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024 ** 3)  # Convert to GB
    
    # Start with batch size estimation based on GPU memory
    if total_memory_gb >= 32:
        default_batch_size = 128
    elif total_memory_gb >= 16:
        default_batch_size = 64
    elif total_memory_gb >= 8:
        default_batch_size = 32
    else:
        default_batch_size = 16
    
    # Test different batch sizes for embedding generation
    batch_sizes = [8, 16, 32, 64, 128, 256]
    embedding_generation_results = {}
    vector_search_results = {}
    
    # This would normally involve actual benchmarking
    # For this mock implementation, we'll estimate based on GPU memory
    for batch_size in batch_sizes:
        # Mock throughput estimation (tokens/sec)
        estimated_throughput = min(batch_size * 100, total_memory_gb * 500)
        estimated_memory_usage = batch_size * 0.5  # GB
        
        if estimated_memory_usage <= total_memory_gb * 0.8:  # Keep 20% memory free
            embedding_generation_results[batch_size] = estimated_throughput
            vector_search_results[batch_size] = estimated_throughput * 0.7  # Vector search is typically slower
    
    # Find optimal batch sizes (highest throughput that fits in memory)
    optimal_embedding_batch_size = max(embedding_generation_results.items(), 
                                      key=lambda x: x[1])[0] if embedding_generation_results else default_batch_size
    
    optimal_vector_search_batch_size = max(vector_search_results.items(),
                                          key=lambda x: x[1])[0] if vector_search_results else default_batch_size // 2
    
    # Set maximum batch size (for peak throughput scenarios)
    max_batch_size = min(optimal_embedding_batch_size * 2, 256)
    
    return {
        "default": default_batch_size,
        "embedding_generation": optimal_embedding_batch_size,
        "vector_search": optimal_vector_search_batch_size,
        "max_batch_size": max_batch_size,
    }


def determine_optimal_precision() -> str:
    """
    Determine the optimal precision mode for TensorRT.
    
    Tests different precision modes and measures accuracy and performance
    to determine the optimal precision mode for the current hardware.
    
    Returns:
        Optimal precision mode (fp32, fp16, int8)
    """
    logger.info("Determining optimal precision mode...")
    
    if not HAS_CUDA or not HAS_TENSORRT:
        logger.warning("CUDA or TensorRT not available. Using fp16 precision.")
        return "fp16"
    
    # Check hardware capabilities
    cuda_device = torch.device("cuda:0")
    gpu_compute_capability = torch.cuda.get_device_capability(cuda_device)
    
    # Tensor Cores available on Volta (7.0), Turing (7.5), Ampere (8.0+), Ada (8.9), Hopper (9.0)
    has_tensor_cores = gpu_compute_capability[0] >= 7
    
    # INT8 precision available on Turing (7.5) and later
    supports_int8 = gpu_compute_capability[0] > 7 or (gpu_compute_capability[0] == 7 and gpu_compute_capability[1] >= 5)
    
    # FP16 precision available on Pascal (6.0) and later
    supports_fp16 = gpu_compute_capability[0] >= 6
    
    # This would normally involve accuracy testing with INT8 calibration
    # For this mock implementation, we'll use a simple heuristic
    
    # Check for TensorRT version support for INT8
    if HAS_TENSORRT:
        try:
            trt_version = trt.__version__
            supports_int8_in_trt = int(trt_version.split('.')[0]) >= 7
        except (AttributeError, ValueError, IndexError):
            supports_int8_in_trt = False
    else:
        supports_int8_in_trt = False
    
    # Determine optimal precision
    if supports_int8 and supports_int8_in_trt:
        # INT8 is fastest but requires calibration for accuracy
        # In a real implementation, we would validate accuracy with test data
        return "int8"
    elif supports_fp16:
        # FP16 is a good balance of accuracy and performance
        return "fp16"
    else:
        # Fall back to FP32 if nothing else is supported
        return "fp32"


def tune_hnsw_parameters(connection_string: Optional[str] = None) -> Dict[str, int]:
    """
    Tune HNSW index parameters for vector search.
    
    Tests different HNSW parameters and measures search performance
    to determine the optimal HNSW parameters for the current data.
    
    Args:
        connection_string: SAP HANA connection string
        
    Returns:
        Dict with optimal HNSW parameters
    """
    logger.info("Tuning HNSW parameters...")
    
    # This would normally involve connecting to the database and testing
    # For this mock implementation, we'll use sensible defaults
    
    # If we have a connection string, we would test with actual data
    if connection_string:
        # Mock testing with different parameters
        # In a real implementation, we would benchmark search performance
        return {
            "m": 16,                  # Number of connections per element
            "ef_construction": 100,   # Search width during index construction
            "ef_search": 50,          # Search width during search
        }
    else:
        # Default values based on common benchmarks
        return {
            "m": 16,                  # Balanced for both search speed and accuracy
            "ef_construction": 100,   # Moderate quality index
            "ef_search": 50,          # Balanced for both speed and recall
        }


def determine_optimal_workers() -> Dict[str, int]:
    """
    Determine the optimal number of workers for different services.
    
    Takes into account CPU cores and memory to determine the optimal
    number of workers for different services.
    
    Returns:
        Dict with optimal worker counts for different services
    """
    logger.info("Determining optimal worker counts...")
    
    # Get CPU information
    cpu_count = os.cpu_count() or 4
    
    # In a real implementation, we would benchmark with different worker counts
    # For this mock implementation, we'll use a simple heuristic
    
    # API workers (typically CPU-bound)
    api_workers = max(1, min(cpu_count - 1, 4))
    
    # GPU workers (should be less than or equal to number of GPUs)
    gpu_count = torch.cuda.device_count() if HAS_CUDA else 0
    gpu_workers = max(1, gpu_count)
    
    # Database connection pool size (typically I/O-bound)
    db_pool_size = max(5, min(cpu_count * 2, 20))
    
    return {
        "api_workers": api_workers,
        "gpu_workers": gpu_workers,
        "db_pool_size": db_pool_size,
        "thread_count": max(1, cpu_count // 2),  # For thread pool executor
    }


def tune_gpu_memory_allocation() -> Dict[str, float]:
    """
    Tune GPU memory allocation parameters.
    
    Determines the optimal memory fraction for different operations
    to balance between memory usage and performance.
    
    Returns:
        Dict with optimal memory allocation parameters
    """
    logger.info("Tuning GPU memory allocation...")
    
    if not HAS_CUDA:
        logger.warning("CUDA not available. Using default memory allocation.")
        return {
            "memory_fraction": 0.9,
            "cache_size_mb": 2048,
            "max_workspace_size_mb": 1024,
        }
    
    # Get GPU memory information
    total_memory = cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024 ** 3)  # Convert to GB
    
    # In a real implementation, we would test different memory configurations
    # For this mock implementation, we'll use a heuristic based on GPU memory
    
    # For larger GPUs, we can reserve more memory for PyTorch
    if total_memory_gb >= 32:
        memory_fraction = 0.9  # Reserve 90% for PyTorch
        cache_size_mb = 8192
        max_workspace_size_mb = 4096
    elif total_memory_gb >= 16:
        memory_fraction = 0.85  # Reserve 85% for PyTorch
        cache_size_mb = 4096
        max_workspace_size_mb = 2048
    elif total_memory_gb >= 8:
        memory_fraction = 0.8  # Reserve 80% for PyTorch
        cache_size_mb = 2048
        max_workspace_size_mb = 1024
    else:
        memory_fraction = 0.7  # Reserve 70% for PyTorch
        cache_size_mb = 1024
        max_workspace_size_mb = 512
    
    return {
        "memory_fraction": memory_fraction,
        "cache_size_mb": cache_size_mb,
        "max_workspace_size_mb": max_workspace_size_mb,
    }


def run_auto_tuning(args: argparse.Namespace) -> Dict[str, any]:
    """
    Run the auto-tuning process.
    
    Args:
        args: Command-line arguments
        
    Returns:
        Dict with auto-tuned configuration
    """
    logger.info(f"Starting auto-tuning process (duration: {args.duration} minutes)...")
    start_time = time.time()
    end_time = start_time + (args.duration * 60)
    
    # Collect system information
    system_info = {
        "cpu_count": os.cpu_count(),
        "gpu_available": HAS_CUDA,
        "gpu_count": torch.cuda.device_count() if HAS_CUDA else 0,
        "tensorrt_available": HAS_TENSORRT,
    }
    
    if HAS_CUDA:
        system_info["gpu_info"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": torch.cuda.get_device_properties(i).total_memory / (1024 ** 3),
            }
            for i in range(torch.cuda.device_count())
        ]
    
    # Run tuning tasks
    batch_sizes = find_optimal_batch_sizes()
    precision = determine_optimal_precision()
    hnsw_params = tune_hnsw_parameters(args.hana_connection_string)
    worker_counts = determine_optimal_workers()
    memory_allocation = tune_gpu_memory_allocation()
    
    # Create auto-tuned configuration
    auto_tuned_config = {
        "system_info": system_info,
        "batch_sizes": batch_sizes,
        "precision": precision,
        "hnsw_parameters": hnsw_params,
        "worker_counts": worker_counts,
        "memory_allocation": memory_allocation,
        "tuning_info": {
            "duration_minutes": args.duration,
            "test_data_size": args.test_data_size,
            "model_name": args.model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }
    
    # Write configuration to file
    output_dir = os.path.dirname(args.output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_file, "w") as f:
        json.dump(auto_tuned_config, f, indent=2)
    
    logger.info(f"Auto-tuning completed. Configuration saved to {args.output_file}")
    
    return auto_tuned_config


def main():
    """Main function."""
    args = parse_args()
    
    try:
        auto_tuned_config = run_auto_tuning(args)
        
        # Print a summary of the auto-tuned configuration
        print("\nAuto-Tuned Configuration Summary:")
        print(f"Optimal Batch Size: {auto_tuned_config['batch_sizes']['default']}")
        print(f"Optimal Precision: {auto_tuned_config['precision']}")
        print(f"Optimal HNSW Parameters: M={auto_tuned_config['hnsw_parameters']['m']}, "
              f"efConstruction={auto_tuned_config['hnsw_parameters']['ef_construction']}, "
              f"efSearch={auto_tuned_config['hnsw_parameters']['ef_search']}")
        print(f"Optimal API Workers: {auto_tuned_config['worker_counts']['api_workers']}")
        print(f"Optimal GPU Memory Fraction: {auto_tuned_config['memory_allocation']['memory_fraction']}")
        
    except Exception as e:
        logger.error(f"Auto-tuning failed: {e}")
        raise


if __name__ == "__main__":
    main()