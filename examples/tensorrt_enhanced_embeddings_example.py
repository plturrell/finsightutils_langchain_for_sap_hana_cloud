"""
TensorRT Enhanced Embeddings Example

This example demonstrates how to use the enhanced TensorRT embeddings module
with advanced T4 GPU optimization for improved performance.

Key features demonstrated:
1. Initialization with different precision modes (FP16, FP32)
2. Tensor Core optimization for NVIDIA T4 GPUs
3. Dynamic batch sizing based on available GPU memory
4. Performance benchmarking and comparison with standard embeddings
5. Memory usage tracking and optimization

Requirements:
- NVIDIA GPU (optimized for T4)
- PyTorch with CUDA support
- TensorRT
- sentence-transformers
"""

import time
import logging
import argparse
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Conditionally import torch
try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch with CUDA support not available. Some features will be disabled.")

# Import embeddings modules (with error handling)
try:
    from api.embeddings.embeddings_tensorrt import TensorRTEmbeddings
    from api.embeddings.embeddings_tensorrt_enhanced import (
        TensorRTEmbeddingsEnhanced,
        TensorRTEmbeddingsWithTensorCores,
        create_tensorrt_embeddings
    )
    HAS_TENSORRT_EMBEDDINGS = True
except ImportError:
    logger.warning("TensorRT embeddings modules not found. Using mock implementations for demonstration.")
    HAS_TENSORRT_EMBEDDINGS = False
    
    # Mock implementations for demonstration
    class MockEmbeddings:
        def __init__(self, **kwargs):
            self.model_name = kwargs.get("model_name", "mock-model")
            self.device = kwargs.get("device", "cpu")
            self.batch_size = kwargs.get("batch_size", 32)
            self.precision = kwargs.get("precision", "fp32")
            self.enable_tensor_cores = kwargs.get("enable_tensor_cores", False)
            logger.info(f"Initialized mock embeddings with {self.precision} precision")
            
        def embed_documents(self, texts: List[str]) -> np.ndarray:
            time.sleep(0.01 * len(texts))  # Simulate processing time
            return np.random.random((len(texts), 384))
            
        def benchmark(self, **kwargs) -> Dict[str, Any]:
            return {
                "model_name": self.model_name,
                "device": self.device,
                "precision": self.precision,
                "batch_results": {
                    "1": {"mean_latency_ms": 15.0, "throughput_samples_per_second": 66.7},
                    "8": {"mean_latency_ms": 40.0, "throughput_samples_per_second": 200.0},
                    "32": {"mean_latency_ms": 120.0, "throughput_samples_per_second": 266.7}
                }
            }
    
    # Mock the enhanced embeddings classes for demo purposes
    TensorRTEmbeddings = MockEmbeddings
    TensorRTEmbeddingsEnhanced = MockEmbeddings
    TensorRTEmbeddingsWithTensorCores = MockEmbeddings
    
    def create_tensorrt_embeddings(**kwargs):
        return MockEmbeddings(**kwargs)


def generate_sample_texts(count: int = 100, length: int = 512) -> List[str]:
    """Generate sample texts for embedding."""
    texts = []
    for i in range(count):
        # Create text with predictable but varied content
        text = f"This is sample text number {i} for embedding generation testing. " * (length // 60)
        texts.append(text[:length])
    return texts


def run_embedding_comparison(
    model_name: str = "all-MiniLM-L6-v2",
    batch_sizes: List[int] = [1, 8, 32, 64],
    precision: str = "fp16",
    enable_tensor_cores: bool = True
):
    """Compare standard and enhanced TensorRT embeddings."""
    logger.info(f"Comparing embeddings with model: {model_name}")
    logger.info(f"Precision: {precision}, Tensor Cores: {enable_tensor_cores}")
    
    # Generate sample texts
    sample_texts = generate_sample_texts(100)
    logger.info(f"Generated {len(sample_texts)} sample texts")
    
    # Initialize standard TensorRT embeddings
    logger.info("Initializing standard TensorRT embeddings")
    standard_embeddings = TensorRTEmbeddings(
        model_name=model_name,
        batch_size=32,
        use_tensorrt=True,
        precision=precision
    )
    
    # Initialize enhanced TensorRT embeddings
    logger.info("Initializing enhanced TensorRT embeddings")
    enhanced_embeddings = create_tensorrt_embeddings(
        model_name=model_name,
        precision=precision,
        enable_tensor_cores=enable_tensor_cores
    )
    
    # Compare embedding quality
    logger.info("Comparing embedding quality")
    standard_embeddings_result = standard_embeddings.embed_documents(sample_texts[:5])
    enhanced_embeddings_result = enhanced_embeddings.encode_texts(sample_texts[:5])
    
    if isinstance(standard_embeddings_result, np.ndarray) and isinstance(enhanced_embeddings_result, np.ndarray):
        logger.info(f"Standard embeddings shape: {standard_embeddings_result.shape}")
        logger.info(f"Enhanced embeddings shape: {enhanced_embeddings_result.shape}")
        
        # Verify dimensions match
        if standard_embeddings_result.shape == enhanced_embeddings_result.shape:
            # Calculate cosine similarity between embeddings
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = []
            for i in range(len(standard_embeddings_result)):
                sim = cosine_similarity(
                    standard_embeddings_result[i].reshape(1, -1),
                    enhanced_embeddings_result[i].reshape(1, -1)
                )[0][0]
                similarities.append(sim)
                
            avg_similarity = sum(similarities) / len(similarities)
            logger.info(f"Average cosine similarity between standard and enhanced embeddings: {avg_similarity:.4f}")
            logger.info(f"Min similarity: {min(similarities):.4f}, Max similarity: {max(similarities):.4f}")
    
    # Benchmark both implementations
    logger.info("Benchmarking standard embeddings")
    standard_benchmark = standard_embeddings.benchmark(
        batch_sizes=batch_sizes,
        iterations=3
    )
    
    logger.info("Benchmarking enhanced embeddings")
    enhanced_benchmark = enhanced_embeddings.benchmark(
        batch_sizes=batch_sizes,
        iterations=3
    )
    
    # Compare performance
    logger.info("\nPerformance Comparison:")
    logger.info("-" * 80)
    logger.info(f"{'Batch Size':<10} {'Standard (items/s)':<20} {'Enhanced (items/s)':<20} {'Speedup':<10}")
    logger.info("-" * 80)
    
    for batch_size in batch_sizes:
        batch_str = str(batch_size)
        if batch_str in standard_benchmark.get("batch_results", {}) and batch_str in enhanced_benchmark.get("batch_results", {}):
            std_throughput = standard_benchmark["batch_results"][batch_str].get("throughput_samples_per_second", 0)
            enh_throughput = enhanced_benchmark["batch_results"][batch_str].get("throughput_samples_per_second", 0)
            speedup = enh_throughput / std_throughput if std_throughput > 0 else 0
            
            logger.info(f"{batch_size:<10} {std_throughput:<20.2f} {enh_throughput:<20.2f} {speedup:<10.2f}x")
    
    logger.info("-" * 80)
    
    # Return results for potential further analysis
    return {
        "standard": standard_benchmark,
        "enhanced": enhanced_benchmark,
        "sample_texts": sample_texts
    }


def tensor_core_optimization_demo(model_name: str = "all-MiniLM-L6-v2"):
    """Demonstrate the effect of tensor core optimization."""
    if not HAS_TORCH or not torch.cuda.is_available():
        logger.warning("CUDA not available. Tensor Core optimization demo requires NVIDIA GPU.")
        return
    
    # Check for Tensor Core support
    device = torch.cuda.current_device()
    cc_major, cc_minor = torch.cuda.get_device_capability(device)
    has_tensor_cores = (cc_major, cc_minor) >= (7, 0)
    
    if not has_tensor_cores:
        logger.warning(f"GPU (compute capability {cc_major}.{cc_minor}) does not support Tensor Cores. Demo will show simulated results.")
    
    logger.info("Initializing embeddings with and without Tensor Core optimization")
    
    # Create embeddings with Tensor Cores
    tc_embeddings = create_tensorrt_embeddings(
        model_name=model_name,
        precision="fp16",  # Tensor Cores work best with FP16
        enable_tensor_cores=True
    )
    
    # Create embeddings without Tensor Cores
    no_tc_embeddings = create_tensorrt_embeddings(
        model_name=model_name,
        precision="fp16",
        enable_tensor_cores=False
    )
    
    # Generate test data
    test_texts = generate_sample_texts(200)
    batch_sizes = [1, 4, 16, 32, 64]
    
    # Benchmark both configurations
    if hasattr(tc_embeddings, "benchmark_tensor_cores"):
        logger.info("Running Tensor Core specific benchmark")
        tc_benchmark = tc_embeddings.benchmark_tensor_cores()
        
        # Display results
        if "batch_results" in tc_benchmark:
            logger.info("\nTensor Core Optimization Results:")
            logger.info("-" * 80)
            logger.info(f"{'Batch Size':<10} {'With TC (ms)':<15} {'Without TC (ms)':<15} {'Speedup':<10}")
            logger.info("-" * 80)
            
            for batch_size, results in tc_benchmark["batch_results"].items():
                with_tc = results.get("tensor_core_time_ms", 0)
                without_tc = results.get("standard_time_ms", 0)
                speedup = results.get("speedup_factor", 0)
                
                logger.info(f"{batch_size:<10} {with_tc:<15.2f} {without_tc:<15.2f} {speedup:<10.2f}x")
            
            logger.info("-" * 80)
            
            if "profiling_data" in tc_benchmark:
                logger.info("\nProfiling Data:")
                for key, value in tc_benchmark["profiling_data"].items():
                    logger.info(f"{key}: {value}")
    else:
        # Fallback to standard benchmark for both
        tc_results = []
        no_tc_results = []
        
        logger.info("Running manual benchmark comparison")
        for batch_size in batch_sizes:
            batch_texts = test_texts[:batch_size]
            
            # Benchmark with Tensor Cores
            start = time.time()
            _ = tc_embeddings.encode_texts(batch_texts)
            tc_time = time.time() - start
            tc_results.append((batch_size, tc_time))
            
            # Benchmark without Tensor Cores
            start = time.time()
            _ = no_tc_embeddings.encode_texts(batch_texts)
            no_tc_time = time.time() - start
            no_tc_results.append((batch_size, no_tc_time))
        
        # Display results
        logger.info("\nTensor Core Optimization Results:")
        logger.info("-" * 80)
        logger.info(f"{'Batch Size':<10} {'With TC (s)':<15} {'Without TC (s)':<15} {'Speedup':<10}")
        logger.info("-" * 80)
        
        for i, batch_size in enumerate(batch_sizes):
            tc_time = tc_results[i][1]
            no_tc_time = no_tc_results[i][1]
            speedup = no_tc_time / tc_time if tc_time > 0 else 0
            
            logger.info(f"{batch_size:<10} {tc_time:<15.4f} {no_tc_time:<15.4f} {speedup:<10.2f}x")
        
        logger.info("-" * 80)


def batch_size_optimization_example(model_name: str = "all-MiniLM-L6-v2"):
    """Demonstrate batch size optimization for TensorRT embeddings."""
    logger.info(f"Running batch size optimization for model: {model_name}")
    
    # Create enhanced embeddings
    embeddings = create_tensorrt_embeddings(
        model_name=model_name,
        precision="fp16",
        enable_tensor_cores=True
    )
    
    # Generate test data
    test_texts = generate_sample_texts(300)
    
    # Define batch sizes to test
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Run benchmark
    benchmark_results = embeddings.benchmark(
        texts=test_texts,
        iterations=3,
        batch_sizes=batch_sizes
    )
    
    # Extract throughput results
    throughputs = []
    latencies = []
    
    logger.info("\nBatch Size Optimization Results:")
    logger.info("-" * 80)
    logger.info(f"{'Batch Size':<10} {'Latency (ms)':<15} {'Throughput (items/s)':<20}")
    logger.info("-" * 80)
    
    for batch_size in batch_sizes:
        batch_str = str(batch_size)
        if batch_str in benchmark_results.get("batch_results", {}):
            latency = benchmark_results["batch_results"][batch_str].get("mean_latency_ms", 0)
            throughput = benchmark_results["batch_results"][batch_str].get("throughput_samples_per_second", 0)
            
            throughputs.append(throughput)
            latencies.append(latency)
            
            logger.info(f"{batch_size:<10} {latency:<15.2f} {throughput:<20.2f}")
    
    logger.info("-" * 80)
    
    # Find optimal batch size
    if throughputs:
        optimal_index = throughputs.index(max(throughputs))
        optimal_batch_size = batch_sizes[optimal_index]
        logger.info(f"Optimal batch size: {optimal_batch_size} (throughput: {throughputs[optimal_index]:.2f} items/s)")
    
    # Plot results if matplotlib is available
    try:
        # Create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Plot throughput
        color = 'tab:blue'
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Throughput (items/s)', color=color)
        ax1.plot(batch_sizes[:len(throughputs)], throughputs, 'o-', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create second y-axis for latency
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Latency (ms)', color=color)
        ax2.plot(batch_sizes[:len(latencies)], latencies, 'o-', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add grid and title
        ax1.grid(True, linestyle='--', alpha=0.7)
        plt.title(f'Batch Size Optimization for {model_name}')
        
        # Add vertical line at optimal batch size
        if throughputs:
            plt.axvline(x=optimal_batch_size, color='green', linestyle='--', alpha=0.7)
            plt.text(optimal_batch_size, max(throughputs) * 0.9, f'Optimal: {optimal_batch_size}', 
                     color='green', fontweight='bold')
        
        # Save the figure
        plt.tight_layout()
        plt.savefig('batch_size_optimization.png')
        logger.info("Saved plot to batch_size_optimization.png")
    except ImportError:
        logger.warning("Matplotlib not available. Skipping plot generation.")
    
    return benchmark_results


def memory_management_example(model_name: str = "all-MiniLM-L6-v2"):
    """Demonstrate memory management features of enhanced embeddings."""
    if not HAS_TORCH or not torch.cuda.is_available():
        logger.warning("CUDA not available. Memory management demo requires NVIDIA GPU.")
        return
    
    logger.info(f"Running memory management example for model: {model_name}")
    
    # Create enhanced embeddings
    embeddings = TensorRTEmbeddingsEnhanced(
        model_name=model_name,
        device="cuda",
        precision="fp16",
        batch_size=32,
        enable_tensor_cores=True
    )
    
    # Generate test data
    test_texts = generate_sample_texts(200)
    
    # Track memory usage
    memory_usage = []
    batch_sizes_to_test = [1, 8, 16, 32, 64, 128]
    
    for batch_size in batch_sizes_to_test:
        # Reset CUDA memory stats
        if hasattr(torch.cuda, "reset_peak_memory_stats"):
            torch.cuda.reset_peak_memory_stats()
        
        # Generate embeddings
        texts = test_texts[:batch_size]
        _ = embeddings.encode_texts(texts)
        
        # Get memory stats
        mem_stats = embeddings.get_memory_usage() if hasattr(embeddings, "get_memory_usage") else {}
        
        # For demo purposes, if get_memory_usage is not available
        if not mem_stats and hasattr(torch.cuda, "max_memory_allocated"):
            mem_stats = {
                "gpu_available": True,
                "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
                "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
                "device": f"cuda:{torch.cuda.current_device()}",
                "device_name": torch.cuda.get_device_name()
            }
        
        memory_usage.append((batch_size, mem_stats))
        logger.info(f"Batch size {batch_size}: {mem_stats}")
    
    # Calculate memory efficiency
    if memory_usage:
        logger.info("\nMemory Efficiency Analysis:")
        logger.info("-" * 80)
        logger.info(f"{'Batch Size':<10} {'Total Memory (MB)':<20} {'Memory per Item (MB)':<25} {'Efficiency Ratio':<20}")
        logger.info("-" * 80)
        
        for i in range(len(memory_usage)):
            batch_size, mem_stats = memory_usage[i]
            
            if "allocated_mb" in mem_stats:
                total_memory = mem_stats["allocated_mb"]
                memory_per_item = total_memory / batch_size
                
                # Calculate efficiency ratio compared to single item
                if i > 0:
                    baseline_memory_per_item = memory_usage[0][1].get("allocated_mb", 0) / memory_usage[0][0]
                    efficiency_ratio = baseline_memory_per_item / memory_per_item if memory_per_item > 0 else 0
                else:
                    efficiency_ratio = 1.0
                
                logger.info(f"{batch_size:<10} {total_memory:<20.2f} {memory_per_item:<25.2f} {efficiency_ratio:<20.2f}")
        
        logger.info("-" * 80)
    
    # Clean up
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="TensorRT Enhanced Embeddings Example")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                        help="Model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"],
                        help="Precision mode (default: fp16)")
    parser.add_argument("--disable-tensor-cores", action="store_true",
                        help="Disable Tensor Core optimization")
    parser.add_argument("--example", type=str, default="all", 
                        choices=["comparison", "tensor-cores", "batch-size", "memory", "all"],
                        help="Example to run (default: all)")
    args = parser.parse_args()
    
    # Check GPU availability
    gpu_info = "Not available"
    if HAS_TORCH and torch.cuda.is_available():
        device = torch.cuda.current_device()
        gpu_info = f"{torch.cuda.get_device_name(device)} (Compute Capability: {torch.cuda.get_device_capability(device)})"
    
    logger.info("=" * 80)
    logger.info("TensorRT Enhanced Embeddings Example")
    logger.info("=" * 80)
    logger.info(f"GPU: {gpu_info}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Precision: {args.precision}")
    logger.info(f"Tensor Cores: {'Disabled' if args.disable_tensor_cores else 'Enabled'}")
    logger.info("=" * 80)
    
    try:
        # Run the selected example
        if args.example in ["comparison", "all"]:
            logger.info("\n\n>> Running Embedding Comparison <<\n")
            run_embedding_comparison(
                model_name=args.model,
                precision=args.precision,
                enable_tensor_cores=not args.disable_tensor_cores
            )
        
        if args.example in ["tensor-cores", "all"]:
            logger.info("\n\n>> Running Tensor Core Optimization Demo <<\n")
            tensor_core_optimization_demo(model_name=args.model)
        
        if args.example in ["batch-size", "all"]:
            logger.info("\n\n>> Running Batch Size Optimization Example <<\n")
            batch_size_optimization_example(model_name=args.model)
        
        if args.example in ["memory", "all"]:
            logger.info("\n\n>> Running Memory Management Example <<\n")
            memory_management_example(model_name=args.model)
        
        logger.info("\nExample completed successfully!")
    
    except Exception as e:
        logger.error(f"Error running example: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()