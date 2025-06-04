#!/usr/bin/env python3
"""
Batch size performance analysis script for SAP HANA Cloud LangChain integration.

This script analyzes the performance of different batch sizes for embedding generation
to identify and understand why smaller batch sizes might perform better than larger ones.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

# Add project root to path to ensure modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_hana.gpu.tensorrt_embeddings import TensorRTEmbeddings, get_available_gpus
from langchain_hana.monitoring.profiler import (
    profile_embedding_model, 
    create_batch_size_comparison_report
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def trace_cuda_kernels(
    embedding_model: TensorRTEmbeddings,
    batch_sizes: List[int],
    output_dir: str,
    text_length: int = 100,
):
    """
    Generate NVTX traces for CUDA kernel analysis with Nsight.
    
    Args:
        embedding_model: The embedding model to profile
        batch_sizes: List of batch sizes to analyze
        output_dir: Directory to save output files
        text_length: Length of the text to use for profiling
    """
    try:
        import torch
        import nvtx
    except ImportError:
        logger.error("NVTX tracing requires torch and nvtx packages. Install with: pip install torch nvtx")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create NVTX ranges for each batch size
    for batch_size in batch_sizes:
        logger.info(f"Tracing batch size: {batch_size}")
        
        # Generate sample texts
        texts = [
            "x" * text_length + f" sample text {i} for tracing"
            for i in range(batch_size)
        ]
        
        # Warmup
        embedding_model.embed_documents(texts)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Trace with NVTX
        with nvtx.annotate(f"batch_size_{batch_size}", color="green"):
            # Pre-tokenization
            with nvtx.annotate("tokenization", color="blue"):
                pass  # Tokenization happens inside embed_documents
            
            # Embedding generation
            with nvtx.annotate("embedding_generation", color="red"):
                embeddings = embedding_model.embed_documents(texts)
            
            # Synchronize to ensure all operations complete
            with nvtx.annotate("synchronize", color="yellow"):
                torch.cuda.synchronize()
        
        logger.info(f"Completed tracing batch size: {batch_size}")
    
    logger.info(f"NVTX tracing complete. Use NVIDIA Nsight Systems to capture the trace.")
    logger.info(f"Run: nsys profile -o {output_dir}/batch_profile python scripts/analyze_batch_performance.py --trace-only")


def analyze_memory_patterns(
    embedding_model: TensorRTEmbeddings,
    batch_sizes: List[int],
    output_dir: str,
    text_length: int = 100,
):
    """
    Analyze memory allocation patterns for different batch sizes.
    
    Args:
        embedding_model: The embedding model to profile
        batch_sizes: List of batch sizes to analyze
        output_dir: Directory to save output files
        text_length: Length of the text to use for profiling
    """
    try:
        import torch
    except ImportError:
        logger.error("Memory analysis requires torch package. Install with: pip install torch")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open a file to write results
    with open(os.path.join(output_dir, "memory_analysis.csv"), "w") as f:
        f.write("batch_size,total_allocations,max_allocation_size,total_allocated_mb,max_allocated_mb,fragmentation_percent\n")
        
        # Analyze each batch size
        for batch_size in batch_sizes:
            logger.info(f"Analyzing memory patterns for batch size: {batch_size}")
            
            # Generate sample texts
            texts = [
                "x" * text_length + f" sample text {i} for memory analysis"
                for i in range(batch_size)
            ]
            
            # Reset memory stats
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            # Run embedding
            embeddings = embedding_model.embed_documents(texts)
            torch.cuda.synchronize()
            
            # Get memory stats
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)  # MB
            current_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)  # MB
            
            # Calculate fragmentation (reserved - allocated) / reserved
            fragmentation = 0
            if reserved > 0:
                fragmentation = (reserved - current_allocated) / reserved * 100
            
            # Try to get detailed allocation info if available
            try:
                stats = torch.cuda.memory_stats()
                total_allocations = stats.get("num_alloc_retries", 0)
                max_allocation_size = stats.get("largest_block_size", 0) / (1024 * 1024)  # MB
            except (AttributeError, RuntimeError):
                # Not all PyTorch versions support detailed memory stats
                total_allocations = 0
                max_allocation_size = 0
            
            # Write results
            f.write(f"{batch_size},{total_allocations},{max_allocation_size:.2f},{current_allocated:.2f},{max_allocated:.2f},{fragmentation:.2f}\n")
            
            logger.info(f"  Max allocated: {max_allocated:.2f} MB")
            logger.info(f"  Current allocated: {current_allocated:.2f} MB")
            logger.info(f"  Reserved: {reserved:.2f} MB")
            logger.info(f"  Fragmentation: {fragmentation:.2f}%")
            
            # Clean up
            torch.cuda.empty_cache()
    
    logger.info(f"Memory pattern analysis complete. Results saved to {os.path.join(output_dir, 'memory_analysis.csv')}")


def analyze_kernel_efficiency(
    embedding_model: TensorRTEmbeddings,
    batch_sizes: List[int],
    output_dir: str,
    text_length: int = 100,
):
    """
    Analyze CUDA kernel efficiency for different batch sizes.
    
    Args:
        embedding_model: The embedding model to profile
        batch_sizes: List of batch sizes to analyze
        output_dir: Directory to save output files
        text_length: Length of the text to use for profiling
    """
    try:
        import torch
        has_pyprof = False
        try:
            from torch.autograd.profiler import profile
            has_pyprof = True
        except ImportError:
            logger.warning("PyTorch profiler not available. Using basic timing instead.")
    except ImportError:
        logger.error("Kernel analysis requires torch package. Install with: pip install torch")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open a file to write results
    with open(os.path.join(output_dir, "kernel_analysis.csv"), "w") as f:
        f.write("batch_size,total_cuda_time_ms,host_time_ms,cuda_util_percent,kernel_count,avg_kernel_time_ms\n")
        
        # Analyze each batch size
        for batch_size in batch_sizes:
            logger.info(f"Analyzing kernel efficiency for batch size: {batch_size}")
            
            # Generate sample texts
            texts = [
                "x" * text_length + f" sample text {i} for kernel analysis"
                for i in range(batch_size)
            ]
            
            # Warmup
            embedding_model.embed_documents(texts)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Run with profiling if available
            if has_pyprof:
                with profile(use_cuda=True) as prof:
                    start_time = time.time()
                    embeddings = embedding_model.embed_documents(texts)
                    torch.cuda.synchronize()
                    host_time_ms = (time.time() - start_time) * 1000
                
                # Process profiling results
                cuda_time_ms = 0
                kernel_count = 0
                avg_kernel_time_ms = 0
                
                for evt in prof.key_averages():
                    if evt.is_cuda:
                        cuda_time_ms += evt.cuda_time_total
                        kernel_count += 1
                
                if kernel_count > 0:
                    avg_kernel_time_ms = cuda_time_ms / kernel_count
                
                cuda_util_percent = 0
                if host_time_ms > 0:
                    cuda_util_percent = (cuda_time_ms / host_time_ms) * 100
                
                # Write results
                f.write(f"{batch_size},{cuda_time_ms:.2f},{host_time_ms:.2f},{cuda_util_percent:.2f},{kernel_count},{avg_kernel_time_ms:.2f}\n")
                
                logger.info(f"  Host time: {host_time_ms:.2f} ms")
                logger.info(f"  CUDA time: {cuda_time_ms:.2f} ms")
                logger.info(f"  CUDA utilization: {cuda_util_percent:.2f}%")
                logger.info(f"  Kernel count: {kernel_count}")
                logger.info(f"  Average kernel time: {avg_kernel_time_ms:.2f} ms")
            else:
                # Basic timing
                start_time = time.time()
                embeddings = embedding_model.embed_documents(texts)
                torch.cuda.synchronize()
                host_time_ms = (time.time() - start_time) * 1000
                
                # Write basic results
                f.write(f"{batch_size},N/A,{host_time_ms:.2f},N/A,N/A,N/A\n")
                
                logger.info(f"  Host time: {host_time_ms:.2f} ms")
            
            # Clean up
            torch.cuda.empty_cache()
    
    logger.info(f"Kernel efficiency analysis complete. Results saved to {os.path.join(output_dir, 'kernel_analysis.csv')}")


def analyze_synchronization_overhead(
    embedding_model: TensorRTEmbeddings,
    batch_sizes: List[int],
    output_dir: str,
    text_length: int = 100,
):
    """
    Analyze synchronization overhead for different batch sizes.
    
    Args:
        embedding_model: The embedding model to profile
        batch_sizes: List of batch sizes to analyze
        output_dir: Directory to save output files
        text_length: Length of the text to use for profiling
    """
    try:
        import torch
    except ImportError:
        logger.error("Synchronization analysis requires torch package. Install with: pip install torch")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Open a file to write results
    with open(os.path.join(output_dir, "sync_analysis.csv"), "w") as f:
        f.write("batch_size,total_time_ms,cuda_time_ms,sync_time_ms,sync_percent,items_per_second\n")
        
        # Analyze each batch size
        for batch_size in batch_sizes:
            logger.info(f"Analyzing synchronization overhead for batch size: {batch_size}")
            
            # Generate sample texts
            texts = [
                "x" * text_length + f" sample text {i} for sync analysis"
                for i in range(batch_size)
            ]
            
            # Warmup
            embedding_model.embed_documents(texts)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            
            # Measure with and without explicit synchronization
            start_time = time.time()
            embeddings = embedding_model.embed_documents(texts)
            pre_sync_time = time.time()
            torch.cuda.synchronize()
            end_time = time.time()
            
            # Calculate times
            total_time_ms = (end_time - start_time) * 1000
            cuda_time_ms = (pre_sync_time - start_time) * 1000
            sync_time_ms = (end_time - pre_sync_time) * 1000
            
            # Calculate percentage
            sync_percent = 0
            if total_time_ms > 0:
                sync_percent = (sync_time_ms / total_time_ms) * 100
            
            # Calculate throughput
            items_per_second = 0
            if total_time_ms > 0:
                items_per_second = (batch_size / total_time_ms) * 1000
            
            # Write results
            f.write(f"{batch_size},{total_time_ms:.2f},{cuda_time_ms:.2f},{sync_time_ms:.2f},{sync_percent:.2f},{items_per_second:.2f}\n")
            
            logger.info(f"  Total time: {total_time_ms:.2f} ms")
            logger.info(f"  CUDA time (pre-sync): {cuda_time_ms:.2f} ms")
            logger.info(f"  Sync time: {sync_time_ms:.2f} ms")
            logger.info(f"  Sync percentage: {sync_percent:.2f}%")
            logger.info(f"  Items per second: {items_per_second:.2f}")
            
            # Clean up
            torch.cuda.empty_cache()
    
    logger.info(f"Synchronization analysis complete. Results saved to {os.path.join(output_dir, 'sync_analysis.csv')}")


def analyze_batch_size_impact(
    model_name: str = "all-MiniLM-L6-v2",
    batch_sizes: List[int] = None,
    output_dir: str = "batch_analysis",
    iterations: int = 5,
    text_length: int = 100,
    device_id: int = 0,
    run_all: bool = True,
    trace_only: bool = False,
):
    """
    Comprehensive analysis of batch size impact on embedding performance.
    
    Args:
        model_name: Name of the model to analyze
        batch_sizes: List of batch sizes to analyze
        output_dir: Directory to save output files
        iterations: Number of iterations for profiling
        text_length: Length of the text to use for profiling
        device_id: CUDA device ID to use
        run_all: Whether to run all analyses
        trace_only: Whether to only run NVTX tracing (for Nsight)
    """
    # Set default batch sizes if not provided
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU
    gpus = get_available_gpus()
    if not gpus:
        logger.error("No GPUs available. This analysis requires a GPU.")
        return
    
    # Log GPU info
    logger.info(f"Using GPU: {gpus[device_id]}")
    
    # Create embedding model
    logger.info(f"Creating embedding model: {model_name}")
    model_name_full = f"sentence-transformers/{model_name}" if "/" not in model_name else model_name
    embedding_model = TensorRTEmbeddings(
        model_name=model_name_full,
        max_batch_size=max(batch_sizes),
        precision="fp16"  # Use FP16 for best performance
    )
    
    # If trace only, just run the NVTX tracing and exit
    if trace_only:
        trace_cuda_kernels(embedding_model, batch_sizes, output_dir, text_length)
        return
    
    # Run the main profiling
    logger.info("Running comprehensive batch size profiling")
    results = profile_embedding_model(
        embedding_model=embedding_model,
        batch_sizes=batch_sizes,
        text_lengths=[text_length],
        iterations=iterations,
        save_path=os.path.join(output_dir, "profile_results"),
        device_id=device_id,
    )
    
    # Generate HTML report
    report_path = os.path.join(output_dir, "batch_size_report.html")
    logger.info(f"Generating HTML report: {report_path}")
    create_batch_size_comparison_report(results, report_path)
    
    # Run additional analyses if requested
    if run_all:
        # Memory pattern analysis
        logger.info("Analyzing memory patterns")
        analyze_memory_patterns(embedding_model, batch_sizes, output_dir, text_length)
        
        # Kernel efficiency analysis
        logger.info("Analyzing kernel efficiency")
        analyze_kernel_efficiency(embedding_model, batch_sizes, output_dir, text_length)
        
        # Synchronization overhead analysis
        logger.info("Analyzing synchronization overhead")
        analyze_synchronization_overhead(embedding_model, batch_sizes, output_dir, text_length)
        
        # CUDA kernel tracing
        logger.info("Setting up CUDA kernel tracing")
        trace_cuda_kernels(embedding_model, batch_sizes, output_dir, text_length)
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")
    logger.info(f"View the HTML report at {report_path}")


def main():
    """Parse command-line arguments and run the analysis."""
    parser = argparse.ArgumentParser(description="Analyze batch size impact on embedding performance")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2",
                        help="Model name to analyze")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128],
                        help="Batch sizes to analyze")
    parser.add_argument("--output-dir", type=str, default="batch_analysis",
                        help="Directory to save output files")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of iterations for profiling")
    parser.add_argument("--text-length", type=int, default=100,
                        help="Length of the text to use for profiling")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device ID to use")
    parser.add_argument("--memory-only", action="store_true",
                        help="Only run memory analysis")
    parser.add_argument("--kernel-only", action="store_true",
                        help="Only run kernel analysis")
    parser.add_argument("--sync-only", action="store_true",
                        help="Only run synchronization analysis")
    parser.add_argument("--trace-only", action="store_true",
                        help="Only run NVTX tracing (for Nsight)")
    args = parser.parse_args()
    
    # Determine whether to run all analyses
    run_all = not (args.memory_only or args.kernel_only or args.sync_only or args.trace_only)
    
    try:
        if args.memory_only:
            # Create embedding model
            model_name_full = f"sentence-transformers/{args.model}" if "/" not in args.model else args.model
            embedding_model = TensorRTEmbeddings(
                model_name=model_name_full,
                max_batch_size=max(args.batch_sizes),
                precision="fp16"
            )
            analyze_memory_patterns(embedding_model, args.batch_sizes, args.output_dir, args.text_length)
        elif args.kernel_only:
            # Create embedding model
            model_name_full = f"sentence-transformers/{args.model}" if "/" not in args.model else args.model
            embedding_model = TensorRTEmbeddings(
                model_name=model_name_full,
                max_batch_size=max(args.batch_sizes),
                precision="fp16"
            )
            analyze_kernel_efficiency(embedding_model, args.batch_sizes, args.output_dir, args.text_length)
        elif args.sync_only:
            # Create embedding model
            model_name_full = f"sentence-transformers/{args.model}" if "/" not in args.model else args.model
            embedding_model = TensorRTEmbeddings(
                model_name=model_name_full,
                max_batch_size=max(args.batch_sizes),
                precision="fp16"
            )
            analyze_synchronization_overhead(embedding_model, args.batch_sizes, args.output_dir, args.text_length)
        else:
            # Run full analysis
            analyze_batch_size_impact(
                model_name=args.model,
                batch_sizes=args.batch_sizes,
                output_dir=args.output_dir,
                iterations=args.iterations,
                text_length=args.text_length,
                device_id=args.device,
                run_all=run_all,
                trace_only=args.trace_only
            )
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()