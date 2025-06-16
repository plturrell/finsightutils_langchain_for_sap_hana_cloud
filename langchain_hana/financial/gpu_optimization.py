"""
Production-grade GPU optimization for financial embeddings.

This module provides advanced GPU optimization techniques for financial embedding models,
including mixed precision inference, optimized batch processing, and memory management.
"""

import os
import gc
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union, Tuple, Callable

import torch
import numpy as np

logger = logging.getLogger(__name__)


class GPUOptimizer:
    """
    Production-grade GPU optimization for embedding models.
    
    This class provides enterprise-ready GPU optimization techniques,
    including memory management, mixed precision inference, dynamic batching,
    and fault tolerance with automatic recovery.
    """
    
    def __init__(
        self,
        device_id: Optional[int] = None,
        memory_fraction: float = 0.9,
        use_mixed_precision: bool = True,
        precision_type: str = "fp16",
        enable_torch_compile: bool = True,
        enable_dynamic_batching: bool = True,
        enable_memory_monitoring: bool = True,
        monitoring_interval: int = 60,
        enable_fault_tolerance: bool = True,
        max_retries: int = 3,
        enable_tensor_cores: bool = True,
    ):
        """
        Initialize the GPU optimizer.
        
        Args:
            device_id: CUDA device ID to use (None for auto-selection)
            memory_fraction: Fraction of GPU memory to use (0.0-1.0)
            use_mixed_precision: Whether to use mixed precision
            precision_type: Precision type ("fp16", "bf16", "int8")
            enable_torch_compile: Whether to use torch.compile for optimization
            enable_dynamic_batching: Whether to use dynamic batch sizing
            enable_memory_monitoring: Whether to monitor GPU memory usage
            monitoring_interval: Memory monitoring interval in seconds
            enable_fault_tolerance: Whether to enable fault tolerance
            max_retries: Maximum retries for failed operations
            enable_tensor_cores: Whether to enable Tensor Core optimizations
        """
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.use_mixed_precision = use_mixed_precision
        self.precision_type = precision_type
        self.enable_torch_compile = enable_torch_compile
        self.enable_dynamic_batching = enable_dynamic_batching
        self.enable_memory_monitoring = enable_memory_monitoring
        self.monitoring_interval = monitoring_interval
        self.enable_fault_tolerance = enable_fault_tolerance
        self.max_retries = max_retries
        self.enable_tensor_cores = enable_tensor_cores
        
        # Initialize GPU settings
        self.device = None
        self.cuda_available = torch.cuda.is_available()
        self.mixed_precision_available = self._check_mixed_precision_support()
        self.tensor_cores_available = self._check_tensor_cores_support()
        self.compile_available = hasattr(torch, 'compile')
        
        # Memory usage tracking
        self.peak_memory_usage = 0
        self.current_memory_usage = 0
        self.memory_tracking_thread = None
        self.stop_monitoring = False
        
        # Performance metrics
        self.inference_times = []
        self.batch_sizes = []
        self.retry_count = 0
        
        # Initialize GPU
        self._initialize_gpu()
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU settings for optimal performance."""
        if not self.cuda_available:
            logger.warning("CUDA not available. Using CPU for processing.")
            self.device = torch.device("cpu")
            return
        
        # Select device
        if self.device_id is not None:
            if self.device_id < torch.cuda.device_count():
                self.device = torch.device(f"cuda:{self.device_id}")
            else:
                logger.warning(
                    f"Device ID {self.device_id} is out of range "
                    f"(max: {torch.cuda.device_count()-1}). Using default device."
                )
                self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cuda:0")
        
        # Set active device
        torch.cuda.set_device(self.device)
        
        # Configure memory usage
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(self.memory_fraction)
            logger.info(f"Set GPU memory fraction to {self.memory_fraction}")
        
        # Log GPU information
        device_name = torch.cuda.get_device_name(self.device)
        device_capability = torch.cuda.get_device_capability(self.device)
        total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024**3)
        
        logger.info(
            f"Using GPU: {device_name} (compute capability: {device_capability[0]}.{device_capability[1]}, "
            f"memory: {total_memory:.1f} GB)"
        )
        
        # Start memory monitoring if enabled
        if self.enable_memory_monitoring:
            self._start_memory_monitoring()
    
    def _check_mixed_precision_support(self) -> bool:
        """Check if mixed precision is supported."""
        if not self.cuda_available:
            return False
        
        # Check FP16 support
        if self.precision_type == "fp16":
            # All CUDA devices support FP16
            return True
        
        # Check BF16 support
        elif self.precision_type == "bf16":
            # BF16 is supported on Ampere (SM 8.0) and newer
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                major, _ = torch.cuda.get_device_capability(device)
                return major >= 8
            return False
        
        # Check INT8 support
        elif self.precision_type == "int8":
            # INT8 requires PyTorch with QNNPACK or FBGEMM
            return (
                hasattr(torch, 'qscheme') and 
                hasattr(torch.backends, 'quantized') and
                (
                    hasattr(torch.backends.quantized, 'engine') or
                    hasattr(torch.backends, 'cudnn')
                )
            )
        
        return False
    
    def _check_tensor_cores_support(self) -> bool:
        """Check if Tensor Cores are supported."""
        if not self.cuda_available:
            return False
        
        # Tensor Cores are available on Volta (SM 7.0) and newer GPUs
        device = torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(device)
        return major >= 7
    
    def _start_memory_monitoring(self) -> None:
        """Start GPU memory monitoring in a background thread."""
        if not self.cuda_available:
            return
        
        def monitor_memory():
            while not self.stop_monitoring:
                try:
                    # Update current memory usage
                    allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
                    reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
                    self.current_memory_usage = allocated
                    
                    # Update peak memory usage
                    if allocated > self.peak_memory_usage:
                        self.peak_memory_usage = allocated
                    
                    # Log memory usage periodically
                    logger.debug(
                        f"GPU memory: {allocated:.2f} GB allocated, "
                        f"{reserved:.2f} GB reserved, "
                        f"{self.peak_memory_usage:.2f} GB peak"
                    )
                except Exception as e:
                    logger.warning(f"Error monitoring GPU memory: {str(e)}")
                
                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)
        
        # Start monitoring thread
        self.stop_monitoring = False
        self.memory_tracking_thread = threading.Thread(
            target=monitor_memory, 
            daemon=True
        )
        self.memory_tracking_thread.start()
        logger.info("GPU memory monitoring started")
    
    def stop_memory_monitoring(self) -> None:
        """Stop GPU memory monitoring."""
        if self.memory_tracking_thread and self.memory_tracking_thread.is_alive():
            self.stop_monitoring = True
            self.memory_tracking_thread.join(timeout=2.0)
            logger.info("GPU memory monitoring stopped")
    
    def clear_gpu_memory(self) -> None:
        """Clear GPU memory caches to free up resources."""
        if not self.cuda_available:
            return
        
        # Clear PyTorch CUDA cache
        torch.cuda.empty_cache()
        
        # Run garbage collection
        gc.collect()
        
        logger.info("GPU memory cleared")
    
    def optimize_model(self, model: Any) -> Any:
        """
        Apply GPU optimizations to a model.
        
        Args:
            model: PyTorch model to optimize
            
        Returns:
            Optimized model
        """
        if not self.cuda_available:
            logger.warning("CUDA not available. Skipping GPU optimizations.")
            return model
        
        logger.info("Applying GPU optimizations to model...")
        
        try:
            # Move model to configured device
            model.to(self.device)
            
            # Apply mixed precision if enabled and supported
            if self.use_mixed_precision and self.mixed_precision_available:
                if self.precision_type == "fp16":
                    model.half()
                    logger.info("Applied FP16 mixed precision")
                elif self.precision_type == "bf16" and hasattr(torch, 'bfloat16'):
                    model.to(torch.bfloat16)
                    logger.info("Applied BF16 mixed precision")
                elif self.precision_type == "int8":
                    # INT8 quantization requires more complex processing
                    logger.warning("INT8 quantization requires specialized model conversion")
            
            # Apply torch.compile if enabled and available
            if self.enable_torch_compile and self.compile_available:
                try:
                    # For embedding models, we typically optimize the encode method
                    if hasattr(model, 'encode'):
                        original_encode = model.encode
                        compiled_encode = torch.compile(
                            original_encode,
                            mode="reduce-overhead",
                            fullgraph=False
                        )
                        model.encode = compiled_encode
                        logger.info("Applied torch.compile to model.encode")
                except Exception as e:
                    logger.warning(f"Failed to apply torch.compile: {str(e)}")
            
            # Configure for Tensor Core usage if available
            if self.enable_tensor_cores and self.tensor_cores_available:
                if hasattr(torch.backends, 'cudnn'):
                    torch.backends.cudnn.benchmark = True
                    torch.backends.cudnn.enabled = True
                    logger.info("Enabled cuDNN benchmark for Tensor Core optimization")
            
            return model
            
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            return model  # Return original model on error
    
    def get_optimal_batch_size(
        self, 
        current_batch_size: int,
        sequence_length: int
    ) -> int:
        """
        Calculate optimal batch size based on GPU memory and sequence length.
        
        Args:
            current_batch_size: Current batch size
            sequence_length: Average sequence length of the batch
            
        Returns:
            Optimal batch size
        """
        if not self.cuda_available or not self.enable_dynamic_batching:
            return current_batch_size
        
        try:
            # Get available GPU memory
            available_memory = (
                (torch.cuda.get_device_properties(self.device).total_memory / (1024**3)) * 
                self.memory_fraction -
                self.current_memory_usage
            )
            
            # Estimate memory requirements (approximate heuristic)
            # This depends on model architecture, but we use a general estimate
            bytes_per_token = 14 if self.use_mixed_precision else 24
            memory_per_sequence = sequence_length * bytes_per_token / (1024**3)
            
            # Calculate maximum batch size based on available memory
            # We leave 20% margin for safety
            max_batch_size = int(available_memory / (memory_per_sequence * 1.2))
            
            # Adjust batch size to be a multiple of 8 for Tensor Core optimization
            if self.enable_tensor_cores and self.tensor_cores_available:
                max_batch_size = max(8, (max_batch_size // 8) * 8)
            
            # Apply reasonable limits
            max_batch_size = max(1, min(max_batch_size, 256))
            
            # If current batch size is reasonable, keep it for stability
            if current_batch_size <= max_batch_size:
                return current_batch_size
            
            logger.info(
                f"Adjusted batch size from {current_batch_size} to {max_batch_size} "
                f"based on available memory ({available_memory:.2f} GB)"
            )
            
            return max_batch_size
            
        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {str(e)}")
            return current_batch_size  # Return current batch size on error
    
    def process_batch_with_fallback(
        self,
        processing_fn: Callable,
        batch: List[Any],
        **kwargs
    ) -> Any:
        """
        Process a batch with fault tolerance and automatic fallback.
        
        Args:
            processing_fn: Function to process the batch
            batch: Batch data to process
            **kwargs: Additional arguments for processing_fn
            
        Returns:
            Processing result
        """
        if not batch:
            return []
        
        # Track start time for performance monitoring
        start_time = time.time()
        batch_size = len(batch)
        self.batch_sizes.append(batch_size)
        
        # Initialize retry counter
        retries = 0
        
        while retries <= self.max_retries:
            try:
                # Process batch
                result = processing_fn(batch, **kwargs)
                
                # Track performance metrics
                inference_time = time.time() - start_time
                self.inference_times.append(inference_time)
                
                # Log performance periodically
                if len(self.inference_times) % 10 == 0:
                    avg_time = sum(self.inference_times[-10:]) / 10
                    avg_batch = sum(self.batch_sizes[-10:]) / 10
                    logger.debug(
                        f"Performance: {avg_time:.3f}s per batch, "
                        f"{avg_batch / avg_time:.1f} items/sec "
                        f"(avg batch size: {avg_batch:.1f})"
                    )
                
                return result
                
            except torch.cuda.OutOfMemoryError:
                retries += 1
                self.retry_count += 1
                
                if retries > self.max_retries or not self.enable_fault_tolerance:
                    logger.error("Maximum retries exceeded for GPU processing")
                    raise
                
                logger.warning(
                    f"CUDA out of memory (retry {retries}/{self.max_retries}). "
                    f"Reducing batch size and clearing memory."
                )
                
                # Clear GPU memory
                self.clear_gpu_memory()
                
                # Reduce batch size
                if len(batch) > 1:
                    mid = len(batch) // 2
                    
                    # Process first half
                    logger.info(f"Processing first half of batch ({mid} items)")
                    first_half = self.process_batch_with_fallback(
                        processing_fn, 
                        batch[:mid],
                        **kwargs
                    )
                    
                    # Process second half
                    logger.info(f"Processing second half of batch ({len(batch) - mid} items)")
                    second_half = self.process_batch_with_fallback(
                        processing_fn,
                        batch[mid:],
                        **kwargs
                    )
                    
                    # Combine results
                    if isinstance(first_half, list) and isinstance(second_half, list):
                        return first_half + second_half
                    else:
                        logger.warning("Unable to combine split batch results")
                        raise RuntimeError("Failed to process batch after splitting")
                else:
                    logger.error("Cannot process even a single item within GPU memory limits")
                    raise
            
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                
                if not self.enable_fault_tolerance or retries >= self.max_retries:
                    raise
                
                retries += 1
                self.retry_count += 1
                
                logger.warning(
                    f"Retrying batch processing ({retries}/{self.max_retries})"
                )
                
                # Wait before retry
                time.sleep(1.0)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        stats = {
            "cuda_available": self.cuda_available,
            "device": str(self.device) if self.device is not None else "cpu",
            "peak_memory_usage_gb": self.peak_memory_usage,
            "current_memory_usage_gb": self.current_memory_usage,
            "retry_count": self.retry_count,
        }
        
        # Add inference statistics if available
        if self.inference_times:
            stats.update({
                "avg_inference_time": sum(self.inference_times) / len(self.inference_times),
                "min_inference_time": min(self.inference_times),
                "max_inference_time": max(self.inference_times),
                "total_batches_processed": len(self.inference_times),
                "avg_batch_size": sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0,
                "items_per_second": (
                    sum(self.batch_sizes) / sum(self.inference_times) 
                    if self.inference_times and sum(self.inference_times) > 0 else 0
                ),
            })
        
        # Add GPU device info if available
        if self.cuda_available:
            device_props = torch.cuda.get_device_properties(self.device)
            stats.update({
                "gpu_name": device_props.name,
                "gpu_memory_gb": device_props.total_memory / (1024**3),
                "compute_capability": f"{device_props.major}.{device_props.minor}",
                "tensor_cores_available": self.tensor_cores_available,
                "mixed_precision_enabled": self.use_mixed_precision and self.mixed_precision_available,
                "precision_type": self.precision_type,
            })
        
        return stats
    
    def __del__(self):
        """Clean up resources when the optimizer is deleted."""
        self.stop_memory_monitoring()
        

def optimize_for_gpu(model: Any) -> Any:
    """
    Quick utility function to optimize a model for GPU processing.
    
    Args:
        model: PyTorch model to optimize
        
    Returns:
        Optimized model
    """
    optimizer = GPUOptimizer()
    return optimizer.optimize_model(model)