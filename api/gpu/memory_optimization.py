"""Memory optimization for large embedding operations."""

import gc
import logging
import threading
import time
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set

import numpy as np

from api.gpu import gpu_utils
from api.gpu.multi_gpu import get_gpu_manager
from api.gpu.dynamic_batching import get_batch_sizer

logger = logging.getLogger(__name__)


class MemoryOptimizer:
    """
    Memory optimizer for large embedding operations.
    
    This class provides techniques to optimize memory usage during
    large embedding operations, including:
    
    1. Automatic garbage collection
    2. GPU memory caching control
    3. Tensor pooling
    4. Memory-aware processing
    5. Progressive loading
    """
    
    def __init__(
        self,
        auto_gc_threshold: int = 1024 * 1024 * 1024,  # 1 GB
        cache_clear_threshold: float = 0.85,  # 85% memory utilization
        enable_tensor_pooling: bool = True,
        enable_progressive_loading: bool = True,
    ):
        """
        Initialize the memory optimizer.
        
        Args:
            auto_gc_threshold: Memory threshold in bytes for auto garbage collection.
            cache_clear_threshold: Memory utilization threshold for cache clearing.
            enable_tensor_pooling: Whether to enable tensor pooling.
            enable_progressive_loading: Whether to enable progressive loading.
        """
        self.auto_gc_threshold = auto_gc_threshold
        self.cache_clear_threshold = cache_clear_threshold
        self.enable_tensor_pooling = enable_tensor_pooling
        self.enable_progressive_loading = enable_progressive_loading
        
        self.gpu_manager = get_gpu_manager()
        self.last_gc_time = time.time()
        self.min_gc_interval = 5.0  # seconds
        
        # Tensor pooling
        self.tensor_pools = {}
        self.tensor_pools_lock = threading.RLock()
        
        # Memory monitoring
        self._start_memory_monitor()
    
    def _start_memory_monitor(self) -> None:
        """Start a background memory monitoring thread."""
        if not gpu_utils.is_torch_available():
            return
        
        def monitor_memory():
            try:
                try:
                    import torch
                except ImportError:
                    logger.warning("torch not available, using CPU-only mode")
                
                while True:
                    try:
                        # Check GPU memory usage
                        for device_name in self.gpu_manager.get_available_devices():
                            device_info = self.gpu_manager.get_device_info(device_name)
                            device_id = device_info.get("id", 0)
                            
                            # Get memory stats
                            total_memory = device_info.get("total_memory", 0)
                            allocated_memory = torch.cuda.memory_allocated(device_id)
                            reserved_memory = torch.cuda.memory_reserved(device_id)
                            
                            # Calculate utilization
                            memory_utilization = allocated_memory / total_memory if total_memory > 0 else 0
                            
                            # Log memory usage if high
                            if memory_utilization > 0.8:
                                logger.warning(
                                    f"High GPU memory usage on {device_name}: "
                                    f"{allocated_memory / 1024**2:.1f}MB / "
                                    f"{total_memory / 1024**2:.1f}MB "
                                    f"({memory_utilization:.1%})"
                                )
                            
                            # Auto clear cache if threshold exceeded
                            if memory_utilization > self.cache_clear_threshold:
                                self.clear_cache(device_id)
                        
                        # Sleep for a while
                        time.sleep(5.0)
                    
                    except Exception as e:
                        logger.error(f"Error in memory monitor: {str(e)}")
                        time.sleep(10.0)
            
            except Exception as e:
                logger.error(f"Memory monitor thread error: {str(e)}")
        
        # Start the monitoring thread
        monitor_thread = threading.Thread(
            target=monitor_memory,
            daemon=True,
        )
        monitor_thread.start()
    
    def before_batch_processing(self, batch_size: int) -> None:
        """
        Prepare for batch processing.
        
        Args:
            batch_size: Size of the batch to be processed.
        """
        # Determine if we should run garbage collection
        current_time = time.time()
        if current_time - self.last_gc_time >= self.min_gc_interval:
            self.auto_gc()
            self.last_gc_time = current_time
    
    def after_batch_processing(self) -> None:
        """Clean up after batch processing."""
        # Nothing to do here currently
        pass
    
    def auto_gc(self) -> None:
        """Run automatic garbage collection if needed."""
        # Python garbage collection
        gc.collect()
        
        # GPU memory garbage collection
        if gpu_utils.is_torch_available():
            try:
                try:
                    import torch
                except ImportError:
                    logger.warning("torch not available, using CPU-only mode")
                
                # Run garbage collection on each device
                for device_name in self.gpu_manager.get_available_devices():
                    device_info = self.gpu_manager.get_device_info(device_name)
                    device_id = device_info.get("id", 0)
                    
                    # Check memory usage
                    allocated_memory = torch.cuda.memory_allocated(device_id)
                    
                    # Run garbage collection if above threshold
                    if allocated_memory > self.auto_gc_threshold:
                        torch.cuda.empty_cache()
                        logger.debug(
                            f"Ran CUDA garbage collection on device {device_name}, "
                            f"freed {allocated_memory - torch.cuda.memory_allocated(device_id)} bytes"
                        )
            
            except Exception as e:
                logger.error(f"Error in auto_gc: {str(e)}")
    
    def clear_cache(self, device_id: Optional[int] = None) -> None:
        """
        Clear GPU memory cache.
        
        Args:
            device_id: Optional device ID to clear cache for. If None, clears all devices.
        """
        if not gpu_utils.is_torch_available():
            return
        
        try:
            try:
                import torch
            except ImportError:
                logger.warning("torch not available, using CPU-only mode")
            
            if device_id is None:
                # Clear cache on all devices
                for device_name in self.gpu_manager.get_available_devices():
                    device_info = self.gpu_manager.get_device_info(device_name)
                    torch.cuda.empty_cache()
                    logger.info(f"Cleared CUDA cache on all devices")
            else:
                # Clear cache on specific device
                current_device = torch.cuda.current_device()
                torch.cuda.set_device(device_id)
                torch.cuda.empty_cache()
                torch.cuda.set_device(current_device)
                logger.info(f"Cleared CUDA cache on device {device_id}")
        
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
    
    def get_tensor_from_pool(
        self, shape: Tuple[int, ...], dtype: str, device: str
    ) -> Optional[Any]:
        """
        Get a tensor from the pool if available.
        
        Args:
            shape: Shape of the tensor.
            dtype: Data type of the tensor.
            device: Device to allocate the tensor on.
            
        Returns:
            Tensor from the pool or None if not available.
        """
        if not self.enable_tensor_pooling:
            return None
        
        try:
            try:
                import torch
            except ImportError:
                logger.warning("torch not available, using CPU-only mode")
            
            pool_key = (shape, dtype, device)
            
            with self.tensor_pools_lock:
                if pool_key in self.tensor_pools and self.tensor_pools[pool_key]:
                    return self.tensor_pools[pool_key].pop()
            
            return None
        
        except Exception as e:
            logger.error(f"Error getting tensor from pool: {str(e)}")
            return None
    
    def return_tensor_to_pool(self, tensor: Any) -> None:
        """
        Return a tensor to the pool for reuse.
        
        Args:
            tensor: Tensor to return to the pool.
        """
        if not self.enable_tensor_pooling:
            return
        
        try:
            try:
                import torch
            except ImportError:
                logger.warning("torch not available, using CPU-only mode")
            
            if not isinstance(tensor, torch.Tensor):
                return
            
            # Get tensor properties
            shape = tuple(tensor.shape)
            dtype = str(tensor.dtype)
            device = str(tensor.device)
            
            pool_key = (shape, dtype, device)
            
            with self.tensor_pools_lock:
                if pool_key not in self.tensor_pools:
                    self.tensor_pools[pool_key] = []
                
                # Limit pool size to prevent memory leaks
                if len(self.tensor_pools[pool_key]) < 10:
                    self.tensor_pools[pool_key].append(tensor)
        
        except Exception as e:
            logger.error(f"Error returning tensor to pool: {str(e)}")
    
    def optimize_embeddings(
        self, embeddings: List[List[float]]
    ) -> Union[List[List[float]], np.ndarray]:
        """
        Optimize memory usage for embeddings.
        
        Args:
            embeddings: List of embedding vectors.
            
        Returns:
            Optimized embeddings.
        """
        if not embeddings:
            return embeddings
        
        # Convert to numpy array for better memory efficiency
        if isinstance(embeddings, list):
            # Determine if we should convert to a more memory-efficient format
            return np.array(embeddings, dtype=np.float32)
        
        return embeddings
    
    def process_in_chunks(
        self,
        items: List[Any],
        process_fn: Callable,
        chunk_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> List:
        """
        Process large collections of items in chunks to optimize memory usage.
        
        Args:
            items: List of items to process.
            process_fn: Function to apply to each chunk.
            chunk_size: Size of each chunk. If None, determines automatically.
            *args: Additional arguments to pass to process_fn.
            **kwargs: Additional keyword arguments to pass to process_fn.
            
        Returns:
            List of results.
        """
        if not items:
            return []
        
        # Determine chunk size if not provided
        if chunk_size is None:
            # Use the batch sizer to determine optimal chunk size
            batch_sizer = get_batch_sizer("default")
            device_name = self.gpu_manager.get_best_device()
            chunk_size = batch_sizer.get_batch_size(items, device_name)
        
        # Process in chunks
        results = []
        
        for i in range(0, len(items), chunk_size):
            # Prepare for batch processing
            self.before_batch_processing(chunk_size)
            
            # Process the chunk
            chunk = items[i:i + chunk_size]
            chunk_results = process_fn(chunk, *args, **kwargs)
            
            # Optimize the results
            chunk_results = self.optimize_embeddings(chunk_results)
            
            # Add to results
            if isinstance(chunk_results, np.ndarray):
                if isinstance(results, list):
                    if results:
                        # Convert previous results to numpy if not already
                        results = np.array(results, dtype=chunk_results.dtype)
                        results = np.vstack([results, chunk_results])
                    else:
                        results = chunk_results
                else:
                    # Append to existing numpy array
                    results = np.vstack([results, chunk_results])
            else:
                if isinstance(results, list):
                    results.extend(chunk_results)
                else:
                    # Convert numpy array to list and extend
                    # Convert numpy array to list if needed
                    if isinstance(results, np.ndarray):
                        results = results.tolist()
                    # Otherwise, results is already a list
                    results.extend(chunk_results)
            
            # Clean up after batch processing
            self.after_batch_processing()
        
        return results


# Global memory optimizer instance
_memory_optimizer = None


def get_memory_optimizer() -> MemoryOptimizer:
    """
    Get the global memory optimizer instance.
    
    Returns:
        The global memory optimizer instance.
    """
    global _memory_optimizer
    
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    
    return _memory_optimizer


def optimize_memory_usage(embeddings=None, **kwargs):
    """
    Optimize memory usage for embedding operations.
    
    This function is a convenience wrapper around the MemoryOptimizer class
    that provides memory optimization capabilities for embedding operations in CPU mode.
    
    Args:
        embeddings: Optional list of embeddings to optimize.
        **kwargs: Additional keyword arguments for memory optimization.
        
    Returns:
        Optimized embeddings if provided, otherwise None.
    """
    optimizer = get_memory_optimizer()
    
    # Run basic memory optimization
    optimizer.auto_gc()
    
    # If embeddings are provided, optimize them
    if embeddings is not None:
        return optimizer.optimize_embeddings(embeddings)
        
    return None