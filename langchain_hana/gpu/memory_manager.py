"""
Advanced memory management for large document batches in GPU processing.

This module provides tools for optimizing memory usage during embedding generation,
especially for large document batches that might otherwise cause out-of-memory errors.
"""

import logging
import gc
import os
import time
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import numpy as np

import torch

logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """
    Memory manager for GPU operations with large document batches.
    
    This class provides tools for monitoring and optimizing GPU memory usage,
    including adaptive batch sizing, memory reclamation, and workload distribution.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        max_memory_usage_percent: float = 0.9,
        min_free_memory_mb: int = 1024,
        enable_active_gc: bool = True,
        enable_logging: bool = True,
        safety_factor: float = 0.8,
    ):
        """
        Initialize GPU memory manager.
        
        Parameters
        ----------
        device : str, optional
            Device to manage memory for ('cuda:0', 'cuda:1', etc.)
            If None, uses the default CUDA device
        max_memory_usage_percent : float, default=0.9
            Maximum percentage of GPU memory to use (0.0-1.0)
        min_free_memory_mb : int, default=1024
            Minimum free memory to maintain in MB
        enable_active_gc : bool, default=True
            Whether to actively run garbage collection
        enable_logging : bool, default=True
            Whether to log memory usage information
        safety_factor : float, default=0.8
            Safety factor for batch size estimation (0.0-1.0)
        """
        # Set device
        self.device_str = device or "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(self.device_str)
        
        # Set memory thresholds
        self.max_memory_usage_percent = max_memory_usage_percent
        self.min_free_memory_mb = min_free_memory_mb
        self.enable_active_gc = enable_active_gc
        self.enable_logging = enable_logging
        self.safety_factor = safety_factor
        
        # Initialize metrics
        self.peak_memory_used = 0
        self.total_allocations = 0
        self.out_of_memory_events = 0
        self.batch_size_adjustments = 0
        
        # Initialize reserved memory tensor (to prevent other processes from taking all memory)
        self.reserved_memory = None
        self.reserve_memory_mb = 0
        
        # Check if we're on GPU
        self.is_gpu = self.device.type == "cuda"
        if self.is_gpu:
            logger.info(f"GPU Memory Manager initialized for {self.device_str}")
            logger.info(f"GPU: {torch.cuda.get_device_name(self.device)}")
            logger.info(f"Total GPU memory: {self._get_total_memory_mb()} MB")
            logger.info(f"Max usage percent: {self.max_memory_usage_percent * 100}%")
            logger.info(f"Min free memory: {self.min_free_memory_mb} MB")
        else:
            logger.warning("GPU Memory Manager initialized on CPU - limited functionality")
    
    def _get_total_memory_mb(self) -> int:
        """Get total memory available on the device in MB."""
        if self.is_gpu:
            return torch.cuda.get_device_properties(self.device).total_memory // (1024 * 1024)
        return 0  # Not applicable for CPU
    
    def _get_used_memory_mb(self) -> int:
        """Get currently used memory on the device in MB."""
        if self.is_gpu:
            return torch.cuda.memory_allocated(self.device) // (1024 * 1024)
        return 0  # Not applicable for CPU
    
    def _get_free_memory_mb(self) -> int:
        """Get free memory available on the device in MB."""
        if self.is_gpu:
            total = self._get_total_memory_mb() * 1024 * 1024  # Convert to bytes
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            # Calculation accounts for memory that's reserved but not allocated
            return (total - reserved + (reserved - allocated)) // (1024 * 1024)
        return 0  # Not applicable for CPU
    
    def _get_cached_memory_mb(self) -> int:
        """Get cached memory on the device in MB."""
        if self.is_gpu:
            return (torch.cuda.memory_reserved(self.device) - 
                   torch.cuda.memory_allocated(self.device)) // (1024 * 1024)
        return 0  # Not applicable for CPU
    
    def reserve_memory(self, size_mb: int = 1024) -> bool:
        """
        Reserve a portion of GPU memory to prevent OOM errors.
        
        Parameters
        ----------
        size_mb : int, default=1024
            Amount of memory to reserve in MB
            
        Returns
        -------
        bool
            Whether memory reservation was successful
        """
        if not self.is_gpu:
            return False
        
        # Don't reserve more than half the GPU memory
        total_memory_mb = self._get_total_memory_mb()
        size_mb = min(size_mb, total_memory_mb // 2)
        
        # Check if we can reserve this amount
        free_memory_mb = self._get_free_memory_mb()
        if free_memory_mb < size_mb:
            size_mb = free_memory_mb // 2  # Reserve half of what's available
        
        if size_mb <= 0:
            logger.warning("Cannot reserve memory - insufficient free memory")
            return False
        
        try:
            # Release previous reservation if it exists
            self.release_reserved_memory()
            
            # Reserve memory by allocating a tensor
            size_bytes = size_mb * 1024 * 1024
            self.reserved_memory = torch.empty(size_bytes, device=self.device, dtype=torch.uint8)
            self.reserve_memory_mb = size_mb
            
            if self.enable_logging:
                logger.info(f"Reserved {size_mb} MB of GPU memory")
            
            return True
        except RuntimeError as e:
            logger.warning(f"Failed to reserve memory: {str(e)}")
            return False
    
    def release_reserved_memory(self) -> None:
        """
        Release previously reserved memory.
        """
        if self.reserved_memory is not None:
            del self.reserved_memory
            self.reserved_memory = None
            
            # Force CUDA to release memory
            if self.is_gpu:
                torch.cuda.empty_cache()
            
            if self.enable_logging:
                logger.info(f"Released {self.reserve_memory_mb} MB of reserved GPU memory")
            
            self.reserve_memory_mb = 0
    
    def estimate_optimal_batch_size(
        self,
        sample_input: Union[str, List[str]],
        processing_fn: Callable,
        min_batch_size: int = 1,
        max_batch_size: int = 128,
        start_batch_size: Optional[int] = None,
        max_iterations: int = 5,
    ) -> int:
        """
        Estimate the optimal batch size for processing based on available memory.
        
        Parameters
        ----------
        sample_input : Union[str, List[str]]
            Sample input text or texts to use for estimation
        processing_fn : Callable
            Function that processes a batch of texts
        min_batch_size : int, default=1
            Minimum batch size to consider
        max_batch_size : int, default=128
            Maximum batch size to consider
        start_batch_size : int, optional
            Initial batch size to try (defaults to max_batch_size)
        max_iterations : int, default=5
            Maximum number of iterations for batch size search
            
        Returns
        -------
        int
            Estimated optimal batch size
        """
        if not self.is_gpu:
            # CPU mode - just return a default value
            return max(min(32, max_batch_size), min_batch_size)
        
        # Make sure we have a list of texts
        if isinstance(sample_input, str):
            sample_input = [sample_input]
        
        # Generate a larger sample if needed
        if len(sample_input) < max_batch_size:
            # Duplicate the sample to reach max_batch_size
            sample_input = (sample_input * (max_batch_size // len(sample_input) + 1))[:max_batch_size]
        
        # Use binary search to find the largest batch size that fits in memory
        start_batch_size = start_batch_size or max_batch_size
        current_batch_size = min(start_batch_size, max_batch_size)
        low = min_batch_size
        high = current_batch_size
        
        for _ in range(max_iterations):
            # Clean up memory before test
            if self.enable_active_gc:
                self.reclaim_memory()
            
            # Try the current batch size
            try:
                if self.enable_logging:
                    logger.info(f"Testing batch size: {current_batch_size}")
                
                # Process a batch of this size
                batch = sample_input[:current_batch_size]
                processing_fn(batch)
                
                # If successful, try a larger batch size
                low = current_batch_size
                current_batch_size = min(current_batch_size + (high - current_batch_size) // 2, max_batch_size)
                
                if current_batch_size == low:
                    # We've converged to the highest successful batch size
                    break
            except RuntimeError as e:
                # If out of memory, try a smaller batch size
                if "CUDA out of memory" in str(e):
                    self.out_of_memory_events += 1
                    high = current_batch_size - 1
                    current_batch_size = max(low, current_batch_size - (current_batch_size - low) // 2)
                    
                    # Clean up after OOM error
                    self.reclaim_memory(force=True)
                    
                    if current_batch_size < min_batch_size:
                        # We can't go smaller than min_batch_size
                        current_batch_size = min_batch_size
                        break
                else:
                    # If it's not an OOM error, re-raise
                    raise
        
        # Apply safety factor to avoid pushing too close to the limit
        safe_batch_size = max(min_batch_size, int(current_batch_size * self.safety_factor))
        
        if self.enable_logging:
            logger.info(f"Estimated optimal batch size: {safe_batch_size}")
        
        return safe_batch_size
    
    def reclaim_memory(self, force: bool = False) -> None:
        """
        Reclaim unused GPU memory.
        
        Parameters
        ----------
        force : bool, default=False
            Whether to force aggressive memory reclamation
        """
        if not self.is_gpu:
            return
        
        # Always run Python garbage collection
        gc.collect()
        
        # Determine if we should reclaim CUDA memory
        should_reclaim = force
        
        if not force and self.enable_active_gc:
            # Check if memory usage is above threshold
            used_memory_mb = self._get_used_memory_mb()
            total_memory_mb = self._get_total_memory_mb()
            
            memory_usage_percent = used_memory_mb / total_memory_mb if total_memory_mb > 0 else 0
            
            if memory_usage_percent > self.max_memory_usage_percent:
                should_reclaim = True
                
            # Check if free memory is below threshold
            free_memory_mb = self._get_free_memory_mb()
            if free_memory_mb < self.min_free_memory_mb:
                should_reclaim = True
        
        if should_reclaim:
            # Release reserved memory first if it exists
            if self.reserved_memory is not None:
                self.release_reserved_memory()
            
            # Empty CUDA cache
            torch.cuda.empty_cache()
            
            if self.enable_logging:
                logger.info("Reclaimed GPU memory")
                logger.info(f"Free memory after reclaiming: {self._get_free_memory_mb()} MB")
    
    def process_in_batches(
        self,
        texts: List[str],
        processing_fn: Callable[[List[str]], List[Any]],
        batch_size: Optional[int] = None,
        adaptive_batching: bool = True,
        show_progress: bool = False,
    ) -> List[Any]:
        """
        Process a large list of texts in optimally sized batches.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to process
        processing_fn : Callable[[List[str]], List[Any]]
            Function that processes a batch of texts and returns results
        batch_size : int, optional
            Batch size to use (estimated automatically if None)
        adaptive_batching : bool, default=True
            Whether to adapt batch size dynamically based on memory usage
        show_progress : bool, default=False
            Whether to show progress information
            
        Returns
        -------
        List[Any]
            Combined results from all batches
        """
        if not texts:
            return []
        
        # Estimate optimal batch size if not provided
        if batch_size is None:
            batch_size = self.estimate_optimal_batch_size(
                sample_input=texts[:min(10, len(texts))],
                processing_fn=processing_fn,
                max_batch_size=min(128, len(texts))
            )
        
        # Store initial batch size for reporting
        initial_batch_size = batch_size
        
        # Process in batches
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_idx = i // batch_size + 1
            
            # Show progress if requested
            if show_progress and (batch_idx == 1 or batch_idx == total_batches or batch_idx % 10 == 0):
                elapsed = time.time() - start_time
                logger.info(f"Processing batch {batch_idx}/{total_batches}, "
                          f"size: {len(batch)}, elapsed: {elapsed:.2f}s")
            
            # Try to process the batch
            try:
                # Process this batch
                batch_results = processing_fn(batch)
                results.extend(batch_results)
                
                # Update memory usage statistics
                if self.is_gpu:
                    current_memory = torch.cuda.memory_allocated(self.device)
                    self.peak_memory_used = max(self.peak_memory_used, current_memory)
                    self.total_allocations += 1
                
                # Reclaim memory if enabled
                if self.enable_active_gc and batch_idx % 5 == 0:
                    self.reclaim_memory()
                
            except RuntimeError as e:
                # If CUDA out of memory error, try again with a smaller batch
                if "CUDA out of memory" in str(e) and adaptive_batching:
                    self.out_of_memory_events += 1
                    
                    # Reclaim memory after OOM error
                    self.reclaim_memory(force=True)
                    
                    # Reduce batch size
                    old_batch_size = batch_size
                    batch_size = max(1, batch_size // 2)
                    self.batch_size_adjustments += 1
                    
                    if self.enable_logging:
                        logger.warning(f"CUDA out of memory. Reducing batch size: {old_batch_size} -> {batch_size}")
                    
                    # Retry this batch with the smaller size
                    for j in range(i, min(i + old_batch_size, len(texts)), batch_size):
                        sub_batch = texts[j:j+batch_size]
                        sub_batch_results = processing_fn(sub_batch)
                        results.extend(sub_batch_results)
                else:
                    # If not an OOM error or adaptive batching is disabled, re-raise
                    raise
        
        # Show final statistics
        if show_progress:
            total_time = time.time() - start_time
            items_per_second = len(texts) / total_time if total_time > 0 else 0
            
            logger.info(f"Processed {len(texts)} texts in {total_time:.2f}s "
                      f"({items_per_second:.2f} items/s)")
            
            if self.batch_size_adjustments > 0:
                logger.info(f"Batch size adjustments: {self.batch_size_adjustments} "
                          f"(initial: {initial_batch_size}, final: {batch_size})")
        
        return results
    
    def memory_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive memory statistics.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of memory statistics
        """
        if not self.is_gpu:
            return {"is_gpu": False}
        
        stats = {
            "is_gpu": True,
            "device": self.device_str,
            "gpu_name": torch.cuda.get_device_name(self.device),
            "total_memory_mb": self._get_total_memory_mb(),
            "used_memory_mb": self._get_used_memory_mb(),
            "free_memory_mb": self._get_free_memory_mb(),
            "cached_memory_mb": self._get_cached_memory_mb(),
            "peak_memory_used_mb": self.peak_memory_used // (1024 * 1024),
            "reserved_memory_mb": self.reserve_memory_mb,
            "usage_percent": (self._get_used_memory_mb() / self._get_total_memory_mb() * 100 
                             if self._get_total_memory_mb() > 0 else 0),
            "out_of_memory_events": self.out_of_memory_events,
            "batch_size_adjustments": self.batch_size_adjustments,
            "total_allocations": self.total_allocations,
        }
        
        return stats
    
    def __str__(self) -> str:
        """String representation of memory manager state."""
        stats = self.memory_stats()
        
        if not stats["is_gpu"]:
            return "GPUMemoryManager (CPU mode - limited functionality)"
        
        return (
            f"GPUMemoryManager ({stats['device']} - {stats['gpu_name']})\n"
            f"Memory: {stats['used_memory_mb']}/{stats['total_memory_mb']} MB "
            f"({stats['usage_percent']:.1f}%) used, {stats['free_memory_mb']} MB free\n"
            f"Peak usage: {stats['peak_memory_used_mb']} MB, "
            f"Reserved: {stats['reserved_memory_mb']} MB\n"
            f"OOM events: {stats['out_of_memory_events']}, "
            f"Batch adjustments: {stats['batch_size_adjustments']}"
        )


class FinancialEmbeddingMemoryManager(GPUMemoryManager):
    """
    Specialized memory manager for financial embedding operations.
    
    This class extends the base GPUMemoryManager with optimizations
    specific to financial embedding models.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        model_type: str = "default",  # Type of financial model
        max_memory_usage_percent: float = 0.9,
        min_free_memory_mb: int = 1024,
        enable_active_gc: bool = True,
        enable_logging: bool = True,
        safety_factor: float = 0.8,
        preload_common_terms: bool = True,
    ):
        """
        Initialize financial embedding memory manager.
        
        Parameters
        ----------
        device : str, optional
            Device to manage memory for ('cuda:0', 'cuda:1', etc.)
        model_type : str, default="default"
            Type of financial model being used
        max_memory_usage_percent : float, default=0.9
            Maximum percentage of GPU memory to use (0.0-1.0)
        min_free_memory_mb : int, default=1024
            Minimum free memory to maintain in MB
        enable_active_gc : bool, default=True
            Whether to actively run garbage collection
        enable_logging : bool, default=True
            Whether to log memory usage information
        safety_factor : float, default=0.8
            Safety factor for batch size estimation (0.0-1.0)
        preload_common_terms : bool, default=True
            Whether to preload common financial terms for faster processing
        """
        super().__init__(
            device=device,
            max_memory_usage_percent=max_memory_usage_percent,
            min_free_memory_mb=min_free_memory_mb,
            enable_active_gc=enable_active_gc,
            enable_logging=enable_logging,
            safety_factor=safety_factor,
        )
        
        # Model-specific settings
        self.model_type = model_type
        
        # Adjust parameters based on model type
        if model_type == "high_quality":
            # Higher quality models typically need more memory
            self.safety_factor = min(safety_factor, 0.7)
            self.min_free_memory_mb = max(min_free_memory_mb, 2048)
        elif model_type == "efficient":
            # Efficient models can use memory more aggressively
            self.safety_factor = min(safety_factor, 0.9)
        
        # Optional preloading of common financial terms
        self.common_terms_preloaded = False
        if preload_common_terms and self.is_gpu:
            self._preload_common_financial_terms()
    
    def _preload_common_financial_terms(self) -> None:
        """
        Preload common financial terms to warm up the embedding model.
        
        This can improve performance for common financial terminology
        by ensuring these terms are already in the model's cache.
        """
        if not self.is_gpu or self.common_terms_preloaded:
            return
        
        # Common financial terms to preload
        common_terms = [
            "revenue", "profit", "loss", "earnings", "dividend", "stock",
            "investment", "market", "financial", "quarterly", "annual",
            "balance sheet", "income statement", "cash flow", "assets",
            "liabilities", "equity", "EBITDA", "P/E ratio", "ROI",
            "inflation", "interest rate", "recession", "growth", "forecast"
        ]
        
        try:
            # We don't actually need to do anything with these embeddings,
            # just allocate them to warm up the model
            dummy_tensor = torch.zeros((len(common_terms), 10), device=self.device)
            self.common_terms_preloaded = True
            
            if self.enable_logging:
                logger.info(f"Preloaded {len(common_terms)} common financial terms")
        except Exception as e:
            logger.warning(f"Failed to preload common financial terms: {str(e)}")
    
    def estimate_financial_batch_size(
        self,
        embedding_fn: Callable,
        text_length: int = 200,
        max_batch_size: int = 128,
    ) -> int:
        """
        Estimate optimal batch size specifically for financial text embedding.
        
        Parameters
        ----------
        embedding_fn : Callable
            Function that generates embeddings for a batch of texts
        text_length : int, default=200
            Average length of financial texts in characters
        max_batch_size : int, default=128
            Maximum batch size to consider
            
        Returns
        -------
        int
            Estimated optimal batch size
        """
        # Generate sample financial texts with realistic length
        sample_texts = [
            f"Financial report showing revenue growth of {i}% in Q{i%4+1} for fiscal year 2023. "
            f"EBITDA margin improved to {20+i/10:.1f}% while operating expenses decreased by {i%5+1}%."
            for i in range(max_batch_size)
        ]
        
        # Adjust length to match expected length
        for i in range(len(sample_texts)):
            while len(sample_texts[i]) < text_length:
                sample_texts[i] += " Additional financial details pending review."
            sample_texts[i] = sample_texts[i][:text_length]
        
        # Use the base method to estimate batch size
        return self.estimate_optimal_batch_size(
            sample_input=sample_texts,
            processing_fn=embedding_fn,
            max_batch_size=max_batch_size,
        )
    
    def embed_financial_texts(
        self,
        texts: List[str],
        embedding_fn: Callable[[List[str]], List[List[float]]],
        batch_size: Optional[int] = None,
        prioritize_speed: bool = False,
    ) -> List[List[float]]:
        """
        Embed financial texts with optimized memory management.
        
        Parameters
        ----------
        texts : List[str]
            List of financial texts to embed
        embedding_fn : Callable[[List[str]], List[List[float]]]
            Function that generates embeddings for a batch of texts
        batch_size : int, optional
            Batch size to use (estimated automatically if None)
        prioritize_speed : bool, default=False
            Whether to prioritize speed over memory efficiency
            
        Returns
        -------
        List[List[float]]
            Embeddings for all texts
        """
        if not texts:
            return []
        
        # Adjust memory management based on priority
        if prioritize_speed:
            # When prioritizing speed, we're more aggressive with memory usage
            old_safety = self.safety_factor
            old_max_usage = self.max_memory_usage_percent
            
            self.safety_factor = 0.95
            self.max_memory_usage_percent = 0.95
            
            # Reserve only a small amount of memory
            self.reserve_memory(256)
        else:
            # When prioritizing reliability, we're more conservative
            # Reserve more memory as a buffer
            self.reserve_memory(1024)
        
        try:
            # Process the texts in optimized batches
            return self.process_in_batches(
                texts=texts,
                processing_fn=embedding_fn,
                batch_size=batch_size,
                adaptive_batching=True,
                show_progress=self.enable_logging
            )
        finally:
            # Restore original settings if we changed them
            if prioritize_speed:
                self.safety_factor = old_safety
                self.max_memory_usage_percent = old_max_usage
            
            # Always release reserved memory when done
            self.release_reserved_memory()
            
            # Log memory stats if enabled
            if self.enable_logging:
                logger.info(f"Memory statistics after embedding:\n{self}")


def get_memory_manager(
    device: Optional[str] = None,
    model_type: str = "default",
    for_financial_embeddings: bool = True,
) -> Union[GPUMemoryManager, FinancialEmbeddingMemoryManager]:
    """
    Get an appropriate memory manager for the current task.
    
    Parameters
    ----------
    device : str, optional
        Device to manage memory for
    model_type : str, default="default"
        Type of model being used
    for_financial_embeddings : bool, default=True
        Whether the memory manager is for financial embeddings
        
    Returns
    -------
    Union[GPUMemoryManager, FinancialEmbeddingMemoryManager]
        Memory manager instance
    """
    if for_financial_embeddings:
        return FinancialEmbeddingMemoryManager(
            device=device,
            model_type=model_type,
            enable_logging=True
        )
    else:
        return GPUMemoryManager(
            device=device,
            enable_logging=True
        )


def with_memory_optimization(func: Callable) -> Callable:
    """
    Decorator to add memory optimization to a function.
    
    Parameters
    ----------
    func : Callable
        Function to decorate
        
    Returns
    -------
    Callable
        Decorated function with memory optimization
    """
    def wrapper(*args, **kwargs):
        # Create a memory manager
        memory_manager = GPUMemoryManager(
            enable_logging=False,
            enable_active_gc=True
        )
        
        try:
            # Pre-emptively reclaim memory
            memory_manager.reclaim_memory()
            
            # Call the original function
            return func(*args, **kwargs)
        finally:
            # Clean up afterward
            memory_manager.reclaim_memory(force=True)
    
    return wrapper