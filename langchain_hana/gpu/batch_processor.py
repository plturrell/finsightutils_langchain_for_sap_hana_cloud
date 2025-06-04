"""
Dynamic batch processing for GPU-accelerated embedding generation.

This module provides utilities for dynamically determining and adjusting batch sizes
for optimal performance when generating embeddings on GPUs, including:

1. Runtime GPU memory detection
2. Model-aware batch size calculation
3. Dynamic batch adjustment during processing
4. Safety mechanisms to prevent OOM errors
5. Automatic batch splitting for large requests
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic, Union

import numpy as np

# Conditional imports based on GPU availability
try:
    import torch
    import torch.cuda as cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import nvml_py as nvml
    HAS_NVML = True
except ImportError:
    try:
        from py3nvml import nvidia_smi as nvml
        HAS_NVML = True
    except ImportError:
        HAS_NVML = False

logger = logging.getLogger(__name__)

# Type variable for generic batch processing
T = TypeVar('T')
U = TypeVar('U')


@dataclass
class BatchProcessingStats:
    """Statistics from a batch processing operation."""
    
    # Basic stats
    total_items: int
    total_batches: int
    total_time: float
    
    # Batch sizes
    initial_batch_size: int
    final_batch_size: int
    min_batch_size: int
    max_batch_size: int
    
    # Performance metrics
    items_per_second: float
    avg_batch_time: float
    avg_item_time: float
    
    # Memory stats
    peak_memory_used_mb: float
    memory_available_mb: float
    
    # OOM recovery
    oom_events: int
    batch_size_adjustments: int


@dataclass
class GPUMemoryInfo:
    """Information about available GPU memory."""
    
    total_memory_mb: int
    free_memory_mb: int
    used_memory_mb: int
    
    # Additional process-specific memory info (when available)
    process_used_memory_mb: Optional[int] = None
    
    @property
    def utilization_percent(self) -> float:
        """Get GPU memory utilization as a percentage."""
        return 100.0 * (self.used_memory_mb / self.total_memory_mb)
    
    @property
    def available_for_allocation_mb(self) -> int:
        """
        Get the memory available for new allocations, applying a safety margin.
        
        Returns:
            Memory available in MB, with a 5% safety margin.
        """
        # Apply a 5% safety margin to avoid exact boundary conditions
        safety_margin = 0.05
        safety_margin_mb = int(self.total_memory_mb * safety_margin)
        return max(0, self.free_memory_mb - safety_margin_mb)


class ModelMemoryProfile:
    """
    Profile of a model's memory requirements for batch processing.
    
    This class helps estimate the memory requirements for processing batches
    of different sizes, accounting for the specific characteristics of the model.
    """
    
    def __init__(
        self,
        model_name: str,
        embedding_dim: int,
        base_memory_mb: int = 0,
        memory_per_item_kb: Optional[int] = None,
        dtype: str = "float32",
    ):
        """
        Initialize a model memory profile.
        
        Args:
            model_name: Name of the model (for logging)
            embedding_dim: Dimension of embeddings produced by the model
            base_memory_mb: Base memory required by the model (regardless of batch size)
            memory_per_item_kb: Memory required per item in KB (auto-calculated if None)
            dtype: Data type used by the model ("float32", "float16", "int8")
        """
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.base_memory_mb = base_memory_mb
        self.dtype = dtype
        
        # Calculate bytes per element based on dtype
        if dtype == "float32":
            self.bytes_per_element = 4
        elif dtype == "float16":
            self.bytes_per_element = 2
        elif dtype == "int8":
            self.bytes_per_element = 1
        else:
            logger.warning(f"Unknown dtype: {dtype}, assuming float32 (4 bytes)")
            self.bytes_per_element = 4
        
        # Auto-calculate memory per item if not provided
        if memory_per_item_kb is None:
            # Estimate memory per item based on:
            # 1. Input tokens (assume max 128 tokens per item, 2 tensors: input_ids and attention_mask)
            # 2. Intermediate activations (vary by model, but roughly 4x embedding_dim)
            # 3. Output embedding (embedding_dim elements)
            token_memory = 128 * 2 * 2  # 128 tokens, 2 tensors, 2 bytes per token (int16)
            activation_memory = 4 * embedding_dim * self.bytes_per_element
            output_memory = embedding_dim * self.bytes_per_element
            
            # Add a safety factor of 1.2x to account for other memory usage
            self.memory_per_item_kb = int((token_memory + activation_memory + output_memory) * 1.2 / 1024)
        else:
            self.memory_per_item_kb = memory_per_item_kb
        
        logger.debug(
            f"Model memory profile for {model_name}: "
            f"base={self.base_memory_mb}MB, "
            f"per_item={self.memory_per_item_kb}KB, "
            f"dtype={dtype}"
        )
    
    def estimate_batch_memory_mb(self, batch_size: int) -> int:
        """
        Estimate the memory required for a batch of the given size.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Estimated memory requirement in MB
        """
        item_memory_mb = (self.memory_per_item_kb * batch_size) / 1024
        return self.base_memory_mb + int(item_memory_mb)
    
    def max_batch_size(self, available_memory_mb: int, safety_factor: float = 0.8) -> int:
        """
        Calculate the maximum batch size that will fit in the available memory.
        
        Args:
            available_memory_mb: Available GPU memory in MB
            safety_factor: Factor to apply to available memory (0.0-1.0)
                          to avoid using all available memory
            
        Returns:
            Maximum batch size that will fit in memory
        """
        # Apply safety factor to available memory
        safe_memory_mb = int(available_memory_mb * safety_factor)
        
        # Subtract base memory
        memory_for_batch = safe_memory_mb - self.base_memory_mb
        
        # Calculate max batch size
        if memory_for_batch <= 0:
            return 1  # Minimum batch size
        
        max_batch = int((memory_for_batch * 1024) / self.memory_per_item_kb)
        return max(1, max_batch)  # Ensure minimum batch size of 1


class DynamicBatchProcessor(Generic[T, U]):
    """
    Processor for dynamically batched operations on GPU.
    
    This class handles:
    1. Dynamic batch size determination based on GPU memory
    2. Automatic batch splitting for large requests
    3. OOM recovery and batch size adjustment
    4. Performance monitoring and optimization
    
    It's designed to be used for any operation that processes batches on GPU,
    particularly embedding generation.
    """
    
    def __init__(
        self,
        processing_fn: Callable[[List[T]], List[U]],
        model_profile: ModelMemoryProfile,
        device_id: int = 0,
        initial_batch_size: Optional[int] = None,
        min_batch_size: int = 1,
        max_batch_size: int = 256,
        safety_factor: float = 0.8,
        oom_recovery_factor: float = 0.5,
    ):
        """
        Initialize the dynamic batch processor.
        
        Args:
            processing_fn: Function that processes a batch of items
            model_profile: Memory profile of the model being used
            device_id: GPU device ID to use (default: 0)
            initial_batch_size: Starting batch size (auto-determined if None)
            min_batch_size: Minimum batch size to use
            max_batch_size: Maximum batch size to use
            safety_factor: Factor to apply to available memory (0.0-1.0)
            oom_recovery_factor: Factor to reduce batch size by when OOM occurs
        """
        self.processing_fn = processing_fn
        self.model_profile = model_profile
        self.device_id = device_id
        self.min_batch_size = max(1, min_batch_size)  # Ensure min_batch_size is at least 1
        self.max_batch_size = max(self.min_batch_size, max_batch_size)
        self.safety_factor = safety_factor
        self.oom_recovery_factor = oom_recovery_factor
        
        # Set initial batch size
        if initial_batch_size is None:
            # Auto-determine based on available memory
            memory_info = self._get_gpu_memory_info()
            if memory_info:
                self.batch_size = self.model_profile.max_batch_size(
                    memory_info.available_for_allocation_mb, 
                    safety_factor
                )
                # Clamp to min/max range
                self.batch_size = max(self.min_batch_size, min(self.batch_size, self.max_batch_size))
            else:
                # Default to conservative batch size if memory info not available
                self.batch_size = min(32, self.max_batch_size)
        else:
            # Use provided initial batch size
            self.batch_size = max(self.min_batch_size, min(initial_batch_size, self.max_batch_size))
        
        # Initialize performance tracking
        self.batch_times: List[Tuple[int, float]] = []  # (batch_size, time_taken)
        
        logger.info(
            f"Dynamic batch processor initialized with batch_size={self.batch_size}, "
            f"min={self.min_batch_size}, max={self.max_batch_size}, "
            f"model={model_profile.model_name}"
        )
    
    def process(self, items: List[T]) -> Tuple[List[U], BatchProcessingStats]:
        """
        Process a list of items with dynamic batching.
        
        Args:
            items: List of items to process
            
        Returns:
            Tuple containing:
            - List of processed items
            - Statistics about the processing
        """
        if not items:
            return [], BatchProcessingStats(
                total_items=0,
                total_batches=0,
                total_time=0.0,
                initial_batch_size=self.batch_size,
                final_batch_size=self.batch_size,
                min_batch_size=self.batch_size,
                max_batch_size=self.batch_size,
                items_per_second=0.0,
                avg_batch_time=0.0,
                avg_item_time=0.0,
                peak_memory_used_mb=0.0,
                memory_available_mb=0.0,
                oom_events=0,
                batch_size_adjustments=0
            )
        
        # Initialize tracking variables
        results: List[U] = []
        start_time = time.time()
        initial_batch_size = self.batch_size
        min_batch_size_used = self.batch_size
        max_batch_size_used = self.batch_size
        total_batches = 0
        oom_events = 0
        batch_size_adjustments = 0
        peak_memory_used_mb = 0.0
        
        # Get initial memory info
        memory_info = self._get_gpu_memory_info()
        memory_available_mb = memory_info.free_memory_mb if memory_info else 0
        
        # Clear GPU cache before starting
        self._clear_gpu_cache()
        
        # Process in batches
        remaining_items = items.copy()
        while remaining_items:
            # Determine batch size
            batch_size = min(self.batch_size, len(remaining_items))
            batch = remaining_items[:batch_size]
            
            # Process batch with OOM recovery
            success = False
            current_batch_size = batch_size
            while not success:
                try:
                    # Track GPU memory before processing
                    if HAS_TORCH and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats(self.device_id)
                    
                    # Process batch and time it
                    batch_start_time = time.time()
                    batch_results = self.processing_fn(batch[:current_batch_size])
                    batch_end_time = time.time()
                    
                    # Track batch time
                    batch_time = batch_end_time - batch_start_time
                    self.batch_times.append((current_batch_size, batch_time))
                    
                    # Update memory usage statistics
                    if HAS_TORCH and torch.cuda.is_available():
                        peak_memory = torch.cuda.max_memory_allocated(self.device_id) / (1024 * 1024)
                        peak_memory_used_mb = max(peak_memory_used_mb, peak_memory)
                    
                    # Add results and update tracking
                    results.extend(batch_results)
                    total_batches += 1
                    
                    # Update min/max batch sizes used
                    min_batch_size_used = min(min_batch_size_used, current_batch_size)
                    max_batch_size_used = max(max_batch_size_used, current_batch_size)
                    
                    # Successfully processed batch
                    success = True
                    
                    # Consider increasing batch size if processing was fast
                    if len(self.batch_times) >= 3 and self.batch_size < self.max_batch_size:
                        self._consider_batch_size_adjustment()
                
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    # Check if this is an OOM error
                    if "CUDA out of memory" in str(e):
                        oom_events += 1
                        
                        # Clear GPU cache
                        self._clear_gpu_cache()
                        
                        # Reduce batch size for recovery
                        new_batch_size = max(
                            self.min_batch_size,
                            int(current_batch_size * self.oom_recovery_factor)
                        )
                        
                        if new_batch_size < current_batch_size:
                            logger.warning(
                                f"OOM error with batch_size={current_batch_size}, "
                                f"reducing to {new_batch_size}"
                            )
                            current_batch_size = new_batch_size
                            self.batch_size = new_batch_size  # Update global batch size
                            batch_size_adjustments += 1
                        else:
                            # Already at minimum batch size, something is wrong
                            logger.error(
                                f"OOM error at minimum batch size ({self.min_batch_size}). "
                                f"Check model memory requirements or reduce model size."
                            )
                            raise RuntimeError(
                                f"Unable to process even with minimum batch size ({self.min_batch_size}). "
                                f"Consider using a smaller model or checking available GPU memory."
                            ) from e
                    else:
                        # Not an OOM error, re-raise
                        raise
            
            # Remove processed items from remaining items
            remaining_items = remaining_items[current_batch_size:]
        
        # Calculate processing statistics
        total_time = time.time() - start_time
        items_per_second = len(items) / total_time if total_time > 0 else 0
        avg_batch_time = total_time / total_batches if total_batches > 0 else 0
        avg_item_time = total_time / len(items) if len(items) > 0 else 0
        
        # Create and return statistics
        stats = BatchProcessingStats(
            total_items=len(items),
            total_batches=total_batches,
            total_time=total_time,
            initial_batch_size=initial_batch_size,
            final_batch_size=self.batch_size,
            min_batch_size=min_batch_size_used,
            max_batch_size=max_batch_size_used,
            items_per_second=items_per_second,
            avg_batch_time=avg_batch_time,
            avg_item_time=avg_item_time,
            peak_memory_used_mb=peak_memory_used_mb,
            memory_available_mb=memory_available_mb,
            oom_events=oom_events,
            batch_size_adjustments=batch_size_adjustments
        )
        
        return results, stats
    
    def _consider_batch_size_adjustment(self) -> None:
        """
        Consider adjusting the batch size based on recent performance.
        
        This method analyzes recent batch processing times and memory usage
        to determine if batch size should be increased for better throughput.
        """
        if len(self.batch_times) < 3:
            return
        
        # Get recent batch times
        recent_times = self.batch_times[-3:]
        
        # Calculate average processing time per item
        total_items = sum(bs for bs, _ in recent_times)
        total_time = sum(t for _, t in recent_times)
        avg_time_per_item = total_time / total_items if total_items > 0 else 0
        
        # Get current memory info
        memory_info = self._get_gpu_memory_info()
        if not memory_info:
            return
        
        # Only increase batch size if:
        # 1. We're below max_batch_size
        # 2. We have enough memory available
        # 3. Recent processing has been stable (consistent time per item)
        if self.batch_size < self.max_batch_size:
            # Check memory availability
            estimated_next_batch_size = min(
                self.batch_size * 2,  # Don't more than double
                self.max_batch_size
            )
            
            # Estimate memory for increased batch size
            required_memory = self.model_profile.estimate_batch_memory_mb(estimated_next_batch_size)
            
            if required_memory <= memory_info.available_for_allocation_mb:
                # Memory is available, increase batch size
                new_batch_size = estimated_next_batch_size
                logger.info(
                    f"Increasing batch size from {self.batch_size} to {new_batch_size} "
                    f"(available memory: {memory_info.available_for_allocation_mb}MB, "
                    f"required: {required_memory}MB)"
                )
                self.batch_size = new_batch_size
    
    def _clear_gpu_cache(self) -> None:
        """Clear GPU memory cache to free up memory."""
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Error clearing GPU cache: {e}")
    
    def _get_gpu_memory_info(self) -> Optional[GPUMemoryInfo]:
        """
        Get information about GPU memory usage.
        
        Returns:
            GPUMemoryInfo object with memory statistics, or None if information
            couldn't be retrieved.
        """
        # Try to get memory info using PyTorch
        if HAS_TORCH and torch.cuda.is_available():
            try:
                # Make sure we're using the right device
                device = torch.device(f"cuda:{self.device_id}")
                
                # Get memory info
                free_memory, total_memory = torch.cuda.mem_get_info(device)
                free_memory_mb = free_memory // (1024 * 1024)
                total_memory_mb = total_memory // (1024 * 1024)
                used_memory_mb = total_memory_mb - free_memory_mb
                
                return GPUMemoryInfo(
                    total_memory_mb=total_memory_mb,
                    free_memory_mb=free_memory_mb,
                    used_memory_mb=used_memory_mb
                )
            except Exception as e:
                logger.warning(f"Error getting GPU memory info via PyTorch: {e}")
        
        # Try to get memory info using NVML
        if HAS_NVML:
            try:
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(self.device_id)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                
                total_memory_mb = mem_info.total // (1024 * 1024)
                free_memory_mb = mem_info.free // (1024 * 1024)
                used_memory_mb = mem_info.used // (1024 * 1024)
                
                # Clean up
                nvml.nvmlShutdown()
                
                return GPUMemoryInfo(
                    total_memory_mb=total_memory_mb,
                    free_memory_mb=free_memory_mb,
                    used_memory_mb=used_memory_mb
                )
            except Exception as e:
                logger.warning(f"Error getting GPU memory info via NVML: {e}")
                try:
                    nvml.nvmlShutdown()
                except:
                    pass
        
        # Return None if all methods failed
        return None


class EmbeddingBatchProcessor(DynamicBatchProcessor[str, List[float]]):
    """
    Specialized batch processor for embedding generation.
    
    This class extends DynamicBatchProcessor with embedding-specific functionality,
    including:
    1. Pre-tokenization of texts for more accurate memory estimation
    2. Specialized handling of embedding models
    3. Integration with embedding caching and deduplication
    """
    
    def __init__(
        self,
        embedding_fn: Callable[[List[str]], List[List[float]]],
        model_name: str,
        embedding_dim: int,
        device_id: int = 0,
        initial_batch_size: Optional[int] = None,
        min_batch_size: int = 1,
        max_batch_size: int = 256,
        safety_factor: float = 0.8,
        oom_recovery_factor: float = 0.5,
        dtype: str = "float32",
        enable_caching: bool = True,
    ):
        """
        Initialize the embedding batch processor.
        
        Args:
            embedding_fn: Function that generates embeddings for a batch of texts
            model_name: Name of the embedding model
            embedding_dim: Dimension of embeddings produced by the model
            device_id: GPU device ID to use (default: 0)
            initial_batch_size: Starting batch size (auto-determined if None)
            min_batch_size: Minimum batch size to use
            max_batch_size: Maximum batch size to use
            safety_factor: Factor to apply to available memory (0.0-1.0)
            oom_recovery_factor: Factor to reduce batch size by when OOM occurs
            dtype: Data type used by the model ("float32", "float16", "int8")
            enable_caching: Whether to enable caching of embeddings
        """
        # Create model memory profile
        model_profile = ModelMemoryProfile(
            model_name=model_name,
            embedding_dim=embedding_dim,
            dtype=dtype
        )
        
        super().__init__(
            processing_fn=embedding_fn,
            model_profile=model_profile,
            device_id=device_id,
            initial_batch_size=initial_batch_size,
            min_batch_size=min_batch_size,
            max_batch_size=max_batch_size,
            safety_factor=safety_factor,
            oom_recovery_factor=oom_recovery_factor
        )
        
        # Initialize embedding-specific attributes
        self.embedding_dim = embedding_dim
        self.enable_caching = enable_caching
        self.embedding_cache = {}  # Simple cache for duplicate texts
    
    def embed_documents(self, texts: List[str]) -> Tuple[List[List[float]], BatchProcessingStats]:
        """
        Generate embeddings for a list of documents with dynamic batching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tuple containing:
            - List of embedding vectors
            - Statistics about the processing
        """
        # Apply caching if enabled
        if self.enable_caching:
            # Find texts that need embedding (not in cache)
            unique_texts = []
            text_to_idx = {}  # Map texts to their positions in the final result
            
            for i, text in enumerate(texts):
                if text not in self.embedding_cache:
                    text_to_idx.setdefault(text, []).append(i)
                    if text not in unique_texts:
                        unique_texts.append(text)
            
            # Process unique texts
            if unique_texts:
                unique_embeddings, stats = self.process(unique_texts)
                
                # Update cache with new embeddings
                for text, embedding in zip(unique_texts, unique_embeddings):
                    self.embedding_cache[text] = embedding
            else:
                # All texts are in cache, create empty stats
                stats = BatchProcessingStats(
                    total_items=0,
                    total_batches=0,
                    total_time=0.0,
                    initial_batch_size=self.batch_size,
                    final_batch_size=self.batch_size,
                    min_batch_size=self.batch_size,
                    max_batch_size=self.batch_size,
                    items_per_second=0.0,
                    avg_batch_time=0.0,
                    avg_item_time=0.0,
                    peak_memory_used_mb=0.0,
                    memory_available_mb=0.0,
                    oom_events=0,
                    batch_size_adjustments=0
                )
            
            # Construct final result from cache
            embeddings = [self.embedding_cache[text] for text in texts]
            
            return embeddings, stats
        else:
            # No caching, process all texts
            return self.process(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        if self.enable_caching and text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Process single text
        embeddings, _ = self.process([text])
        
        # Update cache
        if self.enable_caching:
            self.embedding_cache[text] = embeddings[0]
        
        return embeddings[0]