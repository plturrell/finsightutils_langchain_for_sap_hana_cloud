"""Dynamic batch size adjustment based on GPU memory."""

import logging
import math
import time
from typing import Dict, List, Optional, Union, Any, Callable, Tuple

import numpy as np

import gpu_utils
from multi_gpu import get_gpu_manager

logger = logging.getLogger(__name__)


class DynamicBatchSizer:
    """
    Dynamic batch size adjuster based on GPU memory.
    
    This class automatically determines and adjusts the optimal batch size
    based on available GPU memory and workload characteristics.
    """
    
    def __init__(
        self,
        initial_batch_size: int = 32,
        min_batch_size: int = 1,
        max_batch_size: int = 1024,
        target_memory_utilization: float = 0.75,
        adjustment_factor: float = 0.2,
        warmup_batches: int = 3,
    ):
        """
        Initialize the dynamic batch sizer.
        
        Args:
            initial_batch_size: Initial batch size to use.
            min_batch_size: Minimum allowed batch size.
            max_batch_size: Maximum allowed batch size.
            target_memory_utilization: Target GPU memory utilization (0.0-1.0).
            adjustment_factor: How quickly to adjust batch size (0.0-1.0).
            warmup_batches: Number of batches to process before adjusting.
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_utilization = target_memory_utilization
        self.adjustment_factor = adjustment_factor
        self.warmup_batches = warmup_batches
        
        self.gpu_manager = get_gpu_manager()
        self.batch_history = []
        self.warmup_complete = False
        self.item_memory_estimate = {}  # Memory estimate per item type
        
        # Performance metrics
        self.metrics = {
            "batch_sizes": [],
            "memory_utilizations": [],
            "processing_times": [],
            "throughputs": [],  # items per second
        }
    
    def _estimate_item_size(
        self, item: Any, item_type: Optional[str] = None
    ) -> int:
        """
        Estimate the memory size of an item in bytes.
        
        Args:
            item: The item to estimate size for.
            item_type: Optional string identifying the item type.
            
        Returns:
            Estimated size in bytes.
        """
        # Determine item type if not provided
        if item_type is None:
            item_type = type(item).__name__
        
        # Check if we have a cached estimate
        if item_type in self.item_memory_estimate:
            return self.item_memory_estimate[item_type]
        
        # Estimate based on item type
        if isinstance(item, str):
            # Rough estimate: 4 bytes per character (overestimate to be safe)
            size = len(item) * 4
            # Add overhead for embedding vectors (assume 768-dim float32 embedding)
            size += 768 * 4
        elif isinstance(item, list):
            if item and isinstance(item[0], (int, float)):
                # Vector
                size = len(item) * 4  # 4 bytes per number
            else:
                # General list - conservative estimate
                size = 1024
        elif isinstance(item, np.ndarray):
            # NumPy array
            size = item.nbytes
        elif hasattr(item, "__len__"):
            # Has length attribute
            size = len(item) * 8  # Conservative estimate
        else:
            # Default size - be conservative
            size = 1024
        
        # Cache the estimate
        self.item_memory_estimate[item_type] = size
        return size
    
    def _estimate_batch_memory(
        self, items: List[Any], item_type: Optional[str] = None
    ) -> int:
        """
        Estimate the memory required for a batch of items.
        
        Args:
            items: List of items to estimate memory for.
            item_type: Optional string identifying the item type.
            
        Returns:
            Estimated memory size in bytes.
        """
        if not items:
            return 0
        
        # Sample a few items to estimate size
        sample_size = min(len(items), 10)
        sample_items = items[:sample_size]
        
        # Estimate size of each sample item
        sample_sizes = [
            self._estimate_item_size(item, item_type) for item in sample_items
        ]
        
        # Calculate average size and multiply by batch size
        avg_size = sum(sample_sizes) / len(sample_sizes)
        return int(avg_size * len(items))
    
    def _get_device_memory_info(self, device_name: str) -> Tuple[int, int]:
        """
        Get memory information for a device.
        
        Args:
            device_name: Name of the device.
            
        Returns:
            Tuple of (free_memory, total_memory) in bytes.
        """
        if device_name == "cpu":
            # For CPU, return a large value
            return (8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024)
        
        try:
            import torch
            
            # Get device information
            device_info = self.gpu_manager.get_device_info(device_name)
            device_id = device_info.get("id", 0)
            
            # Get memory information
            total_memory = device_info.get("total_memory", 0)
            allocated_memory = torch.cuda.memory_allocated(device_id)
            reserved_memory = torch.cuda.memory_reserved(device_id)
            
            # Calculate free memory
            free_memory = total_memory - allocated_memory
            
            return (free_memory, total_memory)
        
        except Exception as e:
            logger.warning(f"Error getting memory info for {device_name}: {str(e)}")
            # Default to 8GB free, 16GB total as a fallback
            return (8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024)
    
    def calculate_optimal_batch_size(
        self,
        items: List[Any],
        device_name: str,
        item_type: Optional[str] = None,
        overhead_factor: float = 2.0,
    ) -> int:
        """
        Calculate the optimal batch size based on GPU memory.
        
        Args:
            items: List of items to process.
            device_name: Device to use for processing.
            item_type: Optional string identifying the item type.
            overhead_factor: Factor to account for memory overhead (e.g., 2.0 = 2x).
            
        Returns:
            Optimal batch size.
        """
        if not items:
            return self.current_batch_size
        
        # Get memory information
        free_memory, total_memory = self._get_device_memory_info(device_name)
        
        # Calculate target memory
        target_memory = total_memory * self.target_memory_utilization
        available_memory = min(free_memory, target_memory)
        
        # Estimate item memory requirements
        item_size = self._estimate_item_size(items[0], item_type)
        
        # Apply overhead factor to account for intermediate results and processing
        effective_item_size = item_size * overhead_factor
        
        # Calculate batch size
        if effective_item_size > 0:
            batch_size = int(available_memory / effective_item_size)
            
            # Ensure batch size is within limits
            batch_size = max(self.min_batch_size, min(self.max_batch_size, batch_size))
            
            # Make batch size a power of 2 for better GPU utilization
            batch_size = 2 ** int(math.log2(batch_size))
            
            logger.debug(
                f"Calculated batch size: {batch_size} "
                f"(free: {free_memory / 1024**2:.1f}MB, "
                f"item: {effective_item_size / 1024:.1f}KB, "
                f"util: {available_memory / total_memory:.1%})"
            )
            
            return batch_size
        else:
            return self.current_batch_size
    
    def update_batch_size(
        self,
        items_processed: int,
        processing_time: float,
        peak_memory_used: int,
        total_memory: int,
        success: bool,
    ) -> int:
        """
        Update the batch size based on processing results.
        
        Args:
            items_processed: Number of items processed.
            processing_time: Time taken to process the batch (seconds).
            peak_memory_used: Peak memory usage during processing (bytes).
            total_memory: Total available memory (bytes).
            success: Whether processing was successful.
            
        Returns:
            Updated batch size.
        """
        # If processing failed, reduce batch size significantly
        if not success:
            new_batch_size = max(
                self.min_batch_size, int(self.current_batch_size * 0.5)
            )
            logger.warning(
                f"Processing failed, reducing batch size from "
                f"{self.current_batch_size} to {new_batch_size}"
            )
            self.current_batch_size = new_batch_size
            return new_batch_size
        
        # Calculate memory utilization
        memory_utilization = peak_memory_used / total_memory if total_memory > 0 else 0
        
        # Calculate throughput (items per second)
        throughput = items_processed / processing_time if processing_time > 0 else 0
        
        # Record metrics
        self.metrics["batch_sizes"].append(self.current_batch_size)
        self.metrics["memory_utilizations"].append(memory_utilization)
        self.metrics["processing_times"].append(processing_time)
        self.metrics["throughputs"].append(throughput)
        
        # Add to batch history
        self.batch_history.append({
            "batch_size": self.current_batch_size,
            "memory_utilization": memory_utilization,
            "processing_time": processing_time,
            "throughput": throughput,
        })
        
        # Limit history length
        if len(self.batch_history) > 10:
            self.batch_history.pop(0)
        
        # Check if we're still in warmup phase
        if not self.warmup_complete and len(self.batch_history) >= self.warmup_batches:
            self.warmup_complete = True
        
        # If we're still in warmup, don't adjust batch size yet
        if not self.warmup_complete:
            return self.current_batch_size
        
        # Determine whether to adjust batch size based on memory utilization
        if memory_utilization < self.target_memory_utilization * 0.8:
            # Memory utilization is too low, increase batch size
            adjustment = 1 + self.adjustment_factor
            new_batch_size = min(
                self.max_batch_size, 
                int(self.current_batch_size * adjustment)
            )
        elif memory_utilization > self.target_memory_utilization * 1.1:
            # Memory utilization is too high, decrease batch size
            adjustment = 1 - self.adjustment_factor
            new_batch_size = max(
                self.min_batch_size, 
                int(self.current_batch_size * adjustment)
            )
        else:
            # Memory utilization is in the target range
            new_batch_size = self.current_batch_size
        
        if new_batch_size != self.current_batch_size:
            logger.debug(
                f"Adjusting batch size from {self.current_batch_size} to {new_batch_size} "
                f"(memory util: {memory_utilization:.1%}, target: {self.target_memory_utilization:.1%})"
            )
            
            self.current_batch_size = new_batch_size
        
        return new_batch_size
    
    def get_batch_size(
        self,
        items: List[Any],
        device_name: str,
        item_type: Optional[str] = None,
    ) -> int:
        """
        Get the current optimal batch size.
        
        Args:
            items: List of items to process.
            device_name: Device to use for processing.
            item_type: Optional string identifying the item type.
            
        Returns:
            Current optimal batch size.
        """
        # If we haven't processed any batches yet, calculate initial batch size
        if not self.warmup_complete and not self.batch_history:
            self.current_batch_size = self.calculate_optimal_batch_size(
                items, device_name, item_type
            )
        
        return self.current_batch_size
    
    def get_metrics(self) -> Dict:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics.
        """
        metrics = {k: v.copy() for k, v in self.metrics.items()}
        
        # Add summary statistics
        if self.metrics["batch_sizes"]:
            metrics["current_batch_size"] = self.current_batch_size
            metrics["avg_batch_size"] = sum(self.metrics["batch_sizes"]) / len(self.metrics["batch_sizes"])
            metrics["avg_memory_utilization"] = sum(self.metrics["memory_utilizations"]) / len(self.metrics["memory_utilizations"])
            metrics["avg_throughput"] = sum(self.metrics["throughputs"]) / len(self.metrics["throughputs"])
        
        return metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = {
            "batch_sizes": [],
            "memory_utilizations": [],
            "processing_times": [],
            "throughputs": [],
        }


# Global registry of batch sizers
_batch_sizers = {}


class DynamicBatcher:
    """
    Dynamic batching implementation for GPU processing.
    
    This class handles batching of items for GPU processing to maximize throughput
    while managing GPU memory efficiently.
    """
    
    def __init__(self, batch_size: int = 32):
        """
        Initialize the dynamic batcher.
        
        Args:
            batch_size: Initial batch size to use.
        """
        self.batch_sizer = DynamicBatchSizer(initial_batch_size=batch_size)
        self.current_batch_size = batch_size
    
    def get_optimal_batch_size(self, items, device_name="cuda:0"):
        """
        Get the optimal batch size for the current items and device.
        
        Args:
            items: List of items to process.
            device_name: Device to use for processing.
            
        Returns:
            Optimal batch size.
        """
        return self.batch_sizer.calculate_optimal_batch_size(items, device_name)
    
    def process_batch(self, items, processing_fn, device_name="cuda:0"):
        """
        Process a batch of items using the provided processing function.
        
        Args:
            items: List of items to process.
            processing_fn: Function to process the items.
            device_name: Device to use for processing.
            
        Returns:
            Results from processing function.
        """
        # Get optimal batch size
        batch_size = self.get_optimal_batch_size(items, device_name)
        self.current_batch_size = batch_size
        
        # Process in batches
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = processing_fn(batch)
            results.extend(batch_results)
        
        # Update batch size based on performance
        self.batch_sizer.update(items, device_name)
        
        return results


def calculate_optimal_batch_size(items, device_name="cuda:0", item_type=None, initial_batch_size=32):
    """
    Calculate the optimal batch size for processing items on the specified device.
    
    This is a standalone helper function that creates a temporary DynamicBatchSizer
    to calculate the optimal batch size.
    
    Args:
        items: List of items to process.
        device_name: Device to use for processing.
        item_type: Optional string identifying the item type.
        initial_batch_size: Initial batch size to start with.
        
    Returns:
        Optimal batch size.
    """
    batch_sizer = DynamicBatchSizer(initial_batch_size=initial_batch_size)
    return batch_sizer.calculate_optimal_batch_size(items, device_name, item_type)


def get_batch_sizer(name: str) -> DynamicBatchSizer:
    """
    Get a named batch sizer instance.
    
    Args:
        name: Name of the batch sizer.
        
    Returns:
        DynamicBatchSizer instance.
    """
    global _batch_sizers
    
    if name not in _batch_sizers:
        _batch_sizers[name] = DynamicBatchSizer()
    
    return _batch_sizers[name]