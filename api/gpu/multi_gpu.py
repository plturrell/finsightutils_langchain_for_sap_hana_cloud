"""Multi-GPU management and load balancing."""

import logging
import sys
import os
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time
import queue
import random
from pathlib import Path
import importlib.util

# Add project root to sys.path if not already there
# This ensures absolute imports work in all execution contexts
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logging.info("Adding project root to sys.path: %s", project_root)

import numpy as np

# Import gpu_utils with fallback mechanism
gpu_utils = None
try:
    # Try relative import first (from parent package)
    from . import gpu_utils
    logger = logging.getLogger(__name__)
    logger.info("Imported gpu_utils via relative import")
except ImportError:
    try:
        # Try absolute import
        import api.gpu.gpu_utils as gpu_utils
        logger = logging.getLogger(__name__)
        logger.info("Imported gpu_utils via absolute import")
    except ImportError:
        try:
            # Try direct file import as last resort
            gpu_utils_path = os.path.join(os.path.dirname(__file__), "gpu_utils.py")
            if os.path.exists(gpu_utils_path):
                spec = importlib.util.spec_from_file_location("gpu_utils", gpu_utils_path)
                if spec and spec.loader:
                    gpu_utils = importlib.util.module_from_spec(spec)
                    sys.modules["gpu_utils"] = gpu_utils
                    spec.loader.exec_module(gpu_utils)
                    logger = logging.getLogger(__name__)
                    logger.info("Imported gpu_utils via direct file import")
            
            # Check if we still don't have gpu_utils
            if gpu_utils is None:
                logger = logging.getLogger(__name__)
                logger.warning("Could not import gpu_utils. Using dummy implementation.")
                
                # Define minimal dummy module for CPU-only environments
                class DummyGPUUtils:
                    @staticmethod
                    def is_gpu_available() -> bool:
                        return False
                    
                    @staticmethod
                    def is_torch_available() -> bool:
                        return False
                    
                    @staticmethod
                    def get_available_gpu_memory() -> Dict[int, int]:
                        return {}
                    
                    @staticmethod
                    def detect_gpus() -> List[Dict[str, Any]]:
                        return []
                
                gpu_utils = DummyGPUUtils()
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error importing gpu_utils: {str(e)}")
            raise

logger = logging.getLogger(__name__)


class GPUManager:
    """
    Manages multiple GPUs with load balancing.
    
    This class provides utilities for:
    1. Managing workload distribution across multiple GPUs
    2. Tracking GPU utilization and memory usage
    3. Auto-selecting the best GPU for specific tasks
    4. Monitoring GPU performance
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the GPU manager.
        
        Args:
            enabled: Whether to enable GPU support.
        """
        self.enabled = enabled
        self.devices = []
        self.device_info = {}
        self.worker_threads = {}
        self.task_queues = {}
        self.stop_events = {}
        self.initialized = False
        
        # Initialize GPUs if enabled
        if self.enabled:
            self._initialize_gpus()
    
    def _initialize_gpus(self) -> None:
        """Initialize GPU devices and worker threads."""
        # Check if CUDA is available via PyTorch
        if not gpu_utils.is_torch_available():
            logger.warning("PyTorch CUDA not available. Multi-GPU support disabled.")
            self.enabled = False
            return
        
        try:
            import torch
            
            # Get device count
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.warning("No CUDA devices available. Multi-GPU support disabled.")
                self.enabled = False
                return
            
            logger.info(f"Initializing {device_count} CUDA devices for multi-GPU support")
            
            # Initialize devices
            for device_id in range(device_count):
                device_name = f"cuda:{device_id}"
                self.devices.append(device_name)
                
                # Get device properties
                device_props = torch.cuda.get_device_properties(device_id)
                total_memory = device_props.total_memory
                
                self.device_info[device_name] = {
                    "id": device_id,
                    "name": device_props.name,
                    "total_memory": total_memory,
                    "compute_capability": (device_props.major, device_props.minor),
                    "multi_processor_count": device_props.multi_processor_count,
                    "utilization": 0.0,
                }
                
                # Create task queue and stop event for this device
                self.task_queues[device_name] = queue.Queue()
                self.stop_events[device_name] = threading.Event()
                
                # Start worker thread for this device
                self._start_worker(device_name)
            
            logger.info(f"Initialized {len(self.devices)} GPU devices for multi-GPU support")
            self.initialized = True
        
        except Exception as e:
            logger.error(f"Error initializing GPUs: {str(e)}")
            self.enabled = False
    
    def _start_worker(self, device_name: str) -> None:
        """
        Start a worker thread for a GPU device.
        
        Args:
            device_name: Name of the device to start a worker for.
        """
        thread = threading.Thread(
            target=self._worker_loop,
            args=(device_name,),
            daemon=True,
        )
        thread.start()
        self.worker_threads[device_name] = thread
        logger.debug(f"Started worker thread for {device_name}")
    
    def _worker_loop(self, device_name: str) -> None:
        """
        Worker loop for a GPU device.
        
        Args:
            device_name: Name of the device to run the worker loop for.
        """
        try:
            import torch
            
            device_id = self.device_info[device_name]["id"]
            logger.info(f"Worker thread for {device_name} started")
            
            # Set CUDA device for this thread
            torch.cuda.set_device(device_id)
            
            while not self.stop_events[device_name].is_set():
                try:
                    # Get a task from the queue with a timeout
                    task, args, kwargs, callback = self.task_queues[device_name].get(timeout=0.1)
                    
                    # Update device utilization
                    self.device_info[device_name]["utilization"] = 1.0
                    
                    # Execute the task
                    start_time = time.time()
                    try:
                        result = task(*args, **kwargs)
                        duration = time.time() - start_time
                        success = True
                    except Exception as e:
                        result = e
                        duration = time.time() - start_time
                        success = False
                    
                    # Update device utilization
                    self.device_info[device_name]["utilization"] = 0.0
                    
                    # Get current memory stats
                    memory_stats = torch.cuda.memory_stats(device_id)
                    memory_allocated = torch.cuda.memory_allocated(device_id)
                    
                    # Update device info
                    self.device_info[device_name].update({
                        "last_task_duration": duration,
                        "memory_allocated": memory_allocated,
                        "memory_stats": memory_stats,
                    })
                    
                    # Call the callback with the result
                    if callback is not None:
                        callback(result, success, duration, device_name)
                    
                    # Mark the task as done
                    self.task_queues[device_name].task_done()
                    
                except queue.Empty:
                    # No tasks in the queue
                    pass
                except Exception as e:
                    logger.error(f"Error in worker loop for {device_name}: {str(e)}")
        
        except Exception as e:
            logger.error(f"Fatal error in worker thread for {device_name}: {str(e)}")
    
    def stop(self) -> None:
        """Stop all worker threads."""
        if not self.initialized:
            return
        
        logger.info("Stopping all GPU worker threads")
        for device_name in self.devices:
            self.stop_events[device_name].set()
        
        for device_name, thread in self.worker_threads.items():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Worker thread for {device_name} did not terminate gracefully")
        
        self.initialized = False
        logger.info("All GPU worker threads stopped")
    
    def get_available_devices(self) -> List[str]:
        """
        Get a list of available GPU devices.
        
        Returns:
            List of available device names.
        """
        return self.devices.copy() if self.initialized else []
    
    def get_device_info(self, device_name: Optional[str] = None) -> Dict:
        """
        Get information about a specific device or all devices.
        
        Args:
            device_name: Name of the device to get information for. If None, returns
                         information for all devices.
        
        Returns:
            Dictionary with device information.
        """
        if not self.initialized:
            return {}
        
        if device_name is not None:
            return self.device_info.get(device_name, {})
        
        return self.device_info
    
    def get_best_device(self) -> str:
        """
        Get the best available device based on current utilization and memory.
        
        Returns:
            Name of the best available device, or 'cpu' if no GPUs are available.
        """
        if not self.initialized or not self.devices:
            return "cpu"
        
        # Find the device with the lowest utilization
        best_device = None
        best_score = float("inf")
        
        for device_name in self.devices:
            info = self.device_info[device_name]
            
            # Calculate a score based on utilization and memory usage
            utilization = info.get("utilization", 0.0)
            memory_allocated = info.get("memory_allocated", 0)
            total_memory = info.get("total_memory", 1)
            
            memory_utilization = memory_allocated / total_memory if total_memory > 0 else 0
            
            # Score = 70% weight on utilization, 30% weight on memory
            score = 0.7 * utilization + 0.3 * memory_utilization
            
            if score < best_score:
                best_score = score
                best_device = device_name
        
        return best_device or "cpu"
    
    def submit_task(
        self,
        task: Callable,
        device_name: Optional[str] = None,
        callback: Optional[Callable] = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Submit a task to be executed on a GPU.
        
        Args:
            task: The function to execute.
            device_name: The device to execute the task on. If None, selects the best device.
            callback: Optional callback function to call with the result.
            *args: Arguments to pass to the task.
            **kwargs: Keyword arguments to pass to the task.
        """
        if not self.initialized or not self.devices:
            # Execute on CPU if no GPUs are available
            try:
                result = task(*args, **kwargs)
                if callback is not None:
                    callback(result, True, 0.0, "cpu")
            except Exception as e:
                if callback is not None:
                    callback(e, False, 0.0, "cpu")
            return
        
        # Select the best device if none provided
        if device_name is None or device_name not in self.devices:
            device_name = self.get_best_device()
        
        # Put the task in the queue
        self.task_queues[device_name].put((task, args, kwargs, callback))
    
    def wait_for_all_tasks(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all tasks to complete.
        
        Args:
            timeout: Optional timeout in seconds.
        
        Returns:
            True if all tasks completed, False if timed out.
        """
        if not self.initialized:
            return True
        
        start_time = time.time()
        
        for device_name in self.devices:
            queue_obj = self.task_queues[device_name]
            
            # Calculate remaining timeout
            remaining_timeout = None
            if timeout is not None:
                elapsed = time.time() - start_time
                remaining_timeout = max(0.0, timeout - elapsed)
            
            try:
                queue_obj.join(timeout=remaining_timeout)
                if not queue_obj.empty():
                    logger.warning(f"Queue for {device_name} not empty after join")
                    return False
            except Exception as e:
                logger.error(f"Error waiting for queue {device_name}: {str(e)}")
                return False
        
        return True
    
    def process_batch(
        self,
        items: List[Any],
        process_fn: Callable,
        batch_size: Optional[int] = None,
        *args,
        **kwargs,
    ) -> List:
        """
        Process a batch of items across multiple GPUs with load balancing.
        
        Args:
            items: List of items to process.
            process_fn: Function to apply to each batch of items.
            batch_size: Size of each batch. If None, uses an optimal batch size.
            *args: Additional arguments to pass to process_fn.
            **kwargs: Additional keyword arguments to pass to process_fn.
        
        Returns:
            List of results.
        """
        if not items:
            return []
        
        if not self.initialized or not self.devices:
            # Process on CPU if no GPUs are available
            if batch_size is None:
                batch_size = 32  # Default CPU batch size
            
            results = []
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = process_fn(batch, *args, **kwargs)
                results.extend(batch_results)
            
            return results
        
        # Determine optimal batch size if not provided
        if batch_size is None:
            # Use a different batch size based on item type and GPU memory
            try:
                # Estimate batch size based on first GPU's memory
                first_device = self.devices[0]
                total_memory = self.device_info[first_device].get("total_memory", 0)
                
                # Rough heuristic based on item type and GPU memory
                if isinstance(items[0], str):
                    # Text items - assume about 4KB per item for embedding
                    batch_size = max(1, min(256, int(total_memory / (4 * 1024 * 1024))))
                elif isinstance(items[0], (list, np.ndarray)):
                    # Vector items - calculate based on vector size
                    if isinstance(items[0], list):
                        vector_size = len(items[0]) * 4  # 4 bytes per float
                    else:
                        vector_size = items[0].nbytes
                    
                    batch_size = max(1, min(512, int(total_memory / (vector_size * 10))))
                else:
                    # Default batch size
                    batch_size = 64
                
                logger.debug(f"Automatically determined batch size: {batch_size}")
            
            except Exception as e:
                logger.warning(f"Error determining optimal batch size: {str(e)}")
                batch_size = 64  # Default batch size
        
        # Split items into batches
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        
        # Process batches in parallel across GPUs
        results_queue = queue.Queue()
        batch_count = len(batches)
        
        def process_batch_callback(result, success, duration, device):
            if success:
                results_queue.put((result, None))
            else:
                results_queue.put((None, result))
        
        # Submit batches to GPU devices
        for i, batch in enumerate(batches):
            # Use a lambda to capture the batch
            self.submit_task(
                lambda b=batch: process_fn(b, *args, **kwargs),
                device_name=None,  # Auto-select the best device
                callback=process_batch_callback,
            )
        
        # Wait for all batches to complete
        results = []
        errors = []
        
        for _ in range(batch_count):
            result, error = results_queue.get()
            if error is not None:
                errors.append(error)
            else:
                results.extend(result)
        
        # Check for errors
        if errors:
            error_msg = f"Errors processing batches: {errors[0]}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        return results


# Global instance of the GPU manager
_gpu_manager = None


def get_gpu_manager() -> GPUManager:
    """
    Get the global GPU manager instance.
    
    Returns:
        The global GPU manager instance.
    """
    global _gpu_manager
    
    if _gpu_manager is None:
        _gpu_manager = GPUManager(enabled=gpu_utils.is_gpu_available())
    
    return _gpu_manager


def distribute_workload(
    items: List[Any], 
    process_fn: Callable, 
    batch_size: Optional[int] = None,
    use_gpu: bool = True,
    *args, 
    **kwargs
) -> List[Any]:
    """
    Distribute a workload across multiple GPUs if available.
    
    This is a high-level wrapper around the GPUManager's process_batch method
    that provides a simpler interface for distributing workloads.
    
    Args:
        items: List of items to process
        process_fn: Function to apply to each batch of items
        batch_size: Size of each batch (if None, uses an optimal size based on GPU memory)
        use_gpu: Whether to use GPU acceleration if available
        *args: Additional arguments to pass to process_fn
        **kwargs: Additional keyword arguments to pass to process_fn
        
    Returns:
        List of results from processing all items
    """
    if not use_gpu or not gpu_utils.is_gpu_available():
        # Process on CPU if GPUs not available or not requested
        if batch_size is None:
            # Use a reasonable default batch size for CPU
            batch_size = 32
            
        # Split into batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches sequentially
        results = []
        for batch in batches:
            batch_result = process_fn(batch, *args, **kwargs)
            if isinstance(batch_result, list):
                results.extend(batch_result)
            else:
                results.append(batch_result)
                
        return results
    
    # Process on GPUs
    manager = get_gpu_manager()
    return manager.process_batch(
        items=items,
        process_fn=process_fn,
        batch_size=batch_size,
        *args,
        **kwargs
    )


def setup_multi_gpu(
    enabled: bool = True,
    device_ids: Optional[List[int]] = None,
    memory_fraction: float = 0.9,
    force_reinit: bool = False
) -> bool:
    """
    Initialize the multi-GPU environment and manager.
    
    This function sets up the GPU manager with the specified configuration.
    It is called by modules that need multi-GPU support.
    
    Args:
        enabled: Whether to enable GPU support
        device_ids: List of GPU device IDs to use (None for all available)
        memory_fraction: Fraction of GPU memory to allocate (0.0-1.0)
        force_reinit: Force reinitialization of the GPU manager
    
    Returns:
        True if initialization was successful, False otherwise
    """
    try:
        # Check if GPUs are available at all
        if not enabled or not gpu_utils.is_gpu_available():
            logger.info("GPU support is disabled or not available")
            return False
            
        # Get the GPU manager instance
        manager = get_gpu_manager()
        
        # If already initialized and not forcing reinitialization, return
        if manager.initialized and not force_reinit:
            logger.info("GPU manager already initialized")
            return True
            
        # Set GPU memory fraction through PyTorch if available
        try:
            import torch
            if torch.cuda.is_available():
                # This is a simple approximation - actual memory management is more complex
                torch.cuda.set_per_process_memory_fraction(memory_fraction)
                logger.info(f"Set GPU memory fraction to {memory_fraction}")
        except (ImportError, AttributeError):
            logger.warning("Could not set GPU memory fraction")
            
        # Return initialization status
        return manager.initialized
        
    except Exception as e:
        logger.error(f"Error setting up multi-GPU environment: {str(e)}")
        return False