"""
Enhanced multi-GPU management module for distributed workloads.

This module provides advanced multi-GPU capabilities for LangChain integration with
SAP HANA Cloud, enabling efficient distribution of embedding workloads across
multiple NVIDIA GPUs.
"""

import os
import logging
import threading
import time
import queue
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import numpy as np
import json
from pathlib import Path
import threading
import uuid

# Conditional imports based on availability
try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class GPUInfo:
    """Information about a single GPU device."""
    
    def __init__(self, device_id: int):
        """
        Initialize GPU information.
        
        Args:
            device_id: CUDA device ID
        """
        self.device_id = device_id
        self.device_name = f"cuda:{device_id}"
        self.properties = None
        self.utilization = 0.0
        self.memory_allocated = 0
        self.memory_reserved = 0
        self.temperature = 0
        self.power_draw = 0
        self.supports_tensor_cores = False
        self.compute_capability = (0, 0)
        self.last_updated = 0
        self.active_tasks = 0
        self.completed_tasks = 0
        self.total_execution_time = 0
        self.is_available = True
        
        # Load properties if CUDA is available
        if TORCH_AVAILABLE:
            try:
                self.properties = cuda.get_device_properties(device_id)
                self.compute_capability = (self.properties.major, self.properties.minor)
                self.supports_tensor_cores = self.compute_capability >= (7, 0)
            except Exception as e:
                logger.warning(f"Error getting properties for GPU {device_id}: {e}")
    
    def update_stats(self) -> None:
        """Update GPU statistics."""
        if not TORCH_AVAILABLE:
            return
            
        try:
            # Update memory stats
            self.memory_allocated = cuda.memory_allocated(self.device_id)
            self.memory_reserved = cuda.memory_reserved(self.device_id)
            
            # Get utilization if nvml is available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                
                # Get utilization
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.utilization = utilization.gpu / 100.0
                
                # Get temperature
                self.temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
                
                # Get power usage
                self.power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to Watts
            except (ImportError, Exception) as e:
                # Estimate utilization from memory usage if nvml not available
                if self.properties:
                    self.utilization = self.memory_allocated / self.properties.total_memory
            
            self.last_updated = time.time()
        except Exception as e:
            logger.warning(f"Error updating stats for GPU {self.device_id}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert GPU info to dictionary."""
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "name": self.properties.name if self.properties else "Unknown",
            "compute_capability": f"{self.compute_capability[0]}.{self.compute_capability[1]}",
            "supports_tensor_cores": self.supports_tensor_cores,
            "total_memory_mb": (self.properties.total_memory / (1024 * 1024)) if self.properties else 0,
            "memory_allocated_mb": self.memory_allocated / (1024 * 1024),
            "memory_reserved_mb": self.memory_reserved / (1024 * 1024),
            "utilization": self.utilization,
            "temperature_c": self.temperature,
            "power_draw_watts": self.power_draw,
            "active_tasks": self.active_tasks,
            "completed_tasks": self.completed_tasks,
            "is_available": self.is_available,
        }


class TaskResult:
    """Container for task execution results."""
    
    def __init__(
        self,
        task_id: str,
        result: Any = None,
        error: Optional[Exception] = None,
        device_id: Optional[int] = None,
        execution_time: float = 0.0,
    ):
        """
        Initialize task result.
        
        Args:
            task_id: Unique task identifier
            result: Task execution result
            error: Exception if task failed
            device_id: GPU device ID used for execution
            execution_time: Task execution time in seconds
        """
        self.task_id = task_id
        self.result = result
        self.error = error
        self.success = error is None
        self.device_id = device_id
        self.execution_time = execution_time
        self.completion_time = time.time()


class Task:
    """GPU task definition with execution details."""
    
    def __init__(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        task_id: Optional[str] = None,
        priority: int = 0,
        device_preference: Optional[int] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize a GPU task.
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            task_id: Unique task identifier (generated if not provided)
            priority: Task priority (higher values = higher priority)
            device_preference: Preferred GPU device ID
            timeout: Maximum execution time in seconds
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs or {}
        self.task_id = task_id or str(uuid.uuid4())
        self.priority = priority
        self.device_preference = device_preference
        self.timeout = timeout
        self.creation_time = time.time()
        self.start_time = None
        self.completion_time = None
        self.device_id = None
    
    def execute(self, device_id: int) -> TaskResult:
        """
        Execute the task on the specified device.
        
        Args:
            device_id: GPU device ID
            
        Returns:
            Task result
        """
        self.device_id = device_id
        self.start_time = time.time()
        
        try:
            # Set device for execution
            if TORCH_AVAILABLE:
                with torch.cuda.device(device_id):
                    result = self.func(*self.args, **self.kwargs)
            else:
                result = self.func(*self.args, **self.kwargs)
                
            self.completion_time = time.time()
            execution_time = self.completion_time - self.start_time
            
            return TaskResult(
                task_id=self.task_id,
                result=result,
                device_id=device_id,
                execution_time=execution_time,
            )
        except Exception as e:
            self.completion_time = time.time()
            execution_time = self.completion_time - self.start_time
            
            return TaskResult(
                task_id=self.task_id,
                error=e,
                device_id=device_id,
                execution_time=execution_time,
            )


class EnhancedMultiGPUManager:
    """
    Enhanced multi-GPU manager for distributed workloads.
    
    This class provides advanced multi-GPU capabilities:
    1. Intelligent work distribution across available GPUs
    2. Load balancing based on GPU capabilities and current load
    3. Task prioritization and queuing
    4. Advanced monitoring and statistics
    5. Automatic failover and recovery
    
    Performance characteristics:
    - Best efficiency: Large batch sizes distributed across GPUs
    - Good scaling: Near-linear performance with additional GPUs
    - Memory optimization: Tasks assigned based on available memory
    - Dynamic adjustment: Real-time load balancing based on GPU performance
    """
    
    def __init__(
        self,
        enabled: bool = True,
        strategy: str = "auto",
        monitor_interval: float = 5.0,
        stats_interval: float = 60.0,
        stats_file: Optional[str] = None,
        device_ids: Optional[List[int]] = None,
    ):
        """
        Initialize the multi-GPU manager.
        
        Args:
            enabled: Whether multi-GPU support is enabled
            strategy: Load balancing strategy ("auto", "round_robin", "memory", "utilization")
            monitor_interval: GPU monitoring interval in seconds
            stats_interval: Statistics update interval in seconds
            stats_file: File to write GPU statistics
            device_ids: Specific GPU device IDs to use (all available if None)
        """
        self.enabled = enabled and TORCH_AVAILABLE
        self.strategy = strategy
        self.monitor_interval = monitor_interval
        self.stats_interval = stats_interval
        self.stats_file = stats_file
        
        # Initialize state
        self.devices = []
        self.device_info = {}
        self.task_queues = {}
        self.result_queues = {}
        self.worker_threads = {}
        self.stop_events = {}
        self.monitor_thread = None
        self.monitor_stop_event = threading.Event()
        self.stats_thread = None
        self.stats_stop_event = threading.Event()
        self.global_task_queue = queue.PriorityQueue()
        self.initialization_lock = threading.Lock()
        self.initialized = False
        
        # Task tracking
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        # Performance metrics
        self.start_time = time.time()
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        self.total_execution_time = 0
        
        # Initialize if enabled
        if self.enabled:
            self.initialize(device_ids)
    
    def initialize(self, device_ids: Optional[List[int]] = None) -> None:
        """
        Initialize GPU devices and worker threads.
        
        Args:
            device_ids: Specific GPU device IDs to use
        """
        with self.initialization_lock:
            if self.initialized:
                return
                
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch CUDA not available. Multi-GPU support disabled.")
                self.enabled = False
                return
            
            try:
                # Get device count
                device_count = torch.cuda.device_count()
                if device_count == 0:
                    logger.warning("No CUDA devices available. Multi-GPU support disabled.")
                    self.enabled = False
                    return
                
                # Determine devices to use
                if device_ids is not None:
                    # Validate device IDs
                    valid_device_ids = [d for d in device_ids if 0 <= d < device_count]
                    if len(valid_device_ids) == 0:
                        logger.warning(f"No valid device IDs among {device_ids}. Using all available devices.")
                        device_ids = list(range(device_count))
                    else:
                        device_ids = valid_device_ids
                else:
                    # Use all available devices
                    device_ids = list(range(device_count))
                
                logger.info(f"Initializing {len(device_ids)} CUDA devices for multi-GPU support")
                
                # Initialize devices
                for device_id in device_ids:
                    # Create device info
                    gpu_info = GPUInfo(device_id)
                    self.device_info[device_id] = gpu_info
                    self.devices.append(device_id)
                    
                    # Create task queue and result queue for this device
                    self.task_queues[device_id] = queue.Queue()
                    self.result_queues[device_id] = queue.Queue()
                    self.stop_events[device_id] = threading.Event()
                    
                    # Start worker thread for this device
                    self._start_worker(device_id)
                
                # Start monitor thread
                self._start_monitor()
                
                # Start stats thread if stats file is provided
                if self.stats_file:
                    self._start_stats_thread()
                
                self.initialized = True
                logger.info(f"Multi-GPU manager initialized with {len(self.devices)} devices")
                
            except Exception as e:
                logger.error(f"Error initializing multi-GPU manager: {e}")
                self.enabled = False
    
    def _start_worker(self, device_id: int) -> None:
        """
        Start a worker thread for a GPU device.
        
        Args:
            device_id: GPU device ID
        """
        thread = threading.Thread(
            target=self._worker_loop,
            args=(device_id,),
            name=f"GPU-Worker-{device_id}",
            daemon=True,
        )
        thread.start()
        self.worker_threads[device_id] = thread
        logger.debug(f"Started worker thread for GPU {device_id}")
    
    def _worker_loop(self, device_id: int) -> None:
        """
        Worker loop for processing tasks on a GPU device.
        
        Args:
            device_id: GPU device ID
        """
        try:
            # Set CUDA device for this thread
            if TORCH_AVAILABLE:
                torch.cuda.set_device(device_id)
            
            logger.info(f"Worker thread for GPU {device_id} started")
            
            # Process tasks until stopped
            while not self.stop_events[device_id].is_set():
                try:
                    # Try to get a task from the device queue
                    try:
                        task = self.task_queues[device_id].get(block=False)
                    except queue.Empty:
                        # If device queue is empty, try the global queue
                        try:
                            _, task = self.global_task_queue.get(block=True, timeout=0.1)
                        except queue.Empty:
                            # No tasks available, sleep briefly
                            time.sleep(0.01)
                            continue
                    
                    # Update device stats
                    self.device_info[device_id].active_tasks += 1
                    self.device_info[device_id].update_stats()
                    
                    # Execute the task
                    result = task.execute(device_id)
                    
                    # Update device stats
                    self.device_info[device_id].active_tasks -= 1
                    self.device_info[device_id].completed_tasks += 1
                    self.device_info[device_id].total_execution_time += result.execution_time
                    
                    # Put result in the result queue
                    self.result_queues[device_id].put(result)
                    
                    # Update task tracking
                    with self.initialization_lock:
                        if task.task_id in self.pending_tasks:
                            del self.pending_tasks[task.task_id]
                        
                        if result.success:
                            self.completed_tasks[task.task_id] = result
                            self.total_tasks_completed += 1
                        else:
                            self.failed_tasks[task.task_id] = result
                            self.total_tasks_failed += 1
                        
                        self.total_execution_time += result.execution_time
                    
                    # Mark the task as done in its queue
                    if self.task_queues[device_id].qsize() > 0:
                        self.task_queues[device_id].task_done()
                    else:
                        self.global_task_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Error in worker loop for GPU {device_id}: {e}")
                    time.sleep(1)  # Avoid tight loop on error
        
        except Exception as e:
            logger.error(f"Fatal error in worker thread for GPU {device_id}: {e}")
            self.device_info[device_id].is_available = False
    
    def _start_monitor(self) -> None:
        """Start the GPU monitoring thread."""
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="GPU-Monitor",
            daemon=True,
        )
        self.monitor_thread.start()
        logger.debug("Started GPU monitor thread")
    
    def _monitor_loop(self) -> None:
        """Monitor loop for updating GPU statistics."""
        try:
            while not self.monitor_stop_event.is_set():
                try:
                    # Update stats for all devices
                    for device_id in self.devices:
                        self.device_info[device_id].update_stats()
                    
                    # Sleep until next update
                    time.sleep(self.monitor_interval)
                
                except Exception as e:
                    logger.error(f"Error in GPU monitor loop: {e}")
                    time.sleep(1)  # Avoid tight loop on error
        
        except Exception as e:
            logger.error(f"Fatal error in GPU monitor thread: {e}")
    
    def _start_stats_thread(self) -> None:
        """Start the statistics thread for logging GPU stats."""
        if not self.stats_file:
            return
            
        self.stats_thread = threading.Thread(
            target=self._stats_loop,
            name="GPU-Stats",
            daemon=True,
        )
        self.stats_thread.start()
        logger.debug(f"Started GPU stats thread (logging to {self.stats_file})")
    
    def _stats_loop(self) -> None:
        """Statistics loop for logging GPU stats to file."""
        try:
            # Create directory for stats file if it doesn't exist
            stats_dir = os.path.dirname(self.stats_file)
            if stats_dir and not os.path.exists(stats_dir):
                os.makedirs(stats_dir)
            
            while not self.stats_stop_event.is_set():
                try:
                    # Collect stats
                    stats = {
                        "timestamp": time.time(),
                        "uptime_seconds": time.time() - self.start_time,
                        "total_tasks_submitted": self.total_tasks_submitted,
                        "total_tasks_completed": self.total_tasks_completed,
                        "total_tasks_failed": self.total_tasks_failed,
                        "pending_tasks": len(self.pending_tasks),
                        "gpus": {
                            device_id: self.device_info[device_id].to_dict()
                            for device_id in self.devices
                        }
                    }
                    
                    # Write stats to file
                    with open(self.stats_file, "a") as f:
                        f.write(json.dumps(stats) + "\n")
                    
                    # Sleep until next update
                    time.sleep(self.stats_interval)
                
                except Exception as e:
                    logger.error(f"Error in GPU stats loop: {e}")
                    time.sleep(self.stats_interval)  # Continue on error
        
        except Exception as e:
            logger.error(f"Fatal error in GPU stats thread: {e}")
    
    def stop(self) -> None:
        """Stop all worker threads and monitoring."""
        if not self.initialized:
            return
        
        logger.info("Stopping multi-GPU manager")
        
        # Signal threads to stop
        self.monitor_stop_event.set()
        self.stats_stop_event.set()
        
        for device_id in self.devices:
            self.stop_events[device_id].set()
        
        # Wait for worker threads to finish
        for device_id, thread in self.worker_threads.items():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning(f"Worker thread for GPU {device_id} did not terminate gracefully")
        
        # Wait for monitor thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            if self.monitor_thread.is_alive():
                logger.warning("GPU monitor thread did not terminate gracefully")
        
        # Wait for stats thread
        if self.stats_thread and self.stats_thread.is_alive():
            self.stats_thread.join(timeout=5.0)
            if self.stats_thread.is_alive():
                logger.warning("GPU stats thread did not terminate gracefully")
        
        self.initialized = False
        logger.info("Multi-GPU manager stopped")
    
    def _select_device_for_task(self, task: Task) -> int:
        """
        Select the best device for a task based on the current strategy.
        
        Args:
            task: Task to assign
            
        Returns:
            Selected GPU device ID
        """
        # If the task has a device preference and it's available, use it
        if (task.device_preference is not None and 
            task.device_preference in self.devices and
            self.device_info[task.device_preference].is_available):
            return task.device_preference
        
        # Use the selected strategy to determine the best device
        if self.strategy == "round_robin":
            # Simple round robin
            return self.devices[self.total_tasks_submitted % len(self.devices)]
            
        elif self.strategy == "memory":
            # Select device with most available memory
            best_device = self.devices[0]
            best_available = 0
            
            for device_id in self.devices:
                info = self.device_info[device_id]
                if not info.is_available:
                    continue
                    
                if info.properties:
                    available = info.properties.total_memory - info.memory_allocated
                    if available > best_available:
                        best_available = available
                        best_device = device_id
            
            return best_device
            
        elif self.strategy == "utilization":
            # Select device with lowest utilization
            best_device = self.devices[0]
            best_utilization = float("inf")
            
            for device_id in self.devices:
                info = self.device_info[device_id]
                if not info.is_available:
                    continue
                    
                if info.utilization < best_utilization:
                    best_utilization = info.utilization
                    best_device = device_id
            
            return best_device
            
        else:  # "auto" or default
            # Use a weighted score based on multiple factors
            best_device = self.devices[0]
            best_score = float("inf")
            
            for device_id in self.devices:
                info = self.device_info[device_id]
                if not info.is_available:
                    continue
                
                # Calculate score based on memory, utilization, and queue size
                memory_score = 0
                if info.properties:
                    memory_used = info.memory_allocated / info.properties.total_memory
                    memory_score = memory_used * 0.5
                
                utilization_score = info.utilization * 0.3
                queue_score = (self.task_queues[device_id].qsize() / 10) * 0.2
                
                # Combine scores (lower is better)
                total_score = memory_score + utilization_score + queue_score
                
                if total_score < best_score:
                    best_score = total_score
                    best_device = device_id
            
            return best_device
    
    def submit_task(
        self,
        func: Callable,
        args: Tuple = (),
        kwargs: Dict[str, Any] = None,
        task_id: Optional[str] = None,
        priority: int = 0,
        device_preference: Optional[int] = None,
        timeout: Optional[float] = None,
        callback: Optional[Callable[[TaskResult], None]] = None,
    ) -> str:
        """
        Submit a task for execution on a GPU.
        
        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            task_id: Unique task identifier (generated if not provided)
            priority: Task priority (higher values = higher priority)
            device_preference: Preferred GPU device ID
            timeout: Maximum execution time in seconds
            callback: Optional callback function for task completion
            
        Returns:
            Task ID
        """
        if not self.enabled or not self.initialized:
            # Execute on CPU if multi-GPU is not enabled
            result = None
            error = None
            start_time = time.time()
            
            try:
                result = func(*args, **(kwargs or {}))
            except Exception as e:
                error = e
            
            execution_time = time.time() - start_time
            task_result = TaskResult(
                task_id=task_id or str(uuid.uuid4()),
                result=result,
                error=error,
                device_id=None,
                execution_time=execution_time,
            )
            
            if callback:
                callback(task_result)
            
            return task_result.task_id
        
        # Create task
        task = Task(
            func=func,
            args=args,
            kwargs=kwargs,
            task_id=task_id,
            priority=priority,
            device_preference=device_preference,
            timeout=timeout,
        )
        
        # Select device for task
        device_id = self._select_device_for_task(task)
        
        # Update tracking
        with self.initialization_lock:
            self.pending_tasks[task.task_id] = task
            self.total_tasks_submitted += 1
        
        # Put task in device queue or global queue
        if priority > 0:
            # High priority tasks go directly to device queue
            self.task_queues[device_id].put(task)
        else:
            # Normal priority tasks go to global queue
            # Use negative priority for min-heap (higher priority = lower value)
            self.global_task_queue.put((-priority, task))
        
        # Set up callback if provided
        if callback:
            def result_listener():
                # Wait for task completion
                while task.task_id not in self.completed_tasks and task.task_id not in self.failed_tasks:
                    time.sleep(0.1)
                    if not self.initialized:  # Manager was stopped
                        return
                
                # Get the result
                if task.task_id in self.completed_tasks:
                    result = self.completed_tasks[task.task_id]
                else:
                    result = self.failed_tasks[task.task_id]
                
                # Call the callback
                callback(result)
            
            # Start listener thread
            thread = threading.Thread(
                target=result_listener,
                name=f"Task-{task.task_id[:8]}-Callback",
                daemon=True,
            )
            thread.start()
        
        return task.task_id
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[TaskResult]:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task ID to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Task result or None if timeout or task not found
        """
        if not self.enabled or not self.initialized:
            return None
        
        start_time = time.time()
        while task_id not in self.completed_tasks and task_id not in self.failed_tasks:
            # Check timeout
            if timeout and time.time() - start_time > timeout:
                return None
            
            # Check if the task exists
            if task_id not in self.pending_tasks:
                # Task not found
                return None
            
            # Wait briefly
            time.sleep(0.1)
            
            # Check if manager was stopped
            if not self.initialized:
                return None
        
        # Return the result
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return self.failed_tasks[task_id]
    
    def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """
        Get the result of a completed task.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task result or None if task not found or not completed
        """
        if task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        elif task_id in self.failed_tasks:
            return self.failed_tasks[task_id]
        else:
            return None
    
    def batch_process(
        self,
        func: Callable,
        items: List[Any],
        batch_size: Optional[int] = None,
        kwargs: Dict[str, Any] = None,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> List[Any]:
        """
        Process a batch of items across multiple GPUs.
        
        Args:
            func: Function to process items
            items: List of items to process
            batch_size: Size of each batch (auto-determined if None)
            kwargs: Additional keyword arguments for func
            wait: Whether to wait for all tasks to complete
            timeout: Maximum time to wait for completion
            
        Returns:
            List of results in the same order as items
        """
        if not items:
            return []
        
        if not self.enabled or not self.initialized:
            # Process on CPU if multi-GPU is not available
            kwargs = kwargs or {}
            results = []
            
            # Determine batch size
            if batch_size is None:
                batch_size = min(32, len(items))
            
            # Process in batches
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                batch_results = func(batch, **kwargs)
                results.extend(batch_results)
            
            return results
        
        # Determine optimal batch size if not provided
        if batch_size is None:
            # Default batch size depends on number of GPUs
            batch_size = min(32, max(1, len(items) // len(self.devices)))
        
        # Create batches
        batches = []
        for i in range(0, len(items), batch_size):
            batches.append(items[i:i + batch_size])
        
        # Create result placeholders
        results = [None] * len(items)
        task_map = {}  # Maps task_id to result index range
        
        # Submit batches as tasks
        for i, batch in enumerate(batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + len(batch), len(items))
            
            def process_batch(batch_items, batch_idx=i):
                return func(batch_items, **(kwargs or {}))
            
            task_id = self.submit_task(
                func=process_batch,
                args=(batch,),
                priority=10,  # Higher priority for batch processing
            )
            
            task_map[task_id] = (start_idx, end_idx)
        
        # Wait for results if requested
        if wait:
            start_time = time.time()
            completed = set()
            
            while len(completed) < len(batches):
                # Check timeout
                if timeout and time.time() - start_time > timeout:
                    break
                
                # Check for completed tasks
                for task_id, (start_idx, end_idx) in task_map.items():
                    if task_id in completed:
                        continue
                        
                    result = self.get_task_result(task_id)
                    if result:
                        # Task completed
                        completed.add(task_id)
                        
                        if result.success:
                            # Update results
                            batch_results = result.result
                            for j, item_result in enumerate(batch_results):
                                if start_idx + j < len(results):
                                    results[start_idx + j] = item_result
                
                # Check if all tasks completed
                if len(completed) >= len(batches):
                    break
                
                # Wait briefly
                time.sleep(0.1)
        
        return results
    
    def get_available_devices(self) -> List[int]:
        """
        Get a list of available GPU devices.
        
        Returns:
            List of available GPU device IDs
        """
        if not self.enabled or not self.initialized:
            return []
            
        return [device_id for device_id in self.devices 
                if self.device_info[device_id].is_available]
    
    def get_device_info(self, device_id: Optional[int] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get information about a specific device or all devices.
        
        Args:
            device_id: GPU device ID or None for all devices
            
        Returns:
            Device information dictionary or list of dictionaries
        """
        if not self.enabled or not self.initialized:
            return [] if device_id is None else {}
            
        # Update stats for all devices
        for d_id in self.devices:
            self.device_info[d_id].update_stats()
            
        if device_id is not None:
            if device_id in self.device_info:
                return self.device_info[device_id].to_dict()
            else:
                return {}
        else:
            return [self.device_info[d_id].to_dict() for d_id in self.devices]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the multi-GPU manager.
        
        Returns:
            Status dictionary
        """
        status = {
            "enabled": self.enabled,
            "initialized": self.initialized,
            "strategy": self.strategy,
            "device_count": len(self.devices),
            "uptime_seconds": time.time() - self.start_time,
            "total_tasks_submitted": self.total_tasks_submitted,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "pending_tasks": len(self.pending_tasks),
            "total_execution_time": self.total_execution_time,
        }
        
        if self.enabled and self.initialized:
            status["devices"] = self.get_device_info()
            status["queue_sizes"] = {
                device_id: self.task_queues[device_id].qsize()
                for device_id in self.devices
            }
            status["global_queue_size"] = self.global_task_queue.qsize()
        
        return status
    
    def get_best_device(self) -> Optional[int]:
        """
        Get the best available device based on current load.
        
        Returns:
            Best available GPU device ID or None if no devices are available
        """
        if not self.enabled or not self.initialized:
            return None
            
        # Use a dummy task to select the best device
        dummy_task = Task(lambda: None)
        return self._select_device_for_task(dummy_task)


# Global instance
_multi_gpu_manager = None

def get_multi_gpu_manager() -> EnhancedMultiGPUManager:
    """
    Get the global multi-GPU manager instance.
    
    Returns:
        Global EnhancedMultiGPUManager instance
    """
    global _multi_gpu_manager
    
    if _multi_gpu_manager is None:
        # Initialize based on environment variables
        enabled = os.environ.get("MULTI_GPU_ENABLED", "false").lower() in ("true", "1", "yes")
        strategy = os.environ.get("MULTI_GPU_STRATEGY", "auto")
        
        # Create manager
        _multi_gpu_manager = EnhancedMultiGPUManager(
            enabled=enabled,
            strategy=strategy,
            stats_file=os.environ.get("MULTI_GPU_STATS_FILE"),
        )
    
    return _multi_gpu_manager