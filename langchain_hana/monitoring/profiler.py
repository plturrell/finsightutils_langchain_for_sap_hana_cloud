"""
Profiling system for SAP HANA Cloud LangChain integration GPU operations.

This module provides detailed profiling and analysis tools to understand performance 
characteristics of GPU-accelerated operations, with a focus on understanding batch size 
impact on embedding generation and vector operations.
"""

import logging
import time
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

# Conditional imports based on GPU availability
try:
    import torch
    import torch.cuda as cuda
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pyinstrument import Profiler as PyInstrumentProfiler
    HAS_PYINSTRUMENT = True
except ImportError:
    HAS_PYINSTRUMENT = False

try:
    import nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False

logger = logging.getLogger(__name__)


@dataclass
class GPUEvent:
    """Records a single GPU operation event with timing and memory information."""
    
    name: str
    start_time: float
    end_time: float
    duration_ms: float
    cuda_time_ms: Optional[float] = None
    memory_allocated_mb: Optional[float] = None
    memory_reserved_mb: Optional[float] = None
    max_memory_allocated_mb: Optional[float] = None
    max_memory_reserved_mb: Optional[float] = None
    cuda_sync_time_ms: Optional[float] = None
    kernel_launch_count: Optional[int] = None
    device_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class BatchProfile:
    """Profile of a single batch processing operation."""
    
    batch_size: int
    start_time: float
    end_time: float
    duration_ms: float
    items_per_second: float
    events: List[GPUEvent] = field(default_factory=list)
    memory_peak_mb: Optional[float] = None
    cuda_time_ms: Optional[float] = None
    host_time_ms: Optional[float] = None
    kernel_launches: Optional[int] = None
    host_to_device_time_ms: Optional[float] = None
    device_to_host_time_ms: Optional[float] = None
    compute_time_ms: Optional[float] = None
    tokenization_time_ms: Optional[float] = None
    forward_pass_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["events"] = [e.to_dict() for e in self.events]
        return result
    
    def efficiency_ratio(self) -> float:
        """Calculate the efficiency ratio (CUDA time / total time)."""
        if self.cuda_time_ms is not None and self.duration_ms > 0:
            return self.cuda_time_ms / self.duration_ms
        return 0.0
    
    def time_per_item_ms(self) -> float:
        """Calculate time per item in milliseconds."""
        if self.batch_size > 0:
            return self.duration_ms / self.batch_size
        return 0.0
    
    def cuda_time_per_item_ms(self) -> Optional[float]:
        """Calculate CUDA time per item in milliseconds."""
        if self.cuda_time_ms is not None and self.batch_size > 0:
            return self.cuda_time_ms / self.batch_size
        return None
    
    def memory_per_item_mb(self) -> Optional[float]:
        """Calculate peak memory per item in MB."""
        if self.memory_peak_mb is not None and self.batch_size > 0:
            return self.memory_peak_mb / self.batch_size
        return None


@dataclass
class BatchSizeProfileResult:
    """Results of profiling different batch sizes."""
    
    model_name: str
    device_name: str
    batch_sizes: List[int]
    profiles: List[BatchProfile]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    system_info: Dict[str, Any] = field(default_factory=dict)
    cuda_version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "device_name": self.device_name,
            "batch_sizes": self.batch_sizes,
            "profiles": [p.to_dict() for p in self.profiles],
            "timestamp": self.timestamp,
            "system_info": self.system_info,
            "cuda_version": self.cuda_version
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    def save(self, file_path: str) -> None:
        """Save profiling results to a file."""
        with open(file_path, "w") as f:
            f.write(self.to_json())
    
    def get_optimal_batch_size(self) -> int:
        """
        Determine the optimal batch size based on throughput.
        
        Returns:
            The batch size with the highest items per second.
        """
        if not self.profiles:
            return 1
        
        return max(self.profiles, key=lambda p: p.items_per_second).batch_size
    
    def get_optimal_batch_size_efficiency(self) -> int:
        """
        Determine the optimal batch size based on GPU efficiency.
        
        Returns:
            The batch size with the highest CUDA time / total time ratio.
        """
        if not self.profiles:
            return 1
        
        return max(self.profiles, key=lambda p: p.efficiency_ratio()).batch_size
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        """
        Generate a detailed analysis report of the batch size profiling.
        
        Returns:
            Dictionary with analysis results
        """
        if not self.profiles:
            return {"error": "No profiling data available"}
        
        # Find best batch size for different metrics
        throughput_best = max(self.profiles, key=lambda p: p.items_per_second)
        efficiency_best = max(self.profiles, key=lambda p: p.efficiency_ratio())
        memory_best = min(
            (p for p in self.profiles if p.memory_per_item_mb() is not None), 
            key=lambda p: p.memory_per_item_mb() or float('inf'),
            default=None
        )
        
        # Calculate relative performances
        max_throughput = throughput_best.items_per_second
        relative_performances = {}
        for profile in self.profiles:
            relative_performances[profile.batch_size] = {
                "relative_throughput": profile.items_per_second / max_throughput if max_throughput > 0 else 0,
                "efficiency_ratio": profile.efficiency_ratio(),
                "time_per_item_ms": profile.time_per_item_ms(),
                "cuda_time_per_item_ms": profile.cuda_time_per_item_ms(),
                "memory_per_item_mb": profile.memory_per_item_mb()
            }
        
        # Analyze memory usage patterns
        memory_scaling = {}
        batch_sizes = sorted(p.batch_size for p in self.profiles)
        for i in range(1, len(batch_sizes)):
            prev_size = batch_sizes[i-1]
            curr_size = batch_sizes[i]
            
            prev_profile = next((p for p in self.profiles if p.batch_size == prev_size), None)
            curr_profile = next((p for p in self.profiles if p.batch_size == curr_size), None)
            
            if prev_profile and curr_profile and prev_profile.memory_peak_mb and curr_profile.memory_peak_mb:
                # Calculate memory growth factor
                size_ratio = curr_size / prev_size
                memory_ratio = curr_profile.memory_peak_mb / prev_profile.memory_peak_mb
                memory_scaling[f"{prev_size}->{curr_size}"] = {
                    "batch_size_ratio": size_ratio,
                    "memory_ratio": memory_ratio,
                    "scaling_efficiency": size_ratio / memory_ratio if memory_ratio > 0 else 0
                }
        
        # Identify potential bottlenecks
        bottlenecks = []
        
        # Check for host-device transfer overhead
        for profile in self.profiles:
            if (profile.host_to_device_time_ms is not None and
                profile.device_to_host_time_ms is not None and
                profile.duration_ms > 0):
                
                transfer_ratio = (profile.host_to_device_time_ms + profile.device_to_host_time_ms) / profile.duration_ms
                if transfer_ratio > 0.25:  # More than 25% spent on transfers
                    bottlenecks.append({
                        "type": "transfer_overhead",
                        "batch_size": profile.batch_size,
                        "transfer_ratio": transfer_ratio,
                        "severity": "high" if transfer_ratio > 0.5 else "medium"
                    })
        
        # Check for tokenization overhead
        for profile in self.profiles:
            if profile.tokenization_time_ms is not None and profile.duration_ms > 0:
                tokenization_ratio = profile.tokenization_time_ms / profile.duration_ms
                if tokenization_ratio > 0.2:  # More than 20% spent on tokenization
                    bottlenecks.append({
                        "type": "tokenization_overhead",
                        "batch_size": profile.batch_size,
                        "tokenization_ratio": tokenization_ratio,
                        "severity": "high" if tokenization_ratio > 0.4 else "medium"
                    })
        
        # Check for inefficient GPU utilization
        for profile in self.profiles:
            if profile.cuda_time_ms is not None and profile.duration_ms > 0:
                efficiency = profile.cuda_time_ms / profile.duration_ms
                if efficiency < 0.7:  # Less than 70% GPU utilization
                    bottlenecks.append({
                        "type": "low_gpu_utilization",
                        "batch_size": profile.batch_size,
                        "efficiency": efficiency,
                        "severity": "high" if efficiency < 0.5 else "medium"
                    })
        
        # Check for CUDA synchronization overhead
        for profile in self.profiles:
            sync_time = sum(
                (e.cuda_sync_time_ms for e in profile.events if e.cuda_sync_time_ms is not None),
                0
            )
            if sync_time > 0 and profile.duration_ms > 0:
                sync_ratio = sync_time / profile.duration_ms
                if sync_ratio > 0.1:  # More than 10% spent on CUDA synchronization
                    bottlenecks.append({
                        "type": "sync_overhead",
                        "batch_size": profile.batch_size,
                        "sync_ratio": sync_ratio,
                        "severity": "high" if sync_ratio > 0.2 else "medium"
                    })
        
        # Generate recommendations
        recommendations = []
        
        # Based on findings, make specific recommendations
        if throughput_best.batch_size < max(self.batch_sizes):
            recommendations.append({
                "type": "batch_size",
                "message": f"Use a batch size of {throughput_best.batch_size} for optimal throughput.",
                "details": "Smaller batch sizes may perform better due to better GPU utilization, " +
                           "reduced kernel launch overhead, or less memory fragmentation."
            })
        
        # Memory padding or allocation recommendation
        memory_pattern = "sub_linear" if all(
            v["scaling_efficiency"] >= 0.9 for v in memory_scaling.values()
        ) else "super_linear"
        
        if memory_pattern == "super_linear":
            recommendations.append({
                "type": "memory_management",
                "message": "Memory usage scales non-linearly with batch size, suggesting padding or alignment issues.",
                "details": "Consider implementing custom padding or investigating model's handling of variable-length inputs."
            })
        
        # Identify if there's a pattern of diminishing returns
        if len(self.batch_sizes) > 2:
            diminishing_returns = True
            batch_sizes = sorted(self.batch_sizes)
            throughputs = [
                next((p.items_per_second for p in self.profiles if p.batch_size == bs), 0)
                for bs in batch_sizes
            ]
            
            for i in range(2, len(batch_sizes)):
                prev_improvement = throughputs[i-1] / throughputs[i-2] if throughputs[i-2] > 0 else 0
                curr_improvement = throughputs[i] / throughputs[i-1] if throughputs[i-1] > 0 else 0
                
                if curr_improvement >= prev_improvement:
                    diminishing_returns = False
                    break
            
            if diminishing_returns:
                recommendations.append({
                    "type": "scaling_pattern",
                    "message": "Performance shows diminishing returns as batch size increases.",
                    "details": "Consider using dynamic batch sizing to adapt to workload and available memory."
                })
        
        # Add bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            if bottleneck["type"] == "transfer_overhead":
                recommendations.append({
                    "type": "transfer_optimization",
                    "message": "Reduce host-device memory transfers.",
                    "details": "Pin host memory, use CUDA streams for overlapping transfers, or keep data on device longer."
                })
            elif bottleneck["type"] == "tokenization_overhead":
                recommendations.append({
                    "type": "tokenization_optimization",
                    "message": "Optimize tokenization process.",
                    "details": "Consider pre-tokenizing inputs, using GPU-accelerated tokenization, or batch tokenization."
                })
            elif bottleneck["type"] == "low_gpu_utilization":
                recommendations.append({
                    "type": "utilization_optimization",
                    "message": "Improve GPU utilization.",
                    "details": "Use CUDA streams for overlapping computation, optimize kernel launch parameters, or implement kernel fusion."
                })
            elif bottleneck["type"] == "sync_overhead":
                recommendations.append({
                    "type": "sync_optimization",
                    "message": "Reduce CUDA synchronization points.",
                    "details": "Use asynchronous operations where possible and minimize explicit synchronization points."
                })
        
        return {
            "optimal_batch_size": {
                "throughput": throughput_best.batch_size,
                "efficiency": efficiency_best.batch_size,
                "memory_efficiency": memory_best.batch_size if memory_best else None
            },
            "relative_performances": relative_performances,
            "memory_scaling": memory_scaling,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations
        }


class GPUProfiler:
    """
    Profiler for GPU operations with detailed memory and timing information.
    
    This profiler can measure:
    1. GPU time vs CPU time
    2. Memory allocation patterns
    3. CUDA kernel launches
    4. Host-device transfer times
    5. Synchronization overhead
    6. Per-operation timings
    """
    
    def __init__(
        self,
        device_id: int = 0,
        nvtx_ranges: bool = False,
        enable_pyinstrument: bool = False,
        memory_stats: bool = True,
        collect_cuda_events: bool = True,
        warmup_iterations: int = 2,
        events_dir: Optional[str] = None,
    ):
        """
        Initialize the GPU profiler.
        
        Args:
            device_id: CUDA device ID to profile
            nvtx_ranges: Whether to add NVTX ranges for Nsight profiling
            enable_pyinstrument: Whether to use pyinstrument for Python profiling
            memory_stats: Whether to collect memory statistics
            collect_cuda_events: Whether to collect CUDA events for timing
            warmup_iterations: Number of warmup iterations before profiling
            events_dir: Directory to save CUDA events for later analysis
        """
        self.device_id = device_id
        self.nvtx_ranges = nvtx_ranges and HAS_NVTX
        self.enable_pyinstrument = enable_pyinstrument and HAS_PYINSTRUMENT
        self.memory_stats = memory_stats and HAS_TORCH
        self.collect_cuda_events = collect_cuda_events and HAS_TORCH
        self.warmup_iterations = warmup_iterations
        self.events_dir = events_dir
        
        self.current_events = []
        self.current_stage = None
        self.pyinstrument_profiler = None
        self.cuda_events = {}
        
        if HAS_TORCH and self.memory_stats:
            try:
                torch.cuda.reset_peak_memory_stats(self.device_id)
                logger.info(f"Memory profiling enabled for device {self.device_id}")
            except Exception as e:
                logger.warning(f"Failed to reset memory stats: {e}")
                self.memory_stats = False
    
    def _record_cuda_event(self) -> Optional[torch.cuda.Event]:
        """Record a CUDA event if collection is enabled."""
        if self.collect_cuda_events and HAS_TORCH and torch.cuda.is_available():
            try:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                return event
            except Exception as e:
                logger.warning(f"Failed to record CUDA event: {e}")
        return None
    
    def start_event(self, name: str) -> None:
        """
        Start timing a new event.
        
        Args:
            name: Name of the event
        """
        if self.nvtx_ranges and HAS_NVTX:
            nvtx.push_range(name)
        
        if self.enable_pyinstrument and HAS_PYINSTRUMENT:
            if self.pyinstrument_profiler is None:
                self.pyinstrument_profiler = PyInstrumentProfiler()
                self.pyinstrument_profiler.start()
        
        # Record starting memory
        start_memory_allocated = None
        start_memory_reserved = None
        if self.memory_stats and HAS_TORCH and torch.cuda.is_available():
            try:
                start_memory_allocated = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
                start_memory_reserved = torch.cuda.memory_reserved(self.device_id) / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Failed to get start memory: {e}")
        
        # Record CUDA event
        start_cuda_event = self._record_cuda_event()
        
        # Store event info
        self.current_events.append({
            "name": name,
            "start_time": time.time(),
            "start_memory_allocated": start_memory_allocated,
            "start_memory_reserved": start_memory_reserved,
            "start_cuda_event": start_cuda_event,
        })
    
    def end_event(self) -> GPUEvent:
        """
        End timing the current event.
        
        Returns:
            GPUEvent with timing and memory information
        """
        if not self.current_events:
            logger.warning("Trying to end an event, but no events are active")
            return GPUEvent(
                name="unknown",
                start_time=time.time(),
                end_time=time.time(),
                duration_ms=0
            )
        
        event_info = self.current_events.pop()
        end_time = time.time()
        name = event_info["name"]
        start_time = event_info["start_time"]
        duration_ms = (end_time - start_time) * 1000
        
        # Record ending memory
        end_memory_allocated = None
        end_memory_reserved = None
        max_memory_allocated = None
        max_memory_reserved = None
        if self.memory_stats and HAS_TORCH and torch.cuda.is_available():
            try:
                end_memory_allocated = torch.cuda.memory_allocated(self.device_id) / (1024 * 1024)
                end_memory_reserved = torch.cuda.memory_reserved(self.device_id) / (1024 * 1024)
                max_memory_allocated = torch.cuda.max_memory_allocated(self.device_id) / (1024 * 1024)
                max_memory_reserved = torch.cuda.max_memory_reserved(self.device_id) / (1024 * 1024)
            except Exception as e:
                logger.warning(f"Failed to get end memory: {e}")
        
        # Record CUDA event and synchronize
        end_cuda_event = self._record_cuda_event()
        cuda_time_ms = None
        cuda_sync_time_ms = None
        
        if self.collect_cuda_events and HAS_TORCH and torch.cuda.is_available():
            if "start_cuda_event" in event_info and event_info["start_cuda_event"] and end_cuda_event:
                sync_start = time.time()
                torch.cuda.synchronize(self.device_id)
                sync_end = time.time()
                cuda_sync_time_ms = (sync_end - sync_start) * 1000
                
                try:
                    cuda_time_ms = event_info["start_cuda_event"].elapsed_time(end_cuda_event)
                except Exception as e:
                    logger.warning(f"Failed to get CUDA event elapsed time: {e}")
        
        if self.nvtx_ranges and HAS_NVTX:
            nvtx.pop_range()
        
        # Create and return GPU event
        gpu_event = GPUEvent(
            name=name,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms,
            cuda_time_ms=cuda_time_ms,
            memory_allocated_mb=end_memory_allocated,
            memory_reserved_mb=end_memory_reserved,
            max_memory_allocated_mb=max_memory_allocated,
            max_memory_reserved_mb=max_memory_reserved,
            cuda_sync_time_ms=cuda_sync_time_ms,
            device_id=self.device_id
        )
        
        return gpu_event
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Tuple[Any, GPUEvent]:
        """
        Profile a function call.
        
        Args:
            func: Function to profile
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Tuple of (function result, GPUEvent with profiling information)
        """
        func_name = func.__name__ if hasattr(func, "__name__") else "anonymous_function"
        self.start_event(func_name)
        result = func(*args, **kwargs)
        event = self.end_event()
        return result, event
    
    def get_pyinstrument_result(self) -> Optional[str]:
        """
        Get the HTML result from pyinstrument if enabled.
        
        Returns:
            HTML string with the profiling result, or None if not enabled
        """
        if self.enable_pyinstrument and HAS_PYINSTRUMENT and self.pyinstrument_profiler:
            self.pyinstrument_profiler.stop()
            html = self.pyinstrument_profiler.output_html()
            self.pyinstrument_profiler = None
            return html
        return None
    
    def reset(self) -> None:
        """Reset the profiler state."""
        self.current_events = []
        self.current_stage = None
        self.cuda_events = {}
        
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats(self.device_id)
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to reset CUDA state: {e}")
    
    def profile_batch_sizes(
        self,
        operation_fn: Callable[[int], Any],
        batch_sizes: List[int],
        data_generator_fn: Callable[[int], Any],
        model_name: str,
        iterations: int = 5,
        save_path: Optional[str] = None,
    ) -> BatchSizeProfileResult:
        """
        Profile an operation with different batch sizes.
        
        Args:
            operation_fn: Function that takes a batch and performs the operation
            batch_sizes: List of batch sizes to profile
            data_generator_fn: Function that generates data for a given batch size
            model_name: Name of the model being profiled
            iterations: Number of iterations to run for each batch size
            save_path: Path to save the profiling results
            
        Returns:
            BatchSizeProfileResult with profiling data
        """
        # Get device name
        device_name = "cpu"
        if HAS_TORCH and torch.cuda.is_available():
            try:
                device_name = torch.cuda.get_device_name(self.device_id)
            except Exception as e:
                logger.warning(f"Failed to get device name: {e}")
        
        # Get CUDA version
        cuda_version = None
        if HAS_TORCH:
            cuda_version = torch.version.cuda
        
        # Get system info
        system_info = {}
        if HAS_TORCH and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(self.device_id)
                system_info = {
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory_mb": props.total_memory // (1024 * 1024),
                    "multi_processor_count": props.multi_processor_count,
                    "max_threads_per_block": props.max_threads_per_block,
                    "max_threads_per_multi_processor": props.max_threads_per_multi_processor
                }
            except Exception as e:
                logger.warning(f"Failed to get device properties: {e}")
        
        profiles = []
        
        for batch_size in batch_sizes:
            logger.info(f"Profiling batch size: {batch_size}")
            
            # Reset the profiler state
            self.reset()
            
            # Perform warmup iterations
            for _ in range(self.warmup_iterations):
                batch_data = data_generator_fn(batch_size)
                operation_fn(batch_data)
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device_id)
            
            # Reset stats after warmup
            self.reset()
            
            # Initialize tracking variables
            batch_events = []
            total_duration = 0
            total_items = 0
            total_cuda_time = 0
            total_host_to_device_time = 0
            total_device_to_host_time = 0
            total_tokenization_time = 0
            total_forward_pass_time = 0
            memory_peaks = []
            
            # Run profiling iterations
            for i in range(iterations):
                # Generate batch data
                batch_data = data_generator_fn(batch_size)
                
                # Start batch event
                self.start_event(f"batch_{batch_size}_{i}")
                
                # Tokenization stage
                if callable(getattr(operation_fn, "tokenize", None)):
                    self.start_event("tokenization")
                    tokens = operation_fn.tokenize(batch_data)
                    tokenization_event = self.end_event()
                    batch_events.append(tokenization_event)
                    total_tokenization_time += tokenization_event.duration_ms
                
                # Host to device transfer stage
                if HAS_TORCH and torch.cuda.is_available():
                    self.start_event("host_to_device")
                    # This is a placeholder - in a real implementation, you'd track the actual transfer
                    host_to_device_event = self.end_event()
                    batch_events.append(host_to_device_event)
                    total_host_to_device_time += host_to_device_event.duration_ms
                
                # Forward pass stage
                self.start_event("forward_pass")
                result = operation_fn(batch_data)
                forward_pass_event = self.end_event()
                batch_events.append(forward_pass_event)
                total_forward_pass_time += forward_pass_event.duration_ms
                
                # Device to host transfer stage
                if HAS_TORCH and torch.cuda.is_available():
                    self.start_event("device_to_host")
                    # This is a placeholder - in a real implementation, you'd track the actual transfer
                    device_to_host_event = self.end_event()
                    batch_events.append(device_to_host_event)
                    total_device_to_host_time += device_to_host_event.duration_ms
                
                # End batch event
                batch_event = self.end_event()
                
                # Update tracking variables
                total_duration += batch_event.duration_ms
                total_items += batch_size
                if batch_event.cuda_time_ms is not None:
                    total_cuda_time += batch_event.cuda_time_ms
                
                # Track memory peak
                if batch_event.max_memory_allocated_mb is not None:
                    memory_peaks.append(batch_event.max_memory_allocated_mb)
                
                # Clean up after the iteration
                if HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Calculate averages
            avg_duration_ms = total_duration / iterations
            items_per_second = (total_items / total_duration) * 1000 if total_duration > 0 else 0
            avg_cuda_time_ms = total_cuda_time / iterations if iterations > 0 else None
            avg_host_to_device_time_ms = total_host_to_device_time / iterations if iterations > 0 else None
            avg_device_to_host_time_ms = total_device_to_host_time / iterations if iterations > 0 else None
            avg_tokenization_time_ms = total_tokenization_time / iterations if iterations > 0 else None
            avg_forward_pass_time_ms = total_forward_pass_time / iterations if iterations > 0 else None
            memory_peak_mb = max(memory_peaks) if memory_peaks else None
            
            # Create batch profile
            profile = BatchProfile(
                batch_size=batch_size,
                start_time=time.time(),
                end_time=time.time() + (avg_duration_ms / 1000),
                duration_ms=avg_duration_ms,
                items_per_second=items_per_second,
                events=batch_events,
                memory_peak_mb=memory_peak_mb,
                cuda_time_ms=avg_cuda_time_ms,
                host_time_ms=avg_duration_ms - (avg_cuda_time_ms or 0),
                host_to_device_time_ms=avg_host_to_device_time_ms,
                device_to_host_time_ms=avg_device_to_host_time_ms,
                tokenization_time_ms=avg_tokenization_time_ms,
                forward_pass_time_ms=avg_forward_pass_time_ms,
            )
            
            profiles.append(profile)
            
            # Log batch size results
            logger.info(
                f"Batch size {batch_size}: {items_per_second:.2f} items/s, "
                f"duration: {avg_duration_ms:.2f} ms, "
                f"memory: {memory_peak_mb:.2f} MB"
            )
        
        # Create result
        result = BatchSizeProfileResult(
            model_name=model_name,
            device_name=device_name,
            batch_sizes=batch_sizes,
            profiles=profiles,
            system_info=system_info,
            cuda_version=cuda_version
        )
        
        # Save result if requested
        if save_path:
            result.save(save_path)
        
        return result


def profile_embedding_model(
    embedding_model: Any,
    batch_sizes: List[int] = None,
    text_lengths: List[int] = None,
    iterations: int = 5,
    save_path: Optional[str] = None,
    device_id: int = 0,
):
    """
    Profile an embedding model with different batch sizes and text lengths.
    
    Args:
        embedding_model: Embedding model to profile
        batch_sizes: List of batch sizes to profile (default: [1, 2, 4, 8, 16, 32, 64, 128])
        text_lengths: List of text lengths to profile (default: [100])
        iterations: Number of iterations to run for each configuration
        save_path: Path to save the profiling results
        device_id: CUDA device ID to profile
        
    Returns:
        Dictionary with profiling results for each text length
    """
    if batch_sizes is None:
        batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    
    if text_lengths is None:
        text_lengths = [100]
    
    results = {}
    
    for length in text_lengths:
        logger.info(f"Profiling with text length: {length}")
        
        # Create data generator for this text length
        def generate_data(batch_size):
            return [
                "x" * length + f" sample text {i} for profiling"
                for i in range(batch_size)
            ]
        
        # Create operation function
        def operation_fn(texts):
            return embedding_model.embed_documents(texts)
        
        # Add tokenize method if available
        if hasattr(embedding_model, "_tokenize"):
            operation_fn.tokenize = embedding_model._tokenize
        
        # Create profiler
        profiler = GPUProfiler(
            device_id=device_id,
            nvtx_ranges=True,
            enable_pyinstrument=False,
            memory_stats=True,
            collect_cuda_events=True,
            warmup_iterations=2,
        )
        
        # Run profiling
        result = profiler.profile_batch_sizes(
            operation_fn=operation_fn,
            batch_sizes=batch_sizes,
            data_generator_fn=generate_data,
            model_name=getattr(embedding_model, "model_name", str(type(embedding_model).__name__)),
            iterations=iterations,
            save_path=f"{save_path}_length_{length}.json" if save_path else None,
        )
        
        # Generate analysis
        analysis = result.generate_analysis_report()
        
        # Save analysis
        if save_path:
            with open(f"{save_path}_length_{length}_analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
        
        results[length] = {
            "profile": result,
            "analysis": analysis
        }
    
    return results


def create_batch_size_comparison_report(
    results: Dict[int, Dict[str, Any]],
    output_path: str,
):
    """
    Create a detailed HTML report from batch size profiling results.
    
    Args:
        results: Dictionary mapping text lengths to profiling results
        output_path: Path to save the HTML report
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        import base64
        from io import BytesIO
    except ImportError:
        logger.error("Matplotlib is required for creating reports. Install with: pip install matplotlib")
        return
    
    # Function to convert a figure to a base64 string
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('ascii')
        return img_str
    
    # Create HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Batch Size Performance Analysis</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            .section { margin-bottom: 30px; }
            .graph { margin: 20px 0; text-align: center; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .recommendation { background-color: #e6f7ff; padding: 15px; border-left: 5px solid #1890ff; margin: 10px 0; }
            .bottleneck { background-color: #fff2e8; padding: 15px; border-left: 5px solid #fa541c; margin: 10px 0; }
            .optimal { font-weight: bold; color: #52c41a; }
        </style>
    </head>
    <body>
        <h1>Batch Size Performance Analysis</h1>
    """
    
    # Add timestamp
    html_content += f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
    
    # Process each text length
    for length, result_data in results.items():
        profile = result_data["profile"]
        analysis = result_data["analysis"]
        
        html_content += f"""
        <div class="section">
            <h2>Text Length: {length} characters</h2>
            <p>Model: {profile.model_name}</p>
            <p>Device: {profile.device_name}</p>
        """
        
        # Add optimal batch size section
        html_content += """
            <h3>Optimal Batch Sizes:</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Optimal Batch Size</th>
                </tr>
        """
        
        for metric, batch_size in analysis["optimal_batch_size"].items():
            if batch_size is not None:
                html_content += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td class="optimal">{batch_size}</td>
                </tr>
                """
        
        html_content += "</table>"
        
        # Create throughput chart
        fig, ax = plt.subplots(figsize=(10, 6))
        batch_sizes = []
        throughputs = []
        
        for p in profile.profiles:
            batch_sizes.append(p.batch_size)
            throughputs.append(p.items_per_second)
        
        ax.plot(batch_sizes, throughputs, 'o-', color='#1890ff', linewidth=2)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Items per Second')
        ax.set_title('Throughput by Batch Size')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add the figure to the HTML
        html_content += f"""
            <div class="graph">
                <h3>Throughput by Batch Size</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Throughput Chart">
            </div>
        """
        plt.close(fig)
        
        # Create efficiency chart
        fig, ax = plt.subplots(figsize=(10, 6))
        efficiencies = []
        
        for p in profile.profiles:
            efficiencies.append(p.efficiency_ratio())
        
        ax.plot(batch_sizes, efficiencies, 'o-', color='#52c41a', linewidth=2)
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('GPU Efficiency Ratio')
        ax.set_title('GPU Efficiency by Batch Size')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add the figure to the HTML
        html_content += f"""
            <div class="graph">
                <h3>GPU Efficiency by Batch Size</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Efficiency Chart">
            </div>
        """
        plt.close(fig)
        
        # Create time breakdown chart
        fig, ax = plt.subplots(figsize=(10, 6))
        tokenization_times = []
        forward_times = []
        transfer_times = []
        other_times = []
        
        for p in profile.profiles:
            tokenization_times.append(p.tokenization_time_ms or 0)
            forward_times.append(p.forward_pass_time_ms or 0)
            transfer_time = (p.host_to_device_time_ms or 0) + (p.device_to_host_time_ms or 0)
            transfer_times.append(transfer_time)
            other_time = p.duration_ms - (
                (p.tokenization_time_ms or 0) + 
                (p.forward_pass_time_ms or 0) + 
                transfer_time
            )
            other_times.append(max(0, other_time))
        
        width = 0.6
        
        ax.bar(batch_sizes, tokenization_times, width, label='Tokenization', color='#faad14')
        ax.bar(batch_sizes, forward_times, width, bottom=tokenization_times, label='Forward Pass', color='#1890ff')
        
        bottoms = [t + f for t, f in zip(tokenization_times, forward_times)]
        ax.bar(batch_sizes, transfer_times, width, bottom=bottoms, label='Transfer', color='#f5222d')
        
        bottoms = [b + t for b, t in zip(bottoms, transfer_times)]
        ax.bar(batch_sizes, other_times, width, bottom=bottoms, label='Other', color='#d9d9d9')
        
        ax.set_xlabel('Batch Size')
        ax.set_ylabel('Time (ms)')
        ax.set_title('Time Breakdown by Batch Size')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add the figure to the HTML
        html_content += f"""
            <div class="graph">
                <h3>Time Breakdown by Batch Size</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Time Breakdown Chart">
            </div>
        """
        plt.close(fig)
        
        # Create memory usage chart
        fig, ax = plt.subplots(figsize=(10, 6))
        memory_peaks = []
        memory_per_item = []
        
        for p in profile.profiles:
            memory_peaks.append(p.memory_peak_mb or 0)
            memory_per_item.append((p.memory_peak_mb or 0) / p.batch_size)
        
        ax1 = ax
        ax1.plot(batch_sizes, memory_peaks, 'o-', color='#722ed1', linewidth=2, label='Total Memory')
        ax1.set_xlabel('Batch Size')
        ax1.set_ylabel('Total Memory (MB)', color='#722ed1')
        ax1.tick_params(axis='y', labelcolor='#722ed1')
        
        ax2 = ax1.twinx()
        ax2.plot(batch_sizes, memory_per_item, 'o-', color='#eb2f96', linewidth=2, label='Memory per Item')
        ax2.set_ylabel('Memory per Item (MB)', color='#eb2f96')
        ax2.tick_params(axis='y', labelcolor='#eb2f96')
        
        # Add both legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_title('Memory Usage by Batch Size')
        
        # Add the figure to the HTML
        html_content += f"""
            <div class="graph">
                <h3>Memory Usage by Batch Size</h3>
                <img src="data:image/png;base64,{fig_to_base64(fig)}" alt="Memory Usage Chart">
            </div>
        """
        plt.close(fig)
        
        # Add performance details table
        html_content += """
            <h3>Performance Details:</h3>
            <table>
                <tr>
                    <th>Batch Size</th>
                    <th>Items/Second</th>
                    <th>Time per Item (ms)</th>
                    <th>GPU Efficiency</th>
                    <th>Memory per Item (MB)</th>
                </tr>
        """
        
        for p in profile.profiles:
            html_content += f"""
                <tr>
                    <td>{p.batch_size}</td>
                    <td>{p.items_per_second:.2f}</td>
                    <td>{p.time_per_item_ms():.2f}</td>
                    <td>{p.efficiency_ratio():.2f}</td>
                    <td>{p.memory_per_item_mb() or 0:.2f}</td>
                </tr>
            """
        
        html_content += "</table>"
        
        # Add bottlenecks section
        if analysis["bottlenecks"]:
            html_content += """
                <h3>Identified Bottlenecks:</h3>
            """
            
            for bottleneck in analysis["bottlenecks"]:
                html_content += f"""
                    <div class="bottleneck">
                        <h4>{bottleneck["type"].replace("_", " ").title()} (Batch Size: {bottleneck["batch_size"]})</h4>
                        <p>Severity: {bottleneck["severity"].title()}</p>
                        <p>Details: {", ".join([f"{k}: {v:.2f}" for k, v in bottleneck.items() if k not in ["type", "batch_size", "severity"] and isinstance(v, (int, float))])}</p>
                    </div>
                """
        
        # Add recommendations section
        if analysis["recommendations"]:
            html_content += """
                <h3>Recommendations:</h3>
            """
            
            for recommendation in analysis["recommendations"]:
                html_content += f"""
                    <div class="recommendation">
                        <h4>{recommendation["type"].replace("_", " ").title()}</h4>
                        <p><strong>{recommendation["message"]}</strong></p>
                        <p>{recommendation["details"]}</p>
                    </div>
                """
        
        html_content += "</div>"  # Close section
    
    # Close the HTML
    html_content += """
    </body>
    </html>
    """
    
    # Write the HTML to a file
    with open(output_path, "w") as f:
        f.write(html_content)
    
    logger.info(f"Batch size comparison report saved to {output_path}")


def main():
    """
    Command-line interface for the profiler.
    
    Usage:
        python -m langchain_hana.monitoring.profiler --model all-MiniLM-L6-v2 --batch-sizes 1 2 4 8 16 32 64 --iterations 3
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile embedding model performance with different batch sizes")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="Model name to profile")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128], 
                        help="Batch sizes to profile")
    parser.add_argument("--text-lengths", type=int, nargs="+", default=[100], 
                        help="Text lengths to profile")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per batch size")
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--output", type=str, default="profile_results", help="Output file prefix")
    parser.add_argument("--report", action="store_true", help="Generate HTML report")
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    try:
        from langchain_hana.gpu.tensorrt_embeddings import TensorRTEmbeddings
        
        # Create embedding model
        logger.info(f"Creating embedding model: {args.model}")
        embedding_model = TensorRTEmbeddings(
            model_name=f"sentence-transformers/{args.model}",
            max_batch_size=max(args.batch_sizes),
            precision="fp16"  # Use FP16 for best performance
        )
        
        # Run profiling
        logger.info(f"Profiling batch sizes: {args.batch_sizes}")
        results = profile_embedding_model(
            embedding_model=embedding_model,
            batch_sizes=args.batch_sizes,
            text_lengths=args.text_lengths,
            iterations=args.iterations,
            save_path=args.output,
            device_id=args.device,
        )
        
        # Generate report if requested
        if args.report:
            report_path = f"{args.output}_report.html"
            logger.info(f"Generating HTML report: {report_path}")
            create_batch_size_comparison_report(results, report_path)
        
        logger.info("Profiling complete")
    
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure required dependencies are installed")
    except Exception as e:
        logger.error(f"Error during profiling: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()