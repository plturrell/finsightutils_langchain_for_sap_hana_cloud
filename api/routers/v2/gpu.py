"""
GPU-specific routes for version 2 of the API.

This module provides routes for GPU information, optimization, and monitoring.
"""

import logging
import time
from typing import Dict, List, Any, Optional

from fastapi import Depends, Request, Query
from pydantic import BaseModel, Field

from ...config_standardized import get_standardized_settings
from ...utils.standardized_exceptions import GPUNotAvailableException
from ..base import BaseRouter
from ..dependencies import get_gpu_info, get_admin_user

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)


class GPUDevice(BaseModel):
    """Information about a GPU device."""
    
    id: str = Field(..., description="Device ID")
    name: str = Field(..., description="Device name")
    memory_total: int = Field(..., description="Total memory in bytes")
    memory_used: int = Field(..., description="Used memory in bytes")
    memory_free: int = Field(..., description="Free memory in bytes")
    utilization: Optional[float] = Field(None, description="GPU utilization percentage")
    temperature: Optional[float] = Field(None, description="GPU temperature in Celsius")
    power: Optional[float] = Field(None, description="Power usage in Watts")
    compute_capability: Optional[str] = Field(None, description="Compute capability version")


class GPUInfoResponse(BaseModel):
    """Response model for GPU information."""
    
    available: bool = Field(..., description="Whether GPU is available")
    count: int = Field(..., description="Number of GPUs")
    cuda_version: Optional[str] = Field(None, description="CUDA version")
    devices: Dict[str, GPUDevice] = Field(default_factory=dict, description="GPU devices")
    tensorrt_available: bool = Field(False, description="Whether TensorRT is available")
    tensorrt_version: Optional[str] = Field(None, description="TensorRT version")
    multi_gpu_enabled: bool = Field(False, description="Whether multi-GPU is enabled")
    multi_gpu_strategy: Optional[str] = Field(None, description="Multi-GPU strategy")


class OptimizationParameters(BaseModel):
    """Parameters for GPU optimization."""
    
    clear_cache: bool = Field(True, description="Whether to clear CUDA cache")
    optimize_memory: bool = Field(True, description="Whether to optimize memory allocation")
    optimize_batch_size: bool = Field(True, description="Whether to optimize batch sizes")
    force: bool = Field(False, description="Whether to force optimization even if not needed")


class OptimizationResponse(BaseModel):
    """Response model for GPU optimization."""
    
    success: bool = Field(..., description="Whether optimization was successful")
    actions_taken: List[str] = Field(default_factory=list, description="List of actions taken")
    memory_before: Dict[str, float] = Field(default_factory=dict, description="Memory usage before optimization")
    memory_after: Dict[str, float] = Field(default_factory=dict, description="Memory usage after optimization")
    recommended_batch_sizes: Dict[str, int] = Field(default_factory=dict, description="Recommended batch sizes for each device")


class BatchSizeParameters(BaseModel):
    """Parameters for batch size optimization."""
    
    model_name: str = Field(..., description="Name of the model")
    input_shape: List[int] = Field(..., description="Input shape (excluding batch dimension)")
    max_sequence_length: Optional[int] = Field(None, description="Maximum sequence length")
    precision: str = Field("fp32", description="Precision (fp32, fp16, int8)")


class BatchSizeResponse(BaseModel):
    """Response model for batch size optimization."""
    
    optimal_batch_size: int = Field(..., description="Optimal batch size")
    device: str = Field(..., description="Device used")
    model_name: str = Field(..., description="Model name")
    precision: str = Field(..., description="Precision used")
    estimated_memory: float = Field(..., description="Estimated memory usage in MB")
    throughput_estimate: float = Field(..., description="Estimated throughput in items/second")


# Create router
router = BaseRouter(tags=["GPU"])


@router.get("/gpu/info", response_model=GPUInfoResponse)
async def gpu_info(
    request: Request,
    gpu_info: Dict[str, Any] = Depends(get_gpu_info),
):
    """
    Get detailed GPU information.
    
    Returns information about available GPUs, including memory usage, 
    utilization, and other details.
    """
    if not gpu_info.get("available", False):
        raise GPUNotAvailableException(
            detail="GPU is not available",
            suggestion="Check that your system has CUDA-capable GPUs and that they are properly configured"
        )
    
    # Convert raw GPU info to response model
    devices = {}
    for device_id, device_info in gpu_info.get("devices", {}).items():
        devices[device_id] = GPUDevice(
            id=device_id,
            name=device_info.get("name", "Unknown"),
            memory_total=device_info.get("memory_total", 0),
            memory_used=device_info.get("memory_used", 0) if "memory_used" in device_info else device_info.get("memory_allocated", 0),
            memory_free=device_info.get("memory_free", 0) if "memory_free" in device_info else (device_info.get("memory_total", 0) - device_info.get("memory_allocated", 0)),
            utilization=device_info.get("utilization"),
            temperature=device_info.get("temperature"),
            power=device_info.get("power"),
            compute_capability=device_info.get("compute_capability"),
        )
    
    # Get TensorRT info
    tensorrt_info = gpu_info.get("tensorrt", {})
    
    return GPUInfoResponse(
        available=gpu_info.get("available", False),
        count=gpu_info.get("count", 0),
        cuda_version=gpu_info.get("cuda_version"),
        devices=devices,
        tensorrt_available=gpu_info.get("tensorrt_available", False),
        tensorrt_version=tensorrt_info.get("version") if tensorrt_info else None,
        multi_gpu_enabled=gpu_info.get("multi_gpu", {}).get("enabled", False),
        multi_gpu_strategy=gpu_info.get("multi_gpu", {}).get("strategy"),
    )


@router.post("/gpu/optimize", response_model=OptimizationResponse)
async def optimize_gpu(
    request: Request,
    params: OptimizationParameters,
    gpu_info: Dict[str, Any] = Depends(get_gpu_info),
    user = Depends(get_admin_user),  # Admin only
):
    """
    Optimize GPU resources.
    
    Performs various optimizations like clearing cache, optimizing memory allocation,
    and suggesting optimal batch sizes. Admin privileges required.
    """
    if not gpu_info.get("available", False):
        raise GPUNotAvailableException(
            detail="GPU is not available",
            suggestion="Check that your system has CUDA-capable GPUs and that they are properly configured"
        )
    
    # Get memory usage before optimization
    memory_before = {}
    if hasattr(request.app.state, "gpu_middleware"):
        gpu_middleware = request.app.state.gpu_middleware
        memory_before = gpu_middleware.get_gpu_memory_usage()
    
    # Initialize result
    actions_taken = []
    
    # Perform optimizations
    if params.clear_cache:
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                actions_taken.append(f"Cleared CUDA cache for {torch.cuda.device_count()} device(s)")
        except ImportError:
            actions_taken.append("Failed to clear CUDA cache: PyTorch not available")
    
    if params.optimize_memory and hasattr(request.app.state, "gpu_middleware"):
        gpu_middleware = request.app.state.gpu_middleware
        gpu_middleware.optimize_gpu_memory()
        actions_taken.append("Optimized GPU memory allocation")
    
    # Get memory usage after optimization
    memory_after = {}
    if hasattr(request.app.state, "gpu_middleware"):
        gpu_middleware = request.app.state.gpu_middleware
        memory_after = gpu_middleware.get_gpu_memory_usage()
    
    # Calculate recommended batch sizes
    recommended_batch_sizes = {}
    if params.optimize_batch_size and hasattr(request.app.state, "gpu_middleware"):
        gpu_middleware = request.app.state.gpu_middleware
        
        # Get available devices
        devices = gpu_middleware.get_available_devices()
        
        # Calculate batch size for each device
        for device in devices:
            if device.startswith("cuda:"):
                device_idx = int(device.split(":")[1])
                device_name = f"gpu{device_idx}"
                recommended_batch_sizes[device_name] = gpu_middleware.get_device_batch_size(device)
        
        actions_taken.append("Calculated recommended batch sizes for each GPU")
    
    return OptimizationResponse(
        success=True,
        actions_taken=actions_taken,
        memory_before=memory_before,
        memory_after=memory_after,
        recommended_batch_sizes=recommended_batch_sizes
    )


@router.post("/gpu/batch_size", response_model=BatchSizeResponse)
async def calculate_batch_size(
    request: Request,
    params: BatchSizeParameters,
    gpu_info: Dict[str, Any] = Depends(get_gpu_info),
):
    """
    Calculate optimal batch size for a model.
    
    Determines the optimal batch size for a specific model and hardware configuration
    based on available memory and desired precision.
    """
    if not gpu_info.get("available", False):
        raise GPUNotAvailableException(
            detail="GPU is not available",
            suggestion="Check that your system has CUDA-capable GPUs and that they are properly configured"
        )
    
    # Get GPU middleware
    if not hasattr(request.app.state, "gpu_middleware"):
        raise GPUNotAvailableException(
            detail="GPU middleware not available",
            suggestion="Check your server configuration"
        )
    
    gpu_middleware = request.app.state.gpu_middleware
    
    # Get optimal device
    device = gpu_middleware.get_optimal_device()
    
    # Calculate batch size
    base_batch_size = 32
    
    # Adjust for precision
    precision_factor = 1.0
    if params.precision == "fp16":
        precision_factor = 2.0
    elif params.precision == "int8":
        precision_factor = 4.0
    
    # Adjust for input shape
    input_size = 1
    for dim in params.input_shape:
        input_size *= dim
    
    # Adjust for sequence length
    seq_length = params.max_sequence_length or 128
    
    # Calculate optimal batch size
    optimal_batch_size = gpu_middleware.get_device_batch_size(
        device, 
        base_batch_size=int(base_batch_size * precision_factor)
    )
    
    # Calculate estimated memory
    element_size = 4  # bytes for fp32
    if params.precision == "fp16":
        element_size = 2
    elif params.precision == "int8":
        element_size = 1
    
    estimated_memory = optimal_batch_size * input_size * seq_length * element_size / (1024 * 1024)  # MB
    
    # Estimate throughput (very rough estimate)
    throughput_estimate = optimal_batch_size * 10  # items/second
    
    return BatchSizeResponse(
        optimal_batch_size=optimal_batch_size,
        device=device,
        model_name=params.model_name,
        precision=params.precision,
        estimated_memory=estimated_memory,
        throughput_estimate=throughput_estimate
    )