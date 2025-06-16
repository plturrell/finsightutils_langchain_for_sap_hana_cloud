"""
GPUMiddleware for the SAP HANA LangChain Integration API.

This middleware manages GPU state and provides GPU information for the API responses,
handling GPU availability, utilization tracking, and memory management.
"""

import logging
import os
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ..config_standardized import get_standardized_settings
from ..utils.standardized_exceptions import GPUNotAvailableException, TensorRTNotAvailableException

# Set up logger
logger = logging.getLogger(__name__)

# Get settings
settings = get_standardized_settings()

# Try to import GPU-related modules
try:
    import nvidia_smi
    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False
    logger.warning("nvidia-smi not available, some GPU monitoring features will be disabled")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, some GPU features will be disabled")

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available, TensorRT acceleration will be disabled")


class GPUMiddleware(BaseHTTPMiddleware):
    """Middleware for managing GPU state and information."""
    
    def __init__(
        self,
        app: ASGIApp,
        enabled: bool = None,
        cuda_visible_devices: str = None,
        cuda_device_order: str = None,
        tensorrt_enabled: bool = None,
        memory_fraction: float = None,
        auto_optimize: bool = None,
    ):
        """Initialize the middleware."""
        super().__init__(app)
        
        # Set defaults from settings if not provided
        self.enabled = enabled if enabled is not None else settings.gpu.enabled
        self.cuda_visible_devices = cuda_visible_devices or settings.gpu.cuda_visible_devices
        self.cuda_device_order = cuda_device_order or settings.gpu.cuda_device_order
        self.tensorrt_enabled = tensorrt_enabled if tensorrt_enabled is not None else settings.tensorrt.enabled
        self.memory_fraction = memory_fraction or settings.gpu.cuda_memory_fraction
        self.auto_optimize = auto_optimize if auto_optimize is not None else settings.gpu.auto_optimize
        
        # GPU state and cache
        self.gpu_info = {}
        self.gpu_info_timestamp = 0
        self.gpu_info_ttl = 10  # Refresh GPU info every 10 seconds
        
        # Set GPU environment variables
        if self.enabled:
            self._set_gpu_environment_variables()
            
        # Initialize GPU monitoring if available
        if NVIDIA_SMI_AVAILABLE and self.enabled:
            try:
                nvidia_smi.nvmlInit()
                self.nvidia_smi_initialized = True
                logger.info("NVIDIA SMI initialized successfully")
            except Exception as e:
                self.nvidia_smi_initialized = False
                logger.warning(f"Failed to initialize NVIDIA SMI: {str(e)}")
        else:
            self.nvidia_smi_initialized = False
        
        # Check CUDA availability if torch is available
        if TORCH_AVAILABLE and self.enabled:
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                self.cuda_device_count = torch.cuda.device_count()
                logger.info(f"CUDA is available with {self.cuda_device_count} device(s)")
            else:
                logger.warning("CUDA is not available")
        else:
            self.cuda_available = False
            self.cuda_device_count = 0
        
        # Check TensorRT availability if enabled
        if TENSORRT_AVAILABLE and self.tensorrt_enabled:
            try:
                self.tensorrt_logger = trt.Logger(trt.Logger.WARNING)
                self.tensorrt_available = True
                logger.info(f"TensorRT is available (version: {trt.__version__})")
            except Exception as e:
                self.tensorrt_available = False
                logger.warning(f"Failed to initialize TensorRT: {str(e)}")
        else:
            self.tensorrt_available = False
        
        # Start memory optimization thread if auto-optimize is enabled
        if self.auto_optimize and self.enabled and self.cuda_available:
            self._start_memory_optimization_thread()
    
    def _set_gpu_environment_variables(self) -> None:
        """Set GPU-related environment variables."""
        # Set CUDA environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        os.environ["CUDA_DEVICE_ORDER"] = self.cuda_device_order
        
        # Set TensorFlow environment variables (for TensorRT compatibility)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow logging
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        
        # Set TensorRT environment variables
        if self.tensorrt_enabled:
            os.environ["TRT_LOG_LEVEL"] = "2"  # WARNING level
            os.environ["TRT_MEMORY_POOL_SIZE"] = str(settings.tensorrt.max_workspace_size)
            os.environ["TRT_PLUGIN_DIRECTORY"] = settings.tensorrt.cache_dir
            
            # Set cache path for TensorRT engines
            os.makedirs(settings.tensorrt.cache_dir, exist_ok=True)
            os.environ["TRT_ENGINE_CACHE_PATH"] = settings.tensorrt.cache_dir
        
        # Set CUDA cache parameters
        os.environ["CUDA_CACHE_MAXSIZE"] = str(settings.gpu.cuda_cache_maxsize)
        os.environ["CUDA_CACHE_PATH"] = settings.gpu.cuda_cache_path
        os.makedirs(settings.gpu.cuda_cache_path, exist_ok=True)
    
    def _start_memory_optimization_thread(self) -> None:
        """Start a background thread for GPU memory optimization."""
        def optimize_memory_periodically():
            while True:
                try:
                    # Check GPU memory usage
                    memory_usage = self.get_gpu_memory_usage()
                    
                    # If any GPU is using more than the specified fraction, clear caches
                    for device, usage in memory_usage.items():
                        if usage > self.memory_fraction * 100:
                            logger.info(f"GPU {device} memory usage is high ({usage:.2f}%). Clearing caches.")
                            self.optimize_gpu_memory()
                            break
                except Exception as e:
                    logger.error(f"Error in memory optimization thread: {str(e)}")
                
                # Sleep for 30 seconds
                time.sleep(30)
        
        # Start the thread
        thread = threading.Thread(target=optimize_memory_periodically, daemon=True)
        thread.start()
        logger.info("GPU memory optimization thread started")
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information.
        
        Returns:
            Dict containing GPU information including memory usage, utilization, etc.
        """
        # Check if cached GPU info is still valid
        now = time.time()
        if self.gpu_info and now - self.gpu_info_timestamp < self.gpu_info_ttl:
            return self.gpu_info
        
        gpu_info = {
            "available": False,
            "count": 0,
            "devices": {},
            "cuda_version": None,
            "tensorrt_available": self.tensorrt_available,
        }
        
        # Get GPU information using NVIDIA SMI if available
        if self.nvidia_smi_initialized:
            try:
                device_count = nvidia_smi.nvmlDeviceGetCount()
                gpu_info["available"] = True
                gpu_info["count"] = device_count
                
                for i in range(device_count):
                    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                    
                    # Get device name
                    name = nvidia_smi.nvmlDeviceGetName(handle)
                    
                    # Get memory info
                    mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Get utilization info
                    util_info = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Get temperature
                    temp = nvidia_smi.nvmlDeviceGetTemperature(
                        handle, nvidia_smi.NVML_TEMPERATURE_GPU
                    )
                    
                    # Get power usage
                    try:
                        power = nvidia_smi.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                    except:
                        power = None
                    
                    # Store device information
                    gpu_info["devices"][f"gpu{i}"] = {
                        "name": name,
                        "memory_total": mem_info.total,
                        "memory_used": mem_info.used,
                        "memory_free": mem_info.free,
                        "utilization": util_info.gpu,
                        "memory_utilization": util_info.memory,
                        "temperature": temp,
                        "power": power
                    }
            except Exception as e:
                logger.warning(f"Failed to get GPU information from NVIDIA SMI: {str(e)}")
        
        # Get CUDA information using PyTorch if available
        if TORCH_AVAILABLE and self.cuda_available:
            try:
                gpu_info["available"] = True
                gpu_info["count"] = self.cuda_device_count
                gpu_info["cuda_version"] = torch.version.cuda
                
                # Get memory information for each device
                for i in range(self.cuda_device_count):
                    if f"gpu{i}" not in gpu_info["devices"]:
                        gpu_info["devices"][f"gpu{i}"] = {}
                    
                    # Get device properties
                    props = torch.cuda.get_device_properties(i)
                    
                    # Get memory information
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    max_memory = props.total_memory
                    
                    # Update device information
                    gpu_info["devices"][f"gpu{i}"].update({
                        "name": props.name,
                        "memory_total": max_memory,
                        "memory_allocated": allocated,
                        "memory_reserved": reserved,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "multi_processor_count": props.multi_processor_count,
                    })
            except Exception as e:
                logger.warning(f"Failed to get GPU information from PyTorch: {str(e)}")
        
        # Get TensorRT information if available
        if TENSORRT_AVAILABLE and self.tensorrt_available:
            try:
                # Add TensorRT information
                gpu_info["tensorrt"] = {
                    "version": trt.__version__,
                    "available": True,
                    "max_workspace_size": settings.tensorrt.max_workspace_size,
                    "cache_dir": settings.tensorrt.cache_dir,
                    "fp16_enabled": settings.tensorrt.fp16_enabled,
                    "int8_enabled": settings.tensorrt.int8_enabled,
                }
            except Exception as e:
                logger.warning(f"Failed to get TensorRT information: {str(e)}")
        
        # Add multi-GPU support information
        gpu_info["multi_gpu"] = {
            "enabled": settings.gpu.multi_gpu_enabled,
            "strategy": settings.gpu.multi_gpu_strategy,
            "device_ids": [int(id) for id in self.cuda_visible_devices.split(",") if id.isdigit()]
        }
        
        # Update the cached GPU info
        self.gpu_info = gpu_info
        self.gpu_info_timestamp = now
        
        return gpu_info
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process the request and add GPU information to the request state."""
        # Skip if GPU is disabled
        if not self.enabled:
            return await call_next(request)
        
        # Get GPU information
        gpu_info = self._get_gpu_info()
        
        # Check if GPU is required for this request
        requires_gpu = False
        requires_tensorrt = False
        
        # Check query parameters for GPU requirements
        if "use_gpu" in request.query_params:
            requires_gpu = request.query_params.get("use_gpu").lower() in ("true", "1", "yes")
        
        if "use_tensorrt" in request.query_params:
            requires_tensorrt = request.query_params.get("use_tensorrt").lower() in ("true", "1", "yes")
        
        # Check headers for GPU requirements
        if "X-Use-GPU" in request.headers:
            requires_gpu = request.headers.get("X-Use-GPU").lower() in ("true", "1", "yes")
        
        if "X-Use-TensorRT" in request.headers:
            requires_tensorrt = request.headers.get("X-Use-TensorRT").lower() in ("true", "1", "yes")
        
        # Arrow Flight endpoints may require GPU
        if request.url.path.startswith("/api/flight"):
            if "use_gpu" in request.query_params:
                requires_gpu = request.query_params.get("use_gpu").lower() in ("true", "1", "yes")
        
        # If GPU is required but not available, raise an exception
        if requires_gpu and not gpu_info["available"]:
            raise GPUNotAvailableException(
                detail="GPU acceleration is requested but not available",
                details={"gpu_info": gpu_info}
            )
        
        # If TensorRT is required but not available, raise an exception
        if requires_tensorrt and not gpu_info["tensorrt_available"]:
            raise TensorRTNotAvailableException(
                detail="TensorRT acceleration is requested but not available",
                details={"gpu_info": gpu_info}
            )
        
        # Add GPU information to request state
        request.state.gpu_info = gpu_info
        request.state.requires_gpu = requires_gpu
        request.state.requires_tensorrt = requires_tensorrt
        
        # Process the request
        response = await call_next(request)
        
        # Add GPU information to response headers if GPU was used
        if requires_gpu:
            response.headers["X-GPU-Used"] = "true"
            if requires_tensorrt:
                response.headers["X-TensorRT-Used"] = "true"
        
        return response
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get GPU memory usage as a percentage.
        
        Returns:
            Dict mapping device names to memory usage percentages
        """
        gpu_info = self._get_gpu_info()
        memory_usage = {}
        
        for device, info in gpu_info.get("devices", {}).items():
            if "memory_total" in info and "memory_used" in info:
                memory_usage[device] = (info["memory_used"] / info["memory_total"]) * 100
            elif "memory_total" in info and "memory_allocated" in info:
                memory_usage[device] = (info["memory_allocated"] / info["memory_total"]) * 100
        
        return memory_usage
    
    def optimize_gpu_memory(self) -> None:
        """Optimize GPU memory by clearing caches if needed."""
        if not TORCH_AVAILABLE or not self.cuda_available:
            return
        
        # Get current memory usage
        memory_usage = self.get_gpu_memory_usage()
        
        # Clear caches if memory usage is above threshold
        threshold = self.memory_fraction * 100
        for device, usage in memory_usage.items():
            if usage > threshold:
                device_idx = int(device.replace("gpu", ""))
                logger.warning(f"GPU {device_idx} memory usage is high ({usage:.2f}%). Clearing caches.")
                
                # Clear PyTorch caches
                with torch.cuda.device(device_idx):
                    torch.cuda.empty_cache()
    
    def get_optimal_device(self) -> str:
        """Get the optimal device for computation.
        
        Returns:
            Device string for PyTorch (e.g., 'cuda:0', 'cpu')
        """
        # If GPU is not available or not enabled, use CPU
        if not self.enabled or not self.cuda_available:
            return "cpu"
        
        # Get memory usage for each device
        memory_usage = self.get_gpu_memory_usage()
        
        # Find the device with the lowest memory usage
        if memory_usage:
            best_device = min(memory_usage.items(), key=lambda x: x[1])
            device_idx = int(best_device[0].replace("gpu", ""))
            return f"cuda:{device_idx}"
        
        # If no memory usage information is available, use the first device
        return "cuda:0"
    
    def get_available_devices(self) -> List[str]:
        """Get a list of available devices.
        
        Returns:
            List of available device strings for PyTorch
        """
        if not self.enabled or not self.cuda_available:
            return ["cpu"]
        
        return [f"cuda:{i}" for i in range(self.cuda_device_count)] + ["cpu"]
    
    def get_device_batch_size(self, device: str, base_batch_size: int = 32) -> int:
        """Get the optimal batch size for a device.
        
        Args:
            device: Device string (e.g., 'cuda:0', 'cpu')
            base_batch_size: Base batch size to scale
            
        Returns:
            Optimal batch size for the device
        """
        # For CPU, use a smaller batch size
        if device == "cpu":
            return max(1, base_batch_size // 4)
        
        # For GPU, scale based on memory
        if device.startswith("cuda:"):
            device_idx = int(device.split(":")[1])
            gpu_info = self._get_gpu_info()
            
            # Get device information
            device_info = gpu_info.get("devices", {}).get(f"gpu{device_idx}", {})
            
            # Scale batch size based on memory
            memory_total = device_info.get("memory_total", 0)
            if memory_total > 0:
                # Scale batch size based on total memory (GB)
                memory_gb = memory_total / (1024 ** 3)
                
                # T4 has 16GB, we'll use that as a reference
                scale_factor = memory_gb / 16.0
                
                # Scale batch size, but ensure it's at least 1
                return max(1, int(base_batch_size * scale_factor))
            
            # If memory information is not available, use base batch size
            return base_batch_size
        
        # Default to base batch size
        return base_batch_size


def setup_gpu_middleware(app: ASGIApp, **kwargs) -> GPUMiddleware:
    """
    Configure and add the GPU middleware to the application.
    
    Args:
        app: FastAPI application
        **kwargs: Additional arguments to pass to the GPUMiddleware constructor
        
    Returns:
        Configured GPUMiddleware instance
    """
    middleware = GPUMiddleware(app, **kwargs)
    
    # Add endpoint to get GPU information
    try:
        from fastapi import FastAPI, APIRouter
        
        # Check if app is a FastAPI instance
        if isinstance(app, FastAPI):
            @app.get("/api/gpu/info")
            def gpu_info():
                """Get GPU information."""
                return middleware._get_gpu_info()
    except ImportError:
        logger.warning("FastAPI not available, GPU info endpoint will not be added")
    
    return middleware