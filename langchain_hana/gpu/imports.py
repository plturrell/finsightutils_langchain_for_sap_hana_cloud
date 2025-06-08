"""
Centralized conditional imports for GPU-related functionality.

This module provides a clean way to handle optional dependencies for GPU acceleration.
By centralizing all conditional imports here, we can:

1. Provide detailed error messages with installation instructions
2. Avoid scattered try/except blocks throughout the codebase
3. Make it easier to test code paths with and without GPU dependencies
4. Provide a single point of configuration for GPU feature detection

Usage:
    from langchain_hana.gpu.imports import (
        torch, cuda, tensorrt, pycuda, TORCH_AVAILABLE, TENSORRT_AVAILABLE, PYCUDA_AVAILABLE
    )
    
    if TORCH_AVAILABLE:
        # Use torch features
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        # Handle missing dependency appropriately
"""

import sys
import logging
from typing import Dict, Any, Optional, Tuple, List

logger = logging.getLogger(__name__)

# Dictionary to track import availability
AVAILABLE_IMPORTS = {
    "torch": False,
    "tensorrt": False,
    "pycuda": False,
    "nvml": False,
    "sentence_transformers": False,
}

# Dictionary for providing installation instructions
INSTALL_INSTRUCTIONS = {
    "torch": "pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118",
    "tensorrt": "pip install tensorrt>=8.6.0",
    "pycuda": "pip install pycuda>=2022.2",
    "nvml": "pip install nvidia-ml-py>=11.525.84",
    "sentence_transformers": "pip install sentence-transformers>=2.2.2",
    "all_gpu": "pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118 tensorrt>=8.6.0 pycuda>=2022.2 nvidia-ml-py>=11.525.84",
}

# Placeholder objects for missing dependencies
# These will be replaced with actual modules when available
torch = None
cuda = None
tensorrt = None
pycuda = None
pynvml = None
sentence_transformers = None

# Import PyTorch and CUDA
try:
    import torch
    import torch.cuda as cuda
    AVAILABLE_IMPORTS["torch"] = True
    TORCH_AVAILABLE = True
    
    # Check if CUDA is available
    CUDA_AVAILABLE = torch.cuda.is_available()
    if not CUDA_AVAILABLE:
        logger.info("PyTorch installed, but CUDA is not available.")
except ImportError as e:
    logger.info(f"PyTorch not available: {e}. GPU acceleration disabled.")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False
except Exception as e:
    logger.warning(f"Unexpected error importing PyTorch: {e}. GPU acceleration may not work properly.")
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

# Import TensorRT
try:
    import tensorrt as trt
    from tensorrt.logger import Logger as TRTLogger
    AVAILABLE_IMPORTS["tensorrt"] = True
    TENSORRT_AVAILABLE = True
except ImportError as e:
    logger.info(f"TensorRT not available: {e}. TensorRT optimization disabled.")
    TENSORRT_AVAILABLE = False
    # Create placeholders for missing classes
    class TRTLogger:
        pass
except Exception as e:
    logger.warning(f"Unexpected error importing TensorRT: {e}. TensorRT optimization may not work properly.")
    TENSORRT_AVAILABLE = False
    # Create placeholders for missing classes
    class TRTLogger:
        pass

# Import PyCUDA
try:
    import pycuda.driver as cuda_driver
    import pycuda.autoinit
    AVAILABLE_IMPORTS["pycuda"] = True
    PYCUDA_AVAILABLE = True
except ImportError as e:
    logger.info(f"PyCUDA not available: {e}. Some GPU features disabled.")
    PYCUDA_AVAILABLE = False
    cuda_driver = None
except Exception as e:
    logger.warning(f"Unexpected error importing PyCUDA: {e}. Some GPU features may not work properly.")
    PYCUDA_AVAILABLE = False
    cuda_driver = None

# Import NVML for GPU monitoring
try:
    import pynvml
    AVAILABLE_IMPORTS["nvml"] = True
    NVML_AVAILABLE = True
    
    # Initialize NVML
    try:
        pynvml.nvmlInit()
    except Exception as e:
        logger.warning(f"Error initializing NVML: {e}. GPU monitoring will be limited.")
        NVML_AVAILABLE = False
except ImportError:
    logger.info("NVIDIA Management Library not available. GPU monitoring will be limited.")
    NVML_AVAILABLE = False
except Exception as e:
    logger.warning(f"Unexpected error importing NVML: {e}. GPU monitoring may not work properly.")
    NVML_AVAILABLE = False

# Import SentenceTransformers for CPU fallback
try:
    import sentence_transformers
    AVAILABLE_IMPORTS["sentence_transformers"] = True
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.info("SentenceTransformers not available. CPU fallback for embeddings will be limited.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
except Exception as e:
    logger.warning(f"Unexpected error importing SentenceTransformers: {e}. CPU fallback may not work properly.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


def get_gpu_features_status() -> Dict[str, Any]:
    """
    Get the status of available GPU features and dependencies.
    
    Returns:
        Dict[str, Any]: Dictionary with status of GPU features
    """
    return {
        "torch_available": TORCH_AVAILABLE,
        "cuda_available": CUDA_AVAILABLE,
        "tensorrt_available": TENSORRT_AVAILABLE,
        "pycuda_available": PYCUDA_AVAILABLE,
        "nvml_available": NVML_AVAILABLE,
        "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE,
        "gpu_count": torch.cuda.device_count() if TORCH_AVAILABLE and CUDA_AVAILABLE else 0,
        "cuda_version": torch.version.cuda if TORCH_AVAILABLE else None,
        "imports": AVAILABLE_IMPORTS.copy(),
    }


def check_gpu_requirements(feature: str) -> Tuple[bool, str]:
    """
    Check if all required dependencies are available for a specific GPU feature.
    
    Args:
        feature: Name of the GPU feature to check
            - "tensorrt": TensorRT optimization
            - "multi_gpu": Multi-GPU support
            - "monitoring": GPU monitoring
            - "tensor_cores": Tensor Core optimization
    
    Returns:
        Tuple of (requirements_met: bool, message: str)
    """
    if feature == "tensorrt":
        requirements_met = TORCH_AVAILABLE and TENSORRT_AVAILABLE and PYCUDA_AVAILABLE
        if not requirements_met:
            missing = []
            if not TORCH_AVAILABLE:
                missing.append(f"torch ({INSTALL_INSTRUCTIONS['torch']})")
            if not TENSORRT_AVAILABLE:
                missing.append(f"tensorrt ({INSTALL_INSTRUCTIONS['tensorrt']})")
            if not PYCUDA_AVAILABLE:
                missing.append(f"pycuda ({INSTALL_INSTRUCTIONS['pycuda']})")
                
            message = (
                f"TensorRT optimization requires the following missing dependencies: "
                f"{', '.join(missing)}. Install with: {INSTALL_INSTRUCTIONS['all_gpu']}"
            )
        else:
            message = "All TensorRT dependencies available."
            
        return requirements_met, message
        
    elif feature == "multi_gpu":
        requirements_met = TORCH_AVAILABLE and CUDA_AVAILABLE
        if not requirements_met:
            message = (
                "Multi-GPU support requires PyTorch with CUDA. "
                f"Install with: {INSTALL_INSTRUCTIONS['torch']}"
            )
        else:
            message = "Multi-GPU dependencies available."
            
        return requirements_met, message
        
    elif feature == "monitoring":
        requirements_met = NVML_AVAILABLE
        if not requirements_met:
            message = (
                "Advanced GPU monitoring requires NVIDIA Management Library. "
                f"Install with: {INSTALL_INSTRUCTIONS['nvml']}"
            )
        else:
            message = "GPU monitoring dependencies available."
            
        return requirements_met, message
        
    elif feature == "tensor_cores":
        requirements_met = TORCH_AVAILABLE and CUDA_AVAILABLE and TENSORRT_AVAILABLE
        if not requirements_met:
            message = (
                "Tensor Core optimization requires PyTorch with CUDA and TensorRT. "
                f"Install with: {INSTALL_INSTRUCTIONS['all_gpu']}"
            )
        else:
            # Check for hardware support
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                gpu_count = torch.cuda.device_count()
                if gpu_count > 0:
                    for i in range(gpu_count):
                        props = torch.cuda.get_device_properties(i)
                        # Tensor Cores available on Volta (7.0) and later
                        if props.major >= 7:
                            return True, f"Tensor Cores supported on {props.name} (compute capability {props.major}.{props.minor})."
                    
                    # No GPUs with Tensor Core support found
                    return False, "No GPUs with Tensor Core support detected. Requires NVIDIA Volta (V100) or newer GPU."
                else:
                    return False, "No CUDA-capable GPUs detected."
            else:
                message = "Tensor Core dependencies available, but CUDA not accessible."
                
        return requirements_met, message
        
    else:
        return False, f"Unknown feature: {feature}"


def get_installation_instructions() -> str:
    """
    Get installation instructions for GPU dependencies.
    
    Returns:
        str: Formatted installation instructions
    """
    return f"""
# GPU Acceleration Installation Guide

To enable GPU acceleration features, you need to install the following dependencies:

## Basic GPU Support
```bash
{INSTALL_INSTRUCTIONS['torch']}
```

## TensorRT Optimization (Recommended for inference)
```bash
{INSTALL_INSTRUCTIONS['tensorrt']}
{INSTALL_INSTRUCTIONS['pycuda']}
```

## GPU Monitoring
```bash
{INSTALL_INSTRUCTIONS['nvml']}
```

## CPU Fallback (Used when GPU is unavailable)
```bash
{INSTALL_INSTRUCTIONS['sentence_transformers']}
```

## All GPU Dependencies
```bash
{INSTALL_INSTRUCTIONS['all_gpu']}
{INSTALL_INSTRUCTIONS['sentence_transformers']}
```

Note: TensorRT installation might require additional steps depending on your system.
Please refer to the NVIDIA documentation for detailed instructions.
"""

def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Get detailed information about available GPUs.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries with GPU information
    """
    if not TORCH_AVAILABLE or not CUDA_AVAILABLE:
        return []
        
    gpu_count = torch.cuda.device_count()
    if gpu_count == 0:
        return []
        
    gpu_info = []
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        
        # Get memory info
        total_memory = props.total_memory / (1024 ** 2)  # Convert to MB
        
        # Get free memory
        free_memory = 0
        try:
            with torch.cuda.device(i):
                free_memory, _ = torch.cuda.mem_get_info()
                free_memory = free_memory / (1024 ** 2)  # Convert to MB
        except (RuntimeError, AttributeError):
            # mem_get_info not available in older PyTorch versions
            free_memory = 0
        
        # Check for Tensor Core support
        tensor_cores_supported = props.major >= 7
        
        # Get utilization
        utilization = 0
        temperature = 0
        power_draw = 0
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu / 100.0
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            except Exception as e:
                logger.debug(f"Error getting NVML info for GPU {i}: {e}")
        
        # Create GPU info dictionary
        info = {
            "index": i,
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "memory_total_mb": total_memory,
            "memory_free_mb": free_memory,
            "memory_usage_percent": 100.0 * (1.0 - (free_memory / total_memory)) if total_memory > 0 else 0,
            "cores": props.multi_processor_count,
            "tensor_cores_supported": tensor_cores_supported,
            "utilization": utilization,
            "temperature_c": temperature,
            "power_draw_watts": power_draw,
        }
        
        gpu_info.append(info)
    
    return gpu_info