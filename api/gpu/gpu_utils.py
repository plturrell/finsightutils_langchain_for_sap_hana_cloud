"""GPU utilities for acceleration."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

# Add project root to sys.path if not already there
# This ensures absolute imports work in all execution contexts
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logging.info("Adding project root to sys.path: %s", project_root)

# For CPU-only deployments, we need to provide fallback implementations
import numpy as np

logger = logging.getLogger(__name__)

# Initialize global variables
_gpu_available = False
_cupy_available = False
_torch_available = False
_cuda_version = None
_gpu_info = {}

# Try to import GPU libraries
try:
    import torch
    _torch_available = torch.cuda.is_available()
    if _torch_available:
        _gpu_available = True
        logger.info("PyTorch CUDA is available: %s", torch.version.cuda)
except ImportError:
    logger.info("PyTorch not available")
    _torch_available = False

try:
    import cupy as cp
    _cupy_available = cp.is_available()
    if _cupy_available:
        _gpu_available = True
        _cuda_version = cp.cuda.runtime.runtimeGetVersion()
        logger.info("CuPy is available with CUDA version: %s", _cuda_version)
except ImportError:
    logger.info("CuPy not available")
    _cupy_available = False

# Try to get detailed GPU info
try:
    import pynvml
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    _gpu_info["device_count"] = device_count
    _gpu_info["devices"] = []
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        device_info = {
            "name": pynvml.nvmlDeviceGetName(handle).decode("utf-8"),
            "memory_total": pynvml.nvmlDeviceGetMemoryInfo(handle).total,
            "compute_capability": pynvml.nvmlDeviceGetCudaComputeCapability(handle),
        }
        _gpu_info["devices"].append(device_info)
    
    logger.info("Found %s NVIDIA GPUs", device_count)
    pynvml.nvmlShutdown()
except (ImportError, Exception) as e:
    logger.info("Unable to get detailed GPU info: %s", str(e))


def is_gpu_available() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns:
        bool: True if GPU acceleration is available.
    """
    return _gpu_available


def is_cupy_available() -> bool:
    """
    Check if CuPy is available.
    
    Returns:
        bool: True if CuPy is available.
    """
    return _cupy_available


def is_torch_available() -> bool:
    """
    Check if PyTorch with CUDA is available.
    
    Returns:
        bool: True if PyTorch with CUDA is available.
    """
    return _torch_available


def get_gpu_info() -> Dict:
    """
    Get information about available GPUs.
    
    Returns:
        Dict: Dictionary with GPU information.
    """
    return _gpu_info


def detect_gpus() -> List[Dict[str, Any]]:
    """
    Detect and return information about available GPUs.
    
    Returns:
        List[Dict[str, Any]]: List of dictionaries with GPU information.
        Each dictionary contains at least the following keys:
        - 'name': GPU name
        - 'memory_total': Total memory in bytes
        - 'device_id': GPU device ID
        
        Returns an empty list if no GPUs are detected.
    """
    if not _gpu_available:
        logger.info("No GPUs detected")
        return []
    
    gpus = []
    try:
        if _torch_available:
            for device_idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(device_idx)
                gpus.append({
                    'name': props.name,
                    'memory_total': props.total_memory,
                    'device_id': device_idx,
                    'compute_capability': f"{props.major}.{props.minor}",
                })
        elif 'devices' in _gpu_info and _gpu_info['devices']:
            # Use previously collected GPU info from pynvml
            for i, device in enumerate(_gpu_info['devices']):
                gpus.append({
                    'name': device.get('name', f"GPU {i}"),
                    'memory_total': device.get('memory_total', 0),
                    'device_id': i,
                    'compute_capability': device.get('compute_capability', (0, 0)),
                })
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.warning("GPU detection error: %s", str(e))
        
    return gpus


def to_gpu_array(data: Union[List[float], np.ndarray]) -> Union[np.ndarray, 'cp.ndarray']:
    """
    Convert data to GPU array if possible.
    
    Args:
        data: Data to convert.
        
    Returns:
        Array on GPU if available, otherwise NumPy array.
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    
    if _cupy_available:
        try:
            # Re-import cupy in this scope
            import cupy as cp_local
            return cp_local.asarray(data)
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning("Failed to convert to GPU array: %s", str(e))
    
    return data


def to_cpu_array(data: Union[np.ndarray, 'cp.ndarray']):
    """
    Convert data from GPU array to CPU NumPy array.
    
    Args:
        data: Data to convert.
        
    Returns:
        NumPy array on CPU.
    """
    if _cupy_available and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    elif _torch_available and isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return np.array(data)


def get_available_gpu_memory() -> Dict[int, int]:
    """Get available GPU memory per device.

    Returns:
        Dict[int, int]: Dictionary mapping device ID to available memory in bytes.
        Empty dictionary for CPU-only environments.
    """
    return get_available_memory()


def get_available_memory() -> Dict[int, int]:
    """Alias for get_available_gpu_memory for backward compatibility.
    
    Returns:
        Dict[int, int]: Dictionary mapping device ID to available memory in bytes.
        Empty dictionary for CPU-only environments.
    """
    if not _gpu_available:
        logger.debug("Using GPU: %s", str(is_gpu_available()))
        return {}
    
    result = {}
    try:
        if _torch_available:
            for device_idx in range(torch.cuda.device_count()):
                memory_total = torch.cuda.get_device_properties(device_idx).total_memory
                memory_allocated = torch.cuda.memory_allocated(device_idx)
                result[device_idx] = memory_total - memory_allocated
        elif 'pynvml' in globals():
            # Re-import pynvml in this scope
            import pynvml as pynvml_local
            for i in range(pynvml_local.nvmlDeviceGetCount()):
                device_handle = pynvml_local.nvmlDeviceGetHandleByIndex(i)
                info = pynvml_local.nvmlDeviceGetMemoryInfo(device_handle)

                result[i] = info.free
    except (ImportError, RuntimeError, AttributeError) as e:
        logger.error("Error getting GPU memory: %s", str(e))
        return {}
    
    return result


def gpu_maximal_marginal_relevance(
    query_embedding: Union[List[float], np.ndarray, 'cp.ndarray'],
    embedding_list: Union[List[List[float]], np.ndarray, 'cp.ndarray'],
    lambda_mult: float = 0.5,
    k: int = 4
) -> List[int]:
    """
    GPU-accelerated maximal marginal relevance algorithm for diverse document retrieval.
    
    Args:
        query_embedding: Query embedding.
        embedding_list: List of embeddings to select from.
        lambda_mult: Diversity weight between 0 and 1.
        k: Number of embeddings to select.
        
    Returns:
        List of indices of selected embeddings.
    """
    if _cupy_available:
        try:
            # Re-import cupy in this scope
            import cupy as cp_local
            
            # Convert to GPU arrays
            query_embedding = cp_local.asarray(query_embedding, dtype=cp_local.float32).reshape(1, -1)
            embedding_list = cp_local.asarray(embedding_list, dtype=cp_local.float32)
            
            # Normalize embeddings
            query_norm = cp_local.sqrt(cp_local.sum(query_embedding ** 2))
            embedding_list_norm = cp_local.sqrt(cp_local.sum(embedding_list ** 2, axis=1, keepdims=True))
            embedding_list_norm = embedding_list / cp.linalg.norm(embedding_list, axis=1, keepdims=True)
            
            # Calculate similarities
            normed_query = query_embedding / query_norm
            normed_embeddings = embedding_list / embedding_list_norm
            similarities = cp_local.dot(normed_embeddings, normed_query.T).flatten()
            
            # Select indices
            indices = []
            selected_embeddings = None
            
            for _ in range(min(k, len(embedding_list))):
                if len(indices) == 0:
                    # Select the most similar embedding first
                    idx = int(cp_local.argmax(similarities).get())
                else:
                    # Calculate diversity penalty
                    if selected_embeddings is None:
                        selected_embeddings = embedding_list_norm[indices[0]].reshape(1, -1)
                    
                    # Relevance score
                    relevance_scores = similarities
                    
                    # Diversity score
                    similarity_to_selected = cp_local.max(cp_local.dot(normed_embeddings, selected_embeddings.T), axis=1)
                    max_similarity_to_selected = cp_local.max(similarity_to_selected, axis=1)
                    diversity_scores = 1 - max_similarity_to_selected
                    
                    # Combined score
                    combined_scores = lambda_mult * relevance_scores + (1 - lambda_mult) * diversity_scores
                    
                    # Set already selected indices to large negative value
                    for idx in indices:
                        combined_scores[idx] = -9999
                    
                    idx = int(cp_local.argmax(combined_scores).get())
                
                indices.append(idx)
                
                # Update selected embeddings
                if len(indices) > 1:
                    selected_embeddings = embedding_list[selected_embeddings]
            
            return indices
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.warning(f"GPU MMR calculation failed, falling back to CPU: {str(e)}")
    
    # Fall back to CPU implementation
    from langchain_core.vectorstores.utils import maximal_marginal_relevance
    return maximal_marginal_relevance(
        np.array(query_embedding), 
        np.array(embedding_list), 
        lambda_mult=lambda_mult, 
        k=k
    )