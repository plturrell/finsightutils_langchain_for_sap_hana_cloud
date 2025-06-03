"""GPU utilities for acceleration."""

import logging
from typing import Dict, List, Optional, Tuple, Union

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
        logger.info(f"PyTorch CUDA is available: {torch.version.cuda}")
except ImportError:
    logger.info("PyTorch not available")
    _torch_available = False

try:
    import cupy as cp
    _cupy_available = cp.is_available()
    if _cupy_available:
        _gpu_available = True
        _cuda_version = cp.cuda.runtime.runtimeGetVersion()
        logger.info(f"CuPy is available with CUDA version: {_cuda_version}")
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
    
    logger.info(f"Found {device_count} NVIDIA GPUs")
    pynvml.nvmlShutdown()
except (ImportError, Exception) as e:
    logger.info(f"Unable to get detailed GPU info: {str(e)}")


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
            import cupy as cp
            return cp.asarray(data)
        except Exception as e:
            logger.warning(f"Failed to convert to GPU array: {str(e)}")
    
    return data


def to_cpu_array(data: Union[np.ndarray, 'cp.ndarray']) -> np.ndarray:
    """
    Convert data from GPU array to CPU NumPy array.
    
    Args:
        data: Data to convert.
        
    Returns:
        NumPy array on CPU.
    """
    if _cupy_available:
        import cupy as cp
        if isinstance(data, cp.ndarray):
            return data.get()
    
    if not isinstance(data, np.ndarray):
        return np.array(data, dtype=np.float32)
    
    return data


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
            import cupy as cp
            
            # Convert to GPU arrays
            query_embedding = cp.asarray(query_embedding, dtype=cp.float32).reshape(1, -1)
            embedding_list = cp.asarray(embedding_list, dtype=cp.float32)
            
            # Normalize embeddings
            query_embedding_norm = query_embedding / cp.linalg.norm(query_embedding, axis=1, keepdims=True)
            embedding_list_norm = embedding_list / cp.linalg.norm(embedding_list, axis=1, keepdims=True)
            
            # Calculate similarities
            similarities = cp.matmul(query_embedding_norm, embedding_list_norm.T).flatten()
            
            # Select indices
            indices = []
            selected_embeddings = None
            
            for _ in range(min(k, len(embedding_list))):
                if len(indices) == 0:
                    # Select the most similar embedding first
                    idx = int(cp.argmax(similarities).get())
                else:
                    # Calculate diversity penalty
                    if selected_embeddings is None:
                        selected_embeddings = embedding_list_norm[indices[0]].reshape(1, -1)
                    
                    # Relevance score
                    relevance_scores = similarities
                    
                    # Diversity score
                    similarity_to_selected = cp.matmul(embedding_list_norm, selected_embeddings.T)
                    max_similarity_to_selected = cp.max(similarity_to_selected, axis=1)
                    diversity_scores = 1 - max_similarity_to_selected
                    
                    # Combined score
                    combined_scores = lambda_mult * relevance_scores + (1 - lambda_mult) * diversity_scores
                    
                    # Set already selected indices to large negative value
                    for idx in indices:
                        combined_scores[idx] = -9999
                    
                    idx = int(cp.argmax(combined_scores).get())
                
                indices.append(idx)
                
                # Update selected embeddings
                if len(indices) > 1:
                    selected_embeddings = embedding_list_norm[indices]
            
            return indices
        except Exception as e:
            logger.warning(f"GPU MMR calculation failed, falling back to CPU: {str(e)}")
    
    # Fall back to CPU implementation
    from langchain_core.vectorstores.utils import maximal_marginal_relevance
    return maximal_marginal_relevance(
        np.array(query_embedding), 
        np.array(embedding_list), 
        lambda_mult=lambda_mult, 
        k=k
    )