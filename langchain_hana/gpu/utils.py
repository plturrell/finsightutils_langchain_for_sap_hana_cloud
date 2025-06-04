"""
GPU utilities for SAP HANA Cloud LangChain integration.

This module provides utilities for GPU acceleration, including memory management,
device selection, and optimal batch size determination.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Conditional imports based on GPU availability
try:
    import torch
    import cupy as cp
    HAS_GPU_DEPENDENCIES = True
except ImportError:
    HAS_GPU_DEPENDENCIES = False

logger = logging.getLogger(__name__)


def detect_gpu_capabilities() -> Dict[str, Union[bool, int, str]]:
    """
    Detect GPU capabilities and return information about available hardware.
    
    Returns:
        Dict with the following keys:
        - has_gpu: Whether a GPU is available
        - gpu_count: Number of available GPUs
        - gpu_names: List of GPU names
        - cuda_version: CUDA version (if available)
        - total_gpu_memory: Total GPU memory in MB
        - compute_capabilities: List of compute capabilities
    """
    result = {
        "has_gpu": False,
        "gpu_count": 0,
        "gpu_names": [],
        "cuda_version": "N/A",
        "total_gpu_memory": 0,
        "compute_capabilities": []
    }
    
    if not HAS_GPU_DEPENDENCIES:
        logger.info("GPU detection skipped: required packages not installed")
        return result
    
    try:
        if not torch.cuda.is_available():
            logger.info("No CUDA-capable GPUs detected")
            return result
        
        # Update basic information
        gpu_count = torch.cuda.device_count()
        result.update({
            "has_gpu": True,
            "gpu_count": gpu_count,
            "cuda_version": torch.version.cuda,
        })
        
        # Get detailed information for each GPU
        gpu_names = []
        compute_capabilities = []
        total_memory = 0
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            gpu_names.append(props.name)
            compute_capabilities.append(f"{props.major}.{props.minor}")
            total_memory += props.total_memory // (1024 * 1024)  # Convert to MB
        
        result.update({
            "gpu_names": gpu_names,
            "total_gpu_memory": total_memory,
            "compute_capabilities": compute_capabilities
        })
        
        return result
    
    except Exception as e:
        logger.warning(f"Error detecting GPU capabilities: {str(e)}")
        return result


def get_optimal_batch_size(
    vector_dim: int,
    available_memory_mb: Optional[int] = None,
    headroom_percent: float = 20.0,
    min_batch_size: int = 8,
    max_batch_size: int = 128,
) -> int:
    """
    Calculate the optimal batch size based on available GPU memory.
    
    Args:
        vector_dim: Dimension of embedding vectors
        available_memory_mb: Available GPU memory in MB (auto-detected if None)
        headroom_percent: Percentage of memory to keep free as headroom
        min_batch_size: Minimum batch size to return
        max_batch_size: Maximum batch size to return
        
    Returns:
        Optimal batch size for embedding generation
    """
    if not HAS_GPU_DEPENDENCIES or not torch.cuda.is_available():
        logger.info("No GPU available, using default batch size")
        return min_batch_size
    
    try:
        # Get available memory if not provided
        if available_memory_mb is None:
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            available_memory_mb = free_memory // (1024 * 1024)
        
        # Apply headroom to avoid OOM errors
        usable_memory_mb = available_memory_mb * (1 - (headroom_percent / 100.0))
        
        # Estimate memory needed per sample
        # We use a heuristic that accounts for:
        # - Input tensors (ids, masks): ~8 bytes per token, ~128 tokens per sample
        # - Internal activations: ~4 * vector_dim bytes
        # - Output vector: 4 bytes per dimension
        bytes_per_sample = (8 * 128) + (4 * vector_dim) + (4 * vector_dim)
        mb_per_sample = bytes_per_sample / (1024 * 1024)
        
        # Calculate batch size
        optimal_batch_size = int(usable_memory_mb / mb_per_sample)
        
        # Clamp to min/max range
        return max(min_batch_size, min(optimal_batch_size, max_batch_size))
    
    except Exception as e:
        logger.warning(f"Error calculating optimal batch size: {str(e)}")
        return min_batch_size


def gpu_maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[np.ndarray],
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """
    Calculate maximal marginal relevance using GPU acceleration.
    
    This implementation uses CuPy for GPU-accelerated vector operations,
    which significantly improves performance for large embedding sets.
    
    Args:
        query_embedding: Embedding of the query
        embedding_list: List of embeddings to consider
        lambda_mult: Number between 0 and 1 that determines the degree of diversity
                    (0 = maximum diversity, 1 = maximum similarity)
        k: Number of embeddings to return
        
    Returns:
        List of indices of selected embeddings
    """
    if not HAS_GPU_DEPENDENCIES or not torch.cuda.is_available():
        # Fall back to CPU implementation
        return _cpu_maximal_marginal_relevance(
            query_embedding, embedding_list, lambda_mult, k
        )
    
    try:
        # Convert inputs to CuPy arrays
        query_embedding_cp = cp.array(query_embedding, dtype=cp.float32)
        embedding_mat = cp.array(embedding_list, dtype=cp.float32)
        
        # Calculate similarity scores
        similarity_to_query = cp.dot(embedding_mat, query_embedding_cp)
        
        # Select the first embedding
        most_similar_idx = int(cp.argmax(similarity_to_query).item())
        idxs = [most_similar_idx]
        selected = [embedding_list[most_similar_idx]]
        
        # Build the selected embeddings matrix incrementally
        while len(idxs) < min(k, len(embedding_list)):
            # Calculate similarity to query
            best_score = -1.0
            best_idx = -1
            
            # Create matrix of already selected embeddings
            selected_cp = cp.array(selected, dtype=cp.float32)
            
            # Calculate similarity to already selected embeddings
            similarity_to_selected = cp.dot(embedding_mat, selected_cp.T)
            
            # Get maximum similarity to already selected
            max_similarity_to_selected = cp.max(similarity_to_selected, axis=1)
            
            # Calculate MMR score
            mmr_scores = lambda_mult * similarity_to_query - (1 - lambda_mult) * max_similarity_to_selected
            
            # Mask out already selected indices
            mmr_scores[idxs] = -cp.inf
            
            # Get the next best
            next_idx = int(cp.argmax(mmr_scores).item())
            idxs.append(next_idx)
            selected.append(embedding_list[next_idx])
        
        return idxs
    
    except Exception as e:
        logger.warning(f"GPU MMR calculation failed: {str(e)}. Falling back to CPU.")
        return _cpu_maximal_marginal_relevance(
            query_embedding, embedding_list, lambda_mult, k
        )


def _cpu_maximal_marginal_relevance(
    query_embedding: np.ndarray,
    embedding_list: List[np.ndarray],
    lambda_mult: float = 0.5,
    k: int = 4,
) -> List[int]:
    """CPU fallback implementation of maximal marginal relevance."""
    if len(embedding_list) <= k:
        return list(range(len(embedding_list)))
    
    # Convert list of embeddings to a single matrix
    embeddings_array = np.array(embedding_list, dtype=np.float32)
    
    # Calculate similarity scores
    similarity_to_query = np.dot(embeddings_array, query_embedding)
    
    # Select the first embedding
    most_similar_idx = int(np.argmax(similarity_to_query))
    idxs = [most_similar_idx]
    selected = [embedding_list[most_similar_idx]]
    
    # Build the selected embeddings matrix incrementally
    while len(idxs) < min(k, len(embedding_list)):
        best_score = -1.0
        best_idx = -1
        
        for i, embedding in enumerate(embedding_list):
            if i in idxs:
                continue
            
            # Calculate similarity to query
            similarity_score = np.dot(embedding, query_embedding)
            
            # Calculate maximum similarity to already selected
            max_similarity_to_selected = max(
                np.dot(embedding, selected_embedding)
                for selected_embedding in selected
            )
            
            # Calculate MMR score
            mmr_score = lambda_mult * similarity_score - (1 - lambda_mult) * max_similarity_to_selected
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        idxs.append(best_idx)
        selected.append(embedding_list[best_idx])
    
    return idxs