"""
GPU accelerator interface for SAP HANA Cloud LangChain integration.

This module provides a unified interface for GPU-accelerated operations, including
automatic device selection, memory management, and optimal performance configurations.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

# Conditional imports
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import tensorrt as trt
    HAS_TENSORRT = True
except ImportError:
    HAS_TENSORRT = False

from langchain_hana.config.deployment import get_config
from langchain_hana.gpu.utils import detect_gpu_capabilities, get_optimal_batch_size

logger = logging.getLogger(__name__)


class AcceleratorType(str, Enum):
    """Type of GPU acceleration to use."""
    NONE = "none"
    PYTORCH = "pytorch"
    CUPY = "cupy"
    TENSORRT = "tensorrt"
    TRITON = "triton"


class MemoryStrategy(str, Enum):
    """Memory management strategy for GPU operations."""
    CONSERVATIVE = "conservative"  # Prioritize stability, use less memory
    BALANCED = "balanced"          # Balance between memory usage and performance
    AGGRESSIVE = "aggressive"      # Prioritize performance, use more memory


@dataclass
class AcceleratorConfig:
    """Configuration for GPU acceleration."""
    # Basic settings
    enabled: bool = False
    accelerator_type: AcceleratorType = AcceleratorType.NONE
    device_id: int = 0
    
    # Memory management
    memory_strategy: MemoryStrategy = MemoryStrategy.BALANCED
    max_batch_size: int = 32
    memory_fraction: float = 0.8
    
    # TensorRT settings
    tensorrt_precision: str = "fp16"
    tensorrt_workspace_size: int = 1 << 30  # 1 GB
    tensorrt_cache_dir: Optional[str] = None
    
    # Triton settings
    triton_server_url: Optional[str] = None
    triton_model_name: Optional[str] = None
    triton_timeout: int = 60


class GPUAccelerator:
    """
    Unified interface for GPU-accelerated operations.
    
    This class provides a common interface for GPU acceleration, handling
    device selection, memory management, and optimal performance configurations.
    It supports multiple acceleration backends including PyTorch, CuPy, and TensorRT.
    
    Example:
        ```python
        # Create an accelerator with default configuration
        accelerator = GPUAccelerator()
        
        # Or with specific configuration
        config = AcceleratorConfig(
            enabled=True,
            accelerator_type=AcceleratorType.PYTORCH,
            device_id=0,
            memory_strategy=MemoryStrategy.BALANCED,
        )
        accelerator = GPUAccelerator(config)
        
        # Check if acceleration is available
        if accelerator.is_available:
            # Use the accelerator for operations
            embeddings = accelerator.embed_texts(texts)
        else:
            # Fall back to CPU operations
            pass
        ```
    """
    
    def __init__(self, config: Optional[AcceleratorConfig] = None):
        """
        Initialize the GPU accelerator.
        
        Args:
            config: Accelerator configuration. If None, configuration is loaded
                   from global configuration or environment variables.
        """
        # Initialize configuration
        self.config = config or self._load_config()
        
        # Initialize state
        self.is_available = False
        self.device = None
        self.backend = None
        self.device_properties = {}
        
        # Initialize the accelerator
        self._initialize()
    
    def _load_config(self) -> AcceleratorConfig:
        """Load accelerator configuration from global config or environment."""
        # Get GPU configuration from global config
        gpu_config = get_config().gpu_config
        
        # Determine accelerator type
        accelerator_type = AcceleratorType.NONE
        if gpu_config.enabled:
            if gpu_config.enable_tensorrt and HAS_TENSORRT:
                accelerator_type = AcceleratorType.TENSORRT
            elif HAS_TORCH:
                accelerator_type = AcceleratorType.PYTORCH
            elif HAS_CUPY:
                accelerator_type = AcceleratorType.CUPY
        
        # Get platform-specific settings
        platform_settings = get_config().get_platform_specific_settings()
        triton_server_url = platform_settings.get("triton_server_url")
        triton_model_name = platform_settings.get("triton_model_name")
        
        # Check for Triton configuration
        if (
            platform_settings.get("use_triton_inference", False)
            and triton_server_url
            and triton_model_name
        ):
            accelerator_type = AcceleratorType.TRITON
        
        # Create configuration
        return AcceleratorConfig(
            enabled=gpu_config.enabled,
            accelerator_type=accelerator_type,
            device_id=int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
            memory_strategy=MemoryStrategy(
                os.environ.get("GPU_MEMORY_STRATEGY", MemoryStrategy.BALANCED)
            ),
            max_batch_size=gpu_config.max_batch_size,
            memory_fraction=1.0 - (gpu_config.memory_threshold / 100.0),
            tensorrt_precision=gpu_config.precision,
            tensorrt_cache_dir=gpu_config.tensorrt_cache_dir,
            triton_server_url=triton_server_url,
            triton_model_name=triton_model_name,
            triton_timeout=platform_settings.get("triton_timeout", 60),
        )
    
    def _initialize(self) -> None:
        """Initialize the GPU accelerator based on configuration."""
        # Skip initialization if acceleration is disabled
        if not self.config.enabled:
            logger.info("GPU acceleration is disabled")
            self.is_available = False
            return
        
        # Check for available backends
        has_any_backend = HAS_TORCH or HAS_CUPY or HAS_TENSORRT
        if not has_any_backend:
            logger.warning(
                "GPU acceleration is enabled but no supported backends are available. "
                "Install torch, cupy, or tensorrt to enable GPU acceleration."
            )
            self.is_available = False
            return
        
        # Initialize the appropriate backend
        if self.config.accelerator_type == AcceleratorType.PYTORCH:
            self._initialize_pytorch()
        elif self.config.accelerator_type == AcceleratorType.CUPY:
            self._initialize_cupy()
        elif self.config.accelerator_type == AcceleratorType.TENSORRT:
            self._initialize_tensorrt()
        elif self.config.accelerator_type == AcceleratorType.TRITON:
            self._initialize_triton()
        else:
            logger.warning(
                f"Unsupported accelerator type: {self.config.accelerator_type}. "
                "Falling back to no acceleration."
            )
            self.is_available = False
    
    def _initialize_pytorch(self) -> None:
        """Initialize PyTorch GPU acceleration."""
        if not HAS_TORCH:
            logger.warning("PyTorch not installed, cannot use PyTorch acceleration")
            self.is_available = False
            return
        
        try:
            # Check if CUDA is available
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, cannot use PyTorch acceleration")
                self.is_available = False
                return
            
            # Get device count
            device_count = torch.cuda.device_count()
            if device_count == 0:
                logger.warning("No CUDA devices found, cannot use PyTorch acceleration")
                self.is_available = False
                return
            
            # Check if requested device ID is valid
            if self.config.device_id >= device_count:
                logger.warning(
                    f"Requested device ID {self.config.device_id} is not available. "
                    f"Using device 0 instead."
                )
                self.config.device_id = 0
            
            # Initialize device
            self.device = torch.device(f"cuda:{self.config.device_id}")
            
            # Get device properties
            props = torch.cuda.get_device_properties(self.config.device_id)
            self.device_properties = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            }
            
            # Set memory strategy
            if self.config.memory_strategy == MemoryStrategy.CONSERVATIVE:
                torch.cuda.set_per_process_memory_fraction(
                    min(0.5, self.config.memory_fraction)
                )
            elif self.config.memory_strategy == MemoryStrategy.BALANCED:
                torch.cuda.set_per_process_memory_fraction(
                    min(0.7, self.config.memory_fraction)
                )
            else:  # AGGRESSIVE
                torch.cuda.set_per_process_memory_fraction(
                    min(0.9, self.config.memory_fraction)
                )
            
            # Empty cache
            torch.cuda.empty_cache()
            
            # Set backend
            self.backend = "pytorch"
            self.is_available = True
            
            logger.info(
                f"PyTorch GPU acceleration initialized on {props.name} "
                f"(Device {self.config.device_id})"
            )
        
        except Exception as e:
            logger.error(f"Failed to initialize PyTorch GPU acceleration: {str(e)}")
            self.is_available = False
    
    def _initialize_cupy(self) -> None:
        """Initialize CuPy GPU acceleration."""
        if not HAS_CUPY:
            logger.warning("CuPy not installed, cannot use CuPy acceleration")
            self.is_available = False
            return
        
        try:
            # Check if CUDA is available
            if cp.cuda.runtime.getDeviceCount() == 0:
                logger.warning("No CUDA devices found, cannot use CuPy acceleration")
                self.is_available = False
                return
            
            # Get device count
            device_count = cp.cuda.runtime.getDeviceCount()
            
            # Check if requested device ID is valid
            if self.config.device_id >= device_count:
                logger.warning(
                    f"Requested device ID {self.config.device_id} is not available. "
                    f"Using device 0 instead."
                )
                self.config.device_id = 0
            
            # Initialize device
            cp.cuda.Device(self.config.device_id).use()
            
            # Get device properties
            device = cp.cuda.Device(self.config.device_id)
            attributes = device.attributes
            
            self.device_properties = {
                "name": device.name,
                "compute_capability": f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                "total_memory": device.mem_total,
                "multi_processor_count": attributes.get("MultiProcessorCount", 0),
            }
            
            # Set memory pool limits based on memory strategy
            pool = cp.get_default_memory_pool()
            if self.config.memory_strategy == MemoryStrategy.CONSERVATIVE:
                fraction = min(0.5, self.config.memory_fraction)
            elif self.config.memory_strategy == MemoryStrategy.BALANCED:
                fraction = min(0.7, self.config.memory_fraction)
            else:  # AGGRESSIVE
                fraction = min(0.9, self.config.memory_fraction)
            
            limit = int(device.mem_total * fraction)
            pool.set_limit(limit)
            
            # Set backend
            self.backend = "cupy"
            self.is_available = True
            
            logger.info(
                f"CuPy GPU acceleration initialized on {device.name} "
                f"(Device {self.config.device_id})"
            )
        
        except Exception as e:
            logger.error(f"Failed to initialize CuPy GPU acceleration: {str(e)}")
            self.is_available = False
    
    def _initialize_tensorrt(self) -> None:
        """Initialize TensorRT GPU acceleration."""
        if not HAS_TENSORRT:
            logger.warning("TensorRT not installed, cannot use TensorRT acceleration")
            self.is_available = False
            return
        
        try:
            # Check if CUDA is available (via PyTorch, which is required for TensorRT)
            if not HAS_TORCH or not torch.cuda.is_available():
                logger.warning(
                    "CUDA not available via PyTorch, cannot use TensorRT acceleration. "
                    "TensorRT requires PyTorch for CUDA context management."
                )
                self.is_available = False
                return
            
            # Get device count
            device_count = torch.cuda.device_count()
            
            # Check if requested device ID is valid
            if self.config.device_id >= device_count:
                logger.warning(
                    f"Requested device ID {self.config.device_id} is not available. "
                    f"Using device 0 instead."
                )
                self.config.device_id = 0
            
            # Initialize device
            torch.cuda.set_device(self.config.device_id)
            
            # Get device properties
            props = torch.cuda.get_device_properties(self.config.device_id)
            self.device_properties = {
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
            }
            
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Create TensorRT runtime
            runtime = trt.Runtime(trt_logger)
            
            # Ensure cache directory exists
            if self.config.tensorrt_cache_dir:
                os.makedirs(self.config.tensorrt_cache_dir, exist_ok=True)
            
            # Set backend
            self.backend = "tensorrt"
            self.is_available = True
            
            logger.info(
                f"TensorRT GPU acceleration initialized on {props.name} "
                f"(Device {self.config.device_id})"
            )
        
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT GPU acceleration: {str(e)}")
            self.is_available = False
    
    def _initialize_triton(self) -> None:
        """Initialize Triton Inference Server acceleration."""
        if not self.config.triton_server_url:
            logger.warning("Triton server URL not configured, cannot use Triton acceleration")
            self.is_available = False
            return
        
        try:
            # Import tritonclient
            import tritonclient.http as triton_http
            
            # Parse server URL
            server_url = self.config.triton_server_url
            if not server_url.startswith(("http://", "https://")):
                server_url = f"http://{server_url}"
            
            # Create client
            client = triton_http.InferenceServerClient(
                url=server_url,
                verbose=False,
                connection_timeout=self.config.triton_timeout,
                network_timeout=self.config.triton_timeout,
            )
            
            # Check server readiness
            if not client.is_server_ready():
                logger.warning("Triton server is not ready, cannot use Triton acceleration")
                self.is_available = False
                return
            
            # Check model readiness
            if self.config.triton_model_name:
                if not client.is_model_ready(self.config.triton_model_name):
                    logger.warning(
                        f"Model '{self.config.triton_model_name}' is not ready on Triton server, "
                        "cannot use Triton acceleration"
                    )
                    self.is_available = False
                    return
            
            # Get server metadata
            server_metadata = client.get_server_metadata()
            
            # Store device properties
            self.device_properties = {
                "name": "Triton Inference Server",
                "server_version": server_metadata.version,
                "server_extensions": server_metadata.extensions,
            }
            
            # Set backend
            self.backend = "triton"
            self.is_available = True
            
            logger.info(
                f"Triton Inference Server acceleration initialized at {server_url}"
            )
        
        except ImportError:
            logger.warning(
                "tritonclient not installed, cannot use Triton acceleration. "
                "Install with: pip install tritonclient[http]"
            )
            self.is_available = False
        except Exception as e:
            logger.error(f"Failed to initialize Triton acceleration: {str(e)}")
            self.is_available = False
    
    def get_optimal_batch_size(self, vector_dim: int) -> int:
        """
        Get the optimal batch size for the current device and configuration.
        
        Args:
            vector_dim: Dimension of vectors to process
            
        Returns:
            int: Optimal batch size
        """
        if not self.is_available:
            return 1
        
        # Get free memory
        free_memory_mb = self._get_free_memory_mb()
        
        # Get optimal batch size
        return get_optimal_batch_size(
            vector_dim=vector_dim,
            available_memory_mb=free_memory_mb,
            headroom_percent=20.0,
            min_batch_size=1,
            max_batch_size=self.config.max_batch_size,
        )
    
    def _get_free_memory_mb(self) -> int:
        """Get free memory on the current device in MB."""
        if not self.is_available:
            return 0
        
        if self.backend == "pytorch":
            free_memory, total_memory = torch.cuda.mem_get_info(self.config.device_id)
            return free_memory // (1024 * 1024)
        elif self.backend == "cupy":
            device = cp.cuda.Device(self.config.device_id)
            free_memory, total_memory = device.mem_info
            return free_memory // (1024 * 1024)
        elif self.backend == "tensorrt":
            # TensorRT uses PyTorch for memory management
            free_memory, total_memory = torch.cuda.mem_get_info(self.config.device_id)
            return free_memory // (1024 * 1024)
        elif self.backend == "triton":
            # Triton server runs on a remote machine, so we can't get its memory info
            return 1024  # Assume 1GB available
        
        return 0
    
    def run_with_auto_batch(
        self,
        operation: Callable[[List[Any]], List[Any]],
        items: List[Any],
        vector_dim: int,
    ) -> List[Any]:
        """
        Run an operation with automatic batch size determination.
        
        Args:
            operation: Function that processes a batch of items
            items: Items to process
            vector_dim: Dimension of vectors to process
            
        Returns:
            List[Any]: Results of the operation
        """
        if not items:
            return []
        
        # Determine batch size
        batch_size = self.get_optimal_batch_size(vector_dim)
        
        # Process in batches
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            batch_results = operation(batch)
            results.extend(batch_results)
        
        return results
    
    def to_device(self, data: Any) -> Any:
        """
        Move data to the GPU device.
        
        Args:
            data: Data to move to device
            
        Returns:
            Data on the device
        """
        if not self.is_available:
            return data
        
        if self.backend == "pytorch":
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.tensor(data, device=self.device)
            elif isinstance(data, list):
                if isinstance(data[0], np.ndarray):
                    return torch.tensor(np.array(data), device=self.device)
                elif isinstance(data[0], list):
                    return torch.tensor(np.array(data), device=self.device)
                elif isinstance(data[0], (int, float)):
                    return torch.tensor(data, device=self.device)
            return data
        
        elif self.backend == "cupy":
            if isinstance(data, cp.ndarray):
                return data
            elif isinstance(data, np.ndarray):
                return cp.array(data)
            elif isinstance(data, list):
                if isinstance(data[0], np.ndarray):
                    return cp.array(np.array(data))
                elif isinstance(data[0], list):
                    return cp.array(np.array(data))
                elif isinstance(data[0], (int, float)):
                    return cp.array(data)
            return data
        
        elif self.backend == "tensorrt":
            # TensorRT uses PyTorch tensors for input/output
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.tensor(data, device=self.device)
            elif isinstance(data, list):
                if isinstance(data[0], np.ndarray):
                    return torch.tensor(np.array(data), device=self.device)
                elif isinstance(data[0], list):
                    return torch.tensor(np.array(data), device=self.device)
                elif isinstance(data[0], (int, float)):
                    return torch.tensor(data, device=self.device)
            return data
        
        return data
    
    def to_numpy(self, data: Any) -> np.ndarray:
        """
        Convert data from device to NumPy array.
        
        Args:
            data: Data to convert
            
        Returns:
            np.ndarray: NumPy array
        """
        if not self.is_available:
            if isinstance(data, np.ndarray):
                return data
            elif isinstance(data, list):
                return np.array(data)
            return data
        
        if self.backend == "pytorch":
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, list):
                return np.array(data)
            return data
        
        elif self.backend == "cupy":
            if isinstance(data, cp.ndarray):
                return cp.asnumpy(data)
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, list):
                return np.array(data)
            return data
        
        elif self.backend == "tensorrt":
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            elif isinstance(data, np.ndarray):
                return data
            elif isinstance(data, list):
                return np.array(data)
            return data
        
        return data
    
    def normalize_vectors(self, vectors: Any) -> Any:
        """
        Normalize vectors to unit length.
        
        Args:
            vectors: Vectors to normalize
            
        Returns:
            Normalized vectors
        """
        if not self.is_available:
            vectors_np = np.array(vectors)
            norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
            return vectors_np / np.maximum(norms, 1e-12)
        
        if self.backend == "pytorch":
            vectors_t = self.to_device(vectors)
            norms = torch.norm(vectors_t, dim=1, keepdim=True)
            normalized = vectors_t / torch.clamp(norms, min=1e-12)
            return normalized
        
        elif self.backend == "cupy":
            vectors_cp = self.to_device(vectors)
            norms = cp.linalg.norm(vectors_cp, axis=1, keepdims=True)
            normalized = vectors_cp / cp.maximum(norms, 1e-12)
            return normalized
        
        elif self.backend == "tensorrt":
            vectors_t = self.to_device(vectors)
            norms = torch.norm(vectors_t, dim=1, keepdim=True)
            normalized = vectors_t / torch.clamp(norms, min=1e-12)
            return normalized
        
        vectors_np = np.array(vectors)
        norms = np.linalg.norm(vectors_np, axis=1, keepdims=True)
        return vectors_np / np.maximum(norms, 1e-12)
    
    def vector_similarity(
        self, query_vector: Any, vectors: Any, k: int = 4
    ) -> Tuple[List[int], List[float]]:
        """
        Calculate vector similarity and return top-k indices and scores.
        
        Args:
            query_vector: Query vector
            vectors: Vectors to compare against
            k: Number of top results to return
            
        Returns:
            Tuple of (indices, scores) for top-k results
        """
        if not self.is_available:
            query_np = np.array(query_vector)
            vectors_np = np.array(vectors)
            
            # Calculate cosine similarity
            similarities = np.dot(vectors_np, query_np)
            
            # Find top-k
            if k >= len(similarities):
                indices = list(range(len(similarities)))
                scores = similarities.tolist()
                # Sort by score in descending order
                pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
                indices, scores = zip(*pairs) if pairs else ([], [])
                return list(indices), list(scores)
            
            # Find top-k indices
            indices = np.argsort(-similarities)[:k]
            scores = similarities[indices]
            
            return indices.tolist(), scores.tolist()
        
        if self.backend == "pytorch":
            query_t = self.to_device(query_vector)
            vectors_t = self.to_device(vectors)
            
            # Calculate cosine similarity
            similarities = torch.matmul(vectors_t, query_t)
            
            # Find top-k
            if k >= len(similarities):
                indices = torch.arange(len(similarities), device=self.device)
                scores = similarities
                # Sort by score in descending order
                values, indices = torch.sort(scores, descending=True)
                return indices.cpu().tolist(), values.cpu().tolist()
            
            # Find top-k indices
            values, indices = torch.topk(similarities, k)
            
            return indices.cpu().tolist(), values.cpu().tolist()
        
        elif self.backend == "cupy":
            query_cp = self.to_device(query_vector)
            vectors_cp = self.to_device(vectors)
            
            # Calculate cosine similarity
            similarities = cp.dot(vectors_cp, query_cp)
            
            # Find top-k
            if k >= len(similarities):
                indices = cp.arange(len(similarities))
                scores = similarities
                # Sort by score in descending order
                order = cp.argsort(-scores)
                return cp.asnumpy(indices[order]).tolist(), cp.asnumpy(scores[order]).tolist()
            
            # Find top-k indices
            indices = cp.argsort(-similarities)[:k]
            scores = similarities[indices]
            
            return cp.asnumpy(indices).tolist(), cp.asnumpy(scores).tolist()
        
        elif self.backend == "tensorrt":
            # TensorRT uses PyTorch for this operation
            query_t = self.to_device(query_vector)
            vectors_t = self.to_device(vectors)
            
            # Calculate cosine similarity
            similarities = torch.matmul(vectors_t, query_t)
            
            # Find top-k
            if k >= len(similarities):
                indices = torch.arange(len(similarities), device=self.device)
                scores = similarities
                # Sort by score in descending order
                values, indices = torch.sort(scores, descending=True)
                return indices.cpu().tolist(), values.cpu().tolist()
            
            # Find top-k indices
            values, indices = torch.topk(similarities, k)
            
            return indices.cpu().tolist(), values.cpu().tolist()
        
        # Fall back to NumPy
        query_np = np.array(query_vector)
        vectors_np = np.array(vectors)
        
        # Calculate cosine similarity
        similarities = np.dot(vectors_np, query_np)
        
        # Find top-k
        if k >= len(similarities):
            indices = list(range(len(similarities)))
            scores = similarities.tolist()
            # Sort by score in descending order
            pairs = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
            indices, scores = zip(*pairs) if pairs else ([], [])
            return list(indices), list(scores)
        
        # Find top-k indices
        indices = np.argsort(-similarities)[:k]
        scores = similarities[indices]
        
        return indices.tolist(), scores.tolist()
    
    def maximal_marginal_relevance(
        self,
        query_vector: Any,
        vectors: Any,
        k: int = 4,
        lambda_mult: float = 0.5,
    ) -> List[int]:
        """
        Calculate maximal marginal relevance to get diverse results.
        
        Args:
            query_vector: Query vector
            vectors: Vectors to compare against
            k: Number of results to return
            lambda_mult: Diversity parameter (0 = max diversity, 1 = max relevance)
            
        Returns:
            List[int]: Indices of selected vectors
        """
        if not self.is_available:
            # Fall back to NumPy implementation
            query_np = np.array(query_vector)
            vectors_np = np.array(vectors)
            
            # If fewer vectors than k, return all indices
            if len(vectors_np) <= k:
                return list(range(len(vectors_np)))
            
            # Calculate similarity to query
            similarity_to_query = np.dot(vectors_np, query_np)
            
            # Initialize selection
            selected_indices = []
            selected_vectors = []
            
            # Select first vector (most similar to query)
            first_idx = np.argmax(similarity_to_query)
            selected_indices.append(first_idx)
            selected_vectors.append(vectors_np[first_idx])
            
            # Select remaining vectors
            while len(selected_indices) < k:
                # Calculate MMR scores
                mmr_scores = np.zeros(len(vectors_np))
                
                for i in range(len(vectors_np)):
                    if i in selected_indices:
                        mmr_scores[i] = float('-inf')
                        continue
                    
                    # Calculate similarity to query
                    query_similarity = similarity_to_query[i]
                    
                    # Calculate maximum similarity to selected vectors
                    max_similarity = max(
                        np.dot(vectors_np[i], selected_vector)
                        for selected_vector in selected_vectors
                    )
                    
                    # Calculate MMR score
                    mmr_scores[i] = lambda_mult * query_similarity - (1 - lambda_mult) * max_similarity
                
                # Select vector with highest MMR score
                next_idx = np.argmax(mmr_scores)
                selected_indices.append(next_idx)
                selected_vectors.append(vectors_np[next_idx])
            
            return selected_indices
        
        if self.backend == "pytorch":
            query_t = self.to_device(query_vector)
            vectors_t = self.to_device(vectors)
            
            # If fewer vectors than k, return all indices
            if len(vectors_t) <= k:
                return list(range(len(vectors_t)))
            
            # Calculate similarity to query
            similarity_to_query = torch.matmul(vectors_t, query_t)
            
            # Initialize selection
            selected_indices = []
            selected_vectors = []
            
            # Select first vector (most similar to query)
            first_idx = torch.argmax(similarity_to_query).item()
            selected_indices.append(first_idx)
            selected_vectors.append(vectors_t[first_idx].unsqueeze(0))
            
            # Select remaining vectors
            while len(selected_indices) < k:
                # Calculate MMR scores
                mmr_scores = torch.full(
                    (len(vectors_t),), float('-inf'), device=self.device
                )
                
                # Create mask for unselected vectors
                mask = torch.ones(len(vectors_t), dtype=torch.bool, device=self.device)
                for idx in selected_indices:
                    mask[idx] = False
                
                # Get unselected vectors
                unselected_indices = torch.nonzero(mask).squeeze(-1)
                
                if len(unselected_indices) == 0:
                    break
                
                # Calculate similarity to selected vectors for unselected vectors
                selected_matrix = torch.cat(selected_vectors, dim=0)
                similarity_to_selected = torch.matmul(
                    vectors_t[unselected_indices], selected_matrix.t()
                )
                
                # Get maximum similarity for each unselected vector
                max_similarity_to_selected, _ = torch.max(similarity_to_selected, dim=1)
                
                # Calculate MMR scores
                mmr_scores[unselected_indices] = (
                    lambda_mult * similarity_to_query[unselected_indices]
                    - (1 - lambda_mult) * max_similarity_to_selected
                )
                
                # Select vector with highest MMR score
                next_idx = torch.argmax(mmr_scores).item()
                selected_indices.append(next_idx)
                selected_vectors.append(vectors_t[next_idx].unsqueeze(0))
            
            return selected_indices
        
        elif self.backend == "cupy":
            query_cp = self.to_device(query_vector)
            vectors_cp = self.to_device(vectors)
            
            # If fewer vectors than k, return all indices
            if len(vectors_cp) <= k:
                return list(range(len(vectors_cp)))
            
            # Calculate similarity to query
            similarity_to_query = cp.dot(vectors_cp, query_cp)
            
            # Initialize selection
            selected_indices = []
            selected_vectors = []
            
            # Select first vector (most similar to query)
            first_idx = int(cp.argmax(similarity_to_query).item())
            selected_indices.append(first_idx)
            selected_vectors.append(vectors_cp[first_idx])
            
            # Select remaining vectors
            while len(selected_indices) < k:
                # Calculate MMR scores
                mmr_scores = cp.full((len(vectors_cp),), float('-inf'))
                
                for i in range(len(vectors_cp)):
                    if i in selected_indices:
                        continue
                    
                    # Calculate similarity to query
                    query_similarity = similarity_to_query[i]
                    
                    # Calculate similarity to selected vectors
                    similarities = cp.array([
                        cp.dot(vectors_cp[i], selected_vector)
                        for selected_vector in selected_vectors
                    ])
                    
                    # Calculate maximum similarity
                    max_similarity = cp.max(similarities)
                    
                    # Calculate MMR score
                    mmr_scores[i] = lambda_mult * query_similarity - (1 - lambda_mult) * max_similarity
                
                # Select vector with highest MMR score
                next_idx = int(cp.argmax(mmr_scores).item())
                selected_indices.append(next_idx)
                selected_vectors.append(vectors_cp[next_idx])
            
            return selected_indices
        
        elif self.backend == "tensorrt":
            # TensorRT uses PyTorch for this operation
            query_t = self.to_device(query_vector)
            vectors_t = self.to_device(vectors)
            
            # If fewer vectors than k, return all indices
            if len(vectors_t) <= k:
                return list(range(len(vectors_t)))
            
            # Calculate similarity to query
            similarity_to_query = torch.matmul(vectors_t, query_t)
            
            # Initialize selection
            selected_indices = []
            selected_vectors = []
            
            # Select first vector (most similar to query)
            first_idx = torch.argmax(similarity_to_query).item()
            selected_indices.append(first_idx)
            selected_vectors.append(vectors_t[first_idx].unsqueeze(0))
            
            # Select remaining vectors
            while len(selected_indices) < k:
                # Calculate MMR scores
                mmr_scores = torch.full(
                    (len(vectors_t),), float('-inf'), device=self.device
                )
                
                # Create mask for unselected vectors
                mask = torch.ones(len(vectors_t), dtype=torch.bool, device=self.device)
                for idx in selected_indices:
                    mask[idx] = False
                
                # Get unselected vectors
                unselected_indices = torch.nonzero(mask).squeeze(-1)
                
                if len(unselected_indices) == 0:
                    break
                
                # Calculate similarity to selected vectors for unselected vectors
                selected_matrix = torch.cat(selected_vectors, dim=0)
                similarity_to_selected = torch.matmul(
                    vectors_t[unselected_indices], selected_matrix.t()
                )
                
                # Get maximum similarity for each unselected vector
                max_similarity_to_selected, _ = torch.max(similarity_to_selected, dim=1)
                
                # Calculate MMR scores
                mmr_scores[unselected_indices] = (
                    lambda_mult * similarity_to_query[unselected_indices]
                    - (1 - lambda_mult) * max_similarity_to_selected
                )
                
                # Select vector with highest MMR score
                next_idx = torch.argmax(mmr_scores).item()
                selected_indices.append(next_idx)
                selected_vectors.append(vectors_t[next_idx].unsqueeze(0))
            
            return selected_indices
        
        # Fall back to NumPy implementation
        query_np = np.array(query_vector)
        vectors_np = np.array(vectors)
        
        # If fewer vectors than k, return all indices
        if len(vectors_np) <= k:
            return list(range(len(vectors_np)))
        
        # Calculate similarity to query
        similarity_to_query = np.dot(vectors_np, query_np)
        
        # Initialize selection
        selected_indices = []
        selected_vectors = []
        
        # Select first vector (most similar to query)
        first_idx = np.argmax(similarity_to_query)
        selected_indices.append(first_idx)
        selected_vectors.append(vectors_np[first_idx])
        
        # Select remaining vectors
        while len(selected_indices) < k:
            # Calculate MMR scores
            mmr_scores = np.zeros(len(vectors_np))
            
            for i in range(len(vectors_np)):
                if i in selected_indices:
                    mmr_scores[i] = float('-inf')
                    continue
                
                # Calculate similarity to query
                query_similarity = similarity_to_query[i]
                
                # Calculate maximum similarity to selected vectors
                max_similarity = max(
                    np.dot(vectors_np[i], selected_vector)
                    for selected_vector in selected_vectors
                )
                
                # Calculate MMR score
                mmr_scores[i] = lambda_mult * query_similarity - (1 - lambda_mult) * max_similarity
            
            # Select vector with highest MMR score
            next_idx = np.argmax(mmr_scores)
            selected_indices.append(next_idx)
            selected_vectors.append(vectors_np[next_idx])
        
        return selected_indices