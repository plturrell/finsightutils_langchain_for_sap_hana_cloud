"""
GPU-aware Arrow memory manager for optimized data transfer.

This module provides utilities for efficient memory management 
between Apache Arrow and GPU operations, enabling zero-copy
data transfer where possible.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pyarrow as pa
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    from numba import cuda
    HAS_NUMBA_CUDA = True
except ImportError:
    HAS_NUMBA_CUDA = False

from .utils import get_available_memory

logger = logging.getLogger(__name__)


class ArrowGpuMemoryManager:
    """
    GPU-aware memory manager for Apache Arrow data.
    
    This class provides utilities for efficient memory management and data transfer
    between Apache Arrow columnar format and GPU operations, enabling zero-copy
    transfers where possible and optimizing memory usage for vector operations.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        max_memory_fraction: float = 0.8,
        enable_pinned_memory: bool = True,
        enable_tensor_cores: bool = True,
        batch_size: int = 1024,
        precision: str = "float32"
    ):
        """
        Initialize the GPU-aware Arrow memory manager.
        
        Args:
            device_id: GPU device ID (default: 0)
            max_memory_fraction: Maximum fraction of GPU memory to use (default: 0.8)
            enable_pinned_memory: Whether to use pinned CPU memory for faster transfers
            enable_tensor_cores: Whether to enable tensor core operations when available
            batch_size: Default batch size for operations
            precision: Default precision for vector operations ("float32", "float16", "int8")
            
        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If GPU is not available
        """
        if not HAS_ARROW:
            raise ImportError(
                "The pyarrow package is required for Arrow GPU memory management. "
                "Install it with 'pip install pyarrow'."
            )
        
        self.device_id = device_id
        self.max_memory_fraction = max_memory_fraction
        self.enable_pinned_memory = enable_pinned_memory
        self.enable_tensor_cores = enable_tensor_cores
        self.batch_size = batch_size
        self.precision = precision
        
        # Track allocated GPU buffers
        self._gpu_buffers = {}
        self._pinned_buffers = {}
        
        # Initialize available frameworks
        self._initialize_gpu()
    
    def _initialize_gpu(self):
        """Initialize GPU and available frameworks."""
        self.has_torch = HAS_TORCH
        self.has_cupy = HAS_CUPY
        self.has_numba = HAS_NUMBA_CUDA
        
        # Check for CUDA availability
        self.cuda_available = False
        self.tensor_cores_available = False
        
        if self.has_torch and torch.cuda.is_available():
            self.cuda_available = True
            
            # Set PyTorch device
            self.torch_device = torch.device(f"cuda:{self.device_id}")
            torch.cuda.set_device(self.device_id)
            
            # Check for tensor cores
            if self.enable_tensor_cores:
                device_capability = torch.cuda.get_device_capability(self.device_id)
                self.tensor_cores_available = device_capability[0] >= 7
                
                if self.tensor_cores_available:
                    logger.info(f"Tensor Cores are available on device {self.device_id}")
                    
                    # Configure PyTorch for tensor cores if available
                    if hasattr(torch.backends.cudnn, 'allow_tf32'):
                        torch.backends.cudnn.allow_tf32 = True
                    if hasattr(torch, 'set_float32_matmul_precision'):
                        torch.set_float32_matmul_precision('high')
        
        if not self.cuda_available:
            logger.warning("CUDA is not available. GPU acceleration will not be used.")
            self.has_cupy = False
            self.has_numba = False
    
    def get_arrow_device_type(self) -> Optional[pa.DeviceType]:
        """
        Get the appropriate Arrow device type for current configuration.
        
        Returns:
            Arrow device type if CUDA is available, None otherwise
        """
        if not HAS_ARROW:
            return None
            
        # Check if CUDA device type is available in Arrow
        try:
            return pa.cuda.DeviceType.CUDA
        except (AttributeError, ImportError):
            logger.warning("Arrow CUDA device type not available")
            return None
    
    def get_optimal_batch_size(self) -> int:
        """
        Calculate the optimal batch size based on available GPU memory.
        
        Returns:
            Optimal batch size for vector operations
        """
        if not self.cuda_available:
            return self.batch_size
            
        try:
            # Get available GPU memory
            available_memory = get_available_memory(self.device_id)
            
            # Calculate memory per vector (with some overhead)
            bytes_per_value = 4 if self.precision == "float32" else 2  # float32 or float16
            estimated_dim = 768  # Common embedding dimension, adjust if known
            
            # Memory for one vector
            memory_per_vector = bytes_per_value * estimated_dim * 1.2  # 20% overhead
            
            # Calculate max batch size
            max_batch_size = int((available_memory * self.max_memory_fraction) / memory_per_vector)
            
            # Limit to reasonable range
            optimal_batch_size = max(1, min(max_batch_size, 16384))
            
            logger.debug(f"Calculated optimal batch size: {optimal_batch_size}")
            return optimal_batch_size
            
        except Exception as e:
            logger.warning(f"Error calculating optimal batch size: {str(e)}")
            return self.batch_size
    
    def arrow_to_torch(
        self, 
        arrow_array: Union[pa.Array, pa.ChunkedArray],
        device: Optional[torch.device] = None,
        non_blocking: bool = True
    ) -> torch.Tensor:
        """
        Convert an Arrow array to a PyTorch tensor, optimizing for GPU transfer.
        
        Args:
            arrow_array: Arrow array or chunked array to convert
            device: PyTorch device (default: self.torch_device if CUDA is available)
            non_blocking: Whether to use asynchronous transfer
            
        Returns:
            PyTorch tensor
        """
        if not self.has_torch:
            raise ImportError("PyTorch is required for torch tensor conversion")
            
        device = device or (self.torch_device if self.cuda_available else torch.device("cpu"))
        
        try:
            # Handle chunked arrays
            if isinstance(arrow_array, pa.ChunkedArray):
                if len(arrow_array.chunks) == 0:
                    # Empty array
                    return torch.empty(0, device=device)
                elif len(arrow_array.chunks) == 1:
                    # Single chunk
                    arrow_array = arrow_array.chunks[0]
                else:
                    # Multiple chunks - concatenate them
                    tensor_chunks = [self.arrow_to_torch(chunk, device) for chunk in arrow_array.chunks]
                    return torch.cat(tensor_chunks)
            
            # Convert to numpy array first
            if self.enable_pinned_memory and device.type == "cuda":
                # Use pinned memory for faster GPU transfer
                np_array = arrow_array.to_numpy()
                pinned_array = torch.tensor(np_array, pin_memory=True)
                return pinned_array.to(device, non_blocking=non_blocking)
            else:
                # Direct conversion
                np_array = arrow_array.to_numpy()
                return torch.tensor(np_array, device=device)
                
        except Exception as e:
            logger.error(f"Error converting Arrow array to torch tensor: {str(e)}")
            # Fallback to standard conversion
            np_array = arrow_array.to_numpy()
            return torch.tensor(np_array, device=device)
    
    def torch_to_arrow(
        self, 
        tensor: torch.Tensor,
        type_: Optional[pa.DataType] = None
    ) -> pa.Array:
        """
        Convert a PyTorch tensor to an Arrow array, optimizing for GPU transfer.
        
        Args:
            tensor: PyTorch tensor to convert
            type_: Optional Arrow data type
            
        Returns:
            Arrow array
        """
        if not self.has_torch:
            raise ImportError("PyTorch is required for torch tensor conversion")
            
        try:
            # Move to CPU if on GPU
            if tensor.device.type == "cuda":
                # Use pinned memory for faster transfer if enabled
                if self.enable_pinned_memory:
                    tensor = tensor.pin_memory()
                tensor = tensor.cpu()
            
            # Convert to numpy and then to Arrow
            np_array = tensor.numpy()
            
            if type_ is not None:
                return pa.array(np_array, type=type_)
            else:
                return pa.array(np_array)
                
        except Exception as e:
            logger.error(f"Error converting torch tensor to Arrow array: {str(e)}")
            # Fallback to standard conversion
            np_array = tensor.detach().cpu().numpy()
            return pa.array(np_array)
    
    def arrow_record_batch_to_torch_dict(
        self, 
        batch: pa.RecordBatch,
        device: Optional[torch.device] = None,
        columns: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convert an Arrow RecordBatch to a dictionary of PyTorch tensors.
        
        Args:
            batch: Arrow RecordBatch to convert
            device: PyTorch device (default: self.torch_device if CUDA is available)
            columns: Optional list of column names to include
            
        Returns:
            Dictionary of PyTorch tensors
        """
        if not self.has_torch:
            raise ImportError("PyTorch is required for torch tensor conversion")
            
        device = device or (self.torch_device if self.cuda_available else torch.device("cpu"))
        
        result = {}
        col_names = columns or batch.schema.names
        
        for name in col_names:
            if name in batch.schema.names:
                col_idx = batch.schema.get_field_index(name)
                col = batch.column(col_idx)
                result[name] = self.arrow_to_torch(col, device)
        
        return result
    
    def torch_dict_to_arrow_record_batch(
        self, 
        tensor_dict: Dict[str, torch.Tensor],
        schema: Optional[pa.Schema] = None
    ) -> pa.RecordBatch:
        """
        Convert a dictionary of PyTorch tensors to an Arrow RecordBatch.
        
        Args:
            tensor_dict: Dictionary of PyTorch tensors
            schema: Optional Arrow schema
            
        Returns:
            Arrow RecordBatch
        """
        if not self.has_torch:
            raise ImportError("PyTorch is required for torch tensor conversion")
            
        arrays = []
        names = []
        
        for name, tensor in tensor_dict.items():
            arrays.append(self.torch_to_arrow(tensor))
            names.append(name)
        
        if schema is not None:
            return pa.RecordBatch.from_arrays(arrays, schema=schema)
        else:
            return pa.RecordBatch.from_arrays(arrays, names=names)
    
    def vectors_to_fixed_size_list_array(
        self, 
        vectors: Union[List[List[float]], np.ndarray, torch.Tensor]
    ) -> pa.FixedSizeListArray:
        """
        Convert vectors to an Arrow FixedSizeListArray for efficient transfer.
        
        Args:
            vectors: Vectors as list of lists, numpy array, or PyTorch tensor
            
        Returns:
            Arrow FixedSizeListArray
        """
        if isinstance(vectors, torch.Tensor):
            # Convert PyTorch tensor to numpy
            if vectors.device.type == "cuda":
                vectors = vectors.cpu()
            vectors = vectors.numpy()
        
        if isinstance(vectors, np.ndarray):
            # Ensure 2D array
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
                
            # Create Arrow array from flattened numpy array
            flat_array = pa.array(vectors.flatten(), type=pa.float32())
            return pa.FixedSizeListArray.from_arrays(flat_array, vectors.shape[1])
        
        else:
            # Handle list of lists
            if not vectors:
                raise ValueError("Empty vector list provided")
                
            # Determine dimension from first vector
            dim = len(vectors[0])
            
            # Convert to numpy for efficiency
            np_array = np.array(vectors, dtype=np.float32)
            flat_array = pa.array(np_array.flatten(), type=pa.float32())
            
            return pa.FixedSizeListArray.from_arrays(flat_array, dim)
    
    def fixed_size_list_array_to_numpy(
        self, 
        array: pa.FixedSizeListArray
    ) -> np.ndarray:
        """
        Convert an Arrow FixedSizeListArray to a numpy array efficiently.
        
        Args:
            array: Arrow FixedSizeListArray to convert
            
        Returns:
            Numpy array with shape (n_vectors, vector_dim)
        """
        # Get vector dimension
        vector_dim = array.type.list_size
        
        # Get flattened values and reshape
        flat_values = array.flatten().to_numpy()
        return flat_values.reshape(-1, vector_dim)
    
    def fixed_size_list_array_to_torch(
        self, 
        array: pa.FixedSizeListArray,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """
        Convert an Arrow FixedSizeListArray to a PyTorch tensor efficiently.
        
        Args:
            array: Arrow FixedSizeListArray to convert
            device: PyTorch device (default: self.torch_device if CUDA is available)
            dtype: PyTorch data type
            
        Returns:
            PyTorch tensor with shape (n_vectors, vector_dim)
        """
        if not self.has_torch:
            raise ImportError("PyTorch is required for torch tensor conversion")
            
        device = device or (self.torch_device if self.cuda_available else torch.device("cpu"))
        dtype = dtype or torch.float32
        
        # Convert to numpy first
        np_array = self.fixed_size_list_array_to_numpy(array)
        
        # Use pinned memory for faster GPU transfer if enabled
        if self.enable_pinned_memory and device.type == "cuda":
            tensor = torch.tensor(np_array, pin_memory=True, dtype=dtype)
            return tensor.to(device, non_blocking=True)
        else:
            return torch.tensor(np_array, device=device, dtype=dtype)
    
    def allocate_pinned_buffer(
        self, 
        shape: Tuple[int, ...],
        dtype: np.dtype = np.float32,
        buffer_id: Optional[str] = None
    ) -> np.ndarray:
        """
        Allocate a pinned CPU memory buffer for faster GPU transfers.
        
        Args:
            shape: Buffer shape
            dtype: Numpy data type
            buffer_id: Optional ID for tracking the buffer
            
        Returns:
            Numpy array in pinned memory
        """
        if not self.has_torch:
            raise ImportError("PyTorch is required for pinned memory allocation")
            
        # Create torch tensor in pinned memory
        torch_tensor = torch.empty(shape, dtype=torch.from_numpy(np.array(0, dtype=dtype)).dtype, pin_memory=True)
        
        # Convert to numpy array (still pinned)
        pinned_array = torch_tensor.numpy()
        
        # Track buffer if ID provided
        if buffer_id is not None:
            self._pinned_buffers[buffer_id] = pinned_array
            
        return pinned_array
    
    def allocate_gpu_buffer(
        self, 
        shape: Tuple[int, ...],
        dtype: Union[torch.dtype, np.dtype] = torch.float32,
        buffer_id: Optional[str] = None
    ) -> torch.Tensor:
        """
        Allocate a GPU memory buffer.
        
        Args:
            shape: Buffer shape
            dtype: PyTorch or numpy data type
            buffer_id: Optional ID for tracking the buffer
            
        Returns:
            PyTorch tensor on GPU
        """
        if not self.has_torch or not self.cuda_available:
            raise RuntimeError("PyTorch with CUDA is required for GPU buffer allocation")
            
        # Convert numpy dtype to torch dtype if needed
        if isinstance(dtype, np.dtype):
            if np.issubdtype(dtype, np.float32):
                dtype = torch.float32
            elif np.issubdtype(dtype, np.float16):
                dtype = torch.float16
            elif np.issubdtype(dtype, np.int32):
                dtype = torch.int32
            elif np.issubdtype(dtype, np.int64):
                dtype = torch.int64
            else:
                dtype = torch.float32
        
        # Allocate GPU tensor
        gpu_tensor = torch.empty(shape, dtype=dtype, device=self.torch_device)
        
        # Track buffer if ID provided
        if buffer_id is not None:
            self._gpu_buffers[buffer_id] = gpu_tensor
            
        return gpu_tensor
    
    def release_buffer(self, buffer_id: str):
        """
        Release a tracked buffer.
        
        Args:
            buffer_id: ID of the buffer to release
        """
        if buffer_id in self._gpu_buffers:
            del self._gpu_buffers[buffer_id]
            
        if buffer_id in self._pinned_buffers:
            del self._pinned_buffers[buffer_id]
    
    def release_all_buffers(self):
        """Release all tracked buffers."""
        self._gpu_buffers.clear()
        self._pinned_buffers.clear()
        
        if self.has_torch and self.cuda_available:
            # Clear PyTorch CUDA cache
            torch.cuda.empty_cache()
    
    def batch_similarity_search(
        self, 
        query_vectors: Union[np.ndarray, torch.Tensor, pa.FixedSizeListArray],
        stored_vectors: Union[np.ndarray, torch.Tensor, pa.FixedSizeListArray],
        k: int = 4,
        metric: str = "cosine"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform batch similarity search using GPU acceleration.
        
        Args:
            query_vectors: Query vectors
            stored_vectors: Stored vectors to search against
            k: Number of results to return for each query
            metric: Distance metric ("cosine", "l2", "dot")
            
        Returns:
            Tuple of (distances, indices) as PyTorch tensors
        """
        if not self.has_torch or not self.cuda_available:
            raise RuntimeError("PyTorch with CUDA is required for GPU similarity search")
            
        # Convert inputs to PyTorch tensors if needed
        if isinstance(query_vectors, pa.FixedSizeListArray):
            query_vectors = self.fixed_size_list_array_to_torch(query_vectors)
        elif isinstance(query_vectors, np.ndarray):
            query_vectors = torch.from_numpy(query_vectors).to(self.torch_device)
        elif isinstance(query_vectors, torch.Tensor) and query_vectors.device.type != "cuda":
            query_vectors = query_vectors.to(self.torch_device)
            
        if isinstance(stored_vectors, pa.FixedSizeListArray):
            stored_vectors = self.fixed_size_list_array_to_torch(stored_vectors)
        elif isinstance(stored_vectors, np.ndarray):
            stored_vectors = torch.from_numpy(stored_vectors).to(self.torch_device)
        elif isinstance(stored_vectors, torch.Tensor) and stored_vectors.device.type != "cuda":
            stored_vectors = stored_vectors.to(self.torch_device)
        
        # Ensure vectors are normalized for cosine similarity
        if metric == "cosine":
            query_vectors = torch.nn.functional.normalize(query_vectors, p=2, dim=1)
            stored_vectors = torch.nn.functional.normalize(stored_vectors, p=2, dim=1)
            
            # Compute cosine similarity (dot product of normalized vectors)
            similarity = torch.mm(query_vectors, stored_vectors.t())
            
            # Get top k results (highest similarity)
            distances, indices = torch.topk(similarity, k=min(k, stored_vectors.size(0)), dim=1)
            
            return distances, indices
            
        elif metric == "dot":
            # Compute dot product
            similarity = torch.mm(query_vectors, stored_vectors.t())
            
            # Get top k results (highest dot product)
            distances, indices = torch.topk(similarity, k=min(k, stored_vectors.size(0)), dim=1)
            
            return distances, indices
            
        elif metric == "l2":
            # Compute squared L2 distance efficiently
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x · y
            q_norm = torch.sum(query_vectors ** 2, dim=1, keepdim=True)
            s_norm = torch.sum(stored_vectors ** 2, dim=1, keepdim=True)
            qs_dot = torch.mm(query_vectors, stored_vectors.t())
            
            distances = q_norm - 2 * qs_dot + s_norm.t()
            
            # Get top k results (lowest distance)
            distances, indices = torch.topk(-distances, k=min(k, stored_vectors.size(0)), dim=1)
            distances = -distances  # Convert back to actual distances
            
            return distances, indices
            
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def mmr_rerank(
        self,
        query_vector: Union[np.ndarray, torch.Tensor, List[float]],
        vectors: Union[np.ndarray, torch.Tensor, pa.FixedSizeListArray],
        indices: Union[np.ndarray, torch.Tensor, List[int]],
        k: int = 4,
        lambda_mult: float = 0.5
    ) -> List[int]:
        """
        Perform Maximum Marginal Relevance reranking using GPU acceleration.
        
        MMR optimizes for relevance and diversity by selecting items that
        maximize marginal relevance: relevance to query - redundancy with
        already selected items.
        
        Args:
            query_vector: Query vector
            vectors: All vectors
            indices: Initial indices to consider (e.g., from similarity search)
            k: Number of results to return
            lambda_mult: Balance between relevance and diversity (0-1)
                1 = maximize relevance, 0 = maximize diversity
            
        Returns:
            List of reranked indices
        """
        if not self.has_torch or not self.cuda_available:
            raise RuntimeError("PyTorch with CUDA is required for GPU MMR reranking")
        
        # Convert inputs to PyTorch tensors
        if isinstance(query_vector, list):
            query_vector = torch.tensor(query_vector, device=self.torch_device).float()
        elif isinstance(query_vector, np.ndarray):
            query_vector = torch.from_numpy(query_vector).to(self.torch_device).float()
        elif isinstance(query_vector, torch.Tensor) and query_vector.device.type != "cuda":
            query_vector = query_vector.to(self.torch_device).float()
        
        if isinstance(vectors, pa.FixedSizeListArray):
            vectors = self.fixed_size_list_array_to_torch(vectors)
        elif isinstance(vectors, np.ndarray):
            vectors = torch.from_numpy(vectors).to(self.torch_device).float()
        elif isinstance(vectors, torch.Tensor) and vectors.device.type != "cuda":
            vectors = vectors.to(self.torch_device).float()
        
        if isinstance(indices, list):
            indices = torch.tensor(indices, device=self.torch_device).long()
        elif isinstance(indices, np.ndarray):
            indices = torch.from_numpy(indices).to(self.torch_device).long()
        elif isinstance(indices, torch.Tensor) and indices.device.type != "cuda":
            indices = indices.to(self.torch_device).long()
        
        # Ensure query_vector is normalized and reshaped properly
        if query_vector.dim() == 1:
            query_vector = query_vector.unsqueeze(0)
        query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)
        
        # Get vectors for initial indices
        candidate_vectors = vectors[indices]
        
        # Normalize candidate vectors
        candidate_vectors = torch.nn.functional.normalize(candidate_vectors, p=2, dim=1)
        
        # Calculate relevance scores (cosine similarity with query)
        relevance_scores = torch.mm(candidate_vectors, query_vector.t()).squeeze()
        
        # Initialize selected and remaining indices
        selected_indices = []
        remaining_indices = list(range(len(indices)))
        
        # Select first document with highest relevance
        if len(remaining_indices) > 0:
            best_idx = torch.argmax(relevance_scores).item()
            selected_indices.append(remaining_indices[best_idx])
            remaining_indices.pop(best_idx)
        
        # Select remaining documents using MMR
        while len(selected_indices) < k and len(remaining_indices) > 0:
            # Get selected and candidate vectors
            if len(selected_indices) > 0:
                selected_vectors = candidate_vectors[selected_indices]
            
            # Calculate maximum similarity with selected documents for each candidate
            max_similarities = torch.zeros(len(remaining_indices), device=self.torch_device)
            
            if len(selected_indices) > 0:
                for i, idx in enumerate(remaining_indices):
                    similarity = torch.mm(
                        candidate_vectors[idx].unsqueeze(0),
                        selected_vectors.t()
                    )
                    max_similarities[i] = torch.max(similarity)
            
            # Calculate MMR scores
            mmr_scores = lambda_mult * relevance_scores[remaining_indices] - \
                        (1 - lambda_mult) * max_similarities
            
            # Select document with highest MMR score
            best_mmr_idx = torch.argmax(mmr_scores).item()
            selected_indices.append(remaining_indices[best_mmr_idx])
            remaining_indices.pop(best_mmr_idx)
        
        # Map back to original indices
        return [indices[i].item() for i in selected_indices]
    
    def cleanup(self):
        """Clean up resources and release memory."""
        self.release_all_buffers()
        
        if self.has_torch and self.cuda_available:
            torch.cuda.empty_cache()