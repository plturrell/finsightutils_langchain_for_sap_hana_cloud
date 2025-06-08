"""
GPU-accelerated data layer for SAP HANA Cloud.

This module provides GPU acceleration directly for the data layer,
enabling high-performance vector operations with minimal data transfer.
"""

import logging
import time
import warnings
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import json
import struct
import os
import threading

import numpy as np

from hdbcli import dbapi
from langchain_core.documents import Document

from langchain_hana.utils import DistanceStrategy
from langchain_hana.error_utils import handle_database_error
from langchain_hana.gpu.imports import (
    check_gpu_requirements,
    get_gpu_count,
    get_gpu_info,
    TORCH_AVAILABLE,
    CUDA_AVAILABLE,
    FAISS_GPU_AVAILABLE,
)

# Configure logger
logger = logging.getLogger(__name__)

# Constants
DEFAULT_BATCH_SIZE = 1024
DEFAULT_GPU_CACHE_SIZE_GB = 4.0
DEFAULT_PRECISION = "float32"  # Options: float32, float16, int8
DEFAULT_VECTOR_DIMENSION = 384

# Global variables
_gpu_memory_pool = None
_gpu_engines = {}
_lock = threading.RLock()


def _initialize_module():
    """Initialize the module and check for GPU capabilities."""
    global _torch, _faiss
    
    # Only import these if available to prevent errors
    if TORCH_AVAILABLE and CUDA_AVAILABLE:
        import torch
        _torch = torch
        logger.info(f"PyTorch {torch.__version__} with CUDA {torch.version.cuda} is available")
        
        # Initialize CUDA if available
        if torch.cuda.is_available():
            torch.cuda.init()
            logger.info(f"Initialized CUDA with {torch.cuda.device_count()} GPUs")
        else:
            logger.warning("PyTorch is installed but CUDA is not available")
    else:
        logger.warning("PyTorch with CUDA is not available, GPU acceleration will be limited")
        _torch = None
    
    # Check for FAISS-GPU
    if FAISS_GPU_AVAILABLE:
        import faiss
        _faiss = faiss
        logger.info(f"FAISS-GPU is available for high-performance vector search")
    else:
        logger.warning("FAISS-GPU is not available, index acceleration will be limited")
        _faiss = None


# Attempt to initialize on module load
try:
    _initialize_module()
except Exception as e:
    logger.warning(f"Error initializing GPU data layer accelerator: {str(e)}")
    logger.warning("GPU data layer acceleration will be disabled")


class MemoryManager:
    """
    GPU memory manager for efficient memory allocation and caching.
    
    This class manages GPU memory allocation, caching, and deallocation
    to optimize memory usage for vector operations.
    """
    
    def __init__(self, 
                 gpu_id: int = 0, 
                 cache_size_gb: float = DEFAULT_GPU_CACHE_SIZE_GB,
                 precision: str = DEFAULT_PRECISION):
        """
        Initialize the GPU memory manager.
        
        Args:
            gpu_id: ID of the GPU to use
            cache_size_gb: Size of the cache in GB
            precision: Data precision (float32, float16, int8)
        """
        self.gpu_id = gpu_id
        self.cache_size_gb = cache_size_gb
        self.precision = precision
        
        # Memory management
        self.allocated_memory = 0
        self.max_memory = int(self.cache_size_gb * 1024 * 1024 * 1024)  # Convert GB to bytes
        self.cached_data = {}
        
        # Initialize GPU if available
        if TORCH_AVAILABLE and _torch.cuda.is_available():
            self.device = _torch.device(f"cuda:{gpu_id}")
            # Reserve memory for the cache
            self._reserve_memory()
        else:
            self.device = None
            logger.warning(f"GPU {gpu_id} is not available, falling back to CPU")
    
    def _reserve_memory(self):
        """Reserve GPU memory for the cache."""
        if self.device is None:
            return
            
        try:
            # Clear any existing cached tensors
            _torch.cuda.empty_cache()
            
            # Create a large tensor to reserve memory
            reserved_tensor = _torch.empty(
                size=(self.max_memory // 4,),  # float32 = 4 bytes
                dtype=_torch.float32,
                device=self.device
            )
            
            # Free the tensor immediately
            del reserved_tensor
            _torch.cuda.empty_cache()
            
            logger.info(f"Reserved {self.cache_size_gb}GB of GPU memory on GPU {self.gpu_id}")
        except Exception as e:
            logger.error(f"Failed to reserve GPU memory: {str(e)}")
            self.device = None
    
    def get_vector_tensor(self, 
                         vectors: List[List[float]], 
                         cache_key: Optional[str] = None) -> Any:
        """
        Convert vectors to a GPU tensor, using cache if available.
        
        Args:
            vectors: List of vectors to convert
            cache_key: Optional cache key for reusing tensors
            
        Returns:
            Tensor on GPU, or numpy array if GPU not available
        """
        if self.device is None:
            return np.array(vectors, dtype=np.float32)
            
        # Check cache first
        if cache_key and cache_key in self.cached_data:
            return self.cached_data[cache_key]
            
        # Create new tensor
        tensor = _torch.tensor(vectors, dtype=self._get_torch_dtype(), device=self.device)
        
        # Cache if requested
        if cache_key:
            self.cached_data[cache_key] = tensor
            self.allocated_memory += tensor.element_size() * tensor.nelement()
            
            # Clean cache if needed
            if self.allocated_memory > self.max_memory:
                self._clean_cache()
                
        return tensor
    
    def _get_torch_dtype(self):
        """Get PyTorch dtype based on precision setting."""
        if self.precision == "float16":
            return _torch.float16
        elif self.precision == "int8":
            return _torch.int8
        else:
            return _torch.float32  # Default
    
    def _clean_cache(self):
        """Clean the cache when memory usage exceeds limits."""
        if not self.cached_data:
            return
            
        # Simple LRU-like cleanup - remove oldest entries first
        # In a production system, this would use a proper LRU cache
        logger.info(f"Cleaning GPU memory cache (using {self.allocated_memory/1e9:.2f}GB)")
        
        # Sort by insertion order (simple approach for example)
        keys = list(self.cached_data.keys())
        
        # Remove oldest entries until we're below 75% capacity
        target_memory = int(self.max_memory * 0.75)
        for key in keys:
            if self.allocated_memory <= target_memory:
                break
                
            tensor = self.cached_data.pop(key)
            self.allocated_memory -= tensor.element_size() * tensor.nelement()
            del tensor
            
        # Force CUDA to reclaim memory
        _torch.cuda.empty_cache()
        
        logger.info(f"After cleaning: {self.allocated_memory/1e9:.2f}GB used, {len(self.cached_data)} cached items")
    
    def compute_similarity(self, 
                         query_vector: List[float],
                         document_vectors: Union[List[List[float]], Any],
                         distance_strategy: DistanceStrategy,
                         batch_size: int = DEFAULT_BATCH_SIZE) -> List[float]:
        """
        Compute similarity between query vector and document vectors on GPU.
        
        Args:
            query_vector: Query vector as a list of floats
            document_vectors: Document vectors as list of lists or tensor
            distance_strategy: Distance calculation strategy
            batch_size: Batch size for processing
            
        Returns:
            List of similarity scores for each document vector
        """
        if self.device is None:
            # Fall back to CPU calculation
            return self._compute_similarity_cpu(
                query_vector, document_vectors, distance_strategy
            )
            
        # Convert query vector to tensor if needed
        if not isinstance(query_vector, _torch.Tensor):
            query_tensor = _torch.tensor(
                query_vector, 
                dtype=self._get_torch_dtype(),
                device=self.device
            ).reshape(1, -1)  # Add batch dimension
        else:
            query_tensor = query_vector
            
        # Convert document vectors to tensor if needed
        if not isinstance(document_vectors, _torch.Tensor):
            doc_tensor = self.get_vector_tensor(document_vectors)
        else:
            doc_tensor = document_vectors
            
        # Normalize vectors if using cosine similarity
        if distance_strategy == DistanceStrategy.COSINE:
            query_norm = _torch.norm(query_tensor, p=2, dim=1, keepdim=True)
            query_tensor = query_tensor / query_norm.clamp(min=1e-8)
            
            # Normalize document vectors in batches to avoid OOM
            if len(doc_tensor) > batch_size:
                # Process in batches for large collections
                normalized_chunks = []
                for i in range(0, len(doc_tensor), batch_size):
                    chunk = doc_tensor[i:i+batch_size]
                    chunk_norm = _torch.norm(chunk, p=2, dim=1, keepdim=True)
                    normalized_chunks.append(chunk / chunk_norm.clamp(min=1e-8))
                doc_tensor = _torch.cat(normalized_chunks, dim=0)
            else:
                doc_norm = _torch.norm(doc_tensor, p=2, dim=1, keepdim=True)
                doc_tensor = doc_tensor / doc_norm.clamp(min=1e-8)
                
        # Compute similarity scores
        if distance_strategy == DistanceStrategy.COSINE:
            # Cosine similarity is the dot product of normalized vectors
            similarity = _torch.matmul(query_tensor, doc_tensor.T).squeeze(0)
            # Convert to list and return
            return similarity.cpu().tolist()
        elif distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            # For euclidean distance, we use negative distance 
            # (so higher values are better, consistent with cosine)
            distance = -_torch.cdist(query_tensor, doc_tensor, p=2).squeeze(0)
            return distance.cpu().tolist()
        else:
            raise ValueError(f"Unsupported distance strategy: {distance_strategy}")
    
    def _compute_similarity_cpu(self, 
                              query_vector: List[float],
                              document_vectors: List[List[float]],
                              distance_strategy: DistanceStrategy) -> List[float]:
        """Fall back to CPU-based similarity computation."""
        # Convert to numpy arrays
        query_np = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        docs_np = np.array(document_vectors, dtype=np.float32)
        
        # Compute similarity based on distance strategy
        if distance_strategy == DistanceStrategy.COSINE:
            # Normalize vectors
            query_norm = np.linalg.norm(query_np, axis=1, keepdims=True)
            query_np = query_np / np.clip(query_norm, 1e-8, None)
            
            docs_norm = np.linalg.norm(docs_np, axis=1, keepdims=True)
            docs_np = docs_np / np.clip(docs_norm, 1e-8, None)
            
            # Compute cosine similarity
            similarity = np.dot(query_np, docs_np.T).flatten()
            return similarity.tolist()
        elif distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            # Compute negative euclidean distance
            distance = -np.sqrt(np.sum((query_np - docs_np)**2, axis=1))
            return distance.tolist()
        else:
            raise ValueError(f"Unsupported distance strategy: {distance_strategy}")
    
    def release(self):
        """Release all GPU resources."""
        if self.device is None:
            return
            
        # Clear cache
        self.cached_data.clear()
        self.allocated_memory = 0
        
        # Force CUDA to reclaim memory
        _torch.cuda.empty_cache()
        logger.info(f"Released GPU memory on GPU {self.gpu_id}")


class HanaGPUVectorEngine:
    """
    GPU-accelerated vector engine for SAP HANA Cloud.
    
    This class provides GPU acceleration for vector operations in SAP HANA Cloud,
    including similarity search, index management, and vector transformations.
    """
    
    def __init__(self, 
                 connection: dbapi.Connection,
                 table_name: str,
                 content_column: str,
                 metadata_column: str,
                 vector_column: str,
                 distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
                 gpu_ids: Optional[List[int]] = None,
                 cache_size_gb: float = DEFAULT_GPU_CACHE_SIZE_GB,
                 precision: str = DEFAULT_PRECISION,
                 enable_tensor_cores: bool = True,
                 enable_prefetch: bool = True,
                 prefetch_size: int = 100000,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        """
        Initialize the GPU vector engine.
        
        Args:
            connection: SAP HANA database connection
            table_name: Name of the vector table
            content_column: Name of the content column
            metadata_column: Name of the metadata column
            vector_column: Name of the vector column
            distance_strategy: Distance strategy for similarity calculation
            gpu_ids: IDs of GPUs to use (None = use all available)
            cache_size_gb: Size of GPU memory cache in GB
            precision: Computation precision (float32, float16, int8)
            enable_tensor_cores: Whether to use Tensor Cores if available
            enable_prefetch: Whether to prefetch vectors from database
            prefetch_size: Number of vectors to prefetch
            batch_size: Batch size for processing
        """
        self.connection = connection
        self.table_name = table_name
        self.content_column = content_column
        self.metadata_column = metadata_column
        self.vector_column = vector_column
        self.distance_strategy = distance_strategy
        
        # GPU configuration
        self.batch_size = batch_size
        self.enable_tensor_cores = enable_tensor_cores
        self.enable_prefetch = enable_prefetch
        self.prefetch_size = prefetch_size
        
        # Check GPU availability
        gpu_available, gpu_message = check_gpu_requirements("data_layer")
        if not gpu_available:
            logger.warning(f"GPU acceleration is not available: {gpu_message}")
            logger.warning("Falling back to CPU execution for all operations")
            self.gpu_available = False
            self.memory_managers = []
        else:
            self.gpu_available = True
            
            # Initialize GPU devices
            if gpu_ids is None:
                # Use all available GPUs
                gpu_count = get_gpu_count()
                self.gpu_ids = list(range(gpu_count))
            else:
                self.gpu_ids = gpu_ids
                
            logger.info(f"Using GPUs: {self.gpu_ids}")
                
            # Initialize memory managers for each GPU
            self.memory_managers = []
            for gpu_id in self.gpu_ids:
                try:
                    manager = MemoryManager(
                        gpu_id=gpu_id,
                        cache_size_gb=cache_size_gb,
                        precision=precision
                    )
                    self.memory_managers.append(manager)
                    logger.info(f"Initialized GPU memory manager for GPU {gpu_id}")
                except Exception as e:
                    logger.error(f"Failed to initialize GPU {gpu_id}: {str(e)}")
                    
            if not self.memory_managers:
                logger.warning("No GPU memory managers could be initialized")
                logger.warning("Falling back to CPU execution for all operations")
                self.gpu_available = False
            else:
                # Set primary GPU for operations
                self.primary_gpu = self.memory_managers[0]
                
                # Enable Tensor Cores if requested and available
                if self.enable_tensor_cores and TORCH_AVAILABLE:
                    if hasattr(_torch.backends.cuda, 'matmul') and hasattr(_torch.backends.cuda.matmul, 'allow_tf32'):
                        _torch.backends.cuda.matmul.allow_tf32 = True
                        _torch.backends.cudnn.allow_tf32 = True
                        logger.info("TensorFloat-32 (TF32) enabled for matrix operations")
                
                # Prefetch vectors if enabled
                if self.enable_prefetch:
                    self._prefetch_vectors()
    
    def _prefetch_vectors(self):
        """Prefetch vectors from the database to GPU memory."""
        if not self.gpu_available or not self.enable_prefetch:
            return
            
        logger.info(f"Prefetching up to {self.prefetch_size} vectors to GPU memory")
        
        try:
            # Query to get vectors
            sql_str = f'''
            SELECT "{self.vector_column}" 
            FROM "{self.table_name}" 
            LIMIT {self.prefetch_size}
            '''
            
            cur = self.connection.cursor()
            try:
                cur.execute(sql_str)
                vectors_binary = cur.fetchall()
                
                if vectors_binary:
                    # Deserialize vectors
                    vectors = []
                    for row in vectors_binary:
                        vector_binary = row[0]
                        # Extract vector dimension and values
                        # Assuming binary format starts with a 4-byte integer for dimension
                        # followed by that many float values
                        dim = struct.unpack("<I", vector_binary[:4])[0]
                        vector = list(struct.unpack(f"<{dim}f", vector_binary[4:4+dim*4]))
                        vectors.append(vector)
                    
                    # Load vectors to GPU
                    if vectors:
                        self.primary_gpu.get_vector_tensor(
                            vectors, 
                            cache_key="prefetched_vectors"
                        )
                        logger.info(f"Prefetched {len(vectors)} vectors to GPU memory")
            finally:
                cur.close()
                
        except Exception as e:
            logger.error(f"Error prefetching vectors: {str(e)}")
            logger.warning("Vector prefetching disabled due to errors")
            self.enable_prefetch = False
    
    def similarity_search(self,
                         query_vector: List[float],
                         k: int = 4,
                         filter: Optional[Dict[str, Any]] = None,
                         fetch_all_vectors: bool = False) -> List[Tuple]:
        """
        Perform GPU-accelerated similarity search.
        
        This method has two modes:
        1. Hybrid mode (default): Use SQL for filtering, GPU for similarity calculation
        2. Full GPU mode (fetch_all_vectors=True): Fetch all vectors to GPU and do 
           both filtering and similarity calculation on GPU
        
        Args:
            query_vector: Query vector as a list of floats
            k: Number of results to return
            filter: Filter for metadata
            fetch_all_vectors: Whether to fetch all vectors to GPU memory
            
        Returns:
            List of (document_content, metadata_json, similarity_score) tuples
        """
        if not self.gpu_available:
            logger.warning("GPU acceleration not available, using database for similarity search")
            return self._similarity_search_db(query_vector, k, filter)
            
        # Hybrid approach - use database for filtering, GPU for similarity calculation
        if not fetch_all_vectors:
            return self._similarity_search_hybrid(query_vector, k, filter)
            
        # Full GPU approach - fetch all vectors and perform search entirely on GPU
        return self._similarity_search_full_gpu(query_vector, k, filter)
    
    def _similarity_search_db(self,
                            query_vector: List[float],
                            k: int = 4,
                            filter: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """
        Perform similarity search using database native functions.
        
        This is a fallback method when GPU acceleration is not available.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filter: Metadata filter
            
        Returns:
            List of (document_content, metadata_json, similarity_score) tuples
        """
        # Prepare the vector for the query
        vector_binary = self._serialize_vector(query_vector)
        
        # Build the SQL query
        distance_function, order_direction = (
            ("COSINE_SIMILARITY", "DESC") 
            if self.distance_strategy == DistanceStrategy.COSINE 
            else ("L2DISTANCE", "ASC")
        )
        
        # Construct filter clause
        filter_clause = ""
        filter_params = []
        
        if filter:
            conditions = []
            for key, value in filter.items():
                # Using JSON_VALUE to extract values from metadata JSON
                conditions.append(f'JSON_VALUE("{self.metadata_column}", \'$.{key}\') = ?')
                filter_params.append(str(value))  # Convert all values to string for simplicity
                
            if conditions:
                filter_clause = "WHERE " + " AND ".join(conditions)
                
        # Construct the full query
        sql_str = f'''
        SELECT 
            "{self.content_column}",
            "{self.metadata_column}",
            {distance_function}("{self.vector_column}", ?) as similarity
        FROM "{self.table_name}"
        {filter_clause}
        ORDER BY similarity {order_direction}
        LIMIT {k}
        '''
        
        # Execute the query
        cur = self.connection.cursor()
        try:
            params = [vector_binary] + filter_params
            cur.execute(sql_str, params)
            results = cur.fetchall()
            return results
        except dbapi.Error as e:
            # Handle database errors
            additional_context = {
                "operation": "similarity_search_db",
                "table_name": self.table_name,
                "vector_dimension": len(query_vector)
            }
            handle_database_error(e, "similarity_search", additional_context)
            return []
        finally:
            cur.close()
    
    def _similarity_search_hybrid(self,
                                query_vector: List[float],
                                k: int = 4,
                                filter: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """
        Perform similarity search using hybrid CPU/GPU approach.
        
        This method fetches filtered vectors from the database and then
        performs similarity calculation on the GPU.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filter: Metadata filter
            
        Returns:
            List of (document_content, metadata_json, similarity_score) tuples
        """
        try:
            # Construct filter clause
            filter_clause = ""
            filter_params = []
            
            if filter:
                conditions = []
                for key, value in filter.items():
                    conditions.append(f'JSON_VALUE("{self.metadata_column}", \'$.{key}\') = ?')
                    filter_params.append(str(value))
                    
                if conditions:
                    filter_clause = "WHERE " + " AND ".join(conditions)
            
            # Fetch filtered vectors and metadata
            sql_str = f'''
            SELECT 
                "{self.content_column}",
                "{self.metadata_column}",
                "{self.vector_column}"
            FROM "{self.table_name}"
            {filter_clause}
            '''
            
            # Execute the query
            cur = self.connection.cursor()
            try:
                cur.execute(sql_str, filter_params)
                results = cur.fetchall()
            finally:
                cur.close()
                
            if not results:
                logger.info("No documents match the filter criteria")
                return []
                
            # Extract vectors and compute similarity on GPU
            contents = []
            metadatas = []
            vectors = []
            
            for row in results:
                contents.append(row[0])
                metadatas.append(row[1])
                
                # Deserialize vector
                vector_binary = row[2]
                dim = struct.unpack("<I", vector_binary[:4])[0]
                vector = list(struct.unpack(f"<{dim}f", vector_binary[4:4+dim*4]))
                vectors.append(vector)
            
            # Compute similarity scores on GPU
            similarity_scores = self.primary_gpu.compute_similarity(
                query_vector, vectors, self.distance_strategy, self.batch_size
            )
            
            # Combine results with similarity scores
            scored_results = list(zip(contents, metadatas, similarity_scores))
            
            # Sort by similarity score (descending for cosine, ascending for euclidean)
            reverse_sort = self.distance_strategy == DistanceStrategy.COSINE
            sorted_results = sorted(
                scored_results, 
                key=lambda x: x[2], 
                reverse=reverse_sort
            )
            
            # Return top k results
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Error in hybrid similarity search: {str(e)}")
            # Fall back to database search if GPU search fails
            logger.info("Falling back to database similarity search")
            return self._similarity_search_db(query_vector, k, filter)
    
    def _similarity_search_full_gpu(self,
                                  query_vector: List[float],
                                  k: int = 4,
                                  filter: Optional[Dict[str, Any]] = None) -> List[Tuple]:
        """
        Perform similarity search entirely on GPU.
        
        This method fetches all vectors to GPU memory and performs both
        filtering and similarity calculation on the GPU.
        
        Args:
            query_vector: Query vector
            k: Number of results
            filter: Metadata filter
            
        Returns:
            List of (document_content, metadata_json, similarity_score) tuples
        """
        try:
            # Fetch all vectors and metadata
            sql_str = f'''
            SELECT 
                "{self.content_column}",
                "{self.metadata_column}",
                "{self.vector_column}"
            FROM "{self.table_name}"
            '''
            
            # Execute the query
            cur = self.connection.cursor()
            try:
                cur.execute(sql_str)
                results = cur.fetchall()
            finally:
                cur.close()
                
            if not results:
                logger.info("No documents in the table")
                return []
                
            # Extract contents, metadata, and vectors
            contents = []
            metadatas = []
            vectors = []
            
            for row in results:
                contents.append(row[0])
                metadatas.append(row[1])
                
                # Deserialize vector
                vector_binary = row[2]
                dim = struct.unpack("<I", vector_binary[:4])[0]
                vector = list(struct.unpack(f"<{dim}f", vector_binary[4:4+dim*4]))
                vectors.append(vector)
            
            # Apply filter on GPU if provided
            filtered_indices = None
            if filter:
                filtered_indices = []
                for i, metadata_json in enumerate(metadatas):
                    try:
                        metadata = json.loads(metadata_json)
                        match = True
                        for key, value in filter.items():
                            if key not in metadata or metadata[key] != value:
                                match = False
                                break
                        if match:
                            filtered_indices.append(i)
                    except json.JSONDecodeError:
                        continue
                
                if not filtered_indices:
                    logger.info("No documents match the filter criteria")
                    return []
                    
                # Filter vectors, contents, and metadata
                filtered_vectors = [vectors[i] for i in filtered_indices]
                filtered_contents = [contents[i] for i in filtered_indices]
                filtered_metadatas = [metadatas[i] for i in filtered_indices]
            else:
                filtered_vectors = vectors
                filtered_contents = contents
                filtered_metadatas = metadatas
            
            # Compute similarity scores on GPU
            similarity_scores = self.primary_gpu.compute_similarity(
                query_vector, filtered_vectors, self.distance_strategy, self.batch_size
            )
            
            # Combine results with similarity scores
            scored_results = list(zip(filtered_contents, filtered_metadatas, similarity_scores))
            
            # Sort by similarity score (descending for cosine, ascending for euclidean)
            reverse_sort = self.distance_strategy == DistanceStrategy.COSINE
            sorted_results = sorted(
                scored_results, 
                key=lambda x: x[2], 
                reverse=reverse_sort
            )
            
            # Return top k results
            return sorted_results[:k]
            
        except Exception as e:
            logger.error(f"Error in full GPU similarity search: {str(e)}")
            # Fall back to hybrid search if full GPU search fails
            logger.info("Falling back to hybrid similarity search")
            return self._similarity_search_hybrid(query_vector, k, filter)
    
    def mmr_search(self,
                  query_vector: List[float],
                  k: int = 4,
                  fetch_k: int = 20,
                  lambda_mult: float = 0.5,
                  filter: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Perform Maximal Marginal Relevance search with GPU acceleration.
        
        This method implements MMR search on the GPU for diverse results.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            fetch_k: Number of results to consider for diversity
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            filter: Metadata filter
            
        Returns:
            List of Document objects
        """
        if not self.gpu_available:
            logger.warning("GPU acceleration not available, using CPU for MMR search")
            return self._mmr_search_cpu(query_vector, k, fetch_k, lambda_mult, filter)
            
        try:
            # First get fetch_k most similar vectors
            results = self._similarity_search_hybrid(
                query_vector, 
                min(fetch_k, self.prefetch_size), 
                filter
            )
            
            if not results:
                return []
                
            # Extract contents and metadata for creating Documents
            contents = [r[0] for r in results]
            metadatas = [json.loads(r[1]) if r[1] else {} for r in results]
            
            # Extract vectors for MMR calculation
            vectors = []
            for row in results:
                vector_binary = row[2] if len(row) > 3 else None
                if vector_binary:
                    dim = struct.unpack("<I", vector_binary[:4])[0]
                    vector = list(struct.unpack(f"<{dim}f", vector_binary[4:4+dim*4]))
                    vectors.append(vector)
            
            # If vectors couldn't be extracted, fall back to standard similarity search
            if not vectors:
                logger.warning("Could not extract vectors for MMR, falling back to similarity search")
                documents = [
                    Document(page_content=content, metadata=metadata)
                    for content, metadata in zip(contents[:k], metadatas[:k])
                ]
                return documents
                
            # Move vectors to GPU
            vectors_tensor = self.primary_gpu.get_vector_tensor(vectors)
            query_tensor = self.primary_gpu.get_vector_tensor([query_vector])
            
            # Run MMR algorithm on GPU
            mmr_indices = self._compute_mmr_gpu(
                query_tensor, 
                vectors_tensor, 
                k, 
                lambda_mult
            )
            
            # Create Document objects from selected indices
            documents = [
                Document(page_content=contents[i], metadata=metadatas[i])
                for i in mmr_indices
            ]
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in GPU MMR search: {str(e)}")
            # Fall back to CPU MMR search
            logger.info("Falling back to CPU MMR search")
            return self._mmr_search_cpu(query_vector, k, fetch_k, lambda_mult, filter)
    
    def _compute_mmr_gpu(self,
                       query_vector,
                       document_vectors,
                       k: int,
                       lambda_mult: float) -> List[int]:
        """
        Compute Maximal Marginal Relevance on GPU.
        
        This implementation runs the MMR algorithm directly on GPU
        to avoid transferring data between CPU and GPU.
        
        Args:
            query_vector: Query vector tensor
            document_vectors: Document vector tensor
            k: Number of results to return
            lambda_mult: Diversity parameter
            
        Returns:
            List of indices of selected documents
        """
        if not TORCH_AVAILABLE or not self.gpu_available:
            raise ValueError("GPU acceleration is not available")
            
        # Handle edge cases
        if len(document_vectors) <= k:
            return list(range(len(document_vectors)))
            
        device = self.primary_gpu.device
        
        # Compute query-document similarities
        if self.distance_strategy == DistanceStrategy.COSINE:
            # Normalize vectors first for cosine
            query_norm = _torch.norm(query_vector, p=2, dim=1, keepdim=True)
            query_vector_norm = query_vector / query_norm.clamp(min=1e-8)
            
            doc_norm = _torch.norm(document_vectors, p=2, dim=1, keepdim=True)
            document_vectors_norm = document_vectors / doc_norm.clamp(min=1e-8)
            
            # Compute similarities
            query_doc_similarities = _torch.matmul(
                query_vector_norm, document_vectors_norm.T
            ).squeeze(0)
        else:
            # For euclidean, use negative distance
            query_doc_similarities = -_torch.cdist(
                query_vector, document_vectors, p=2
            ).squeeze(0)
        
        # Initialize selected and remaining indices
        selected_indices = []
        remaining_indices = list(range(len(document_vectors)))
        
        # Select first document with highest similarity to query
        similarities = query_doc_similarities.cpu().tolist()
        first_idx = max(range(len(similarities)), key=lambda i: similarities[i])
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Convert to PyTorch tensors for GPU computation
        selected_tensor = _torch.tensor([first_idx], device=device)
        
        # Pre-compute document-document similarities
        doc_doc_similarities = None  # Compute lazily when needed
        
        # Select remaining documents
        for _ in range(min(k - 1, len(remaining_indices))):
            if not remaining_indices:
                break
                
            # Compute document-document similarities if not already computed
            if doc_doc_similarities is None:
                # Compute full similarity matrix
                if self.distance_strategy == DistanceStrategy.COSINE:
                    doc_doc_similarities = _torch.matmul(
                        document_vectors_norm, document_vectors_norm.T
                    )
                else:
                    doc_doc_similarities = -_torch.cdist(
                        document_vectors, document_vectors, p=2
                    )
            
            # Get similarity to query for all remaining documents
            query_similarity = query_doc_similarities[remaining_indices]
            
            # Get maximum similarity to any already selected document
            selected_indices_tensor = _torch.tensor(selected_indices, device=device)
            remaining_indices_tensor = _torch.tensor(remaining_indices, device=device)
            
            # Extract similarities between remaining and selected docs
            doc_doc_sim = doc_doc_similarities[remaining_indices_tensor][:, selected_indices_tensor]
            max_doc_sim, _ = _torch.max(doc_doc_sim, dim=1)
            
            # Compute MMR score
            mmr_scores = lambda_mult * query_similarity - (1 - lambda_mult) * max_doc_sim
            
            # Get index with highest MMR score
            mmr_scores_cpu = mmr_scores.cpu().tolist()
            next_idx_local = max(range(len(mmr_scores_cpu)), key=lambda i: mmr_scores_cpu[i])
            next_idx = remaining_indices[next_idx_local]
            
            # Update indices
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        return selected_indices
    
    def _mmr_search_cpu(self,
                      query_vector: List[float],
                      k: int,
                      fetch_k: int,
                      lambda_mult: float,
                      filter: Optional[Dict[str, Any]]) -> List[Document]:
        """Fall back to CPU-based MMR search."""
        # Implementation omitted for brevity - would use the standard
        # LangChain MMR implementation
        # For a real implementation, this would be filled in with CPU-based MMR
        # This is just a placeholder to show the fallback mechanism
        logger.warning("CPU-based MMR search not implemented in this example")
        return []
    
    def build_index(self,
                   index_type: str = "hnsw",
                   m: int = 16,
                   ef_construction: int = 200,
                   ef_search: int = 100) -> None:
        """
        Build a GPU-accelerated vector index.
        
        This method creates an index structure for efficient similarity search.
        
        Args:
            index_type: Type of index to build ('hnsw' or 'flat')
            m: HNSW parameter - number of connections per node
            ef_construction: HNSW parameter - size of dynamic candidate list
            ef_search: HNSW parameter - size of dynamic candidate list for search
        """
        if not FAISS_GPU_AVAILABLE or not self.gpu_available:
            logger.warning("FAISS-GPU not available, index building will use CPU only")
            return self._build_index_cpu(index_type, m, ef_construction, ef_search)
            
        try:
            # Fetch all vectors
            sql_str = f'SELECT "{self.vector_column}" FROM "{self.table_name}"'
            
            cur = self.connection.cursor()
            try:
                cur.execute(sql_str)
                vector_rows = cur.fetchall()
            finally:
                cur.close()
                
            if not vector_rows:
                logger.warning("No vectors found in the table")
                return
                
            # Deserialize vectors
            vectors = []
            for row in vector_rows:
                vector_binary = row[0]
                dim = struct.unpack("<I", vector_binary[:4])[0]
                vector = list(struct.unpack(f"<{dim}f", vector_binary[4:4+dim*4]))
                vectors.append(vector)
                
            # Convert to numpy array
            vectors_np = np.array(vectors, dtype=np.float32)
            dim = vectors_np.shape[1]
            
            # Initialize FAISS resources
            res = _faiss.StandardGpuResources()
            
            # Create index based on type
            if index_type.lower() == "hnsw":
                # CPU index that will be moved to GPU
                cpu_index = _faiss.IndexHNSWFlat(dim, m, _faiss.METRIC_INNER_PRODUCT)
                cpu_index.hnsw.efConstruction = ef_construction
                cpu_index.hnsw.efSearch = ef_search
                
                # Convert to GPU index
                index = _faiss.index_cpu_to_gpu(res, self.gpu_ids[0], cpu_index)
            else:
                # Flat index
                cpu_index = _faiss.IndexFlatIP(dim)  # Inner product (for cosine, vectors must be normalized)
                index = _faiss.index_cpu_to_gpu(res, self.gpu_ids[0], cpu_index)
            
            # Normalize vectors if using cosine similarity
            if self.distance_strategy == DistanceStrategy.COSINE:
                _faiss.normalize_L2(vectors_np)
            
            # Train the index if necessary
            if hasattr(index, 'train'):
                index.train(vectors_np)
                
            # Add vectors to the index
            index.add(vectors_np)
            
            logger.info(f"Built {index_type} index with {len(vectors)} vectors on GPU {self.gpu_ids[0]}")
            
            # Store the index for later use
            self._index = index
            self._index_type = index_type
            self._vectors_np = vectors_np  # Keep reference to vectors
            
        except Exception as e:
            logger.error(f"Error building GPU index: {str(e)}")
            logger.warning("Falling back to database for similarity search")
            self._index = None
    
    def _build_index_cpu(self,
                        index_type: str = "hnsw",
                        m: int = 16,
                        ef_construction: int = 200,
                        ef_search: int = 100) -> None:
        """Fall back to CPU-based index building."""
        # Implementation omitted for brevity - would build a CPU-based FAISS index
        # For a real implementation, this would be filled in with CPU-based indexing
        # This is just a placeholder to show the fallback mechanism
        logger.warning("CPU-based index building not implemented in this example")
        self._index = None
    
    def _serialize_vector(self, vector: List[float]) -> bytes:
        """
        Serialize a vector to SAP HANA binary format.
        
        Args:
            vector: Vector as a list of floats
            
        Returns:
            Binary representation of the vector
        """
        # Binary format for REAL_VECTOR:
        # - First 4 bytes: dimension (uint32)
        # - Followed by dimension * 4 bytes of float values
        return struct.pack(f"<I{len(vector)}f", len(vector), *vector)
    
    def release(self):
        """Release all GPU resources."""
        for manager in self.memory_managers:
            manager.release()
            
        # Clear FAISS index if it exists
        if hasattr(self, '_index') and self._index is not None:
            del self._index
            if hasattr(self, '_vectors_np'):
                del self._vectors_np
                
        logger.info("Released all GPU resources")


# Factory function to get or create vector engine
def get_vector_engine(
    connection: dbapi.Connection,
    table_name: str,
    content_column: str,
    metadata_column: str,
    vector_column: str,
    distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
    gpu_ids: Optional[List[int]] = None,
    cache_size_gb: float = DEFAULT_GPU_CACHE_SIZE_GB,
    precision: str = DEFAULT_PRECISION,
    **kwargs
) -> HanaGPUVectorEngine:
    """
    Get or create a GPU vector engine.
    
    This factory function creates or reuses a vector engine instance
    for the specified table.
    
    Args:
        connection: SAP HANA database connection
        table_name: Name of the vector table
        content_column: Name of the content column
        metadata_column: Name of the metadata column
        vector_column: Name of the vector column
        distance_strategy: Distance strategy for similarity calculation
        gpu_ids: IDs of GPUs to use (None = use all available)
        cache_size_gb: Size of GPU memory cache in GB
        precision: Computation precision (float32, float16, int8)
        **kwargs: Additional arguments to pass to the vector engine
        
    Returns:
        HanaGPUVectorEngine instance
    """
    global _gpu_engines, _lock
    
    # Create cache key for this engine
    cache_key = f"{table_name}:{content_column}:{metadata_column}:{vector_column}:{distance_strategy}"
    
    with _lock:
        # Check if we already have an engine for this table
        if cache_key in _gpu_engines:
            return _gpu_engines[cache_key]
            
        # Create new engine
        engine = HanaGPUVectorEngine(
            connection=connection,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            vector_column=vector_column,
            distance_strategy=distance_strategy,
            gpu_ids=gpu_ids,
            cache_size_gb=cache_size_gb,
            precision=precision,
            **kwargs
        )
        
        # Cache the engine
        _gpu_engines[cache_key] = engine
        
        return engine