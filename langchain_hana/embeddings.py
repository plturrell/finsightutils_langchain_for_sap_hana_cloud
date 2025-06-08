from typing import List, Dict, Any, Optional, Union, Callable, Tuple, cast
import time
import os
import numpy as np
import logging
import threading
import uuid
from functools import lru_cache
from abc import ABC, abstractmethod

from langchain_core.embeddings import Embeddings

# Import GPU dependencies from centralized imports
from langchain_hana.gpu.imports import (
    torch, 
    TORCH_AVAILABLE,
    get_gpu_features_status
)

# Define Tensor type for type hints
if TORCH_AVAILABLE:
    from torch import Tensor
else:
    Tensor = Any  # type: ignore

# Import multi-GPU manager
from langchain_hana.gpu.multi_gpu_manager import (
    EnhancedMultiGPUManager,
    get_multi_gpu_manager,
    Task,
    TaskResult
)

logger = logging.getLogger(__name__)


class HanaInternalEmbeddings(Embeddings):
    """
    A specialized embeddings class designed to work with SAP HANA Cloud's internal embedding functionality.
    
    Unlike standard embedding classes that perform embedding generation in Python,
    this class delegates embedding generation to SAP HANA's native VECTOR_EMBEDDING function.
    
    This architecture provides several advantages:
    1. Performance: Embeddings are generated directly in the database, reducing data transfer overhead
    2. Resource efficiency: Database CPU/GPU resources are used instead of application resources
    3. Consistency: Embeddings are generated using the same model in both search and insertion operations
    4. Scalability: Can leverage SAP HANA's distributed computing capabilities for large workloads
    
    The class intentionally raises NotImplementedError for standard embedding methods to ensure
    that all embedding operations are performed by the database engine via SQL queries.
    
    Example:
        ```python
        from langchain_hana import HanaVectorStore
        from langchain_hana.embeddings import HanaInternalEmbeddings
        
        # Use SAP HANA's internal embedding model
        embeddings = HanaInternalEmbeddings(internal_embedding_model_id="SAP_NEB.20240715")
        
        # Create vector store with internal embeddings
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embeddings,
            table_name="MY_VECTORS"
        )
        
        # When similarity_search is called, embedding generation will happen in the database
        results = vector_store.similarity_search("What is SAP HANA?")
        ```
    """

    def __init__(self, internal_embedding_model_id: str):
        """
        Initialize the HanaInternalEmbeddings instance.
        
        Args:
            internal_embedding_model_id (str): The ID of the internal embedding model
                                               used by the HANA database. This should match a
                                               valid model ID in your SAP HANA Cloud instance,
                                               such as "SAP_NEB.20240715".
                                               
        Notes:
            - The model_id is passed to the VECTOR_EMBEDDING function in HANA SQL queries
            - The validity of the model_id is checked when the first query is executed
            - Available models depend on your SAP HANA Cloud version and configuration
        """
        self.model_id = internal_embedding_model_id

    def embed_query(self, text: str) -> list[float]:
        """
        Override the embed_query method to raise an error.
        
        This method is intentionally not implemented for HanaInternalEmbeddings because
        query embedding generation is delegated to SAP HANA's VECTOR_EMBEDDING function
        and executed directly in the database through SQL queries.
        
        When using HanaInternalEmbeddings with HanaVectorStore, the vectorstore will
        automatically generate embeddings through SQL by calling VECTOR_EMBEDDING in
        similarity_search_with_score_and_vector_by_query.
        
        Raises:
            NotImplementedError: Always raised to enforce database-side embedding generation.
        """
        raise NotImplementedError(
            "Internal embeddings cannot be used externally. "
            "Use HANA's internal embedding functionality instead."
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Override the embed_documents method to raise an error.
        
        This method is intentionally not implemented for HanaInternalEmbeddings because
        document embedding generation is delegated to SAP HANA's VECTOR_EMBEDDING function
        and executed directly in the database through SQL queries.
        
        When using HanaInternalEmbeddings with HanaVectorStore, the vectorstore's add_texts
        method will automatically generate embeddings through SQL by calling VECTOR_EMBEDDING
        in _add_texts_using_internal_embedding.
        
        Raises:
            NotImplementedError: Always raised to enforce database-side embedding generation.
        """
        raise NotImplementedError(
            "Internal embeddings cannot be used externally. "
            "Use HANA's internal embedding functionality instead."
        )

    def get_model_id(self) -> str:
        """
        Retrieve the internal embedding model ID.
        
        This method is used by HanaVectorStore to get the model ID that should be
        passed to the VECTOR_EMBEDDING function in SQL queries.
        
        Returns:
            str: The ID of the internal embedding model (e.g., "SAP_NEB.20240715").
            
        Notes:
            - This model ID must match one of the embedding models available in your
              SAP HANA Cloud instance
            - The model ID is validated when the first query is executed
            - Available models may vary depending on your SAP HANA Cloud version and configuration
        """
        return self.model_id


class CacheConfig:
    """Configuration for embedding cache behavior."""
    
    def __init__(
        self,
        enabled: bool = True,
        max_size: int = 10000,
        ttl_seconds: Optional[int] = 3600,
        persist_path: Optional[str] = None,
        load_on_init: bool = True
    ):
        """
        Initialize cache configuration.
        
        Args:
            enabled: Whether caching is enabled
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live in seconds (None for no expiration)
            persist_path: Path to persist cache to disk (None for in-memory only)
            load_on_init: Whether to load cache from disk on initialization
        """
        self.enabled = enabled
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.persist_path = persist_path
        self.load_on_init = load_on_init and persist_path is not None


class EmbeddingCache:
    """LRU cache for embedding vectors with optional persistence and TTL."""
    
    def __init__(self, config: CacheConfig):
        """
        Initialize embedding cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        self.cache: Dict[str, Tuple[List[float], float]] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
        # Load cache from disk if enabled
        if self.config.enabled and self.config.persist_path and self.config.load_on_init:
            self._load_from_disk()
    
    def get(self, key: str) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            key: Cache key (usually text)
            
        Returns:
            Embedding vector or None if not in cache
        """
        if not self.config.enabled:
            return None
        
        with self.lock:
            if key in self.cache:
                vector, timestamp = self.cache[key]
                
                # Check TTL
                if self.config.ttl_seconds is not None:
                    if time.time() - timestamp > self.config.ttl_seconds:
                        # Expired
                        del self.cache[key]
                        self.misses += 1
                        return None
                
                # Update timestamp (mark as recently used)
                self.cache[key] = (vector, time.time())
                self.hits += 1
                return vector
            
            self.misses += 1
            return None
    
    def put(self, key: str, vector: List[float]) -> None:
        """
        Put embedding in cache.
        
        Args:
            key: Cache key (usually text)
            vector: Embedding vector
        """
        if not self.config.enabled:
            return
        
        with self.lock:
            # Enforce max size by removing oldest entries
            if len(self.cache) >= self.config.max_size:
                # Sort by timestamp (oldest first)
                sorted_keys = sorted(self.cache.keys(), 
                                    key=lambda k: self.cache[k][1])
                
                # Remove oldest entries
                for old_key in sorted_keys[:len(self.cache) - self.config.max_size + 1]:
                    del self.cache[old_key]
            
            # Add new entry
            self.cache[key] = (vector, time.time())
            
            # Persist to disk if configured
            if self.config.persist_path:
                self._persist_to_disk()
    
    def _persist_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.config.persist_path:
            return
        
        try:
            import pickle
            
            with open(self.config.persist_path, "wb") as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Error persisting embedding cache to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.config.persist_path or not os.path.exists(self.config.persist_path):
            return
        
        try:
            import pickle
            
            with open(self.config.persist_path, "rb") as f:
                self.cache = pickle.load(f)
                
                # Apply TTL to loaded cache
                if self.config.ttl_seconds is not None:
                    current_time = time.time()
                    expired_keys = []
                    
                    for key, (vector, timestamp) in self.cache.items():
                        if current_time - timestamp > self.config.ttl_seconds:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.cache[key]
        except Exception as e:
            logger.warning(f"Error loading embedding cache from disk: {e}")
    
    def clear(self) -> None:
        """Clear the cache."""
        with self.lock:
            self.cache.clear()
            
            if self.config.persist_path:
                self._persist_to_disk()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.config.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "ttl_seconds": self.config.ttl_seconds,
                "persistent": self.config.persist_path is not None
            }


class MultiGPUEmbeddings(Embeddings):
    """
    Multi-GPU enabled embeddings that distributes workloads across available GPUs.
    
    This class is designed to efficiently utilize multiple GPUs for high-throughput
    embedding generation. It leverages the EnhancedMultiGPUManager to distribute
    workloads across available GPUs based on their capabilities and current load.
    
    Features:
    1. Automatic GPU selection based on workload and available resources
    2. Dynamic batch sizing for optimal throughput
    3. Query embedding caching for improved performance on repeated queries
    4. Statistics tracking for performance monitoring
    5. Support for any embeddings model that can be run on GPU
    
    Example:
        ```python
        from langchain_hana.embeddings import MultiGPUEmbeddings
        from langchain_core.embeddings import HuggingFaceEmbeddings
        
        # Create base embeddings models
        base_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Create multi-GPU embeddings
        multi_gpu_embeddings = MultiGPUEmbeddings(
            base_embeddings=base_embeddings,
            batch_size=32,
            enable_caching=True
        )
        
        # Generate embeddings across multiple GPUs
        embeddings = multi_gpu_embeddings.embed_documents(
            ["Text 1", "Text 2", "Text 3", ..., "Text 1000"]
        )
        ```
    """
    
    def __init__(
        self,
        base_embeddings: Embeddings,
        batch_size: int = 32,
        enable_caching: bool = True,
        cache_config: Optional[CacheConfig] = None,
        gpu_manager: Optional[EnhancedMultiGPUManager] = None,
        normalize_embeddings: bool = False,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MultiGPUEmbeddings.
        
        Args:
            base_embeddings: Base embeddings implementation to use
            batch_size: Batch size for embedding generation
            enable_caching: Whether to cache embeddings
            cache_config: Cache configuration
            gpu_manager: Multi-GPU manager to use (creates a default one if None)
            normalize_embeddings: Whether to normalize embeddings
            model_kwargs: Additional keyword arguments for the model
        """
        self.base_embeddings = base_embeddings
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.model_kwargs = model_kwargs or {}
        
        # Initialize cache
        self.enable_caching = enable_caching
        self.cache_config = cache_config or CacheConfig()
        self.cache = EmbeddingCache(self.cache_config) if enable_caching else None
        
        # Get or create GPU manager
        self.gpu_manager = gpu_manager or get_multi_gpu_manager()
        
        # Statistics
        self.stats = {
            "embed_query_calls": 0,
            "embed_documents_calls": 0,
            "documents_embedded": 0,
            "queries_embedded": 0,
            "total_embedding_time": 0.0,
            "batches_processed": 0,
        }
    
    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """
        Normalize a vector to unit length.
        
        Args:
            vector: Input vector
            
        Returns:
            Normalized vector
        """
        if not self.normalize_embeddings:
            return vector
            
        import numpy as np
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        if norm > 0:
            return (arr / norm).tolist()
        return vector
    
    def _batch_embed_texts(
        self, 
        texts: List[str],
        is_query: bool = False
    ) -> List[List[float]]:
        """
        Embed a batch of texts using the base embeddings model.
        
        Args:
            texts: List of texts to embed
            is_query: Whether the texts are queries (vs. documents)
            
        Returns:
            List of embedding vectors
        """
        try:
            if is_query:
                results = []
                for text in texts:
                    results.append(self.base_embeddings.embed_query(text))
                return results
            else:
                return self.base_embeddings.embed_documents(texts)
        except Exception as e:
            logger.error(f"Error embedding batch: {e}")
            raise
    
    def _process_batch(
        self,
        batch: List[str],
        is_query: bool = False
    ) -> List[List[float]]:
        """
        Process a batch of texts, distributing to GPU.
        
        Args:
            batch: Batch of texts
            is_query: Whether the texts are queries
            
        Returns:
            List of embedding vectors
        """
        def embed_batch_on_gpu(texts_batch: List[str]) -> List[List[float]]:
            # This function will be executed on a GPU
            embeddings = self._batch_embed_texts(texts_batch, is_query)
            
            # Normalize if requested
            if self.normalize_embeddings:
                return [self._normalize_vector(emb) for emb in embeddings]
            return embeddings
        
        # Submit task to GPU manager
        task_id = self.gpu_manager.submit_task(
            func=embed_batch_on_gpu,
            args=(batch,),
            priority=5 if is_query else 1,  # Prioritize queries over documents
        )
        
        # Wait for result
        result = self.gpu_manager.wait_for_task(task_id)
        
        if result and result.success:
            return result.result
        elif result and result.error:
            raise result.error
        else:
            raise RuntimeError("GPU task failed with unknown error")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents, distributing workload across GPUs.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        start_time = time.time()
        self.stats["embed_documents_calls"] += 1
        self.stats["documents_embedded"] += len(texts)
        
        # Use cached embeddings if available
        if self.cache and self.enable_caching:
            cached_results = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                if cached:
                    cached_results.append((i, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
            cached_results = []
        
        # Process uncached texts in batches
        all_results: List[Optional[List[float]]] = [None] * len(texts)
        
        # Add cached results
        for idx, embedding in cached_results:
            all_results[idx] = embedding
        
        # Process uncached in batches
        if uncached_texts:
            batches = []
            for i in range(0, len(uncached_texts), self.batch_size):
                batches.append(uncached_texts[i:i + self.batch_size])
            
            self.stats["batches_processed"] += len(batches)
            
            # Process batches in parallel
            batch_results = self.gpu_manager.batch_process(
                func=lambda batch: self._process_batch(batch, is_query=False),
                items=batches,
                wait=True
            )
            
            # Flatten batch results
            flat_results = []
            for batch_result in batch_results:
                flat_results.extend(batch_result)
            
            # Update cache
            if self.cache and self.enable_caching:
                for text, embedding in zip(uncached_texts, flat_results):
                    self.cache.put(text, embedding)
            
            # Update results
            for i, embedding in zip(uncached_indices, flat_results):
                all_results[i] = embedding
        
        # Ensure all results are populated
        final_results = [emb for emb in all_results if emb is not None]
        
        self.stats["total_embedding_time"] += time.time() - start_time
        return final_results
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query, using GPU acceleration.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        start_time = time.time()
        self.stats["embed_query_calls"] += 1
        self.stats["queries_embedded"] += 1
        
        # Check cache first
        if self.cache and self.enable_caching:
            cached = self.cache.get(text)
            if cached:
                return cached
        
        # Use GPU to embed
        result = self._process_batch([text], is_query=True)[0]
        
        # Cache result
        if self.cache and self.enable_caching:
            self.cache.put(text, result)
        
        self.stats["total_embedding_time"] += time.time() - start_time
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Add cache stats if available
        if self.cache:
            stats["cache"] = self.cache.get_stats()
        
        # Add GPU stats
        stats["gpu"] = self.gpu_manager.get_status()
        
        # Add performance metrics
        if stats["documents_embedded"] > 0:
            stats["avg_time_per_document"] = (
                stats["total_embedding_time"] / stats["documents_embedded"]
            )
        
        if stats["queries_embedded"] > 0:
            stats["avg_time_per_query"] = (
                stats["total_embedding_time"] / stats["queries_embedded"]
            )
        
        return stats
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()


class HanaTensorRTMultiGPUEmbeddings(MultiGPUEmbeddings):
    """
    Multi-GPU TensorRT-optimized embeddings for SAP HANA Cloud.
    
    This class extends MultiGPUEmbeddings to leverage TensorRT optimization
    for maximum performance on NVIDIA GPUs. It is specifically designed for
    use with SAP HANA Cloud, including features like paged batching for
    handling large document collections and TensorRT engine management.
    
    Features:
    1. TensorRT engine optimization for maximum throughput
    2. Multi-GPU distribution with dynamic load balancing
    3. Tensor Core optimization for NVIDIA T4 and newer GPUs
    4. FP16 precision for improved performance with minimal accuracy loss
    5. Engine caching for fast startup
    
    Example:
        ```python
        from langchain_hana.embeddings import HanaTensorRTMultiGPUEmbeddings
        
        # Create TensorRT embeddings with multi-GPU support
        embeddings = HanaTensorRTMultiGPUEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            batch_size=64,
            use_fp16=True,
            enable_caching=True
        )
        
        # Create vector store with optimized embeddings
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embeddings,
            table_name="MY_VECTORS"
        )
        
        # Add documents with multi-GPU acceleration
        vector_store.add_texts(["Text 1", "Text 2", ..., "Text 1000"])
        ```
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 32,
        use_fp16: bool = True,
        use_tensorrt: bool = True,
        enable_tensor_cores: bool = True,
        tensorrt_cache_path: Optional[str] = None,
        enable_caching: bool = True,
        cache_config: Optional[CacheConfig] = None,
        gpu_manager: Optional[EnhancedMultiGPUManager] = None,
        normalize_embeddings: bool = True,
        max_sequence_length: Optional[int] = None,
    ):
        """
        Initialize TensorRT multi-GPU embeddings.
        
        Args:
            model_name: HuggingFace model name
            batch_size: Batch size for embedding generation
            use_fp16: Whether to use FP16 precision (faster)
            use_tensorrt: Whether to use TensorRT (if available)
            enable_tensor_cores: Whether to enable Tensor Core optimizations
            tensorrt_cache_path: Path to cache TensorRT engines
            enable_caching: Whether to cache embeddings
            cache_config: Cache configuration
            gpu_manager: Multi-GPU manager (creates default if None)
            normalize_embeddings: Whether to normalize embeddings
            max_sequence_length: Maximum sequence length (None for model default)
        """
        # Import here to avoid import errors if TensorRT is not available
        try:
            from langchain_hana.gpu import TensorRTEmbeddings
            from langchain_hana.gpu import TensorCoreOptimizer
            
            # Create TensorRT embeddings
            trt_embeddings = TensorRTEmbeddings(
                model_name=model_name,
                use_fp16=use_fp16,
                use_tensorrt=use_tensorrt,
                cache_folder=tensorrt_cache_path,
                max_seq_length=max_sequence_length
            )
            
            # Enable Tensor Core optimizations if requested
            if enable_tensor_cores and use_tensorrt:
                optimizer = TensorCoreOptimizer()
                if optimizer.is_supported():
                    trt_embeddings = optimizer.optimize_embeddings(trt_embeddings)
            
            # Initialize base class
            super().__init__(
                base_embeddings=trt_embeddings,
                batch_size=batch_size,
                enable_caching=enable_caching,
                cache_config=cache_config,
                gpu_manager=gpu_manager,
                normalize_embeddings=normalize_embeddings
            )
            
            self.model_name = model_name
            self.use_tensorrt = use_tensorrt
            self.use_fp16 = use_fp16
            
        except ImportError as e:
            error_msg = (
                f"Error initializing TensorRT embeddings: {e}. "
                "Make sure TensorRT and related dependencies are installed."
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e
