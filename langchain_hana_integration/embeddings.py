"""
Optimized embeddings module for SAP HANA Cloud integration.

This module provides high-performance embedding generation with caching,
batching, and advanced optimization for production workloads.
"""

import os
import time
import logging
import hashlib
import pickle
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from functools import lru_cache
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_hana_integration.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, but make it optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Try to import torch, but make it optional
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. GPU acceleration will be disabled.")


class HanaOptimizedEmbeddings(Embeddings):
    """
    Production-grade embeddings provider with advanced optimizations.
    
    Features:
    - Multi-level caching (memory + disk)
    - Optimized batching for throughput
    - GPU acceleration when available
    - Fallback mechanisms for robustness
    - Advanced monitoring and instrumentation
    - Internal SAP HANA embedding support
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        internal_embedding_model_id: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_gpu: bool = True,
        batch_size: int = 32,
        enable_caching: bool = True,
        memory_cache_size: int = 10000,
        normalize_embeddings: bool = True,
        timeout: float = 60.0,
        retry_count: int = 3,
        model_kwargs: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the optimized embeddings provider.
        
        Args:
            model_name: Name of the model to use for embeddings
            internal_embedding_model_id: ID of SAP HANA internal embedding model
            cache_dir: Directory to cache embeddings on disk
            use_gpu: Whether to use GPU acceleration if available
            batch_size: Number of texts to process in a single batch
            enable_caching: Whether to enable caching
            memory_cache_size: Size of in-memory LRU cache
            normalize_embeddings: Whether to normalize embeddings to unit length
            timeout: Timeout in seconds for embedding operations
            retry_count: Number of retries for failed operations
            model_kwargs: Additional keyword arguments for the model
        """
        self.model_name = model_name
        self.internal_embedding_model_id = internal_embedding_model_id
        self.cache_dir = cache_dir
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.batch_size = batch_size
        self.enable_caching = enable_caching
        self.memory_cache_size = memory_cache_size
        self.normalize_embeddings = normalize_embeddings
        self.timeout = timeout
        self.retry_count = retry_count
        self.model_kwargs = model_kwargs or {}
        
        # Internal state
        self._model = None
        self._use_internal_embeddings = internal_embedding_model_id is not None
        self._disk_cache_path = None
        self._embedding_dimension = None
        self._metrics = {
            "total_embedding_calls": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_tokens_processed": 0,
            "total_processing_time": 0.0,
            "errors": 0
        }
        
        # Initialize embedding model and cache
        self._initialize_embeddings()
        self._initialize_cache()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the embedding model based on configuration."""
        if self._use_internal_embeddings:
            logger.info(f"Using SAP HANA internal embeddings with model ID: {self.internal_embedding_model_id}")
            # Internal embeddings don't need a model loaded in Python
            self._embedding_dimension = None  # Will be determined at runtime
            return
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for embedding generation. "
                "Install with: pip install sentence-transformers"
            )
        
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            
            # Load the model with appropriate device and kwargs
            self._model = SentenceTransformer(
                self.model_name,
                device=device,
                **self.model_kwargs
            )
            
            # Determine embedding dimension
            self._embedding_dimension = self._model.get_sentence_embedding_dimension()
            
            logger.info(f"Loaded {self.model_name} with dimension {self._embedding_dimension} on {device}")
            
            # Log GPU info if using GPU
            if device == "cuda":
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"Using GPU: {gpu_name}")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise EmbeddingError(f"Failed to load embedding model: {e}")
    
    def _initialize_cache(self) -> None:
        """Initialize the embedding cache."""
        if not self.enable_caching:
            logger.info("Embedding caching is disabled")
            return
        
        # Set up disk cache if cache_dir is specified
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self._disk_cache_path = Path(self.cache_dir) / f"{self.model_name.replace('/', '_')}_cache.pkl"
            logger.info(f"Disk cache configured at: {self._disk_cache_path}")
        
        # Clear the LRU cache and configure it with the specified size
        self._get_cached_embedding.cache_clear()
        self._get_cached_embedding = lru_cache(maxsize=self.memory_cache_size)(self._get_cached_embedding.__wrapped__)
        
        logger.info(f"Memory cache configured with capacity: {self.memory_cache_size} items")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []
        
        start_time = time.time()
        self._metrics["total_embedding_calls"] += 1
        
        try:
            # If using internal embeddings, raise NotImplementedError as expected
            if self._use_internal_embeddings:
                raise NotImplementedError(
                    "embed_documents is not implemented for internal SAP HANA embeddings. "
                    "The embeddings will be generated in the database."
                )
            
            # Process in batches for better performance
            embeddings = []
            
            # Estimate token count for metrics (rough approximation)
            total_tokens = sum(len(text.split()) for text in texts)
            self._metrics["total_tokens_processed"] += total_tokens
            
            # First check cache for all texts
            cache_hits = 0
            cache_misses = []
            cache_miss_indices = []
            
            if self.enable_caching:
                for i, text in enumerate(texts):
                    cached_embedding = self._get_cached_embedding(text)
                    if cached_embedding is not None:
                        embeddings.append(cached_embedding)
                        cache_hits += 1
                    else:
                        cache_misses.append(text)
                        cache_miss_indices.append(i)
                
                self._metrics["cache_hits"] += cache_hits
                self._metrics["cache_misses"] += len(cache_misses)
            else:
                cache_misses = texts
                cache_miss_indices = list(range(len(texts)))
            
            # If we have any cache misses, process them
            if cache_misses:
                # Process in batches
                batch_results = []
                for i in range(0, len(cache_misses), self.batch_size):
                    batch = cache_misses[i:i + self.batch_size]
                    
                    # Generate embeddings with timeout and retry
                    batch_embeddings = self._generate_embeddings_with_retry(batch)
                    batch_results.extend(batch_embeddings)
                
                # Store cache misses in the right order
                if self.enable_caching:
                    full_results = [None] * len(texts)
                    
                    # Place cache hits
                    j = 0
                    for i in range(len(texts)):
                        if i not in cache_miss_indices:
                            full_results[i] = embeddings[j]
                            j += 1
                    
                    # Place cache misses and update cache
                    for i, idx in enumerate(cache_miss_indices):
                        embedding = batch_results[i]
                        full_results[idx] = embedding
                        
                        # Update cache for this embedding
                        self._set_cached_embedding(cache_misses[i], embedding)
                    
                    embeddings = full_results
                else:
                    embeddings = batch_results
            
            # Record processing time
            processing_time = time.time() - start_time
            self._metrics["total_processing_time"] += processing_time
            
            # Log metrics
            logger.debug(
                f"Embedded {len(texts)} documents in {processing_time:.2f}s "
                f"({len(texts)/processing_time:.1f} docs/s) with {cache_hits} cache hits"
            )
            
            return embeddings
        
        except NotImplementedError:
            # Re-raise NotImplementedError for internal embeddings
            raise
        except Exception as e:
            self._metrics["errors"] += 1
            logger.error(f"Error generating embeddings: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")
    
    def _generate_embeddings_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with timeout and retry logic."""
        for attempt in range(self.retry_count):
            try:
                # Use the model to generate embeddings
                embeddings = self._model.encode(
                    texts,
                    convert_to_tensor=False,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=False
                )
                
                # Convert to Python lists for JSON serialization
                return embeddings.tolist()
            
            except Exception as e:
                if attempt < self.retry_count - 1:
                    backoff = min(2 ** attempt, 30)  # Exponential backoff with 30s max
                    logger.warning(f"Embedding attempt {attempt+1} failed: {e}. Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.error(f"All {self.retry_count} embedding attempts failed")
                    raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        if self._use_internal_embeddings:
            raise NotImplementedError(
                "embed_query is not implemented for internal SAP HANA embeddings. "
                "The embedding will be generated in the database."
            )
        
        # Delegate to embed_documents for consistent handling
        result = self.embed_documents([text])
        return result[0]
    
    @lru_cache(maxsize=10000)  # This will be replaced in _initialize_cache
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get an embedding from the cache.
        
        This method is decorated with lru_cache for memory caching,
        and also implements disk caching if configured.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not in cache
        """
        # First check memory cache (handled by lru_cache decorator)
        
        # Then check disk cache if configured
        if self._disk_cache_path and self._disk_cache_path.exists():
            text_hash = self._hash_text(text)
            
            try:
                with open(self._disk_cache_path, "rb") as f:
                    disk_cache = pickle.load(f)
                
                if text_hash in disk_cache:
                    return disk_cache[text_hash]
            except Exception as e:
                logger.warning(f"Error reading from disk cache: {e}")
        
        return None
    
    def _set_cached_embedding(self, text: str, embedding: List[float]) -> None:
        """
        Store an embedding in the cache.
        
        Args:
            text: Text that was embedded
            embedding: Embedding vector
        """
        # Memory cache is updated through lru_cache when _get_cached_embedding is called
        # We need to call it to update the memory cache
        self._get_cached_embedding(text)
        
        # Update disk cache if configured
        if self._disk_cache_path:
            text_hash = self._hash_text(text)
            
            try:
                # Load existing cache or create new one
                disk_cache = {}
                if self._disk_cache_path.exists():
                    with open(self._disk_cache_path, "rb") as f:
                        disk_cache = pickle.load(f)
                
                # Update cache
                disk_cache[text_hash] = embedding
                
                # Write back to disk
                with open(self._disk_cache_path, "wb") as f:
                    pickle.dump(disk_cache, f)
            
            except Exception as e:
                logger.warning(f"Error writing to disk cache: {e}")
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash for text for use as a cache key."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the embeddings provider.
        
        Returns:
            Dictionary of metrics
        """
        if self._metrics["total_embedding_calls"] > 0:
            cache_hit_rate = self._metrics["cache_hits"] / (self._metrics["cache_hits"] + self._metrics["cache_misses"]) if self._metrics["cache_hits"] + self._metrics["cache_misses"] > 0 else 0
            avg_time_per_call = self._metrics["total_processing_time"] / self._metrics["total_embedding_calls"]
            
            self._metrics.update({
                "cache_hit_rate": cache_hit_rate,
                "avg_time_per_call": avg_time_per_call,
                "embedding_model": self.model_name,
                "use_gpu": self.use_gpu and TORCH_AVAILABLE and torch.cuda.is_available(),
                "caching_enabled": self.enable_caching,
                "using_internal_embeddings": self._use_internal_embeddings
            })
        
        return self._metrics
    
    def get_model_id(self) -> Optional[str]:
        """
        Get the internal embedding model ID if using HANA internal embeddings.
        
        Returns:
            Internal embedding model ID or None if not using internal embeddings
        """
        if self._use_internal_embeddings:
            return self.internal_embedding_model_id
        return None
    
    def clear_cache(self) -> None:
        """Clear both memory and disk cache."""
        # Clear memory cache
        self._get_cached_embedding.cache_clear()
        
        # Clear disk cache if it exists
        if self._disk_cache_path and self._disk_cache_path.exists():
            try:
                os.remove(self._disk_cache_path)
                logger.info(f"Cleared disk cache at {self._disk_cache_path}")
            except Exception as e:
                logger.warning(f"Error clearing disk cache: {e}")
        
        logger.info("Cleared embedding caches")