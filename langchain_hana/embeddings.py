"""
SAP HANA Cloud Embeddings for LangChain

This module provides embedding implementations specifically designed for use with
SAP HANA Cloud's vector capabilities, including:

1. HanaInternalEmbeddings: Uses SAP HANA's internal embedding functionality
2. HanaEmbeddingsCache: Caching layer for embedding models to improve performance

These classes are designed to be used with the HanaVectorStore class for optimal
performance and integration with SAP HANA Cloud.
"""

import time
import logging
import pickle
import os
import json
from typing import Dict, List, Optional, Any, Union, Tuple

from langchain_core.embeddings import Embeddings

# Configure logging
logger = logging.getLogger(__name__)

class HanaInternalEmbeddings(Embeddings):
    """
    Class that leverages SAP HANA Cloud's internal embedding functionality.
    
    This class is designed to be used with SAP HANA Cloud's vector capabilities,
    delegating the embedding generation to SAP HANA's internal VECTOR_EMBEDDING
    function for improved performance.
    
    When using this class with HanaVectorStore, embeddings are generated directly
    within the database, reducing data transfer and improving performance.
    
    Note: This class deliberately raises NotImplementedError for standard embedding
    methods to ensure all embedding operations are performed by the database engine
    via SQL queries.
    
    Examples:
        ```python
        from langchain_hana.embeddings import HanaInternalEmbeddings
        from langchain_hana.vectorstore import HanaVectorStore
        
        # Initialize with SAP HANA's internal embedding model ID
        embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")
        
        # Create vector store with internal embeddings
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embeddings,
            table_name="MY_VECTORS"
        )
        
        # When similarity_search is called, embedding generation happens
        # directly in the database for maximum performance
        results = vector_store.similarity_search("What is SAP HANA?")
        ```
    """
    
    def __init__(self, model_id: str):
        """
        Initialize the HanaInternalEmbeddings instance.
        
        Args:
            model_id: The ID of the internal embedding model to use.
                     This should match a valid model ID in your SAP HANA Cloud instance,
                     such as "SAP_NEB.20240301".
        """
        self.model_id = model_id
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Not implemented for HanaInternalEmbeddings.
        
        This method is deliberately not implemented because document embedding
        generation is delegated to SAP HANA's internal functionality.
        
        When using HanaInternalEmbeddings with HanaVectorStore, embeddings are
        generated directly within the database via SQL queries.
        
        Raises:
            NotImplementedError: Always raised to enforce database-side embedding generation.
        """
        raise NotImplementedError(
            "embed_documents is not implemented for HanaInternalEmbeddings. "
            "Embeddings are generated directly in the database when used with HanaVectorStore. "
            "For external embedding generation, use a different Embeddings implementation."
        )
    
    def embed_query(self, text: str) -> List[float]:
        """
        Not implemented for HanaInternalEmbeddings.
        
        This method is deliberately not implemented because query embedding
        generation is delegated to SAP HANA's internal functionality.
        
        When using HanaInternalEmbeddings with HanaVectorStore, embeddings are
        generated directly within the database via SQL queries.
        
        Raises:
            NotImplementedError: Always raised to enforce database-side embedding generation.
        """
        raise NotImplementedError(
            "embed_query is not implemented for HanaInternalEmbeddings. "
            "Embeddings are generated directly in the database when used with HanaVectorStore. "
            "For external embedding generation, use a different Embeddings implementation."
        )
    
    def get_model_id(self) -> str:
        """
        Get the model ID for use with SAP HANA's VECTOR_EMBEDDING function.
        
        Returns:
            The model ID string.
        """
        return self.model_id


class HanaEmbeddingsCache:
    """
    Caching layer for embedding models to improve performance.
    
    This class provides a local cache for embeddings to avoid redundant
    embedding generation, which can significantly improve performance
    when the same texts are embedded multiple times.
    
    Features:
    - In-memory LRU cache with configurable size
    - Time-based expiration (TTL)
    - Optional disk persistence
    - Thread-safe implementation
    - Performance statistics
    
    Examples:
        ```python
        from langchain_core.embeddings import HuggingFaceEmbeddings
        from langchain_hana.embeddings import HanaEmbeddingsCache
        
        # Create base embeddings model
        base_embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create cached embeddings with 1-hour TTL and disk persistence
        cached_embeddings = HanaEmbeddingsCache(
            base_embeddings=base_embeddings,
            ttl_seconds=3600,
            max_size=10000,
            persist_path="/path/to/cache.pkl"
        )
        
        # Use like any other embeddings model
        embedding = cached_embeddings.embed_query("Hello, world!")
        ```
    """
    
    def __init__(
        self,
        base_embeddings: Embeddings,
        ttl_seconds: Optional[int] = 3600,
        max_size: int = 10000,
        persist_path: Optional[str] = None,
        load_on_init: bool = True,
    ):
        """
        Initialize the embeddings cache.
        
        Args:
            base_embeddings: The underlying embeddings model to use for cache misses.
            ttl_seconds: Time-to-live for cache entries in seconds. Set to None for no expiration.
            max_size: Maximum number of entries to keep in the cache.
            persist_path: Path to save the cache to disk. Set to None to disable persistence.
            load_on_init: Whether to load the cache from disk on initialization.
        """
        self.base_embeddings = base_embeddings
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.persist_path = persist_path
        
        # Initialize cache
        self.query_cache: Dict[str, Tuple[List[float], float]] = {}  # (vector, timestamp)
        self.document_cache: Dict[str, Tuple[List[float], float]] = {}  # (vector, timestamp)
        
        # Statistics
        self.query_hits = 0
        self.query_misses = 0
        self.document_hits = 0
        self.document_misses = 0
        
        # Load cache from disk if enabled
        if persist_path and load_on_init and os.path.exists(persist_path):
            self._load_cache()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text, using the cache if available.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector.
        """
        # Check cache
        if text in self.query_cache:
            vector, timestamp = self.query_cache[text]
            
            # Check TTL if enabled
            if self.ttl_seconds is not None and time.time() - timestamp > self.ttl_seconds:
                # Expired, remove from cache
                del self.query_cache[text]
                self.query_misses += 1
            else:
                # Cache hit
                self.query_hits += 1
                return vector
        
        # Cache miss, generate embedding
        self.query_misses += 1
        vector = self.base_embeddings.embed_query(text)
        
        # Add to cache
        self._add_to_query_cache(text, vector)
        
        return vector
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents, using the cache if available.
        
        Args:
            texts: The texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if text in self.document_cache:
                vector, timestamp = self.document_cache[text]
                
                # Check TTL if enabled
                if self.ttl_seconds is not None and time.time() - timestamp > self.ttl_seconds:
                    # Expired, remove from cache
                    del self.document_cache[text]
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
                    self.document_misses += 1
                else:
                    # Cache hit
                    results.append(vector)
                    self.document_hits += 1
            else:
                # Cache miss
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                self.document_misses += 1
        
        # Generate embeddings for cache misses
        if texts_to_embed:
            new_embeddings = self.base_embeddings.embed_documents(texts_to_embed)
            
            # Add to results and cache
            for text, vector in zip(texts_to_embed, new_embeddings):
                self._add_to_document_cache(text, vector)
        
        # Assemble final results in the correct order
        final_results = [None] * len(texts)
        
        # Add cached results
        cached_count = 0
        for i, text in enumerate(texts):
            if i not in indices_to_embed:
                final_results[i] = self.document_cache[text][0]
                cached_count += 1
        
        # Add newly embedded results
        for i, vector in zip(indices_to_embed, new_embeddings):
            final_results[i] = vector
        
        # Persist cache if enabled
        if self.persist_path and (self.query_misses > 0 or self.document_misses > 0):
            self._persist_cache()
        
        return final_results
    
    def _add_to_query_cache(self, text: str, vector: List[float]) -> None:
        """
        Add an entry to the query cache, enforcing size limits.
        
        Args:
            text: The text to cache.
            vector: The embedding vector.
        """
        # Enforce cache size limit
        if len(self.query_cache) >= self.max_size:
            # Remove oldest entries (by timestamp)
            oldest_entries = sorted(
                self.query_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )[:len(self.query_cache) - self.max_size + 1]
            
            for key, _ in oldest_entries:
                del self.query_cache[key]
        
        # Add to cache with current timestamp
        self.query_cache[text] = (vector, time.time())
    
    def _add_to_document_cache(self, text: str, vector: List[float]) -> None:
        """
        Add an entry to the document cache, enforcing size limits.
        
        Args:
            text: The text to cache.
            vector: The embedding vector.
        """
        # Enforce cache size limit
        if len(self.document_cache) >= self.max_size:
            # Remove oldest entries (by timestamp)
            oldest_entries = sorted(
                self.document_cache.items(),
                key=lambda x: x[1][1]  # Sort by timestamp
            )[:len(self.document_cache) - self.max_size + 1]
            
            for key, _ in oldest_entries:
                del self.document_cache[key]
        
        # Add to cache with current timestamp
        self.document_cache[text] = (vector, time.time())
    
    def _persist_cache(self) -> None:
        """
        Save the cache to disk if persistence is enabled.
        """
        if not self.persist_path:
            return
        
        try:
            with open(self.persist_path, "wb") as f:
                pickle.dump({
                    "query_cache": self.query_cache,
                    "document_cache": self.document_cache,
                    "stats": self.get_stats(),
                }, f)
            
            logger.debug(f"Persisted cache to {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to persist cache: {str(e)}")
    
    def _load_cache(self) -> None:
        """
        Load the cache from disk if available.
        """
        if not self.persist_path or not os.path.exists(self.persist_path):
            return
        
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
                
                # Apply TTL if enabled
                current_time = time.time()
                if self.ttl_seconds is not None:
                    # Filter out expired entries
                    self.query_cache = {
                        k: v for k, v in data.get("query_cache", {}).items()
                        if current_time - v[1] <= self.ttl_seconds
                    }
                    
                    self.document_cache = {
                        k: v for k, v in data.get("document_cache", {}).items()
                        if current_time - v[1] <= self.ttl_seconds
                    }
                else:
                    # Load all entries
                    self.query_cache = data.get("query_cache", {})
                    self.document_cache = data.get("document_cache", {})
                
                # Restore stats if available
                stats = data.get("stats", {})
                self.query_hits = stats.get("query_hits", 0)
                self.query_misses = stats.get("query_misses", 0)
                self.document_hits = stats.get("document_hits", 0)
                self.document_misses = stats.get("document_misses", 0)
                
                logger.debug(
                    f"Loaded cache from {self.persist_path} with "
                    f"{len(self.query_cache)} query entries and "
                    f"{len(self.document_cache)} document entries"
                )
        except Exception as e:
            logger.warning(f"Failed to load cache: {str(e)}")
            
            # Initialize empty caches
            self.query_cache = {}
            self.document_cache = {}
    
    def clear_cache(self) -> None:
        """
        Clear the in-memory and on-disk cache.
        """
        self.query_cache = {}
        self.document_cache = {}
        
        # Remove persisted cache if applicable
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                os.remove(self.persist_path)
                logger.debug(f"Removed persisted cache at {self.persist_path}")
            except Exception as e:
                logger.warning(f"Failed to remove persisted cache: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics.
        """
        query_total = self.query_hits + self.query_misses
        query_hit_rate = self.query_hits / query_total if query_total > 0 else 0
        
        document_total = self.document_hits + self.document_misses
        document_hit_rate = self.document_hits / document_total if document_total > 0 else 0
        
        return {
            "query_cache_size": len(self.query_cache),
            "document_cache_size": len(self.document_cache),
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds,
            "query_hits": self.query_hits,
            "query_misses": self.query_misses,
            "query_hit_rate": query_hit_rate,
            "document_hits": self.document_hits,
            "document_misses": self.document_misses,
            "document_hit_rate": document_hit_rate,
            "persistence_enabled": self.persist_path is not None,
        }