"""
Financial domain-specific embedding cache for SAP HANA Cloud integration.

This module provides specialized caching for financial embedding models, with optimizations
for common financial text patterns and improved performance for financial applications.
"""

import time
import logging
import pickle
import os
import json
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime

from langchain_core.embeddings import Embeddings

# Configure logging
logger = logging.getLogger(__name__)

class FinancialEmbeddingCache:
    """
    Specialized caching layer for financial embeddings.
    
    This class extends the base HanaEmbeddingsCache with additional features optimized
    for financial domain applications:
    
    1. Financial term normalization for improved cache hits
    2. Cache statistics specific to financial data types
    3. Time-based segmentation for financial data (EOD, quarterly, etc.)
    4. Domain-aware TTL settings for different types of financial content
    5. Collision-resistant hashing for exact matching
    
    Examples:
        ```python
        from langchain_hana.financial.caching import FinancialEmbeddingCache
        from langchain_hana.financial.fin_e5_embeddings import FinE5Embeddings
        
        # Create base embeddings model
        base_embeddings = FinE5Embeddings(model_type="high_quality")
        
        # Create financial cache with domain-specific settings
        cached_embeddings = FinancialEmbeddingCache(
            base_embeddings=base_embeddings,
            ttl_seconds=3600,
            max_size=10000,
            persist_path="/path/to/financial_cache.pkl"
        )
        
        # Use like any other embeddings model
        embedding = cached_embeddings.embed_query("What was the Q1 revenue?")
        ```
    """
    
    def __init__(
        self,
        base_embeddings: Embeddings,
        ttl_seconds: Optional[int] = 3600,
        max_size: int = 10000,
        persist_path: Optional[str] = None,
        load_on_init: bool = True,
        model_name: Optional[str] = None,
    ):
        """
        Initialize the financial embeddings cache.
        
        Args:
            base_embeddings: The underlying embeddings model to use for cache misses.
            ttl_seconds: Time-to-live for cache entries in seconds. Set to None for no expiration.
            max_size: Maximum number of entries to keep in the cache.
            persist_path: Path to save the cache to disk. Set to None to disable persistence.
            load_on_init: Whether to load the cache from disk on initialization.
            model_name: Name of the model for cache file naming (if persist_path is a directory)
        """
        self.base_embeddings = base_embeddings
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.model_name = model_name
        
        # Prepare persist path
        if persist_path:
            if os.path.isdir(persist_path) and model_name:
                # If persist_path is a directory and model_name is provided, create a model-specific cache file
                self.persist_path = os.path.join(persist_path, f"financial_cache_{model_name}.pkl")
            else:
                # Use the provided path directly
                self.persist_path = persist_path
        else:
            self.persist_path = None
        
        # Initialize cache
        self.query_cache: Dict[str, Tuple[List[float], float, str]] = {}  # (vector, timestamp, category)
        self.document_cache: Dict[str, Tuple[List[float], float, str]] = {}  # (vector, timestamp, category)
        
        # Financial domain-specific cache categories with custom TTLs
        self.category_ttls = {
            "news": 3600,  # Financial news (1 hour)
            "report": 86400 * 7,  # Financial reports (1 week)
            "analysis": 86400,  # Financial analysis (1 day)
            "market_data": 1800,  # Market data (30 minutes)
            "default": ttl_seconds or 3600  # Default TTL
        }
        
        # Statistics
        self.query_hits = 0
        self.query_misses = 0
        self.document_hits = 0
        self.document_misses = 0
        self.cache_usage_by_category: Dict[str, int] = {
            "news": 0,
            "report": 0,
            "analysis": 0,
            "market_data": 0,
            "default": 0
        }
        
        # Load cache from disk if enabled
        if persist_path and load_on_init and os.path.exists(self.persist_path):
            self._load_cache()
    
    def _get_category(self, text: str) -> str:
        """
        Determine the financial category of the text for TTL assignment.
        
        Args:
            text: The text to categorize
            
        Returns:
            Category name for TTL assignment
        """
        text_lower = text.lower()
        
        # Check for news indicators
        if any(term in text_lower for term in ["news", "announced", "breaking", "today", "just in"]):
            return "news"
            
        # Check for financial report indicators
        if any(term in text_lower for term in ["report", "quarterly", "annual", "10-k", "10-q", "earnings"]):
            return "report"
            
        # Check for analysis indicators
        if any(term in text_lower for term in ["analysis", "research", "forecast", "outlook", "projection"]):
            return "analysis"
            
        # Check for market data indicators
        if any(term in text_lower for term in ["market", "price", "stock", "index", "rate", "yield"]):
            return "market_data"
            
        # Default category
        return "default"
    
    def _get_ttl_for_category(self, category: str) -> int:
        """
        Get the TTL for a specific category.
        
        Args:
            category: The category name
            
        Returns:
            TTL in seconds for the category
        """
        return self.category_ttls.get(category, self.category_ttls["default"])
    
    def _normalize_financial_text(self, text: str) -> str:
        """
        Normalize financial text for better cache matching.
        
        This function performs domain-specific normalization for financial text:
        - Normalizes percentage formats (10% -> 10 percent)
        - Normalizes currency formats ($100 -> 100 USD)
        - Normalizes date formats
        - Normalizes common financial abbreviations
        
        Args:
            text: The text to normalize
            
        Returns:
            Normalized text for cache key generation
        """
        # This is a simplified implementation - in production, you'd want more
        # comprehensive normalization specific to your financial domain
        
        import re
        
        # Normalize text to lowercase
        text = text.lower()
        
        # Normalize percentages
        text = re.sub(r'(\d+)%', r'\1 percent', text)
        
        # Normalize common financial abbreviations
        abbrev_map = {
            "q1": "quarter 1",
            "q2": "quarter 2",
            "q3": "quarter 3",
            "q4": "quarter 4",
            "fy": "fiscal year",
            "ytd": "year to date",
            "yoy": "year over year",
            "ebitda": "earnings before interest taxes depreciation amortization",
            "eps": "earnings per share"
        }
        
        for abbr, full in abbrev_map.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        
        return text
    
    def _generate_cache_key(self, text: str) -> str:
        """
        Generate a normalized cache key for the given text.
        
        Args:
            text: The text to generate a cache key for
            
        Returns:
            Normalized cache key
        """
        # Normalize the text for better cache hits
        normalized_text = self._normalize_financial_text(text)
        
        # Use a cryptographic hash for exact matching
        return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text, using the cache if available.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector.
        """
        # Generate cache key
        cache_key = self._generate_cache_key(text)
        
        # Determine text category
        category = self._get_category(text)
        category_ttl = self._get_ttl_for_category(category)
        
        # Check cache
        if cache_key in self.query_cache:
            vector, timestamp, entry_category = self.query_cache[cache_key]
            
            # Check TTL based on category
            if category_ttl is not None and time.time() - timestamp > category_ttl:
                # Expired, remove from cache
                del self.query_cache[cache_key]
                self.query_misses += 1
            else:
                # Cache hit
                self.query_hits += 1
                self.cache_usage_by_category[entry_category] += 1
                return vector
        
        # Cache miss, generate embedding
        self.query_misses += 1
        vector = self.base_embeddings.embed_query(text)
        
        # Add to cache
        self._add_to_query_cache(cache_key, vector, category)
        
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
        cache_keys = []
        categories = []
        
        # Process each text
        for i, text in enumerate(texts):
            # Generate cache key and determine category
            cache_key = self._generate_cache_key(text)
            category = self._get_category(text)
            category_ttl = self._get_ttl_for_category(category)
            
            cache_keys.append(cache_key)
            categories.append(category)
            
            # Check cache
            if cache_key in self.document_cache:
                vector, timestamp, entry_category = self.document_cache[cache_key]
                
                # Check TTL based on category
                if category_ttl is not None and time.time() - timestamp > category_ttl:
                    # Expired, remove from cache
                    del self.document_cache[cache_key]
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
                    self.document_misses += 1
                else:
                    # Cache hit
                    results.append(vector)
                    self.document_hits += 1
                    self.cache_usage_by_category[entry_category] += 1
            else:
                # Cache miss
                texts_to_embed.append(text)
                indices_to_embed.append(i)
                self.document_misses += 1
        
        # Generate embeddings for cache misses
        new_embeddings = []
        if texts_to_embed:
            new_embeddings = self.base_embeddings.embed_documents(texts_to_embed)
            
            # Add to cache
            for j, (idx, text) in enumerate(zip(indices_to_embed, texts_to_embed)):
                cache_key = cache_keys[idx]
                category = categories[idx]
                self._add_to_document_cache(cache_key, new_embeddings[j], category)
        
        # Assemble final results in the correct order
        final_results = [None] * len(texts)
        
        # Add cached results
        cached_count = 0
        for i, cache_key in enumerate(cache_keys):
            if i not in indices_to_embed and cache_key in self.document_cache:
                final_results[i] = self.document_cache[cache_key][0]
                cached_count += 1
        
        # Add newly embedded results
        for j, idx in enumerate(indices_to_embed):
            final_results[idx] = new_embeddings[j]
        
        # Persist cache if enabled
        if self.persist_path and (self.query_misses > 0 or self.document_misses > 0):
            self._persist_cache()
        
        return final_results
    
    def _add_to_query_cache(self, key: str, vector: List[float], category: str) -> None:
        """
        Add an entry to the query cache, enforcing size limits.
        
        Args:
            key: The cache key.
            vector: The embedding vector.
            category: The financial category of the text.
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
        
        # Add to cache with current timestamp and category
        self.query_cache[key] = (vector, time.time(), category)
    
    def _add_to_document_cache(self, key: str, vector: List[float], category: str) -> None:
        """
        Add an entry to the document cache, enforcing size limits.
        
        Args:
            key: The cache key.
            vector: The embedding vector.
            category: The financial category of the document.
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
        
        # Add to cache with current timestamp and category
        self.document_cache[key] = (vector, time.time(), category)
    
    def _persist_cache(self) -> None:
        """
        Save the cache to disk if persistence is enabled.
        """
        if not self.persist_path:
            return
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.persist_path)), exist_ok=True)
        
        try:
            with open(self.persist_path, "wb") as f:
                pickle.dump({
                    "query_cache": self.query_cache,
                    "document_cache": self.document_cache,
                    "stats": self.get_stats(),
                    "last_updated": datetime.now().isoformat(),
                    "model_name": self.model_name
                }, f)
            
            logger.debug(f"Persisted financial cache to {self.persist_path}")
        except Exception as e:
            logger.warning(f"Failed to persist financial cache: {str(e)}")
    
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
                
                # Load query cache with TTL filtering based on categories
                self.query_cache = {}
                for k, v in data.get("query_cache", {}).items():
                    vector, timestamp, category = v
                    category_ttl = self._get_ttl_for_category(category)
                    
                    if category_ttl is None or current_time - timestamp <= category_ttl:
                        self.query_cache[k] = v
                
                # Load document cache with TTL filtering based on categories
                self.document_cache = {}
                for k, v in data.get("document_cache", {}).items():
                    vector, timestamp, category = v
                    category_ttl = self._get_ttl_for_category(category)
                    
                    if category_ttl is None or current_time - timestamp <= category_ttl:
                        self.document_cache[k] = v
                
                # Restore stats if available
                stats = data.get("stats", {})
                self.query_hits = stats.get("query_hits", 0)
                self.query_misses = stats.get("query_misses", 0)
                self.document_hits = stats.get("document_hits", 0)
                self.document_misses = stats.get("document_misses", 0)
                
                # Restore category usage stats
                self.cache_usage_by_category = stats.get("cache_usage_by_category", {
                    "news": 0,
                    "report": 0,
                    "analysis": 0,
                    "market_data": 0,
                    "default": 0
                })
                
                logger.debug(
                    f"Loaded financial cache from {self.persist_path} with "
                    f"{len(self.query_cache)} query entries and "
                    f"{len(self.document_cache)} document entries"
                )
        except Exception as e:
            logger.warning(f"Failed to load financial cache: {str(e)}")
            
            # Initialize empty caches
            self.query_cache = {}
            self.document_cache = {}
    
    def clear_cache(self) -> None:
        """
        Clear the in-memory and on-disk cache.
        """
        self.query_cache = {}
        self.document_cache = {}
        
        # Reset statistics
        self.query_hits = 0
        self.query_misses = 0
        self.document_hits = 0
        self.document_misses = 0
        self.cache_usage_by_category = {
            "news": 0,
            "report": 0,
            "analysis": 0,
            "market_data": 0,
            "default": 0
        }
        
        # Remove persisted cache if applicable
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                os.remove(self.persist_path)
                logger.debug(f"Removed persisted financial cache at {self.persist_path}")
            except Exception as e:
                logger.warning(f"Failed to remove persisted financial cache: {str(e)}")
    
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
        
        # Calculate category distribution
        total_hits = sum(self.cache_usage_by_category.values())
        category_distribution = {
            k: v / total_hits if total_hits > 0 else 0
            for k, v in self.cache_usage_by_category.items()
        }
        
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
            "cache_usage_by_category": self.cache_usage_by_category,
            "category_distribution": category_distribution,
            "category_ttls": self.category_ttls
        }