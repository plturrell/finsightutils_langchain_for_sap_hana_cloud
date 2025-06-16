"""
Production-ready integration for financial embeddings with SAP HANA Cloud.

This module provides a complete, turnkey solution for deploying financial
embeddings in production environments with SAP HANA Cloud.
"""

import os
import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Import core components
from langchain_hana.financial.embeddings import (
    FinancialEmbeddings,
    create_production_financial_embeddings,
)
from langchain_hana.financial.gpu_optimization import GPUOptimizer
from langchain_hana.financial.vector_store import (
    FinancialVectorStore,
    create_financial_vector_store,
)
from langchain_hana.financial.caching import (
    FinancialQueryCache,
    create_query_cache,
)

# Initialize module logger
logger = logging.getLogger(__name__)


class FinancialEmbeddingSystem:
    """
    Production-ready embedding system for financial applications.
    
    This class provides a comprehensive solution for deploying financial
    embeddings in production environments, with built-in support for
    connection management, caching, monitoring, and error handling.
    """
    
    def __init__(
        self,
        connection_params: Dict[str, Any],
        model_name: Optional[str] = None,
        quality_tier: str = "balanced",
        table_name: str = "FINANCIAL_DOCUMENTS",
        cache_dir: Optional[str] = None,
        redis_url: Optional[str] = None,
        log_file: Optional[str] = None,
        connection_pool_size: int = 5,
        enable_monitoring: bool = True,
        auto_reconnect: bool = True,
        request_timeout: int = 60,
        enable_hnsw_index: bool = True,
        enable_semantic_cache: bool = True,
        max_cache_size: int = 10000,
        cache_ttl_hours: int = 24,
        enable_background_jobs: bool = True,
        backup_interval_hours: int = 24,
        metrics_export_interval: int = 300,
        auto_refresh_connection: bool = True,
        connection_refresh_interval: int = 1800,
    ):
        """
        Initialize the financial embedding system.
        
        Args:
            connection_params: SAP HANA connection parameters
            model_name: Custom model name (overrides quality_tier if provided)
            quality_tier: Quality tier ('high', 'balanced', 'efficient')
            table_name: Table name for the vector store
            cache_dir: Directory for disk cache
            redis_url: Redis URL for distributed cache
            log_file: Log file path
            connection_pool_size: Connection pool size
            enable_monitoring: Whether to enable performance monitoring
            auto_reconnect: Whether to automatically reconnect on connection loss
            request_timeout: Request timeout in seconds
            enable_hnsw_index: Whether to create HNSW index
            enable_semantic_cache: Whether to enable semantic caching
            max_cache_size: Maximum cache size
            cache_ttl_hours: Cache TTL in hours
            enable_background_jobs: Whether to enable background jobs
            backup_interval_hours: Backup interval in hours
            metrics_export_interval: Metrics export interval in seconds
            auto_refresh_connection: Whether to refresh connection periodically
            connection_refresh_interval: Connection refresh interval in seconds
        """
        self.connection_params = connection_params
        self.model_name = model_name
        self.quality_tier = quality_tier
        self.table_name = table_name
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), "cache", "financial_system")
        self.redis_url = redis_url
        self.log_file = log_file
        self.connection_pool_size = connection_pool_size
        self.enable_monitoring = enable_monitoring
        self.auto_reconnect = auto_reconnect
        self.request_timeout = request_timeout
        self.enable_hnsw_index = enable_hnsw_index
        self.enable_semantic_cache = enable_semantic_cache
        self.max_cache_size = max_cache_size
        self.cache_ttl_hours = cache_ttl_hours
        self.enable_background_jobs = enable_background_jobs
        self.backup_interval_hours = backup_interval_hours
        self.metrics_export_interval = metrics_export_interval
        self.auto_refresh_connection = auto_refresh_connection
        self.connection_refresh_interval = connection_refresh_interval
        
        # Configure logging
        self._configure_logging()
        
        # Initialize components
        self.connection_pool = self._initialize_connection_pool()
        self.embedding_model = self._initialize_embedding_model()
        self.vector_store = self._initialize_vector_store()
        self.query_cache = self._initialize_query_cache()
        
        # Initialize monitoring
        if self.enable_monitoring:
            self.metrics = {
                "queries_processed": 0,
                "documents_added": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "errors": 0,
                "avg_query_time": 0,
                "total_query_time": 0,
                "start_time": time.time(),
            }
            self.metrics_lock = threading.RLock()
        
        # Start background jobs
        if self.enable_background_jobs:
            self._start_background_jobs()
        
        logger.info(f"Financial embedding system initialized (model: {self.model_name or quality_tier})")
    
    def _configure_logging(self) -> None:
        """Configure logging for the system."""
        if self.log_file:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            
            # Configure file handler
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            
            # Add handler to logger
            logger.addHandler(file_handler)
            
            # Set level to INFO
            logger.setLevel(logging.INFO)
    
    def _initialize_connection_pool(self) -> List[dbapi.Connection]:
        """
        Initialize connection pool for SAP HANA.
        
        Returns:
            List of connections
        """
        logger.info(f"Initializing connection pool (size: {self.connection_pool_size})")
        
        # Create connections
        connections = []
        for i in range(self.connection_pool_size):
            try:
                conn = dbapi.connect(**self.connection_params)
                connections.append(conn)
            except Exception as e:
                logger.error(f"Failed to create connection {i}: {str(e)}")
        
        if not connections:
            raise RuntimeError("Failed to create any connections in the pool")
        
        return connections
    
    def _get_connection(self) -> dbapi.Connection:
        """
        Get a connection from the pool.
        
        Returns:
            SAP HANA connection
        """
        # Simple round-robin for now
        conn = self.connection_pool[0]
        
        # Check if connection is still alive
        if not conn.isconnected():
            logger.warning("Connection lost, reconnecting...")
            if self.auto_reconnect:
                try:
                    conn = dbapi.connect(**self.connection_params)
                    self.connection_pool[0] = conn
                except Exception as e:
                    logger.error(f"Failed to reconnect: {str(e)}")
                    raise
        
        # Move to end of list for round-robin
        self.connection_pool.append(self.connection_pool.pop(0))
        
        return conn
    
    def _initialize_embedding_model(self) -> Embeddings:
        """
        Initialize the embedding model.
        
        Returns:
            Embedding model
        """
        logger.info(f"Initializing embedding model (quality tier: {self.quality_tier})")
        
        # Check if model_name is a local path
        if self.model_name and (self.model_name.startswith('./') or os.path.isabs(self.model_name)):
            # Check if path exists
            if os.path.exists(self.model_name):
                logger.info(f"Using local model at {self.model_name}")
                return FinancialEmbeddings(
                    model_name=self.model_name,
                    enterprise_mode=True,
                )
            else:
                logger.warning(f"Local model path {self.model_name} does not exist")
        
        # Try to use local model manager if available
        try:
            from langchain_hana.financial.local_models import create_local_model_manager
            
            # Create local model manager
            model_manager = create_local_model_manager(
                default_model=self.model_name or self.quality_tier,
            )
            
            # Get model path or download if needed
            try:
                model_path = model_manager.get_model_path(self.model_name or self.quality_tier)
                
                logger.info(f"Using model from local manager: {model_path}")
                return FinancialEmbeddings(
                    model_name=model_path,
                    enterprise_mode=True,
                )
            except Exception as e:
                logger.warning(f"Failed to get model from local manager: {str(e)}")
        except ImportError:
            logger.debug("Local model manager not available, using standard embedding model")
        
        # Fallback to standard embedding model
        return create_production_financial_embeddings(
            model_name=self.model_name,
            quality_tier=self.quality_tier,
            memory_tier="auto",
            enterprise_mode=True,
        )
    
    def _initialize_vector_store(self) -> FinancialVectorStore:
        """
        Initialize the vector store.
        
        Returns:
            Vector store
        """
        logger.info(f"Initializing vector store (table: {self.table_name})")
        
        # Get connection
        connection = self._get_connection()
        
        # Create vector store
        return create_financial_vector_store(
            connection=connection,
            embedding_model=self.embedding_model,
            table_name=self.table_name,
            create_hnsw_index=self.enable_hnsw_index,
            enterprise_mode=True,
            enable_monitoring=self.enable_monitoring,
            auto_reconnect=self.auto_reconnect,
        )
    
    def _initialize_query_cache(self) -> FinancialQueryCache:
        """
        Initialize the query cache.
        
        Returns:
            Query cache
        """
        if not self.enable_semantic_cache:
            logger.info("Semantic cache disabled")
            return None
        
        logger.info(f"Initializing query cache (TTL: {self.cache_ttl_hours}h)")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create query cache
        return create_query_cache(
            cache_dir=self.cache_dir,
            redis_url=self.redis_url,
            ttl_hours=self.cache_ttl_hours,
            semantic_threshold=0.92,
            enable_cross_user=False,
        )
    
    def _start_background_jobs(self) -> None:
        """Start background jobs for maintenance tasks."""
        # Connection refresh job
        if self.auto_refresh_connection:
            def refresh_connections():
                while True:
                    try:
                        logger.info("Refreshing connections...")
                        for i, conn in enumerate(self.connection_pool):
                            if not conn.isconnected():
                                try:
                                    self.connection_pool[i] = dbapi.connect(**self.connection_params)
                                    logger.info(f"Reconnected connection {i}")
                                except Exception as e:
                                    logger.error(f"Failed to reconnect connection {i}: {str(e)}")
                    except Exception as e:
                        logger.error(f"Error in connection refresh job: {str(e)}")
                    
                    # Sleep for refresh interval
                    time.sleep(self.connection_refresh_interval)
            
            # Start connection refresh thread
            connection_thread = threading.Thread(
                target=refresh_connections,
                daemon=True
            )
            connection_thread.start()
            logger.info(f"Connection refresh job started (interval: {self.connection_refresh_interval}s)")
        
        # Metrics export job
        if self.enable_monitoring:
            def export_metrics():
                while True:
                    try:
                        # Export metrics to log
                        metrics = self.get_metrics()
                        logger.info(f"Performance metrics: {json.dumps(metrics)}")
                    except Exception as e:
                        logger.error(f"Error in metrics export job: {str(e)}")
                    
                    # Sleep for export interval
                    time.sleep(self.metrics_export_interval)
            
            # Start metrics export thread
            metrics_thread = threading.Thread(
                target=export_metrics,
                daemon=True
            )
            metrics_thread.start()
            logger.info(f"Metrics export job started (interval: {self.metrics_export_interval}s)")
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: Documents to add
            ids: Optional document IDs
            batch_size: Batch size for bulk operations
            
        Returns:
            List of document IDs
        """
        start_time = time.time()
        
        try:
            # Add documents to vector store
            result_ids = self.vector_store.add_documents(
                documents=documents,
                ids=ids,
                batch_size=batch_size,
            )
            
            # Update metrics
            if self.enable_monitoring:
                with self.metrics_lock:
                    self.metrics["documents_added"] += len(documents)
            
            return result_ids
            
        except Exception as e:
            # Update error metrics
            if self.enable_monitoring:
                with self.metrics_lock:
                    self.metrics["errors"] += 1
            
            # Log error
            logger.error(f"Error adding documents: {str(e)}")
            raise
            
        finally:
            # Update metrics
            if self.enable_monitoring:
                elapsed_time = time.time() - start_time
                logger.debug(f"Document addition took {elapsed_time:.3f}s")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        use_cache: bool = True,
    ) -> List[Document]:
        """
        Perform similarity search with caching.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Filter to apply to search
            user_id: User ID for user-specific cache
            use_cache: Whether to use cache
            
        Returns:
            List of documents
        """
        start_time = time.time()
        cache_hit = False
        
        try:
            # Check cache if enabled
            if use_cache and self.enable_semantic_cache and self.query_cache:
                cached_result = self.query_cache.get_query_result(
                    query=query,
                    user_id=user_id,
                    filter_params=filter,
                )
                
                if cached_result:
                    # Update metrics
                    if self.enable_monitoring:
                        with self.metrics_lock:
                            self.metrics["queries_processed"] += 1
                            self.metrics["cache_hits"] += 1
                    
                    # Return cached result
                    cache_hit = True
                    return cached_result
            
            # Cache miss, perform search
            if self.enable_monitoring and use_cache and self.enable_semantic_cache:
                with self.metrics_lock:
                    self.metrics["cache_misses"] += 1
            
            # Get embedding for semantic caching
            query_embedding = None
            if self.enable_semantic_cache:
                query_embedding = self.embedding_model.embed_query(query)
            
            # Perform search
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter,
            )
            
            # Cache result if enabled
            if use_cache and self.enable_semantic_cache and self.query_cache and not cache_hit:
                self.query_cache.set_query_result(
                    query=query,
                    result=results,
                    user_id=user_id,
                    filter_params=filter,
                    vector=query_embedding,
                )
            
            # Update metrics
            if self.enable_monitoring:
                with self.metrics_lock:
                    self.metrics["queries_processed"] += 1
            
            return results
            
        except Exception as e:
            # Update error metrics
            if self.enable_monitoring:
                with self.metrics_lock:
                    self.metrics["errors"] += 1
            
            # Log error
            logger.error(f"Error in similarity search: {str(e)}")
            raise
            
        finally:
            # Update metrics
            if self.enable_monitoring:
                elapsed_time = time.time() - start_time
                with self.metrics_lock:
                    query_count = self.metrics["queries_processed"]
                    total_time = self.metrics["total_query_time"] + elapsed_time
                    self.metrics["total_query_time"] = total_time
                    self.metrics["avg_query_time"] = total_time / query_count if query_count > 0 else 0
                
                cache_status = "hit" if cache_hit else "miss"
                logger.debug(f"Query took {elapsed_time:.3f}s (cache: {cache_status})")
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform max marginal relevance search for diverse results.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of results to fetch
            lambda_mult: Diversity vs. relevance parameter
            filter: Filter to apply to search
            
        Returns:
            List of documents
        """
        start_time = time.time()
        
        try:
            # Perform MMR search
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
            )
            
            # Update metrics
            if self.enable_monitoring:
                with self.metrics_lock:
                    self.metrics["queries_processed"] += 1
            
            return results
            
        except Exception as e:
            # Update error metrics
            if self.enable_monitoring:
                with self.metrics_lock:
                    self.metrics["errors"] += 1
            
            # Log error
            logger.error(f"Error in MMR search: {str(e)}")
            raise
            
        finally:
            # Update metrics
            if self.enable_monitoring:
                elapsed_time = time.time() - start_time
                with self.metrics_lock:
                    query_count = self.metrics["queries_processed"]
                    total_time = self.metrics["total_query_time"] + elapsed_time
                    self.metrics["total_query_time"] = total_time
                    self.metrics["avg_query_time"] = total_time / query_count if query_count > 0 else 0
                
                logger.debug(f"MMR query took {elapsed_time:.3f}s")
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[bool]:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Document IDs to delete
            filter: Filter to apply for deletion
            
        Returns:
            True if deletion was successful
        """
        try:
            # Delete from vector store
            return self.vector_store.delete(ids=ids, filter=filter)
            
        except Exception as e:
            # Update error metrics
            if self.enable_monitoring:
                with self.metrics_lock:
                    self.metrics["errors"] += 1
            
            # Log error
            logger.error(f"Error deleting documents: {str(e)}")
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.enable_monitoring:
            return {}
        
        with self.metrics_lock:
            metrics = self.metrics.copy()
        
        # Add uptime
        uptime = time.time() - metrics["start_time"]
        metrics["uptime_seconds"] = uptime
        metrics["uptime_hours"] = uptime / 3600
        metrics["uptime_days"] = uptime / 86400
        
        # Add cache stats if available
        if self.enable_semantic_cache and self.query_cache:
            cache_stats = self.query_cache.get_stats()
            metrics["cache"] = cache_stats
        
        # Add vector store stats if available
        if hasattr(self.vector_store, "get_performance_stats"):
            vector_store_stats = self.vector_store.get_performance_stats()
            metrics["vector_store"] = vector_store_stats
        
        # Add embedding model stats if available
        if hasattr(self.embedding_model, "get_performance_stats"):
            embedding_stats = self.embedding_model.get_performance_stats()
            metrics["embedding_model"] = embedding_stats
        
        return metrics
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status
        """
        health = {
            "status": "healthy",
            "components": {
                "connections": "healthy",
                "embedding_model": "healthy",
                "vector_store": "healthy",
                "query_cache": "healthy" if self.enable_semantic_cache else "disabled",
            },
            "timestamp": time.time(),
        }
        
        # Check connections
        healthy_connections = 0
        for conn in self.connection_pool:
            if conn.isconnected():
                healthy_connections += 1
        
        if healthy_connections == 0:
            health["status"] = "unhealthy"
            health["components"]["connections"] = "unhealthy"
        elif healthy_connections < self.connection_pool_size:
            health["components"]["connections"] = "degraded"
        
        health["connections"] = {
            "total": self.connection_pool_size,
            "healthy": healthy_connections,
        }
        
        # Check embedding model
        try:
            self.embedding_model.embed_query("Test query")
        except Exception as e:
            health["status"] = "unhealthy"
            health["components"]["embedding_model"] = "unhealthy"
            health["embedding_model_error"] = str(e)
        
        # Check vector store
        try:
            # Simple query to test vector store
            self.vector_store.similarity_search("Test query", k=1)
        except Exception as e:
            health["status"] = "unhealthy"
            health["components"]["vector_store"] = "unhealthy"
            health["vector_store_error"] = str(e)
        
        # Check query cache
        if self.enable_semantic_cache and self.query_cache:
            try:
                self.query_cache.get_stats()
            except Exception as e:
                health["status"] = "degraded"  # Cache is non-critical
                health["components"]["query_cache"] = "unhealthy"
                health["query_cache_error"] = str(e)
        
        return health
    
    def shutdown(self) -> None:
        """Gracefully shut down the system."""
        logger.info("Shutting down financial embedding system...")
        
        # Close connections
        for conn in self.connection_pool:
            try:
                if conn.isconnected():
                    conn.close()
            except Exception as e:
                logger.error(f"Error closing connection: {str(e)}")
        
        # Clear caches
        if self.enable_semantic_cache and self.query_cache:
            try:
                # Persist cache
                pass
            except Exception as e:
                logger.error(f"Error persisting cache: {str(e)}")
        
        logger.info("Financial embedding system shut down")


# Factory function to create a production-ready system
def create_financial_system(
    host: str,
    port: int,
    user: str,
    password: str,
    encrypt: bool = True,
    ssl_validate: bool = False,
    model_name: Optional[str] = None,
    quality_tier: str = "balanced",
    table_name: str = "FINANCIAL_DOCUMENTS",
    log_file: Optional[str] = None,
    connection_pool_size: int = 5,
    enable_semantic_cache: bool = True,
    cache_dir: Optional[str] = None,
    redis_url: Optional[str] = None,
) -> FinancialEmbeddingSystem:
    """
    Create a production-ready financial embedding system.
    
    Args:
        host: SAP HANA hostname
        port: SAP HANA port
        user: SAP HANA username
        password: SAP HANA password
        encrypt: Whether to use encryption
        ssl_validate: Whether to validate SSL certificates
        model_name: Custom model name (overrides quality_tier if provided)
        quality_tier: Quality tier ('high', 'balanced', 'efficient')
        table_name: Table name for the vector store
        log_file: Log file path
        connection_pool_size: Connection pool size
        enable_semantic_cache: Whether to enable semantic caching
        cache_dir: Directory for disk cache
        redis_url: Redis URL for distributed cache
        
    Returns:
        Production-ready FinancialEmbeddingSystem instance
    """
    # Create connection parameters
    connection_params = {
        "address": host,
        "port": port,
        "user": user,
        "password": password,
        "encrypt": encrypt,
        "sslValidateCertificate": ssl_validate,
    }
    
    # Create system
    return FinancialEmbeddingSystem(
        connection_params=connection_params,
        model_name=model_name,
        quality_tier=quality_tier,
        table_name=table_name,
        cache_dir=cache_dir,
        redis_url=redis_url,
        log_file=log_file,
        connection_pool_size=connection_pool_size,
        enable_semantic_cache=enable_semantic_cache,
    )