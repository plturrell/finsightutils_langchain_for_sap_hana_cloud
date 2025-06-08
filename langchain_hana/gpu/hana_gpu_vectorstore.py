"""
GPU-accelerated vector store for SAP HANA Cloud.

This module provides a vector store implementation that uses GPU acceleration
for vector operations in the data layer, significantly improving performance
for large vector collections.

Key features:
- GPU-accelerated similarity search for faster query processing
- GPU-accelerated MMR search for diverse results
- Memory-efficient vector operations for large collections
- Support for multiple execution modes (full GPU, hybrid, database fallback)
- Async support for non-blocking operations
- Complete CRUD operations with GPU acceleration
- Performance profiling and monitoring capabilities
- Automatic batch processing for large collections
"""

import json
import logging
import time
import asyncio
import functools
import datetime
import threading
import os
from typing import Any, Dict, List, Optional, Tuple, Union, Type, Callable, Iterable

from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables.config import run_in_executor

from langchain_hana.utils import DistanceStrategy
from langchain_hana.error_utils import handle_database_error
from langchain_hana.gpu.data_layer_accelerator import (
    HanaGPUVectorEngine,
    get_vector_engine,
    check_gpu_requirements,
)

# Configure logger
logger = logging.getLogger(__name__)

# Performance profiling globals
_performance_stats = {}
_performance_lock = threading.RLock()
_enable_profiling = os.environ.get("HANA_GPU_PROFILING", "0") == "1"


def profile(method_name=None):
    """
    Decorator for profiling method performance.
    
    Usage:
        @profile
        def my_method(self, ...):
            ...
            
        @profile("custom_name")
        def my_method(self, ...):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Skip profiling if disabled
            if not _enable_profiling:
                return func(*args, **kwargs)
            
            # Get method name
            name = method_name or func.__name__
            
            # Start timing
            start_time = time.time()
            
            try:
                # Call the original function
                result = func(*args, **kwargs)
                success = True
            except Exception as e:
                success = False
                raise e
            finally:
                # End timing
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000
                
                # Update stats
                with _performance_lock:
                    if name not in _performance_stats:
                        _performance_stats[name] = {
                            "count": 0,
                            "success_count": 0,
                            "error_count": 0,
                            "total_time_ms": 0,
                            "min_time_ms": float('inf'),
                            "max_time_ms": 0,
                            "last_run_time": datetime.datetime.now().isoformat(),
                        }
                    
                    stats = _performance_stats[name]
                    stats["count"] += 1
                    if success:
                        stats["success_count"] += 1
                    else:
                        stats["error_count"] += 1
                    stats["total_time_ms"] += duration_ms
                    stats["min_time_ms"] = min(stats["min_time_ms"], duration_ms)
                    stats["max_time_ms"] = max(stats["max_time_ms"], duration_ms)
                    stats["last_run_time"] = datetime.datetime.now().isoformat()
                    stats["avg_time_ms"] = stats["total_time_ms"] / stats["count"]
            
            return result
        
        return wrapper
    
    # Handle both @profile and @profile("name") syntax
    if callable(method_name):
        func = method_name
        method_name = func.__name__
        return decorator(func)
    
    return decorator


def get_performance_stats():
    """Get a copy of the current performance statistics."""
    with _performance_lock:
        return dict(_performance_stats)


def reset_performance_stats():
    """Reset all performance statistics."""
    with _performance_lock:
        _performance_stats.clear()


def enable_profiling(enable=True):
    """Enable or disable performance profiling."""
    global _enable_profiling
    _enable_profiling = enable


class HanaGPUVectorStore(VectorStore):
    """
    GPU-accelerated vector store for SAP HANA Cloud.
    
    This class provides a LangChain vector store implementation that uses
    GPU acceleration for vector operations in the data layer.
    """
    
    def __init__(
        self,
        connection: dbapi.Connection,
        embedding: Embeddings,
        table_name: str = "EMBEDDINGS",
        content_column: str = "VEC_TEXT",
        metadata_column: str = "VEC_META",
        vector_column: str = "VEC_VECTOR",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        gpu_acceleration_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GPU-accelerated vector store.
        
        Args:
            connection: SAP HANA database connection
            embedding: Embedding model
            table_name: Name of the vector table
            content_column: Name of the content column
            metadata_column: Name of the metadata column
            vector_column: Name of the vector column
            distance_strategy: Distance strategy for similarity calculation
            gpu_acceleration_config: GPU acceleration configuration
        """
        self.connection = connection
        self.embedding = embedding
        self.table_name = table_name
        self.content_column = content_column
        self.metadata_column = metadata_column
        self.vector_column = vector_column
        self.distance_strategy = distance_strategy
        
        # GPU acceleration configuration
        self.gpu_acceleration_config = gpu_acceleration_config or {}
        
        # Default GPU configuration
        self.gpu_ids = self.gpu_acceleration_config.get("gpu_ids", None)
        self.cache_size_gb = self.gpu_acceleration_config.get("memory_limit_gb", 4.0)
        self.precision = self.gpu_acceleration_config.get("precision", "float32")
        self.enable_tensor_cores = self.gpu_acceleration_config.get("enable_tensor_cores", True)
        self.enable_prefetch = self.gpu_acceleration_config.get("enable_prefetch", True)
        self.prefetch_size = self.gpu_acceleration_config.get("prefetch_size", 100000)
        self.batch_size = self.gpu_acceleration_config.get("batch_size", 1024)
        
        # Create GPU vector engine
        self.vector_engine = get_vector_engine(
            connection=connection,
            table_name=table_name,
            content_column=content_column,
            metadata_column=metadata_column,
            vector_column=vector_column,
            distance_strategy=distance_strategy,
            gpu_ids=self.gpu_ids,
            cache_size_gb=self.cache_size_gb,
            precision=self.precision,
            enable_tensor_cores=self.enable_tensor_cores,
            enable_prefetch=self.enable_prefetch,
            prefetch_size=self.prefetch_size,
            batch_size=self.batch_size,
        )
        
        # Initialize table if it doesn't exist
        self._initialize_table()
        
        # Build index if configured
        if self.gpu_acceleration_config.get("build_index", False):
            self._build_index()
    
    def _initialize_table(self):
        """Initialize the vector table if it doesn't exist."""
        try:
            # Check if the table exists
            cur = self.connection.cursor()
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{self.table_name}"')
                # Table exists, no need to create it
                cur.close()
                return
            except dbapi.Error:
                # Table doesn't exist, create it
                pass
                
            # Create the table with the required columns
            create_table_sql = f'''
            CREATE TABLE "{self.table_name}" (
                "ID" INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                "{self.content_column}" NCLOB,
                "{self.metadata_column}" NCLOB,
                "{self.vector_column}" REAL_VECTOR(384)
            )
            '''
            
            cur.execute(create_table_sql)
            self.connection.commit()
            logger.info(f"Created table {self.table_name}")
            
        except dbapi.Error as e:
            handle_database_error(e, "initialize_table", {"table_name": self.table_name})
        finally:
            cur.close()
    
    def _build_index(self):
        """Build a GPU-accelerated index for the vector store."""
        index_type = self.gpu_acceleration_config.get("index_type", "hnsw")
        m = self.gpu_acceleration_config.get("hnsw_m", 16)
        ef_construction = self.gpu_acceleration_config.get("hnsw_ef_construction", 200)
        ef_search = self.gpu_acceleration_config.get("hnsw_ef_search", 100)
        
        self.vector_engine.build_index(
            index_type=index_type,
            m=m,
            ef_construction=ef_construction,
            ef_search=ef_search
        )
    
    @profile
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store with GPU-accelerated batch processing.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of IDs of added texts
        """
        # Generate embeddings if not provided
        if embeddings is None:
            # Get batch size from config
            batch_size = self.gpu_acceleration_config.get("embedding_batch_size", 32)
            use_gpu_batching = self.gpu_acceleration_config.get("use_gpu_batching", True)
            
            if use_gpu_batching and len(texts) > batch_size:
                # Process in batches for large collections to avoid OOM
                logger.info(f"Processing {len(texts)} documents in batches of {batch_size}")
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    batch_embeddings = self.embedding.embed_documents(batch_texts)
                    all_embeddings.extend(batch_embeddings)
                    
                    # Log progress for long-running operations
                    if i > 0 and i % (batch_size * 10) == 0:
                        logger.info(f"Processed {i}/{len(texts)} documents")
                        
                embeddings = all_embeddings
            else:
                # Process all at once for smaller collections
                embeddings = self.embedding.embed_documents(texts)
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        try:
            # Get batch size for database operations
            db_batch_size = self.gpu_acceleration_config.get("db_batch_size", 1000)
            
            # Batch insert for better performance
            cur = self.connection.cursor()
            try:
                # Prepare all parameters first
                all_params = []
                for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                    # Convert metadata to JSON string
                    metadata_json = json.dumps(metadata)
                    
                    # Prepare binary format for the vector
                    import struct
                    vector_binary = struct.pack(f"<I{len(embedding)}f", len(embedding), *embedding)
                    
                    all_params.append((text, metadata_json, vector_binary))
                
                # SQL for batch insert
                sql_str = f'''
                INSERT INTO "{self.table_name}" (
                    "{self.content_column}", 
                    "{self.metadata_column}", 
                    "{self.vector_column}"
                ) VALUES (?, ?, ?)
                '''
                
                # Process in batches for very large collections
                if len(all_params) > db_batch_size:
                    total_processed = 0
                    
                    for i in range(0, len(all_params), db_batch_size):
                        batch_params = all_params[i:i+db_batch_size]
                        cur.executemany(sql_str, batch_params)
                        self.connection.commit()
                        
                        total_processed += len(batch_params)
                        logger.info(f"Inserted batch: {total_processed}/{len(all_params)} documents")
                else:
                    # Process all at once for smaller collections
                    cur.executemany(sql_str, all_params)
                    self.connection.commit()
                
                logger.info(f"Added {len(texts)} documents to {self.table_name}")
                
                # Optionally rebuild the index if needed
                if self.gpu_acceleration_config.get("rebuild_index_on_add", False):
                    self._build_index()
                
                # Return empty list as required by the interface
                return []
                
            except dbapi.Error as e:
                self.connection.rollback()
                handle_database_error(e, "add_texts", {"table_name": self.table_name})
                return []
            finally:
                cur.close()
                
        except Exception as e:
            logger.error(f"Error adding texts: {str(e)}")
            return []
            
    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add texts to the vector store.
        
        Args:
            texts: List of texts to add
            metadatas: Optional list of metadata dictionaries
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of IDs of added texts
        """
        # Generate embeddings if not provided
        if embeddings is None:
            # Get batch size from config
            batch_size = self.gpu_acceleration_config.get("embedding_batch_size", 32)
            use_gpu_batching = self.gpu_acceleration_config.get("use_gpu_batching", True)
            
            if use_gpu_batching and len(texts) > batch_size:
                # Process in batches for large collections to avoid OOM
                logger.info(f"Processing {len(texts)} documents in batches of {batch_size}")
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    # Run in executor to not block the event loop
                    batch_embeddings = await run_in_executor(
                        None,
                        self.embedding.embed_documents,
                        batch_texts
                    )
                    all_embeddings.extend(batch_embeddings)
                    
                    # Log progress for long-running operations
                    if i > 0 and i % (batch_size * 10) == 0:
                        logger.info(f"Processed {i}/{len(texts)} documents")
                        
                embeddings = all_embeddings
            else:
                # Process all at once for smaller collections
                embeddings = await run_in_executor(
                    None,
                    self.embedding.embed_documents,
                    texts
                )
                
        # Run the rest of the add operation in an executor
        return await run_in_executor(
            None,
            self._add_texts_with_embeddings,
            texts,
            metadatas,
            embeddings,
            **kwargs
        )
    
    def _add_texts_with_embeddings(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]],
        embeddings: List[List[float]],
        **kwargs: Any,
    ) -> List[str]:
        """
        Helper method for adding texts with pre-computed embeddings.
        Used by aadd_texts to avoid duplicating code.
        """
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        try:
            # Get batch size for database operations
            db_batch_size = self.gpu_acceleration_config.get("db_batch_size", 1000)
            
            # Batch insert for better performance
            cur = self.connection.cursor()
            try:
                # Prepare all parameters first
                all_params = []
                for i, (text, metadata, embedding) in enumerate(zip(texts, metadatas, embeddings)):
                    # Convert metadata to JSON string
                    metadata_json = json.dumps(metadata)
                    
                    # Prepare binary format for the vector
                    import struct
                    vector_binary = struct.pack(f"<I{len(embedding)}f", len(embedding), *embedding)
                    
                    all_params.append((text, metadata_json, vector_binary))
                
                # SQL for batch insert
                sql_str = f'''
                INSERT INTO "{self.table_name}" (
                    "{self.content_column}", 
                    "{self.metadata_column}", 
                    "{self.vector_column}"
                ) VALUES (?, ?, ?)
                '''
                
                # Process in batches for very large collections
                if len(all_params) > db_batch_size:
                    total_processed = 0
                    
                    for i in range(0, len(all_params), db_batch_size):
                        batch_params = all_params[i:i+db_batch_size]
                        cur.executemany(sql_str, batch_params)
                        self.connection.commit()
                        
                        total_processed += len(batch_params)
                        logger.info(f"Inserted batch: {total_processed}/{len(all_params)} documents")
                else:
                    # Process all at once for smaller collections
                    cur.executemany(sql_str, all_params)
                    self.connection.commit()
                
                logger.info(f"Added {len(texts)} documents to {self.table_name}")
                
                # Optionally rebuild the index if needed
                if self.gpu_acceleration_config.get("rebuild_index_on_add", False):
                    self._build_index()
                
                # Return empty list as required by the interface
                return []
                
            except dbapi.Error as e:
                self.connection.rollback()
                handle_database_error(e, "add_texts", {"table_name": self.table_name})
                return []
            finally:
                cur.close()
                
        except Exception as e:
            logger.error(f"Error adding texts: {str(e)}")
            return []
    
    @profile
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search to find similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Documents similar to the query
        """
        # Generate query embedding
        query_embedding = self.embedding.embed_query(query)
        
        # Perform similarity search by vector
        return self.similarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs
        )
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search using a vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Documents similar to the query vector
        """
        try:
            # Additional parameters
            fetch_all_vectors = kwargs.get("fetch_all_vectors", False)
            
            # Perform GPU-accelerated similarity search
            results = self.vector_engine.similarity_search(
                query_vector=embedding,
                k=k,
                filter=filter,
                fetch_all_vectors=fetch_all_vectors
            )
            
            # Convert results to Documents
            documents = []
            for content, metadata_json, score in results:
                # Parse metadata
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}
                
                # Add similarity score to metadata
                metadata["score"] = score
                
                # Create Document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedding.embed_query(query)
        
        try:
            # Additional parameters
            fetch_all_vectors = kwargs.get("fetch_all_vectors", False)
            
            # Perform GPU-accelerated similarity search
            results = self.vector_engine.similarity_search(
                query_vector=query_embedding,
                k=k,
                filter=filter,
                fetch_all_vectors=fetch_all_vectors
            )
            
            # Convert results to (Document, score) tuples
            doc_scores = []
            for content, metadata_json, score in results:
                # Parse metadata
                try:
                    metadata = json.loads(metadata_json) if metadata_json else {}
                except json.JSONDecodeError:
                    metadata = {}
                
                # Create Document
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                doc_scores.append((doc, score))
            
            return doc_scores
            
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            return []
    
    @profile
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform Maximal Marginal Relevance search for diverse results.
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of results to consider for diversity
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            filter: Optional metadata filter
            
        Returns:
            List of diverse Documents similar to the query
        """
        # Generate query embedding
        query_embedding = self.embedding.embed_query(query)
        
        try:
            # Perform GPU-accelerated MMR search
            return self.vector_engine.mmr_search(
                query_vector=query_embedding,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter
            )
            
        except Exception as e:
            logger.error(f"Error in MMR search: {str(e)}")
            return []
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[bool]:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Not supported, raises ValueError
            filter: Metadata filter to identify documents to delete
            
        Returns:
            True if deletion was successful
        """
        if ids is not None:
            raise ValueError("Deletion via IDs is not supported")
            
        if filter is None:
            raise ValueError("Filter is required for deletion")
            
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
            
            # Execute delete query
            sql_str = f'DELETE FROM "{self.table_name}" {filter_clause}'
            
            cur = self.connection.cursor()
            try:
                cur.execute(sql_str, filter_params)
                self.connection.commit()
                
                count = cur.rowcount
                logger.info(f"Deleted {count} documents from {self.table_name}")
                
                # Optionally rebuild the index if needed
                if count > 0 and self.gpu_acceleration_config.get("rebuild_index_on_delete", False):
                    self._build_index()
                
                return True
                
            except dbapi.Error as e:
                self.connection.rollback()
                handle_database_error(e, "delete", {"table_name": self.table_name})
                return False
            finally:
                cur.close()
                
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def update_texts(
        self,
        texts: List[str],
        filter: Dict[str, Any],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        update_embeddings: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Update documents in the vector store.
        
        Args:
            texts: List of texts to update
            filter: Metadata filter to identify documents to update
            metadatas: Optional list of metadata dictionaries
            embeddings: Optional pre-computed embeddings
            update_embeddings: Whether to update embeddings
            
        Returns:
            True if update was successful
        """
        if not texts:
            return False
            
        # Generate embeddings if needed and not provided
        if update_embeddings and embeddings is None:
            embeddings = self.embedding.embed_documents(texts)
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
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
            
            # Construct update SQL based on whether to update embeddings
            if update_embeddings:
                # Update text, metadata, and vector
                text = texts[0]
                metadata_json = json.dumps(metadatas[0])
                
                if embeddings:
                    # Use provided embedding
                    embedding = embeddings[0]
                    
                    # Prepare binary format for the vector
                    import struct
                    vector_binary = struct.pack(f"<I{len(embedding)}f", len(embedding), *embedding)
                    
                    sql_str = f'''
                    UPDATE "{self.table_name}" 
                    SET 
                        "{self.content_column}" = ?, 
                        "{self.metadata_column}" = ?,
                        "{self.vector_column}" = ?
                    {filter_clause}
                    '''
                    
                    params = [text, metadata_json, vector_binary] + filter_params
                else:
                    # Regenerate embedding using the database function
                    # Note: This assumes the database has a VECTOR_EMBEDDING function
                    sql_str = f'''
                    UPDATE "{self.table_name}" 
                    SET 
                        "{self.content_column}" = ?, 
                        "{self.metadata_column}" = ?,
                        "{self.vector_column}" = VECTOR_EMBEDDING(?, 'QUERY', 'TEXT_EMBEDDING')
                    {filter_clause}
                    '''
                    
                    params = [text, metadata_json, text] + filter_params
            else:
                # Update only text and metadata
                text = texts[0]
                metadata_json = json.dumps(metadatas[0])
                
                sql_str = f'''
                UPDATE "{self.table_name}" 
                SET 
                    "{self.content_column}" = ?, 
                    "{self.metadata_column}" = ?
                {filter_clause}
                '''
                
                params = [text, metadata_json] + filter_params
            
            # Execute update
            cur = self.connection.cursor()
            try:
                cur.execute(sql_str, params)
                self.connection.commit()
                
                count = cur.rowcount
                logger.info(f"Updated {count} documents in {self.table_name}")
                
                # Optionally rebuild the index if needed
                if count > 0 and self.gpu_acceleration_config.get("rebuild_index_on_update", False):
                    self._build_index()
                
                return True
                
            except dbapi.Error as e:
                self.connection.rollback()
                handle_database_error(e, "update_texts", {"table_name": self.table_name})
                return False
            finally:
                cur.close()
                
        except Exception as e:
            logger.error(f"Error updating documents: {str(e)}")
            return False
            
    async def aupdate_texts(
        self,
        texts: List[str],
        filter: Dict[str, Any],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        update_embeddings: bool = True,
        **kwargs: Any,
    ) -> bool:
        """
        Asynchronously update documents in the vector store.
        
        Args:
            texts: List of texts to update
            filter: Metadata filter to identify documents to update
            metadatas: Optional list of metadata dictionaries
            embeddings: Optional pre-computed embeddings
            update_embeddings: Whether to update embeddings
            
        Returns:
            True if update was successful
        """
        return await run_in_executor(
            None, 
            self.update_texts, 
            texts=texts,
            filter=filter,
            metadatas=metadatas,
            embeddings=embeddings,
            update_embeddings=update_embeddings,
            **kwargs
        )
    
    @profile
    def upsert_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        filter: Optional[Dict[str, Any]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add or update texts in the vector store.
        
        This method first checks if documents matching the filter exist:
        - If they exist, it updates them with the new content and metadata
        - If they don't exist, it adds the documents as new entries
        
        Args:
            texts: Iterable of strings to add/update in the vectorstore
            metadatas: Optional list of metadata dictionaries
            filter: Filter criteria to identify existing documents to update
                   If None, documents will be added as new entries
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of IDs of added/updated texts (empty list for compatibility)
        """
        # If no filter is provided, just add as new documents
        if filter is None:
            return self.add_texts(
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings,
                **kwargs
            )
            
        # Convert texts to list for proper indexing
        text_list = list(texts)
        
        # Check if documents matching the filter exist
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
            
            # Execute count query
            sql_str = f'SELECT COUNT(*) FROM "{self.table_name}" {filter_clause}'
            
            cur = self.connection.cursor()
            try:
                cur.execute(sql_str, filter_params)
                count = cur.fetchone()[0]
            finally:
                cur.close()
                
            # If matching documents exist, update them
            if count > 0:
                self.update_texts(
                    texts=text_list,
                    filter=filter,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    **kwargs
                )
            # Otherwise, add as new documents
            else:
                self.add_texts(
                    texts=text_list,
                    metadatas=metadatas,
                    embeddings=embeddings,
                    **kwargs
                )
                
            return []
                
        except Exception as e:
            logger.error(f"Error in upsert_texts: {str(e)}")
            return []
            
    async def aupsert_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        filter: Optional[Dict[str, Any]] = None,
        embeddings: Optional[List[List[float]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Asynchronously add or update texts in the vector store.
        
        Args:
            texts: Iterable of strings to add/update
            metadatas: Optional list of metadata dictionaries
            filter: Filter criteria to identify existing documents
            embeddings: Optional pre-computed embeddings
            
        Returns:
            List of IDs of added/updated texts (empty list for compatibility)
        """
        return await run_in_executor(
            None,
            self.upsert_texts,
            texts=texts,
            metadatas=metadatas,
            filter=filter,
            embeddings=embeddings,
            **kwargs
        )
        
    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Optional[bool]:
        """
        Asynchronously delete documents from the vector store.
        
        Args:
            ids: Not supported, raises ValueError
            filter: Metadata filter to identify documents to delete
            
        Returns:
            True if deletion was successful
        """
        return await run_in_executor(
            None,
            self.delete,
            ids=ids,
            filter=filter
        )
        
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Asynchronously perform similarity search.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Documents similar to the query
        """
        # The embedding step can be computationally expensive, so we do it in an executor
        query_embedding = await run_in_executor(
            None,
            self.embedding.embed_query,
            query
        )
        
        # Then perform the search asynchronously
        return await self.asimilarity_search_by_vector(
            embedding=query_embedding,
            k=k,
            filter=filter,
            **kwargs
        )
        
    async def asimilarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Asynchronously perform similarity search using a vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of Documents similar to the query vector
        """
        return await run_in_executor(
            None,
            self.similarity_search_by_vector,
            embedding=embedding,
            k=k,
            filter=filter,
            **kwargs
        )
        
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Asynchronously perform similarity search and return documents with scores.
        
        Args:
            query: Search query
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (Document, score) tuples
        """
        query_embedding = await run_in_executor(
            None,
            self.embedding.embed_query,
            query
        )
        
        return await run_in_executor(
            None,
            self.similarity_search_with_score,
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
        
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Asynchronously perform Maximal Marginal Relevance search.
        
        Args:
            query: Search query
            k: Number of results to return
            fetch_k: Number of results to consider for diversity
            lambda_mult: Diversity parameter (0=max diversity, 1=max relevance)
            filter: Optional metadata filter
            
        Returns:
            List of diverse Documents similar to the query
        """
        query_embedding = await run_in_executor(
            None,
            self.embedding.embed_query,
            query
        )
        
        return await run_in_executor(
            None,
            self.vector_engine.mmr_search,
            query_vector=query_embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter
        )
    
    @profile
    def release_resources(self):
        """Release all GPU resources."""
        if hasattr(self, 'vector_engine'):
            self.vector_engine.release()
            
        logger.info("Released all GPU resources")
        
    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            self.release_resources()
        except:
            pass
            
    def get_performance_stats(self):
        """
        Get performance statistics for this vector store.
        
        Returns:
            Dict: Performance statistics for all profiled operations
        """
        return get_performance_stats()
        
    def reset_performance_stats(self):
        """Reset all performance statistics for this vector store."""
        reset_performance_stats()
        
    def enable_profiling(self, enable=True):
        """
        Enable or disable performance profiling.
        
        Args:
            enable: Whether to enable profiling
        """
        enable_profiling(enable)
        
    def get_gpu_info(self):
        """
        Get information about the GPUs being used.
        
        Returns:
            Dict: Information about GPU usage and configuration
        """
        if not hasattr(self, 'vector_engine'):
            return {"gpu_available": False}
            
        info = {
            "gpu_available": self.vector_engine.gpu_available,
            "gpu_ids": getattr(self.vector_engine, 'gpu_ids', []),
            "gpu_config": {
                "cache_size_gb": self.cache_size_gb,
                "precision": self.precision,
                "enable_tensor_cores": self.enable_tensor_cores,
                "batch_size": self.batch_size,
                "prefetch_enabled": self.enable_prefetch,
                "prefetch_size": self.prefetch_size,
            }
        }
        
        # Add memory stats if available
        if hasattr(self.vector_engine, 'memory_managers') and self.vector_engine.memory_managers:
            memory_stats = []
            for i, manager in enumerate(self.vector_engine.memory_managers):
                if hasattr(manager, 'allocated_memory') and hasattr(manager, 'max_memory'):
                    memory_stats.append({
                        "gpu_id": getattr(manager, 'gpu_id', i),
                        "allocated_memory_gb": getattr(manager, 'allocated_memory', 0) / (1024**3),
                        "max_memory_gb": getattr(manager, 'max_memory', 0) / (1024**3),
                        "utilization": getattr(manager, 'allocated_memory', 0) / max(getattr(manager, 'max_memory', 1), 1),
                        "cached_items": len(getattr(manager, 'cached_data', {})),
                    })
            
            if memory_stats:
                info["memory_stats"] = memory_stats
                
        return info
            

# Factory function for backward compatibility with HanaDB class
def from_hana_db(
    hana_db,
    gpu_acceleration_config: Optional[Dict[str, Any]] = None,
) -> HanaGPUVectorStore:
    """
    Create a GPU-accelerated vector store from an existing HanaDB instance.
    
    Args:
        hana_db: Existing HanaDB instance
        gpu_acceleration_config: GPU acceleration configuration
        
    Returns:
        HanaGPUVectorStore instance
    """
    return HanaGPUVectorStore(
        connection=hana_db.connection,
        embedding=hana_db.embedding,
        table_name=hana_db.table_name,
        content_column=hana_db.content_column,
        metadata_column=hana_db.metadata_column,
        vector_column=hana_db.vector_column,
        distance_strategy=hana_db.distance_strategy,
        gpu_acceleration_config=gpu_acceleration_config,
    )