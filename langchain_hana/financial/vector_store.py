"""
Production-grade vector store integration with SAP HANA Cloud for financial embeddings.

This module provides enterprise-ready integration between financial embeddings
and SAP HANA Cloud's vector capabilities, with optimizations for production use.
"""

import json
import logging
import time
import threading
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Iterator, Type

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from hdbcli import dbapi

from langchain_hana.vectorstores import HanaDB
from langchain_hana.utils import DistanceStrategy
from langchain_hana.financial.embeddings import FinancialEmbeddings

logger = logging.getLogger(__name__)


class FinancialVectorStore:
    """
    Production-grade vector store for financial embeddings.
    
    This class provides a high-performance, enterprise-ready vector store
    implementation for financial embeddings, with optimizations for SAP HANA Cloud.
    """
    
    def __init__(
        self,
        connection: dbapi.Connection,
        embedding_model: Union[str, Embeddings],
        table_name: str = "FINANCIAL_DOCUMENTS",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        content_column: str = "TEXT",
        metadata_column: str = "METADATA",
        vector_column: str = "EMBEDDING",
        create_hnsw_index: bool = True,
        hnsw_index_params: Optional[Dict[str, Any]] = None,
        enable_bulk_operations: bool = True,
        bulk_batch_size: int = 1000,
        enable_transactions: bool = True,
        enable_connection_check: bool = True,
        connection_timeout: int = 60,
        enable_monitoring: bool = True,
        normalize_metadata_keys: bool = True,
        auto_reconnect: bool = True,
        current_user_id: Optional[str] = None,
        application_name: Optional[str] = None,
        enable_lineage: bool = False,
    ):
        """
        Initialize the financial vector store.
        
        Args:
            connection: SAP HANA Cloud connection
            embedding_model: Embedding model or model name
            table_name: Table name for the vector store
            distance_strategy: Distance strategy for similarity search
            content_column: Column name for document content
            metadata_column: Column name for document metadata
            vector_column: Column name for embedding vectors
            create_hnsw_index: Whether to create HNSW index
            hnsw_index_params: Parameters for HNSW index creation
            enable_bulk_operations: Whether to enable bulk operations
            bulk_batch_size: Batch size for bulk operations
            enable_transactions: Whether to enable transactions
            enable_connection_check: Whether to check connection periodically
            connection_timeout: Connection timeout in seconds
            enable_monitoring: Whether to enable performance monitoring
            normalize_metadata_keys: Whether to normalize metadata keys
            auto_reconnect: Whether to automatically reconnect on connection loss
            current_user_id: Current user ID for tracking
            application_name: Application name for tracking
            enable_lineage: Whether to enable data lineage tracking
        """
        self.connection = connection
        self.table_name = table_name.upper()
        self.distance_strategy = distance_strategy
        self.content_column = content_column.upper()
        self.metadata_column = metadata_column.upper()
        self.vector_column = vector_column.upper()
        self.enable_bulk_operations = enable_bulk_operations
        self.bulk_batch_size = bulk_batch_size
        self.enable_transactions = enable_transactions
        self.enable_connection_check = enable_connection_check
        self.connection_timeout = connection_timeout
        self.enable_monitoring = enable_monitoring
        self.normalize_metadata_keys = normalize_metadata_keys
        self.auto_reconnect = auto_reconnect
        self.current_user_id = current_user_id
        self.application_name = application_name
        self.enable_lineage = enable_lineage
        
        # Initialize embedding model
        self.embedding_model = self._initialize_embedding_model(embedding_model)
        
        # Initialize vector store with LangChain integration
        self.vector_store = self._initialize_vector_store()
        
        # Create HNSW index if requested
        if create_hnsw_index:
            self._create_hnsw_index(hnsw_index_params or {})
        
        # Initialize performance monitoring
        if self.enable_monitoring:
            self.operation_times = {}
            self.operation_counts = {}
            self._monitoring_lock = threading.RLock()
        
        # Initialize connection checking
        if self.enable_connection_check:
            self._start_connection_checking()
    
    def _initialize_embedding_model(
        self, 
        embedding_model: Union[str, Embeddings]
    ) -> Embeddings:
        """
        Initialize the embedding model.
        
        Args:
            embedding_model: Embedding model or model name
            
        Returns:
            Initialized embedding model
        """
        if isinstance(embedding_model, str):
            # Create financial embeddings with the specified model
            logger.info(f"Initializing financial embeddings with model: {embedding_model}")
            return FinancialEmbeddings(
                model_name=embedding_model,
                enterprise_mode=True,
            )
        else:
            # Use provided embedding model
            logger.info(f"Using provided embedding model: {type(embedding_model).__name__}")
            return embedding_model
    
    def _initialize_vector_store(self) -> HanaDB:
        """
        Initialize the vector store with LangChain integration.
        
        Returns:
            Initialized vector store
        """
        logger.info(f"Initializing vector store in table: {self.table_name}")
        
        vector_store = HanaDB(
            connection=self.connection,
            embedding=self.embedding_model,
            distance_strategy=self.distance_strategy,
            table_name=self.table_name,
            content_column=self.content_column,
            metadata_column=self.metadata_column,
            vector_column=self.vector_column,
            enable_lineage=self.enable_lineage,
            current_user_id=self.current_user_id,
            current_application=self.application_name,
        )
        
        return vector_store
    
    def _create_hnsw_index(self, index_params: Dict[str, Any]) -> None:
        """
        Create HNSW index for fast similarity search.
        
        Args:
            index_params: Parameters for HNSW index creation
        """
        # Check if index already exists
        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT COUNT(*) FROM INDEXES WHERE INDEX_NAME = 'HNSW_{self.table_name}' "
            f"AND TABLE_NAME = '{self.table_name}'"
        )
        index_exists = cursor.fetchone()[0] > 0
        
        if index_exists:
            logger.info(f"HNSW index already exists for {self.table_name}")
            return
        
        # Get embedding dimension from model if available
        embedding_dimension = None
        if hasattr(self.embedding_model, 'get_embedding_dimension'):
            embedding_dimension = self.embedding_model.get_embedding_dimension()
        else:
            # Generate a sample embedding to determine dimension
            sample_text = "Sample text for dimension detection"
            sample_embedding = self.embedding_model.embed_query(sample_text)
            embedding_dimension = len(sample_embedding)
        
        # Set default HNSW parameters
        hnsw_params = {
            "dims": embedding_dimension,
            "m": 64,  # Number of connections per layer
            "ef_construction": 128,  # Size of the dynamic list for nearest neighbors
            "ef": 200,  # Size of the dynamic list for nearest neighbors at query time
        }
        
        # Override with provided parameters
        hnsw_params.update(index_params)
        
        # Create index
        logger.info(f"Creating HNSW index for {self.table_name} with dimension {embedding_dimension}")
        
        # Convert parameters to string format
        params_str = ';'.join([f"{k.upper()}={v}" for k, v in hnsw_params.items()])
        
        index_sql = (
            f"CREATE HNSW INDEX \"HNSW_{self.table_name}\" "
            f"ON \"{self.table_name}\" (\"{self.vector_column}\") "
            f"PARAMETERS '{params_str}'"
        )
        
        try:
            cursor.execute(index_sql)
            self.connection.commit()
            logger.info(f"HNSW index created for {self.table_name}")
        except Exception as e:
            logger.error(f"Failed to create HNSW index: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def _start_connection_checking(self) -> None:
        """Start periodic connection checking in a background thread."""
        def check_connection():
            while True:
                try:
                    # Check if connection is alive
                    if not self.connection.isconnected():
                        logger.warning("Connection lost, attempting to reconnect")
                        if self.auto_reconnect:
                            self._reconnect()
                except Exception as e:
                    logger.error(f"Error checking connection: {str(e)}")
                
                # Sleep for check interval
                time.sleep(self.connection_timeout)
        
        # Start connection checking thread
        connection_thread = threading.Thread(
            target=check_connection,
            daemon=True
        )
        connection_thread.start()
        logger.info("Connection checking started")
    
    def _reconnect(self) -> None:
        """Reconnect to SAP HANA Cloud."""
        try:
            # Get connection parameters
            connection_params = {}
            cursor = self.connection.cursor()
            cursor.execute("SELECT * FROM SYS.M_CONNECTIONS WHERE CONNECTION_ID = CURRENT_CONNECTION")
            conn_info = cursor.fetchone()
            if conn_info:
                connection_params = {
                    "address": conn_info["HOST"],
                    "port": conn_info["PORT"],
                    "user": conn_info["USER_NAME"],
                }
            cursor.close()
            
            # Reconnect
            self.connection = dbapi.connect(**connection_params)
            logger.info("Successfully reconnected to SAP HANA Cloud")
            
            # Reinitialize vector store
            self.vector_store = self._initialize_vector_store()
        except Exception as e:
            logger.error(f"Failed to reconnect: {str(e)}")
    
    def _track_operation_time(self, operation: str, start_time: float) -> None:
        """
        Track operation time for performance monitoring.
        
        Args:
            operation: Operation name
            start_time: Operation start time
        """
        if not self.enable_monitoring:
            return
        
        elapsed_time = time.time() - start_time
        
        with self._monitoring_lock:
            if operation not in self.operation_times:
                self.operation_times[operation] = []
                self.operation_counts[operation] = 0
            
            self.operation_times[operation].append(elapsed_time)
            self.operation_counts[operation] += 1
            
            # Keep only the last 100 times
            if len(self.operation_times[operation]) > 100:
                self.operation_times[operation] = self.operation_times[operation][-100:]
    
    def _normalize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize metadata for storage.
        
        Args:
            metadata: Metadata to normalize
            
        Returns:
            Normalized metadata
        """
        if not metadata or not self.normalize_metadata_keys:
            return metadata
        
        # Normalize keys to uppercase for consistency with HANA
        return {k.upper(): v for k, v in metadata.items()}
    
    def add_documents(
        self, 
        documents: List[Document], 
        ids: Optional[List[str]] = None,
        batch_size: Optional[int] = None
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
        
        # Normalize metadata if enabled
        if self.normalize_metadata_keys:
            for doc in documents:
                doc.metadata = self._normalize_metadata(doc.metadata)
        
        # Use bulk operations if enabled
        if self.enable_bulk_operations and len(documents) > 1:
            batch_size = batch_size or self.bulk_batch_size
            
            result_ids = []
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_ids = ids[i:i + batch_size] if ids else None
                
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} documents)")
                
                # Add batch
                batch_result_ids = self.vector_store.add_documents(batch, batch_ids)
                result_ids.extend(batch_result_ids)
                
                # Commit transaction if enabled
                if self.enable_transactions:
                    self.connection.commit()
            
            # Track operation time
            self._track_operation_time("add_documents", start_time)
            
            return result_ids
        else:
            # Add documents using standard method
            result_ids = self.vector_store.add_documents(documents, ids)
            
            # Commit transaction if enabled
            if self.enable_transactions:
                self.connection.commit()
            
            # Track operation time
            self._track_operation_time("add_documents", start_time)
            
            return result_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        fetch_k: Optional[int] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform similarity search.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Filter to apply to search
            fetch_k: Number of results to fetch (for MMR)
            **kwargs: Additional arguments for similarity search
            
        Returns:
            List of documents
        """
        start_time = time.time()
        
        # Normalize filter if enabled
        if filter and self.normalize_metadata_keys:
            filter = self._normalize_metadata(filter)
        
        # Perform search
        result = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs
        )
        
        # Track operation time
        self._track_operation_time("similarity_search", start_time)
        
        return result
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Filter to apply to search
            **kwargs: Additional arguments for similarity search
            
        Returns:
            List of document-score tuples
        """
        start_time = time.time()
        
        # Normalize filter if enabled
        if filter and self.normalize_metadata_keys:
            filter = self._normalize_metadata(filter)
        
        # Perform search
        result = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            **kwargs
        )
        
        # Track operation time
        self._track_operation_time("similarity_search_with_score", start_time)
        
        return result
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[Document]:
        """
        Perform max marginal relevance search for diverse results.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of results to fetch
            lambda_mult: Diversity vs. relevance parameter
            filter: Filter to apply to search
            **kwargs: Additional arguments for MMR search
            
        Returns:
            List of documents
        """
        start_time = time.time()
        
        # Normalize filter if enabled
        if filter and self.normalize_metadata_keys:
            filter = self._normalize_metadata(filter)
        
        # Perform search
        result = self.vector_store.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs
        )
        
        # Track operation time
        self._track_operation_time("max_marginal_relevance_search", start_time)
        
        return result
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[bool]:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Document IDs to delete
            filter: Filter to apply for deletion
            **kwargs: Additional arguments for deletion
            
        Returns:
            True if deletion was successful
        """
        start_time = time.time()
        
        # Normalize filter if enabled
        if filter and self.normalize_metadata_keys:
            filter = self._normalize_metadata(filter)
        
        # Perform deletion
        result = self.vector_store.delete(ids=ids, filter=filter, **kwargs)
        
        # Commit transaction if enabled
        if self.enable_transactions:
            self.connection.commit()
        
        # Track operation time
        self._track_operation_time("delete", start_time)
        
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for monitoring.
        
        Returns:
            Dictionary of performance statistics
        """
        if not self.enable_monitoring:
            return {}
        
        stats = {}
        
        with self._monitoring_lock:
            for operation, times in self.operation_times.items():
                if not times:
                    continue
                
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                count = self.operation_counts[operation]
                
                stats[operation] = {
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "count": count,
                    "ops_per_second": 1.0 / avg_time if avg_time > 0 else 0,
                }
        
        # Add embedding model stats if available
        if hasattr(self.embedding_model, "get_performance_stats"):
            embedding_stats = self.embedding_model.get_performance_stats()
            stats["embedding_model"] = embedding_stats
        
        return stats
    
    def create_custom_index(self, index_definition: str) -> None:
        """
        Create a custom index on the vector store.
        
        Args:
            index_definition: SQL statement for index creation
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(index_definition)
            self.connection.commit()
            logger.info("Custom index created successfully")
        except Exception as e:
            logger.error(f"Failed to create custom index: {str(e)}")
            self.connection.rollback()
            raise
        finally:
            cursor.close()


# Factory function to create a production-ready financial vector store
def create_financial_vector_store(
    connection: dbapi.Connection,
    embedding_model: Union[str, Embeddings] = "FinMTEB/Fin-E5-small",
    table_name: str = "FINANCIAL_DOCUMENTS",
    create_hnsw_index: bool = True,
    enterprise_mode: bool = True,
    **kwargs
) -> FinancialVectorStore:
    """
    Create a production-ready financial vector store.
    
    Args:
        connection: SAP HANA Cloud connection
        embedding_model: Embedding model or model name
        table_name: Table name for the vector store
        create_hnsw_index: Whether to create HNSW index
        enterprise_mode: Whether to enable enterprise features
        **kwargs: Additional arguments for FinancialVectorStore
        
    Returns:
        Production-ready FinancialVectorStore instance
    """
    enterprise_kwargs = {}
    
    if enterprise_mode:
        enterprise_kwargs = {
            "enable_bulk_operations": True,
            "enable_transactions": True,
            "enable_connection_check": True,
            "enable_monitoring": True,
            "auto_reconnect": True,
        }
    
    # Override with provided kwargs
    enterprise_kwargs.update(kwargs)
    
    return FinancialVectorStore(
        connection=connection,
        embedding_model=embedding_model,
        table_name=table_name,
        create_hnsw_index=create_hnsw_index,
        **enterprise_kwargs
    )