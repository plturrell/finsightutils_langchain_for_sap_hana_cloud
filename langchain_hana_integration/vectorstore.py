"""
Production-grade vector store implementation for SAP HANA Cloud.

This module provides a robust vector store implementation that integrates
LangChain with SAP HANA Cloud's vector capabilities.
"""

import os
import time
import json
import logging
import threading
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.runnables.config import run_in_executor

from langchain_hana_integration.connection import get_connection, create_connection_pool
from langchain_hana_integration.embeddings import HanaOptimizedEmbeddings
from langchain_hana_integration.utils.distance import DistanceStrategy, compute_similarity
from langchain_hana_integration.utils.serialization import serialize_vector, deserialize_vector, serialize_metadata, deserialize_metadata
from langchain_hana_integration.utils.query import build_vector_search_query, build_filter_clause, build_metadata_projection
from langchain_hana_integration.exceptions import DatabaseError, VectorOperationError, convert_db_error, InvalidSchemaError

logger = logging.getLogger(__name__)

# Default values
DEFAULT_TABLE_NAME = "LANGCHAIN_VECTORS"
DEFAULT_CONTENT_COLUMN = "VEC_TEXT"
DEFAULT_METADATA_COLUMN = "VEC_META"
DEFAULT_VECTOR_COLUMN = "VEC_VECTOR"
DEFAULT_VECTOR_COLUMN_TYPE = "REAL_VECTOR"


class SAP_HANA_VectorStore(VectorStore):
    """
    Production-grade vector store implementation for SAP HANA Cloud.
    
    Features:
    - High-performance vector operations
    - Connection pooling and management
    - Comprehensive error handling
    - Advanced filtering and search capabilities
    - Performance optimization and monitoring
    - Asynchronous operations
    - Automatic schema management
    - HNSW indexing
    """
    
    def __init__(
        self,
        embedding: Embeddings,
        connection_params: Optional[Dict[str, Any]] = None,
        pool_name: str = "default",
        distance_strategy: DistanceStrategy = DistanceStrategy.COSINE,
        table_name: str = DEFAULT_TABLE_NAME,
        content_column: str = DEFAULT_CONTENT_COLUMN,
        metadata_column: str = DEFAULT_METADATA_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        vector_column_type: str = DEFAULT_VECTOR_COLUMN_TYPE,
        specific_metadata_columns: Optional[List[str]] = None,
        create_table: bool = True,
        batch_size: int = 100,
        timeout: float = 30.0,
        retry_count: int = 3,
        enable_logging: bool = True,
        auto_create_index: bool = False
    ):
        """
        Initialize the SAP HANA vector store.
        
        Args:
            embedding: Embeddings provider
            connection_params: Connection parameters or None to use default pool
            pool_name: Name of the connection pool to use
            distance_strategy: Distance strategy for similarity search
            table_name: Name of the table to store vectors
            content_column: Name of the column for document content
            metadata_column: Name of the column for metadata
            vector_column: Name of the column for vector data
            vector_column_type: Type of vector column ('REAL_VECTOR' or 'HALF_VECTOR')
            specific_metadata_columns: Optional list of metadata fields to store in separate columns
            create_table: Whether to create the table if it doesn't exist
            batch_size: Batch size for database operations
            timeout: Timeout in seconds for database operations
            retry_count: Number of retries for failed operations
            enable_logging: Whether to enable detailed logging
            auto_create_index: Whether to automatically create an HNSW index
        """
        # Store configuration
        self.embedding = embedding
        self.pool_name = pool_name
        self.distance_strategy = distance_strategy
        self.table_name = table_name
        self.content_column = content_column
        self.metadata_column = metadata_column
        self.vector_column = vector_column
        self.vector_column_type = vector_column_type
        self.specific_metadata_columns = specific_metadata_columns or []
        self.batch_size = batch_size
        self.timeout = timeout
        self.retry_count = retry_count
        self.enable_logging = enable_logging
        self.auto_create_index = auto_create_index
        
        # Initialize connection pool if needed
        if connection_params:
            create_connection_pool(
                connection_params=connection_params,
                pool_name=pool_name
            )
        
        # Determine if we're using internal embeddings
        self._use_internal_embeddings = False
        if hasattr(self.embedding, "get_model_id") and callable(self.embedding.get_model_id):
            model_id = self.embedding.get_model_id()
            if model_id:
                self._use_internal_embeddings = True
                self._internal_embedding_model_id = model_id
        
        # Initialize performance metrics
        self._metrics = {
            "add_texts_calls": 0,
            "search_calls": 0,
            "total_documents_added": 0,
            "total_search_time": 0.0,
            "total_add_time": 0.0
        }
        
        # Create the table if requested
        if create_table:
            self._create_table_if_not_exists()
            
            # Create index if requested
            if auto_create_index:
                self.create_hnsw_index()
    
    def _create_table_if_not_exists(self) -> None:
        """Create the vector table if it doesn't exist."""
        with get_connection(self.pool_name) as connection:
            cursor = connection.cursor()
            try:
                # Check if table exists
                cursor.execute(
                    "SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
                    "AND TABLE_NAME = ?",
                    (self.table_name,)
                )
                if cursor.fetchone()[0] > 0:
                    if self.enable_logging:
                        logger.info(f"Table '{self.table_name}' already exists")
                    
                    # Validate columns
                    self._validate_table_schema(cursor)
                    return
                
                # Build CREATE TABLE statement
                columns = [
                    f'"{self.content_column}" NCLOB',
                    f'"{self.metadata_column}" NCLOB',
                    f'"{self.vector_column}" {self.vector_column_type}'
                ]
                
                # Add specific metadata columns
                for col in self.specific_metadata_columns:
                    columns.append(f'"{col}" NVARCHAR(2000)')
                
                create_stmt = f'CREATE TABLE "{self.table_name}" ({", ".join(columns)})'
                
                # Execute create statement
                cursor.execute(create_stmt)
                connection.commit()
                
                if self.enable_logging:
                    logger.info(f"Created table '{self.table_name}'")
            
            except dbapi.Error as e:
                connection.rollback()
                raise convert_db_error(e, "table_creation")
            
            finally:
                cursor.close()
    
    def _validate_table_schema(self, cursor) -> None:
        """
        Validate that the existing table has the expected schema.
        
        Args:
            cursor: Database cursor
            
        Raises:
            InvalidSchemaError: If the table schema doesn't match expected
        """
        try:
            # Check required columns
            required_columns = [
                (self.content_column, ["NCLOB", "NVARCHAR", "VARCHAR"]),
                (self.metadata_column, ["NCLOB", "NVARCHAR", "VARCHAR"]),
                (self.vector_column, [self.vector_column_type]),
            ]
            
            for col_name, valid_types in required_columns:
                cursor.execute(
                    "SELECT DATA_TYPE_NAME FROM SYS.TABLE_COLUMNS "
                    "WHERE SCHEMA_NAME = CURRENT_SCHEMA "
                    "AND TABLE_NAME = ? AND COLUMN_NAME = ?",
                    (self.table_name, col_name)
                )
                
                result = cursor.fetchone()
                if not result:
                    raise InvalidSchemaError(
                        f"Column '{col_name}' does not exist in table '{self.table_name}'",
                        {"table_name": self.table_name, "missing_column": col_name}
                    )
                
                col_type = result[0]
                if col_type not in valid_types:
                    raise InvalidSchemaError(
                        f"Column '{col_name}' has type '{col_type}', expected one of {valid_types}",
                        {"table_name": self.table_name, "column_name": col_name, 
                         "actual_type": col_type, "expected_types": valid_types}
                    )
            
            # Check specific metadata columns
            for col_name in self.specific_metadata_columns:
                cursor.execute(
                    "SELECT COUNT(*) FROM SYS.TABLE_COLUMNS "
                    "WHERE SCHEMA_NAME = CURRENT_SCHEMA "
                    "AND TABLE_NAME = ? AND COLUMN_NAME = ?",
                    (self.table_name, col_name)
                )
                
                if cursor.fetchone()[0] == 0:
                    raise InvalidSchemaError(
                        f"Specific metadata column '{col_name}' does not exist in table '{self.table_name}'",
                        {"table_name": self.table_name, "missing_column": col_name}
                    )
        
        except dbapi.Error as e:
            raise convert_db_error(e, "schema_validation")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Add texts and their embeddings to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries, one per text
            ids: Optional list of IDs (not used, included for compatibility)
            **kwargs: Additional keyword arguments
            
        Returns:
            Empty list (IDs are managed by the database)
            
        Raises:
            VectorOperationError: If adding texts fails
        """
        if not texts:
            return []
        
        start_time = time.time()
        self._metrics["add_texts_calls"] += 1
        self._metrics["total_documents_added"] += len(texts)
        
        # If using internal embeddings, delegate to specialized method
        if self._use_internal_embeddings:
            return self._add_texts_with_internal_embeddings(texts, metadatas)
        
        try:
            # Generate embeddings
            embeddings = self.embedding.embed_documents(list(texts))
            
            # Prepare parameters for batch insert
            params = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas else {}
                
                # Extract specific metadata fields
                specific_metadata = []
                for col in self.specific_metadata_columns:
                    specific_metadata.append(metadata.get(col))
                
                # Serialize vector and metadata
                vector_binary = serialize_vector(embeddings[i], self.vector_column_type)
                metadata_json = serialize_metadata(metadata)
                
                # Add parameters for this document
                row_params = [text, metadata_json, vector_binary] + specific_metadata
                params.append(row_params)
            
            # Insert in batches
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # Build insert statement
                    columns = [self.content_column, self.metadata_column, self.vector_column]
                    columns.extend(self.specific_metadata_columns)
                    
                    placeholders = ", ".join(["?"] * len(columns))
                    insert_stmt = f'INSERT INTO "{self.table_name}" ("{self.content_column}", "{self.metadata_column}", "{self.vector_column}"'
                    
                    if self.specific_metadata_columns:
                        insert_stmt += ', "' + '", "'.join(self.specific_metadata_columns) + '"'
                    
                    insert_stmt += f") VALUES ({placeholders})"
                    
                    # Execute in batches
                    for i in range(0, len(params), self.batch_size):
                        batch = params[i:i + self.batch_size]
                        cursor.executemany(insert_stmt, batch)
                    
                    connection.commit()
                    
                    if self.enable_logging:
                        logger.info(f"Added {len(texts)} documents to '{self.table_name}'")
                
                except dbapi.Error as e:
                    connection.rollback()
                    raise convert_db_error(e, "add_texts")
                
                finally:
                    cursor.close()
            
            self._metrics["total_add_time"] += time.time() - start_time
            return []
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to add texts: {e}", {"error": str(e)})
            raise
    
    def _add_texts_with_internal_embeddings(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Add texts using SAP HANA internal embedding generation.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries, one per text
            
        Returns:
            Empty list (IDs are managed by the database)
            
        Raises:
            VectorOperationError: If adding texts fails
        """
        try:
            # Prepare parameters for batch insert
            params = []
            for i, text in enumerate(texts):
                metadata = metadatas[i] if metadatas else {}
                
                # Extract specific metadata fields
                specific_metadata = []
                for col in self.specific_metadata_columns:
                    specific_metadata.append(metadata.get(col))
                
                # Serialize metadata
                metadata_json = serialize_metadata(metadata)
                
                # Add parameters for this document
                row_params = {
                    "content": text,
                    "metadata": metadata_json,
                    "model_id": self._internal_embedding_model_id
                }
                
                # Add specific metadata columns
                for j, col in enumerate(self.specific_metadata_columns):
                    row_params[col] = specific_metadata[j]
                
                params.append(row_params)
            
            # Insert in batches
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # Build insert statement with VECTOR_EMBEDDING function
                    insert_stmt = f'INSERT INTO "{self.table_name}" ("{self.content_column}", "{self.metadata_column}", "{self.vector_column}"'
                    
                    if self.specific_metadata_columns:
                        insert_stmt += ', "' + '", "'.join(self.specific_metadata_columns) + '"'
                    
                    insert_stmt += f") VALUES (:content, :metadata, VECTOR_EMBEDDING(:content, 'DOCUMENT', :model_id)"
                    
                    if self.specific_metadata_columns:
                        insert_stmt += ", " + ", ".join([f":{col}" for col in self.specific_metadata_columns])
                    
                    insert_stmt += ")"
                    
                    # Execute in batches
                    for i in range(0, len(params), self.batch_size):
                        batch = params[i:i + self.batch_size]
                        cursor.executemany(insert_stmt, batch)
                    
                    connection.commit()
                    
                    if self.enable_logging:
                        logger.info(f"Added {len(texts)} documents to '{self.table_name}' using internal embeddings")
                
                except dbapi.Error as e:
                    connection.rollback()
                    raise convert_db_error(e, "add_texts_internal")
                
                finally:
                    cursor.close()
            
            return []
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to add texts with internal embeddings: {e}", {"error": str(e)})
            raise
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Search for documents similar to the query text.
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional filter to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            List of documents most similar to the query
            
        Raises:
            VectorOperationError: If search fails
        """
        results = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in results]
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the query text and return with scores.
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional filter to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            VectorOperationError: If search fails
        """
        start_time = time.time()
        self._metrics["search_calls"] += 1
        
        try:
            # If using internal embeddings, use HANA's embedding function
            if self._use_internal_embeddings:
                results = self._similarity_search_with_internal_embeddings(query, k, filter)
            else:
                # Generate query embedding
                query_embedding = self.embedding.embed_query(query)
                
                # Search by vector
                results = self.similarity_search_by_vector(query_embedding, k, filter)
            
            search_time = time.time() - start_time
            self._metrics["total_search_time"] += search_time
            
            if self.enable_logging:
                logger.info(f"Similarity search completed in {search_time:.4f}s, found {len(results)} results")
            
            return results
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to perform similarity search: {e}", {"error": str(e)})
            raise
    
    def _similarity_search_with_internal_embeddings(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search using HANA's internal embedding function.
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional filter to apply
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            VectorOperationError: If search fails
        """
        try:
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # Build filter clause if needed
                    filter_clause = ""
                    filter_params = []
                    
                    if filter:
                        filter_clause, filter_params = build_filter_clause(
                            filter,
                            self.metadata_column,
                            self.specific_metadata_columns
                        )
                    
                    # Build the query with VECTOR_EMBEDDING function
                    embedding_expr = "VECTOR_EMBEDDING(?, 'QUERY', ?)"
                    embedding_params = [query, self._internal_embedding_model_id]
                    
                    sql, params = build_vector_search_query(
                        table_name=self.table_name,
                        vector_column=self.vector_column,
                        content_column=self.content_column,
                        metadata_column=self.metadata_column,
                        embedding_expr=embedding_expr,
                        distance_strategy=self.distance_strategy,
                        limit=k,
                        filter_clause=filter_clause,
                        filter_params=filter_params,
                        specific_metadata_columns=self.specific_metadata_columns,
                        embedding_params=embedding_params
                    )
                    
                    # Execute the query
                    cursor.execute(sql, params)
                    
                    # Process results
                    results = []
                    for row in cursor.fetchall():
                        content = row[0]
                        metadata_json = row[1]
                        vector_binary = row[2]
                        score = row[3]
                        
                        # Deserialize metadata
                        metadata = deserialize_metadata(metadata_json)
                        
                        # Add specific metadata columns if present
                        if self.specific_metadata_columns:
                            for i, col in enumerate(self.specific_metadata_columns):
                                if row[4 + i] is not None:
                                    metadata[col] = row[4 + i]
                        
                        # Create document
                        doc = Document(page_content=content, metadata=metadata)
                        results.append((doc, score))
                    
                    return results
                
                except dbapi.Error as e:
                    raise convert_db_error(e, "similarity_search_internal")
                
                finally:
                    cursor.close()
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to perform similarity search with internal embeddings: {e}", {"error": str(e)})
            raise
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to the embedding vector.
        
        Args:
            embedding: Query embedding vector
            k: Number of documents to return
            filter: Optional filter to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            List of (document, score) tuples
            
        Raises:
            VectorOperationError: If search fails
        """
        try:
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # Build filter clause if needed
                    filter_clause = ""
                    filter_params = []
                    
                    if filter:
                        filter_clause, filter_params = build_filter_clause(
                            filter,
                            self.metadata_column,
                            self.specific_metadata_columns
                        )
                    
                    # Convert embedding to string for SQL
                    embedding_str = str(embedding)
                    
                    # Build the query
                    if self.vector_column_type == "REAL_VECTOR":
                        embedding_expr = f"TO_REAL_VECTOR('{embedding_str}')"
                    elif self.vector_column_type == "HALF_VECTOR":
                        embedding_expr = f"TO_HALF_VECTOR('{embedding_str}')"
                    else:
                        raise ValueError(f"Unsupported vector type: {self.vector_column_type}")
                    
                    sql, params = build_vector_search_query(
                        table_name=self.table_name,
                        vector_column=self.vector_column,
                        content_column=self.content_column,
                        metadata_column=self.metadata_column,
                        embedding_expr=embedding_expr,
                        distance_strategy=self.distance_strategy,
                        limit=k,
                        filter_clause=filter_clause,
                        filter_params=filter_params,
                        specific_metadata_columns=self.specific_metadata_columns
                    )
                    
                    # Execute the query
                    cursor.execute(sql, params)
                    
                    # Process results
                    results = []
                    for row in cursor.fetchall():
                        content = row[0]
                        metadata_json = row[1]
                        vector_binary = row[2]
                        score = row[3]
                        
                        # Deserialize metadata
                        metadata = deserialize_metadata(metadata_json)
                        
                        # Add specific metadata columns if present
                        if self.specific_metadata_columns:
                            for i, col in enumerate(self.specific_metadata_columns):
                                if row[4 + i] is not None:
                                    metadata[col] = row[4 + i]
                        
                        # Create document
                        doc = Document(page_content=content, metadata=metadata)
                        results.append((doc, score))
                    
                    return results
                
                except dbapi.Error as e:
                    raise convert_db_error(e, "similarity_search_vector")
                
                finally:
                    cursor.close()
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to perform similarity search by vector: {e}", {"error": str(e)})
            raise
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Search for documents with Maximal Marginal Relevance.
        
        MMR optimizes for similarity to the query AND diversity among results.
        
        Args:
            query: Query text
            k: Number of documents to return
            fetch_k: Number of documents to fetch for MMR calculation
            lambda_mult: Balance between relevance and diversity (0-1)
            filter: Optional filter to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            List of documents selected by MMR
            
        Raises:
            VectorOperationError: If search fails
        """
        if k > fetch_k:
            raise ValueError(f"k ({k}) must be less than or equal to fetch_k ({fetch_k})")
        
        try:
            # Generate query embedding
            if self._use_internal_embeddings:
                # We need to extract embedding for MMR calculation
                query_embedding = self._extract_internal_embedding(query)
            else:
                query_embedding = self.embedding.embed_query(query)
            
            # Fetch candidates
            candidate_docs_with_scores = self.similarity_search_by_vector(
                embedding=query_embedding,
                k=fetch_k,
                filter=filter
            )
            
            # Extract documents and embeddings
            candidate_docs = []
            candidate_embeddings = []
            
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # For each candidate, retrieve its embedding vector
                    for doc, _ in candidate_docs_with_scores:
                        candidate_docs.append(doc)
                        
                        # We need to extract the actual embedding for each document
                        cursor.execute(
                            f'SELECT "{self.vector_column}" FROM "{self.table_name}" '
                            f'WHERE "{self.content_column}" = ?',
                            (doc.page_content,)
                        )
                        
                        row = cursor.fetchone()
                        if row:
                            vector_binary = row[0]
                            embedding = deserialize_vector(vector_binary, self.vector_column_type)
                            candidate_embeddings.append(embedding)
                
                except dbapi.Error as e:
                    raise convert_db_error(e, "mmr_search")
                
                finally:
                    cursor.close()
            
            # Calculate MMR
            mmr_indices = self._calculate_mmr(
                query_embedding=query_embedding,
                doc_embeddings=candidate_embeddings,
                lambda_mult=lambda_mult,
                k=k
            )
            
            # Return selected documents
            return [candidate_docs[i] for i in mmr_indices]
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to perform MMR search: {e}", {"error": str(e)})
            raise
    
    def _extract_internal_embedding(self, text: str) -> List[float]:
        """
        Extract embedding vector using HANA's internal embedding function.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
            
        Raises:
            VectorOperationError: If embedding extraction fails
        """
        try:
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # Execute query to generate embedding
                    cursor.execute(
                        f"SELECT VECTOR_EMBEDDING(?, 'QUERY', ?) FROM SYS.DUMMY",
                        (text, self._internal_embedding_model_id)
                    )
                    
                    row = cursor.fetchone()
                    if not row or not row[0]:
                        raise VectorOperationError("Failed to generate internal embedding", 
                                                 {"text": text[:100] + "..." if len(text) > 100 else text})
                    
                    # Deserialize the binary vector
                    vector_binary = row[0]
                    return deserialize_vector(vector_binary, "REAL_VECTOR")
                
                except dbapi.Error as e:
                    raise convert_db_error(e, "extract_internal_embedding")
                
                finally:
                    cursor.close()
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to extract internal embedding: {e}", {"error": str(e)})
            raise
    
    def _calculate_mmr(
        self,
        query_embedding: List[float],
        doc_embeddings: List[List[float]],
        lambda_mult: float = 0.5,
        k: int = 4
    ) -> List[int]:
        """
        Calculate Maximum Marginal Relevance.
        
        Args:
            query_embedding: Query embedding
            doc_embeddings: Document embeddings
            lambda_mult: Balance between relevance and diversity (0-1)
            k: Number of documents to select
            
        Returns:
            Indices of selected documents
        """
        if not doc_embeddings:
            return []
        
        if len(doc_embeddings) <= k:
            return list(range(len(doc_embeddings)))
        
        # Convert to numpy for efficient computation
        query_embedding_np = np.array(query_embedding)
        doc_embeddings_np = np.array(doc_embeddings)
        
        # Calculate similarity to query
        similarity_to_query = []
        for doc_embedding in doc_embeddings_np:
            similarity = compute_similarity(query_embedding, doc_embedding.tolist(), self.distance_strategy)
            similarity_to_query.append(similarity)
        
        # Initialize selected indices and remaining indices
        selected_indices = []
        remaining_indices = list(range(len(doc_embeddings)))
        
        # Select first document with highest similarity to query
        first_idx = np.argmax(similarity_to_query)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select the rest
        for _ in range(k - 1):
            if not remaining_indices:
                break
            
            # Calculate MMR scores
            mmr_scores = []
            for idx in remaining_indices:
                # Similarity to query component
                sim_query = similarity_to_query[idx]
                
                # Similarity to already selected documents
                max_sim_selected = 0
                for selected_idx in selected_indices:
                    sim = compute_similarity(
                        doc_embeddings[idx],
                        doc_embeddings[selected_idx],
                        self.distance_strategy
                    )
                    max_sim_selected = max(max_sim_selected, sim)
                
                # MMR score
                mmr_score = lambda_mult * sim_query - (1 - lambda_mult) * max_sim_selected
                mmr_scores.append(mmr_score)
            
            # Select document with highest MMR score
            next_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)
        
        return selected_indices
    
    def create_hnsw_index(
        self,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        ef_search: Optional[int] = None,
        index_name: Optional[str] = None
    ) -> None:
        """
        Create an HNSW vector index for faster similarity search.
        
        Args:
            m: Maximum number of connections per node (defaults to HANA default)
            ef_construction: Size of dynamic candidate list during construction
            ef_search: Size of dynamic candidate list during search
            index_name: Optional custom name for the index
            
        Raises:
            VectorOperationError: If index creation fails
        """
        try:
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # Check if index already exists
                    distance_func = "COSINE_SIMILARITY" if self.distance_strategy == DistanceStrategy.COSINE else "L2DISTANCE"
                    default_index_name = f"{self.table_name}_{distance_func}_IDX"
                    index_name = index_name or default_index_name
                    
                    cursor.execute(
                        "SELECT COUNT(*) FROM SYS.INDEXES "
                        "WHERE SCHEMA_NAME = CURRENT_SCHEMA "
                        "AND TABLE_NAME = ? AND INDEX_NAME = ?",
                        (self.table_name, index_name)
                    )
                    
                    if cursor.fetchone()[0] > 0:
                        if self.enable_logging:
                            logger.info(f"Index '{index_name}' already exists")
                        return
                    
                    # Build CREATE INDEX statement
                    create_stmt = (
                        f"CREATE HNSW VECTOR INDEX {index_name} "
                        f'ON "{self.table_name}" ("{self.vector_column}") '
                        f"SIMILARITY FUNCTION {distance_func}"
                    )
                    
                    # Add build configuration if provided
                    build_config = {}
                    if m is not None:
                        build_config["M"] = m
                    if ef_construction is not None:
                        build_config["efConstruction"] = ef_construction
                    
                    if build_config:
                        create_stmt += f" BUILD CONFIGURATION '{json.dumps(build_config)}'"
                    
                    # Add search configuration if provided
                    search_config = {}
                    if ef_search is not None:
                        search_config["efSearch"] = ef_search
                    
                    if search_config:
                        create_stmt += f" SEARCH CONFIGURATION '{json.dumps(search_config)}'"
                    
                    # Always create in ONLINE mode
                    create_stmt += " ONLINE"
                    
                    # Execute create index statement
                    cursor.execute(create_stmt)
                    connection.commit()
                    
                    if self.enable_logging:
                        logger.info(f"Created HNSW index '{index_name}' on '{self.table_name}'")
                
                except dbapi.Error as e:
                    connection.rollback()
                    raise convert_db_error(e, "create_index")
                
                finally:
                    cursor.close()
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to create HNSW index: {e}", {"error": str(e)})
            raise
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[bool]:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Not used, included for compatibility
            filter: Filter to identify documents to delete
            **kwargs: Additional keyword arguments
            
        Returns:
            True if successful
            
        Raises:
            VectorOperationError: If deletion fails
            ValueError: If neither ids nor filter is provided
        """
        if ids:
            logger.warning("Deletion by IDs is not supported, use filter instead")
        
        if not filter:
            raise ValueError("Filter is required for deletion")
        
        try:
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # Build filter clause
                    filter_clause, filter_params = build_filter_clause(
                        filter,
                        self.metadata_column,
                        self.specific_metadata_columns
                    )
                    
                    # Build delete statement
                    delete_stmt = f'DELETE FROM "{self.table_name}"'
                    if filter_clause:
                        delete_stmt += f" WHERE {filter_clause}"
                    
                    # Execute delete statement
                    cursor.execute(delete_stmt, filter_params)
                    rows_affected = cursor.rowcount
                    connection.commit()
                    
                    if self.enable_logging:
                        logger.info(f"Deleted {rows_affected} documents from '{self.table_name}'")
                    
                    return True
                
                except dbapi.Error as e:
                    connection.rollback()
                    raise convert_db_error(e, "delete")
                
                finally:
                    cursor.close()
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to delete documents: {e}", {"error": str(e)})
            raise
    
    def update_texts(
        self,
        texts: List[str],
        filter: Dict[str, Any],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        update_embeddings: bool = True,
        **kwargs: Any
    ) -> bool:
        """
        Update documents in the vector store.
        
        Args:
            texts: List of text strings to update to
            filter: Filter to identify documents to update
            metadatas: Optional list of metadata dictionaries to update to
            update_embeddings: Whether to update embeddings
            **kwargs: Additional keyword arguments
            
        Returns:
            True if successful
            
        Raises:
            VectorOperationError: If update fails
            ValueError: If filter is not provided
        """
        if not filter:
            raise ValueError("Filter is required for update")
        
        if not texts:
            return True
        
        try:
            with get_connection(self.pool_name) as connection:
                cursor = connection.cursor()
                try:
                    # Build filter clause
                    filter_clause, filter_params = build_filter_clause(
                        filter,
                        self.metadata_column,
                        self.specific_metadata_columns
                    )
                    
                    # Handle text and metadata
                    text = texts[0]  # Currently supporting only single text update
                    metadata = metadatas[0] if metadatas else None
                    
                    # Prepare update parameters
                    update_params = []
                    
                    # Build update statement
                    update_stmt = f'UPDATE "{self.table_name}" SET "{self.content_column}" = ?'
                    update_params.append(text)
                    
                    # Update metadata if provided
                    if metadata:
                        # Extract specific metadata fields
                        specific_metadata = {}
                        for col in self.specific_metadata_columns:
                            if col in metadata:
                                specific_metadata[col] = metadata[col]
                        
                        # Update metadata JSON
                        update_stmt += f', "{self.metadata_column}" = ?'
                        update_params.append(serialize_metadata(metadata))
                        
                        # Update specific metadata columns
                        for col, value in specific_metadata.items():
                            update_stmt += f', "{col}" = ?'
                            update_params.append(value)
                    
                    # Update embedding if requested
                    if update_embeddings:
                        if self._use_internal_embeddings:
                            # Use HANA's embedding function
                            update_stmt += f', "{self.vector_column}" = VECTOR_EMBEDDING(?, \'DOCUMENT\', ?)'
                            update_params.extend([text, self._internal_embedding_model_id])
                        else:
                            # Generate embedding and serialize
                            embedding = self.embedding.embed_query(text)
                            vector_binary = serialize_vector(embedding, self.vector_column_type)
                            
                            update_stmt += f', "{self.vector_column}" = ?'
                            update_params.append(vector_binary)
                    
                    # Add WHERE clause
                    if filter_clause:
                        update_stmt += f" WHERE {filter_clause}"
                    
                    # Combine parameters
                    all_params = update_params + filter_params
                    
                    # Execute update statement
                    cursor.execute(update_stmt, all_params)
                    rows_affected = cursor.rowcount
                    connection.commit()
                    
                    if self.enable_logging:
                        logger.info(f"Updated {rows_affected} documents in '{self.table_name}'")
                    
                    return rows_affected > 0
                
                except dbapi.Error as e:
                    connection.rollback()
                    raise convert_db_error(e, "update")
                
                finally:
                    cursor.close()
        
        except Exception as e:
            if not isinstance(e, VectorOperationError):
                raise VectorOperationError(f"Failed to update documents: {e}", {"error": str(e)})
            raise
    
    async def aadd_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Asynchronously add texts to the vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs (not used, included for compatibility)
            **kwargs: Additional keyword arguments
            
        Returns:
            Empty list (IDs are managed by the database)
        """
        return await run_in_executor(None, self.add_texts, texts, metadatas, ids, **kwargs)
    
    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Asynchronously search for documents similar to the query text.
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional filter to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            List of documents most similar to the query
        """
        return await run_in_executor(None, self.similarity_search, query, k, filter, **kwargs)
    
    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """
        Asynchronously search for documents with scores.
        
        Args:
            query: Query text
            k: Number of documents to return
            filter: Optional filter to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            List of (document, score) tuples
        """
        return await run_in_executor(None, self.similarity_search_with_score, query, k, filter, **kwargs)
    
    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Document]:
        """
        Asynchronously search for documents with Maximal Marginal Relevance.
        
        Args:
            query: Query text
            k: Number of documents to return
            fetch_k: Number of documents to fetch for MMR calculation
            lambda_mult: Balance between relevance and diversity (0-1)
            filter: Optional filter to apply
            **kwargs: Additional keyword arguments
            
        Returns:
            List of documents selected by MMR
        """
        return await run_in_executor(
            None, 
            self.max_marginal_relevance_search,
            query, k, fetch_k, lambda_mult, filter, **kwargs
        )
    
    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Optional[bool]:
        """
        Asynchronously delete documents from the vector store.
        
        Args:
            ids: Not used, included for compatibility
            filter: Filter to identify documents to delete
            **kwargs: Additional keyword arguments
            
        Returns:
            True if successful
        """
        return await run_in_executor(None, self.delete, ids, filter, **kwargs)
    
    async def aupdate_texts(
        self,
        texts: List[str],
        filter: Dict[str, Any],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        update_embeddings: bool = True,
        **kwargs: Any
    ) -> bool:
        """
        Asynchronously update documents in the vector store.
        
        Args:
            texts: List of text strings to update to
            filter: Filter to identify documents to update
            metadatas: Optional list of metadata dictionaries to update to
            update_embeddings: Whether to update embeddings
            **kwargs: Additional keyword arguments
            
        Returns:
            True if successful
        """
        return await run_in_executor(
            None,
            self.update_texts,
            texts, filter, metadatas, update_embeddings, **kwargs
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the vector store.
        
        Returns:
            Dictionary of metrics
        """
        # Calculate derived metrics
        if self._metrics["search_calls"] > 0:
            avg_search_time = self._metrics["total_search_time"] / self._metrics["search_calls"]
            self._metrics["avg_search_time"] = avg_search_time
        
        if self._metrics["add_texts_calls"] > 0:
            avg_add_time = self._metrics["total_add_time"] / self._metrics["add_texts_calls"]
            self._metrics["avg_add_time"] = avg_add_time
        
        # Add configuration info
        self._metrics.update({
            "table_name": self.table_name,
            "distance_strategy": str(self.distance_strategy),
            "vector_column_type": self.vector_column_type,
            "using_internal_embeddings": self._use_internal_embeddings
        })
        
        return self._metrics
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        connection_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> "SAP_HANA_VectorStore":
        """
        Create a vector store from texts.
        
        Args:
            texts: List of text strings
            embedding: Embeddings provider
            metadatas: Optional list of metadata dictionaries
            connection_params: Optional connection parameters
            **kwargs: Additional arguments for vector store initialization
            
        Returns:
            Initialized vector store
        """
        # Create vector store
        vector_store = cls(
            embedding=embedding,
            connection_params=connection_params,
            **kwargs
        )
        
        # Add texts
        vector_store.add_texts(texts, metadatas)
        
        return vector_store