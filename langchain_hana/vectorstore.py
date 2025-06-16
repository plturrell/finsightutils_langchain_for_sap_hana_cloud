"""
SAP HANA Cloud Vector Store for LangChain

This module provides a LangChain VectorStore implementation for SAP HANA Cloud,
enabling storage and retrieval of embeddings using SAP HANA's vector capabilities.

Key features:
- Vector similarity search with various distance metrics
- Rich metadata filtering
- Maximal Marginal Relevance (MMR) search for diverse results
- HNSW vector indexing for performance optimization
- Support for both internal and external embeddings
"""

from __future__ import annotations

import json
import logging
import re
import struct
import time
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
    Callable,
    Type,
)

import numpy as np
from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from langchain_hana.connection import (
    create_connection, 
    test_connection, 
    create_connection_pool,
    get_connection_pool
)
from langchain_hana.utils import (
    DistanceStrategy,
    convert_distance_strategy_to_sql,
    create_vector_table,
    create_hnsw_index,
    serialize_vector,
    deserialize_vector,
)

logger = logging.getLogger(__name__)

# Default values
DEFAULT_TABLE_NAME = "LANGCHAIN_VECTORS"
DEFAULT_CONTENT_COLUMN = "CONTENT"
DEFAULT_METADATA_COLUMN = "METADATA"
DEFAULT_VECTOR_COLUMN = "VECTOR"
DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE

class HanaVectorStore(VectorStore):
    """
    Vector store implementation for SAP HANA Cloud.
    
    This class provides a LangChain-compatible vector store that uses SAP HANA Cloud's
    vector capabilities for storing and retrieving embeddings.
    
    Features:
    - Stores document text, metadata, and embeddings
    - Performs similarity search with various distance metrics
    - Supports rich metadata filtering
    - Optimized with HNSW indexing for fast similarity search
    - Provides maximal marginal relevance search for diverse results
    
    Prerequisites:
    - SAP HANA Cloud instance with vector capabilities
    - hdbcli Python package installed (`pip install hdbcli`)
    
    Examples:
        ```python
        from langchain_hana.vectorstore import HanaVectorStore
        from langchain_core.embeddings import HuggingFaceEmbeddings
        
        # Initialize embedding model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create vector store
        vector_store = HanaVectorStore(
            host="your-hana-host.hanacloud.ondemand.com",
            port=443,
            user="your-username",
            password="your-password",
            embedding=embeddings,
            table_name="MY_VECTORS",
            create_table=True  # Create table if it doesn't exist
        )
        
        # Add documents
        docs = ["Document 1", "Document 2", "Document 3"]
        metadata = [{"source": "wiki"}, {"source": "web"}, {"source": "book"}]
        vector_store.add_texts(docs, metadata)
        
        # Search
        results = vector_store.similarity_search("query text", k=3)
        for doc in results:
            print(doc.page_content, doc.metadata)
        ```
    """

    def __init__(
        self,
        embedding: Embeddings,
        connection: Optional[dbapi.Connection] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        schema_name: Optional[str] = None,
        table_name: str = DEFAULT_TABLE_NAME,
        content_column: str = DEFAULT_CONTENT_COLUMN,
        metadata_column: str = DEFAULT_METADATA_COLUMN,
        vector_column: str = DEFAULT_VECTOR_COLUMN,
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        vector_column_type: str = "REAL_VECTOR",
        vector_column_length: int = -1,
        create_table: bool = False,
        create_hnsw_index: bool = False,
        connection_args: Optional[Dict[str, Any]] = None,
        use_connection_pool: bool = False,
        connection_pool_name: str = "default",
        connection_pool_min: int = 1,
        connection_pool_max: int = 10,
        retry_attempts: int = 3,
    ) -> None:
        """
        Initialize the HANA vector store.
        
        Args:
            embedding: LangChain embeddings model to use
            connection: Existing SAP HANA connection (optional)
            host: SAP HANA host (required if connection not provided)
            port: SAP HANA port (required if connection not provided)
            user: SAP HANA username (required if connection not provided)
            password: SAP HANA password (required if connection not provided)
            schema_name: Database schema to use (optional, uses current schema if not specified)
            table_name: Name of the table to store vectors
            content_column: Name of the column to store document content
            metadata_column: Name of the column to store document metadata
            vector_column: Name of the column to store embeddings
            distance_strategy: Distance strategy to use
            vector_column_type: Type of vector column (REAL_VECTOR or HALF_VECTOR)
            vector_column_length: Length of vectors, or -1 for dynamic length
            create_table: Whether to create the table if it doesn't exist
            create_hnsw_index: Whether to create an HNSW index on the vector column
            connection_args: Additional arguments to pass to the connection
            use_connection_pool: Whether to use a connection pool (recommended for production)
            connection_pool_name: Name of the connection pool to use or create
            connection_pool_min: Minimum number of connections in the pool
            connection_pool_max: Maximum number of connections in the pool
            retry_attempts: Number of retry attempts for database operations
        
        Raises:
            ValueError: If connection parameters are invalid
            ConnectionError: If connection to SAP HANA fails
        """
        self.embedding = embedding
        self.schema_name = schema_name
        self.table_name = self._sanitize_identifier(table_name)
        self.content_column = self._sanitize_identifier(content_column)
        self.metadata_column = self._sanitize_identifier(metadata_column)
        self.vector_column = self._sanitize_identifier(vector_column)
        self.distance_strategy = distance_strategy
        self.vector_column_type = vector_column_type
        self.vector_column_length = vector_column_length
        self.retry_attempts = retry_attempts
        self.use_connection_pool = use_connection_pool
        self.connection_pool_name = connection_pool_name
        
        # Check if we're using internal embeddings
        from langchain_hana.embeddings import HanaInternalEmbeddings
        self.use_internal_embeddings = isinstance(embedding, HanaInternalEmbeddings)
        self.internal_embedding_model_id = (
            embedding.get_model_id() if self.use_internal_embeddings else None
        )
        
        # Set up connection
        connection_args = connection_args or {}
        
        if use_connection_pool:
            # Set up connection pool
            if connection:
                logger.warning(
                    "A connection was provided but use_connection_pool=True. "
                    "The provided connection will be ignored and a pooled connection will be used."
                )
            
            # Check if a pool with this name already exists
            pool = get_connection_pool(connection_pool_name)
            
            if not pool:
                # Create a new connection pool
                if not (host and port and user and password):
                    raise ValueError(
                        "Connection parameters (host, port, user, password) are required "
                        "when creating a new connection pool."
                    )
                
                # Create the connection pool
                pool = create_connection_pool(
                    pool_name=connection_pool_name,
                    min_connections=connection_pool_min,
                    max_connections=connection_pool_max,
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    **connection_args
                )
            
            # Store pool reference
            self.connection_pool = pool
            
            # Get a connection from the pool for initial validation
            self.connection = pool.get_connection()
            self.owns_connection = False  # The pool owns the connection
        else:
            # Standard connection handling (non-pooled)
            if connection:
                self.connection = connection
                self.owns_connection = False
            elif host and port and user and password:
                self.connection = create_connection(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    **connection_args
                )
                self.owns_connection = True
            else:
                raise ValueError(
                    "Either an existing connection or connection parameters "
                    "(host, port, user, password) must be provided."
                )
        
        # Test connection
        connection_valid, info = test_connection(self.connection)
        if not connection_valid:
            # Release the connection back to the pool if using one
            if use_connection_pool:
                self.connection_pool.release_connection(self.connection)
            
            raise ConnectionError(f"Invalid connection: {info.get('error', 'Unknown error')}")
        
        # Store connection info
        self.connection_info = info
        
        # Set the full table name (with schema if provided)
        if not self.schema_name:
            self.schema_name = info.get("current_schema")
        
        self.full_table_name = (
            f'"{self.schema_name}"."{self.table_name}"' 
            if self.schema_name 
            else f'"{self.table_name}"'
        )
        
        # Create table if requested
        if create_table:
            self._create_table()
            
            # Create HNSW index if requested
            if create_hnsw_index:
                self._create_hnsw_index()
        
        # Release the connection back to the pool if using one
        if use_connection_pool:
            self.connection_pool.release_connection(self.connection)
    
    def _sanitize_identifier(self, identifier: str) -> str:
        """
        Sanitize an SQL identifier to prevent SQL injection.
        
        Args:
            identifier: The identifier to sanitize
            
        Returns:
            Sanitized identifier
        """
        # Remove any character that's not alphanumeric or underscore
        sanitized = re.sub(r'[^\w]', '', identifier)
        
        # Ensure the identifier starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != '_':
            sanitized = 'X' + sanitized
            
        # Ensure the identifier is not empty
        if not sanitized:
            sanitized = 'X'
            
        return sanitized
    
    def _get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding model.
        
        Returns:
            Dimension of embeddings
        """
        if self.vector_column_length > 0:
            return self.vector_column_length
        
        try:
            if self.use_internal_embeddings:
                # For internal embeddings, query the model capabilities
                cursor = self.connection.cursor()
                try:
                    model_id = self.internal_embedding_model_id
                    cursor.execute(
                        "SELECT LENGTH(VECTOR_EMBEDDING('test', 'QUERY', ?)) FROM sys.DUMMY",
                        (model_id,)
                    )
                    if cursor.has_result_set():
                        row = cursor.fetchone()
                        # Convert bytes to vector length (subtract 4 bytes for the header)
                        if row and row[0]:
                            # The first 4 bytes contain the vector dimension
                            dim_bytes = row[0][:4]
                            dimension = int.from_bytes(dim_bytes, byteorder='little')
                            self.vector_column_length = dimension
                            return dimension
                finally:
                    cursor.close()
            
            # For external embeddings, generate a sample embedding
            sample_embedding = self.embedding.embed_query("test")
            self.vector_column_length = len(sample_embedding)
            return self.vector_column_length
        except Exception as e:
            logger.warning(f"Failed to determine embedding dimension: {str(e)}")
            # Default to a common dimension size as fallback
            self.vector_column_length = 768
            return 768
    
    def _create_table(self) -> bool:
        """
        Create the vector table if it doesn't exist.
        
        Returns:
            True if the table was created successfully, False otherwise
        """
        cursor = self.connection.cursor()
        try:
            # Create the table
            create_vector_table(
                cursor=cursor,
                table_name=self.full_table_name,
                content_column=self.content_column,
                metadata_column=self.metadata_column,
                vector_column=self.vector_column,
                vector_column_type=self.vector_column_type,
                vector_column_length=self._get_embedding_dimension(),
                if_not_exists=True
            )
            
            logger.info(f"Created table {self.full_table_name} (if it didn't exist)")
            return True
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            return False
        finally:
            cursor.close()
    
    def _create_hnsw_index(self) -> bool:
        """
        Create an HNSW vector index on the vector column.
        
        Returns:
            True if the index was created successfully, False otherwise
        """
        cursor = self.connection.cursor()
        try:
            # Convert distance strategy to SQL function name
            distance_function, _ = convert_distance_strategy_to_sql(self.distance_strategy)
            
            # Create the index
            create_hnsw_index(
                cursor=cursor,
                table_name=self.full_table_name,
                vector_column=self.vector_column,
                distance_function=distance_function,
                # Use default parameters for the index
            )
            
            logger.info(f"Created HNSW index on {self.full_table_name}.{self.vector_column}")
            return True
        except Exception as e:
            logger.warning(f"Failed to create HNSW index: {str(e)}")
            return False
        finally:
            cursor.close()
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: Iterable of strings to add
            metadatas: Optional list of metadatas associated with the texts
            ids: Optional list of IDs to associate with the texts
            
        Returns:
            List of IDs of the added texts
        """
        # Convert texts to list for indexing
        texts_list = list(texts)
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(texts_list))]
        
        # Ensure we have the right number of IDs
        if len(ids) != len(texts_list):
            raise ValueError(f"Number of ids ({len(ids)}) must match number of texts ({len(texts_list)})")
        
        # Handle metadatas
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts_list))]
        
        # Ensure we have the right number of metadatas
        if len(metadatas) != len(texts_list):
            raise ValueError(f"Number of metadatas ({len(metadatas)}) must match number of texts ({len(texts_list)})")
        
        # Add IDs to metadata
        for i, metadata in enumerate(metadatas):
            metadata["id"] = ids[i]
        
        # Handle internal vs external embeddings
        if self.use_internal_embeddings:
            self._add_texts_internal(texts_list, metadatas)
        else:
            self._add_texts_external(texts_list, metadatas)
        
        return ids
    
    def _get_db_cursor(self):
        """
        Get a database cursor, either from the direct connection or from the connection pool.
        
        Returns:
            Tuple containing:
                - Database cursor
                - Connection that owns the cursor
                - Boolean indicating if the connection should be released to the pool
        """
        if self.use_connection_pool:
            # Get a connection from the pool
            conn = self.connection_pool.get_connection()
            return conn.cursor(), conn, True
        else:
            # Use the existing connection
            return self.connection.cursor(), self.connection, False
    
    def _execute_with_retry(self, operation_name, operation_func):
        """
        Execute a database operation with retry logic.
        
        Args:
            operation_name: Name of the operation for logging
            operation_func: Function that performs the database operation
                            Should accept a cursor and return a result
        
        Returns:
            The result of the operation function
        
        Raises:
            Exception: If the operation fails after all retry attempts
        """
        attempt = 0
        last_exception = None
        
        while attempt < self.retry_attempts:
            cursor = None
            conn = None
            release_conn = False
            
            try:
                # Get a cursor
                cursor, conn, release_conn = self._get_db_cursor()
                
                # Execute the operation
                result = operation_func(cursor, conn)
                
                # If we got here, the operation succeeded
                return result
                
            except dbapi.Error as e:
                last_exception = e
                attempt += 1
                
                error_msg = f"{operation_name} failed (attempt {attempt}/{self.retry_attempts}): {str(e)}"
                if attempt < self.retry_attempts:
                    logger.warning(f"{error_msg}, retrying...")
                    # Sleep with exponential backoff
                    time.sleep(2 ** attempt * 0.1)  # 0.2s, 0.4s, 0.8s, ...
                else:
                    logger.error(f"{error_msg}, no more retries.")
            
            except Exception as e:
                # For non-database errors, don't retry
                logger.error(f"{operation_name} failed with non-database error: {str(e)}")
                if release_conn and conn:
                    self.connection_pool.release_connection(conn)
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                raise
            
            finally:
                # Clean up
                if cursor:
                    try:
                        cursor.close()
                    except:
                        pass
                
                if release_conn and conn:
                    self.connection_pool.release_connection(conn)
        
        # If we get here, all attempts failed
        raise last_exception or Exception(f"{operation_name} failed after {self.retry_attempts} attempts")
    
    def _add_texts_internal(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add texts using SAP HANA's internal embedding functionality.
        
        Args:
            texts: List of strings to add
            metadatas: List of metadata dicts, one for each text
        """
        # Get the model ID from the embedding instance
        model_id = self.internal_embedding_model_id
        
        # Prepare SQL parameters
        sql_params = []
        for i, text in enumerate(texts):
            metadata = metadatas[i]
            
            # Prepare parameters for this document
            params = {
                "content": text,
                "metadata": json.dumps(metadata),
                "model_id": model_id,
            }
            
            sql_params.append(params)
        
        # Build the SQL INSERT statement
        sql = f"""
        INSERT INTO {self.full_table_name} (
            "{self.content_column}", "{self.metadata_column}", "{self.vector_column}"
        ) VALUES (
            :content, 
            :metadata, 
            VECTOR_EMBEDDING(:content, 'DOCUMENT', :model_id)
        )
        """
        
        # Define the operation function
        def operation_func(cursor, conn):
            # Execute the SQL for each document
            for params in sql_params:
                cursor.execute(sql, params)
            
            # Commit the transaction
            conn.commit()
            
            logger.debug(f"Added {len(texts)} documents with internal embeddings")
            return None
        
        # Execute with retry
        self._execute_with_retry("Add texts with internal embeddings", operation_func)
    
    def _add_texts_external(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Add texts using external embedding model.
        
        Args:
            texts: List of strings to add
            metadatas: List of metadata dicts, one for each text
        """
        # Generate embeddings using the external model
        embeddings = self.embedding.embed_documents(texts)
        
        # Prepare SQL parameters
        sql_params = []
        for i, text in enumerate(texts):
            metadata = metadatas[i]
            
            # Convert embedding to binary format
            vector_binary = serialize_vector(
                embeddings[i], vector_type=self.vector_column_type
            )
            
            # Add parameters for this document
            params = (
                text,
                json.dumps(metadata),
                vector_binary,
            )
            
            sql_params.append(params)
        
        # Build the SQL INSERT statement
        sql = f"""
        INSERT INTO {self.full_table_name} (
            "{self.content_column}", "{self.metadata_column}", "{self.vector_column}"
        ) VALUES (?, ?, ?)
        """
        
        # Define the operation function
        def operation_func(cursor, conn):
            # Execute the SQL for all documents at once
            cursor.executemany(sql, sql_params)
            
            # Commit the transaction
            conn.commit()
            
            logger.debug(f"Added {len(texts)} documents with external embeddings")
            return None
        
        # Execute with retry
        self._execute_with_retry("Add texts with external embeddings", operation_func)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for documents similar to a query string.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter: Optional filter on metadata fields
            
        Returns:
            List of Documents most similar to the query
        """
        docs_with_scores = self.similarity_search_with_score(query, k, filter)
        return [doc for doc, _ in docs_with_scores]
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to a query string and return with scores.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter: Optional filter on metadata fields
            
        Returns:
            List of (Document, score) tuples
        """
        # Handle internal vs external embeddings
        if self.use_internal_embeddings:
            return self._similarity_search_internal(query, k, filter)
        else:
            embedding = self.embedding.embed_query(query)
            return self.similarity_search_by_vector_with_scores(embedding, k, filter)
    
    def _similarity_search_internal(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to a query using SAP HANA's internal embedding functionality.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            filter: Optional filter on metadata fields
            
        Returns:
            List of (Document, score) tuples
        """
        # Get the model ID from the embedding instance
        model_id = self.internal_embedding_model_id
        
        # Get the distance function and sort order
        distance_func, sort_order = convert_distance_strategy_to_sql(self.distance_strategy)
        
        # Build the SQL query
        sql = f"""
        SELECT TOP {k}
            "{self.content_column}",
            "{self.metadata_column}",
            {distance_func}("{self.vector_column}", 
                            VECTOR_EMBEDDING(?, 'QUERY', ?)) AS score
        FROM {self.full_table_name}
        """
        
        # Add filter if provided
        params = [query, model_id]
        where_clause = ""
        
        if filter:
            where_clause, filter_params = self._build_filter_clause(filter)
            if where_clause:
                sql += f" WHERE {where_clause}"
                params.extend(filter_params)
        
        # Add order by clause
        sql += f" ORDER BY score {sort_order}"
        
        # Define the operation function
        def operation_func(cursor, conn):
            # Execute the query
            cursor.execute(sql, params)
            
            # Process results
            results = []
            for row in cursor.fetchall():
                content = row[0]
                metadata = json.loads(row[1])
                score = row[2]
                
                doc = Document(page_content=content, metadata=metadata)
                results.append((doc, score))
            
            return results
        
        # Execute with retry
        return self._execute_with_retry("Similarity search (internal embeddings)", operation_func)
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for documents similar to an embedding vector.
        
        Args:
            embedding: Embedding vector to search for
            k: Number of results to return
            filter: Optional filter on metadata fields
            
        Returns:
            List of Documents most similar to the embedding
        """
        docs_with_scores = self.similarity_search_by_vector_with_scores(embedding, k, filter)
        return [doc for doc, _ in docs_with_scores]
    
    def similarity_search_by_vector_with_scores(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Search for documents similar to an embedding vector and return with scores.
        
        Args:
            embedding: Embedding vector to search for
            k: Number of results to return
            filter: Optional filter on metadata fields
            
        Returns:
            List of (Document, score) tuples
        """
        # Get the distance function and sort order
        distance_func, sort_order = convert_distance_strategy_to_sql(self.distance_strategy)
        
        # Convert embedding to the format expected by HANA
        vector_binary = serialize_vector(embedding, vector_type=self.vector_column_type)
        
        # Build the SQL query
        sql = f"""
        SELECT TOP {k}
            "{self.content_column}",
            "{self.metadata_column}",
            {distance_func}("{self.vector_column}", ?) AS score
        FROM {self.full_table_name}
        """
        
        # Add filter if provided
        params = [vector_binary]
        where_clause = ""
        
        if filter:
            where_clause, filter_params = self._build_filter_clause(filter)
            if where_clause:
                sql += f" WHERE {where_clause}"
                params.extend(filter_params)
        
        # Add order by clause
        sql += f" ORDER BY score {sort_order}"
        
        # Define the operation function
        def operation_func(cursor, conn):
            # Execute the query
            cursor.execute(sql, params)
            
            # Process results
            results = []
            for row in cursor.fetchall():
                content = row[0]
                metadata = json.loads(row[1])
                score = row[2]
                
                doc = Document(page_content=content, metadata=metadata)
                results.append((doc, score))
            
            return results
        
        # Execute with retry
        return self._execute_with_retry("Similarity search by vector", operation_func)
    
    def _build_filter_clause(
        self, filter: Optional[Dict[str, Any]]
    ) -> Tuple[str, List[Any]]:
        """
        Build a WHERE clause from a filter dictionary.
        
        Args:
            filter: Filter dictionary
            
        Returns:
            Tuple of (WHERE clause string, parameters list)
        """
        if not filter:
            return "", []
        
        clauses = []
        params = []
        
        for key, value in filter.items():
            # Handle special operators
            if isinstance(value, dict) and len(value) == 1:
                op, op_value = next(iter(value.items()))
                
                if op == "$eq":
                    clauses.append(f'JSON_VALUE("{self.metadata_column}", \'$."{key}"\') = ?')
                    params.append(json.dumps(op_value))
                elif op == "$ne":
                    clauses.append(f'JSON_VALUE("{self.metadata_column}", \'$."{key}"\') != ?')
                    params.append(json.dumps(op_value))
                elif op == "$gt":
                    clauses.append(f'CAST(JSON_VALUE("{self.metadata_column}", \'$."{key}"\') AS FLOAT) > ?')
                    params.append(float(op_value))
                elif op == "$gte":
                    clauses.append(f'CAST(JSON_VALUE("{self.metadata_column}", \'$."{key}"\') AS FLOAT) >= ?')
                    params.append(float(op_value))
                elif op == "$lt":
                    clauses.append(f'CAST(JSON_VALUE("{self.metadata_column}", \'$."{key}"\') AS FLOAT) < ?')
                    params.append(float(op_value))
                elif op == "$lte":
                    clauses.append(f'CAST(JSON_VALUE("{self.metadata_column}", \'$."{key}"\') AS FLOAT) <= ?')
                    params.append(float(op_value))
                elif op == "$contains":
                    clauses.append(f'CONTAINS(JSON_VALUE("{self.metadata_column}", \'$."{key}"\'), ?)')
                    params.append(str(op_value))
                elif op == "$in":
                    placeholders = ", ".join(["?"] * len(op_value))
                    clauses.append(f'JSON_VALUE("{self.metadata_column}", \'$."{key}"\') IN ({placeholders})')
                    params.extend([json.dumps(v) for v in op_value])
            else:
                # Simple equality
                clauses.append(f'JSON_VALUE("{self.metadata_column}", \'$."{key}"\') = ?')
                params.append(json.dumps(value))
        
        if clauses:
            return " AND ".join(clauses), params
        else:
            return "", []
    
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
        Search for documents similar to a query string using MMR.
        
        Args:
            query: Query string to search for
            k: Number of results to return
            fetch_k: Number of results to fetch before applying MMR
            lambda_mult: MMR lambda parameter (0 = max diversity, 1 = max relevance)
            filter: Optional filter on metadata fields
            
        Returns:
            List of Documents selected by MMR
        """
        # Get the query embedding
        if self.use_internal_embeddings:
            # For internal embeddings, we need to make a special query to get the embedding
            embedding = self._get_internal_embedding(query)
        else:
            # For external embeddings, use the embedding model
            embedding = self.embedding.embed_query(query)
        
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter
        )
    
    def _get_internal_embedding(self, text: str) -> List[float]:
        """
        Get an embedding vector for a text using SAP HANA's internal embedding functionality.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        cursor = self.connection.cursor()
        try:
            # Get the model ID from the embedding instance
            model_id = self.internal_embedding_model_id
            
            # Execute query to get the embedding
            cursor.execute(
                "SELECT VECTOR_EMBEDDING(?, 'QUERY', ?) FROM sys.DUMMY",
                (text, model_id)
            )
            
            # Get the result
            result = cursor.fetchone()
            if result and result[0]:
                # Deserialize the binary vector
                return deserialize_vector(result[0], vector_type=self.vector_column_type)
            else:
                raise ValueError("Failed to generate embedding with internal function")
        finally:
            cursor.close()
    
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for documents similar to an embedding vector using MMR.
        
        Args:
            embedding: Embedding vector to search for
            k: Number of results to return
            fetch_k: Number of results to fetch before applying MMR
            lambda_mult: MMR lambda parameter (0 = max diversity, 1 = max relevance)
            filter: Optional filter on metadata fields
            
        Returns:
            List of Documents selected by MMR
        """
        # First, get fetch_k results with their embeddings
        results_with_embeddings = self._similarity_search_with_embeddings(
            embedding=embedding,
            k=fetch_k,
            filter=filter
        )
        
        if not results_with_embeddings:
            return []
        
        # Split results into docs, scores, and embeddings
        docs = [item[0] for item in results_with_embeddings]
        embeddings = [item[2] for item in results_with_embeddings]
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply MMR
        mmr_indices = maximal_marginal_relevance(
            np.array(embedding),
            embeddings_array,
            k=k,
            lambda_mult=lambda_mult
        )
        
        # Return documents in MMR order
        return [docs[i] for i in mmr_indices]
    
    def _similarity_search_with_embeddings(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, List[float]]]:
        """
        Search for documents similar to an embedding vector and return docs, scores, and embeddings.
        
        Args:
            embedding: Embedding vector to search for
            k: Number of results to return
            filter: Optional filter on metadata fields
            
        Returns:
            List of (Document, score, embedding) tuples
        """
        # Get the distance function and sort order
        distance_func, sort_order = convert_distance_strategy_to_sql(self.distance_strategy)
        
        # Convert embedding to the format expected by HANA
        vector_binary = serialize_vector(embedding, vector_type=self.vector_column_type)
        
        # Build the SQL query
        sql = f"""
        SELECT TOP {k}
            "{self.content_column}",
            "{self.metadata_column}",
            "{self.vector_column}",
            {distance_func}("{self.vector_column}", ?) AS score
        FROM {self.full_table_name}
        """
        
        # Add filter if provided
        params = [vector_binary]
        where_clause = ""
        
        if filter:
            where_clause, filter_params = self._build_filter_clause(filter)
            if where_clause:
                sql += f" WHERE {where_clause}"
                params.extend(filter_params)
        
        # Add order by clause
        sql += f" ORDER BY score {sort_order}"
        
        # Define the operation function
        def operation_func(cursor, conn):
            # Execute the query
            cursor.execute(sql, params)
            
            # Process results
            results = []
            for row in cursor.fetchall():
                content = row[0]
                metadata = json.loads(row[1])
                vector_binary = row[2]
                score = row[3]
                
                # Deserialize the binary vector
                vector = deserialize_vector(vector_binary, vector_type=self.vector_column_type)
                
                doc = Document(page_content=content, metadata=metadata)
                results.append((doc, score, vector))
            
            return results
        
        # Execute with retry
        return self._execute_with_retry("Similarity search with embeddings", operation_func)
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[bool]:
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
            filter: Filter to use for deletion
            
        Returns:
            True if successful
        """
        if ids is None and filter is None:
            raise ValueError("Either ids or filter must be provided")
        
        # Define SQL based on deletion method
        if ids is not None:
            # Delete by IDs
            placeholders = ", ".join(["?"] * len(ids))
            sql = f"""
            DELETE FROM {self.full_table_name}
            WHERE JSON_VALUE("{self.metadata_column}", '$.id') IN ({placeholders})
            """
            params = ids
        else:
            # Delete by filter
            where_clause, filter_params = self._build_filter_clause(filter)
            if where_clause:
                sql = f"DELETE FROM {self.full_table_name} WHERE {where_clause}"
                params = filter_params
            else:
                # No filter - delete all
                sql = f"DELETE FROM {self.full_table_name}"
                params = []
        
        # Define the operation function
        def operation_func(cursor, conn):
            # Execute the delete
            cursor.execute(sql, params)
            
            # Get row count if available
            row_count = getattr(cursor, 'rowcount', -1)
            
            # Commit the transaction
            conn.commit()
            
            logger.info(f"Deleted {row_count if row_count >= 0 else 'unknown number of'} documents")
            return True
        
        # Execute with retry
        return self._execute_with_retry("Delete documents", operation_func)
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        connection: Optional[dbapi.Connection] = None,
        **kwargs: Any,
    ) -> "HanaVectorStore":
        """
        Create a HanaVectorStore from a list of texts.
        
        Args:
            texts: List of texts to add
            embedding: Embedding model to use
            metadatas: Optional list of metadatas associated with the texts
            ids: Optional list of IDs to associate with the texts
            host: SAP HANA host (required if connection not provided)
            port: SAP HANA port (required if connection not provided)
            user: SAP HANA username (required if connection not provided)
            password: SAP HANA password (required if connection not provided)
            connection: Existing SAP HANA connection (optional)
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            HanaVectorStore instance
        """
        # Create the vector store
        vector_store = cls(
            embedding=embedding,
            host=host,
            port=port,
            user=user,
            password=password,
            connection=connection,
            create_table=True,  # Create table by default
            **kwargs,
        )
        
        # Add texts
        vector_store.add_texts(texts, metadatas, ids)
        
        return vector_store
    
    def __del__(self) -> None:
        """
        Clean up resources when the object is destroyed.
        """
        # Only close the connection if we own it and it's not from a pool
        if hasattr(self, "connection") and hasattr(self, "owns_connection") and not hasattr(self, "use_connection_pool"):
            if self.owns_connection and self.connection:
                try:
                    self.connection.close()
                    logger.debug("Closed SAP HANA connection")
                except:
                    pass
        
        # Connection pools are managed globally, so we don't close them here

# Alias for backward compatibility
HanaDB = HanaVectorStore