"""
Arrow Flight-based SAP HANA vector store integration for LangChain.

This module provides an optimized vector store implementation that uses
Apache Arrow Flight for efficient data transfer between SAP HANA Cloud
and GPU-accelerated vector operations.
"""

import logging
import time
import uuid
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.flight as flight
    HAS_ARROW_FLIGHT = True
except ImportError:
    HAS_ARROW_FLIGHT = False

from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import VectorStore
from langchain.vectorstores.utils import maximal_marginal_relevance

from ..error_utils import wrap_hana_errors
from .arrow_flight_client import ArrowFlightClient
from .arrow_gpu_memory_manager import ArrowGpuMemoryManager

logger = logging.getLogger(__name__)


class HanaArrowFlightVectorStore(VectorStore):
    """
    Vector store implementation using Apache Arrow Flight for efficient data transfer with SAP HANA.
    
    This implementation provides optimized vector operations by leveraging Apache Arrow's
    columnar format and the Arrow Flight RPC framework for efficient data transfer between
    SAP HANA Cloud and GPU-accelerated vector operations.
    """
    
    def __init__(
        self,
        embedding: Embeddings,
        host: str,
        port: int = 8815,
        table_name: str = "LANGCHAIN_VECTORS",
        text_column: str = "TEXT",
        vector_column: str = "VEC_VECTOR",
        metadata_column: str = "METADATA",
        id_column: str = "ID",
        distance_strategy: str = "cosine",
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000,
        device_id: int = 0,
        use_tls: bool = False,
        pre_delete_collection: bool = False,
    ):
        """
        Initialize the SAP HANA vector store with Arrow Flight integration.
        
        Args:
            embedding: Embedding function to use
            host: SAP HANA host address
            port: Arrow Flight server port
            table_name: Table name for vector storage
            text_column: Column name for document text
            vector_column: Column name for vector data
            metadata_column: Column name for metadata
            id_column: Column name for document ID
            distance_strategy: Distance strategy for similarity search ("cosine", "l2", "dot")
            username: SAP HANA username
            password: SAP HANA password
            connection_args: Additional connection arguments
            batch_size: Batch size for operations
            device_id: GPU device ID
            use_tls: Whether to use TLS for secure connections
            pre_delete_collection: Whether to delete the collection before initialization
            
        Raises:
            ImportError: If required dependencies are not installed
        """
        if not HAS_ARROW_FLIGHT:
            raise ImportError(
                "The pyarrow and pyarrow.flight packages are required for Arrow Flight integration. "
                "Install them with 'pip install pyarrow pyarrow.flight'."
            )
        
        self.embedding = embedding
        self.table_name = table_name
        self.text_column = text_column
        self.vector_column = vector_column
        self.metadata_column = metadata_column
        self.id_column = id_column
        self.distance_strategy = distance_strategy
        self.batch_size = batch_size
        self.device_id = device_id
        
        # Initialize Arrow Flight client
        self.client = ArrowFlightClient(
            host=host,
            port=port,
            use_tls=use_tls,
            username=username,
            password=password,
            connection_args=connection_args
        )
        
        # Initialize GPU memory manager
        self.memory_manager = ArrowGpuMemoryManager(
            device_id=device_id,
            batch_size=batch_size
        )
        
        # Initialize collection
        self._initialize_collection(pre_delete_collection)
    
    @wrap_hana_errors
    def _initialize_collection(self, pre_delete_collection: bool = False):
        """Initialize the vector collection in SAP HANA."""
        try:
            # Check if collection exists by querying its schema
            try:
                schema = self.client.get_schema(self.table_name)
                logger.info(f"Found existing table: {self.table_name}")
                
                if pre_delete_collection:
                    # Delete the collection if requested
                    self._execute_command(f"DROP TABLE {self.table_name}")
                    logger.info(f"Dropped existing table: {self.table_name}")
                    schema = None
            except Exception:
                schema = None
                
            if schema is None:
                # Create the table if it doesn't exist
                create_table_cmd = f"""
                CREATE TABLE {self.table_name} (
                    {self.id_column} VARCHAR(100) PRIMARY KEY,
                    {self.text_column} NVARCHAR(5000),
                    {self.metadata_column} NCLOB,
                    {self.vector_column} VARBINARY(32000)
                )
                """
                
                self._execute_command(create_table_cmd)
                logger.info(f"Created new table: {self.table_name}")
                
                # Register the table with the Arrow Flight server
                self._register_table()
        except Exception as e:
            logger.error(f"Error initializing collection: {str(e)}")
            raise RuntimeError(f"Failed to initialize collection: {str(e)}")
    
    def _register_table(self):
        """Register the table with the Arrow Flight server."""
        try:
            # Perform a "register_table" action
            action = flight.Action("register_table", json.dumps({
                "table_name": self.table_name
            }).encode('utf-8'))
            
            results = list(self.client.client.do_action(action))
            if results and len(results) > 0:
                logger.info(f"Registered table with Arrow Flight server: {self.table_name}")
        except Exception as e:
            logger.warning(f"Error registering table with Arrow Flight server: {str(e)}")
            # Continue anyway, as this is not critical
    
    def _execute_command(self, command: str, params: Optional[List[Any]] = None) -> Any:
        """Execute a command against the SAP HANA database."""
        # Use the client's database connection to execute the command
        if not hasattr(self.client, '_db_connection') or self.client._db_connection is None:
            raise RuntimeError("No database connection available")
            
        cursor = self.client._db_connection.cursor()
        try:
            if params:
                cursor.execute(command, params)
            else:
                cursor.execute(command)
                
            if cursor.description:
                return cursor.fetchall()
            return None
        finally:
            cursor.close()
    
    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        host: str,
        port: int = 8815,
        table_name: str = "LANGCHAIN_VECTORS",
        text_column: str = "TEXT",
        vector_column: str = "VEC_VECTOR",
        metadata_column: str = "METADATA",
        id_column: str = "ID",
        distance_strategy: str = "cosine",
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000,
        device_id: int = 0,
        use_tls: bool = False,
        pre_delete_collection: bool = False,
        **kwargs: Any,
    ) -> "HanaArrowFlightVectorStore":
        """
        Create a vector store from documents.
        
        Args:
            documents: List of documents to add to the vector store
            embedding: Embedding function to use
            host: SAP HANA host address
            port: Arrow Flight server port
            table_name: Table name for vector storage
            text_column: Column name for document text
            vector_column: Column name for vector data
            metadata_column: Column name for metadata
            id_column: Column name for document ID
            distance_strategy: Distance strategy for similarity search
            username: SAP HANA username
            password: SAP HANA password
            connection_args: Additional connection arguments
            batch_size: Batch size for operations
            device_id: GPU device ID
            use_tls: Whether to use TLS for secure connections
            pre_delete_collection: Whether to delete the collection before initialization
            
        Returns:
            HanaArrowFlightVectorStore instance
        """
        store = cls(
            embedding=embedding,
            host=host,
            port=port,
            table_name=table_name,
            text_column=text_column,
            vector_column=vector_column,
            metadata_column=metadata_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            username=username,
            password=password,
            connection_args=connection_args,
            batch_size=batch_size,
            device_id=device_id,
            use_tls=use_tls,
            pre_delete_collection=pre_delete_collection,
        )
        store.add_documents(documents)
        return store
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        host: str = None,
        port: int = 8815,
        table_name: str = "LANGCHAIN_VECTORS",
        text_column: str = "TEXT",
        vector_column: str = "VEC_VECTOR",
        metadata_column: str = "METADATA",
        id_column: str = "ID",
        distance_strategy: str = "cosine",
        username: Optional[str] = None,
        password: Optional[str] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        batch_size: int = 1000,
        device_id: int = 0,
        use_tls: bool = False,
        pre_delete_collection: bool = False,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "HanaArrowFlightVectorStore":
        """
        Create a vector store from texts.
        
        Args:
            texts: List of texts to add to the vector store
            embedding: Embedding function to use
            metadatas: Optional list of metadata dictionaries
            host: SAP HANA host address
            port: Arrow Flight server port
            table_name: Table name for vector storage
            text_column: Column name for document text
            vector_column: Column name for vector data
            metadata_column: Column name for metadata
            id_column: Column name for document ID
            distance_strategy: Distance strategy for similarity search
            username: SAP HANA username
            password: SAP HANA password
            connection_args: Additional connection arguments
            batch_size: Batch size for operations
            device_id: GPU device ID
            use_tls: Whether to use TLS for secure connections
            pre_delete_collection: Whether to delete the collection before initialization
            ids: Optional list of document IDs
            
        Returns:
            HanaArrowFlightVectorStore instance
        """
        store = cls(
            embedding=embedding,
            host=host,
            port=port,
            table_name=table_name,
            text_column=text_column,
            vector_column=vector_column,
            metadata_column=metadata_column,
            id_column=id_column,
            distance_strategy=distance_strategy,
            username=username,
            password=password,
            connection_args=connection_args,
            batch_size=batch_size,
            device_id=device_id,
            use_tls=use_tls,
            pre_delete_collection=pre_delete_collection,
        )
        store.add_texts(texts, metadatas, ids)
        return store
    
    def _metadata_to_json(self, metadata: Dict[str, Any]) -> str:
        """Convert metadata to JSON string."""
        import json
        if metadata is None:
            return "{}"
        return json.dumps(metadata)
    
    def _json_to_metadata(self, json_str: str) -> Dict[str, Any]:
        """Convert JSON string to metadata dictionary."""
        import json
        if not json_str:
            return {}
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"Error decoding metadata JSON: {json_str}")
            return {}
    
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add texts to the vector store.
        
        Args:
            texts: Iterable of texts to add
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        # Convert texts to list if it's an iterable
        if not isinstance(texts, list):
            texts = list(texts)
            
        if not texts:
            return []
            
        # Generate embeddings
        embeddings = self.embedding.embed_documents(texts)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Standardize metadata to list
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        # Convert metadata to JSON strings
        metadata_jsons = [self._metadata_to_json(m) for m in metadatas]
        
        # Upload vectors using Arrow Flight
        return self.client.upload_vectors(
            table_name=self.table_name,
            vectors=embeddings,
            texts=texts,
            metadata=metadatas,
            ids=ids,
            batch_size=self.batch_size
        )
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, ids, **kwargs)
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Perform similarity search using the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter for metadata
            
        Returns:
            List of Document objects
        """
        docs_with_scores = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in docs_with_scores]
    
    def _filter_dict_to_sql(self, filter_dict: Dict[str, Any]) -> str:
        """Convert a filter dictionary to SQL WHERE clause."""
        if not filter_dict:
            return None
            
        import json
        
        # Build filter condition based on metadata JSON contents
        conditions = []
        for key, value in filter_dict.items():
            # Create a condition that checks for the key-value pair in the JSON
            if isinstance(value, str):
                # For string values, escape single quotes
                escaped_value = value.replace("'", "''")
                conditions.append(f"JSON_VALUE({self.metadata_column}, '$.{key}') = '{escaped_value}'")
            elif value is None:
                # For null values
                conditions.append(f"JSON_VALUE({self.metadata_column}, '$.{key}') IS NULL")
            else:
                # For numeric or boolean values
                conditions.append(f"JSON_VALUE({self.metadata_column}, '$.{key}') = {value}")
        
        if conditions:
            return " AND ".join(conditions)
        return None
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with scores using the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter for metadata
            
        Returns:
            List of (Document, score) tuples
        """
        # Generate embedding for query
        embedding = self.embedding.embed_query(query)
        
        # Convert filter dictionary to SQL WHERE clause if provided
        filter_query = None
        if filter:
            filter_query = self._filter_dict_to_sql(filter)
        
        # Perform similarity search using Arrow Flight
        results = self.client.similarity_search(
            table_name=self.table_name,
            query_vector=embedding,
            k=k,
            filter_query=filter_query,
            include_metadata=True,
            include_vectors=False,
            distance_strategy=self.distance_strategy
        )
        
        # Convert results to Documents with scores
        docs_with_scores = []
        for result in results:
            if 'text' in result and 'metadata' in result and 'score' in result:
                doc = Document(
                    page_content=result['text'],
                    metadata=result['metadata']
                )
                score = result['score']
                docs_with_scores.append((doc, score))
        
        return docs_with_scores
    
    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Return documents and relevance scores in the range [0, 1].
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Optional filter for metadata
            
        Returns:
            List of (Document, relevance_score) tuples
        """
        docs_and_scores = self.similarity_search_with_score(query, k, filter, **kwargs)
        
        # Normalize scores based on distance strategy
        if self.distance_strategy == "cosine" or self.distance_strategy == "dot":
            # For cosine and dot product, higher is better and scores are in [-1, 1]
            min_score = -1.0
            max_score = 1.0
        else:
            # For L2 distance, lower is better and scores are in [0, inf)
            # Use a heuristic to set max_score
            scores = [score for _, score in docs_and_scores]
            min_score = 0.0
            max_score = max(scores) * 1.5 if scores else 1.0
        
        # Normalize to [0, 1]
        normalized_results = []
        for doc, score in docs_and_scores:
            if self.distance_strategy == "cosine" or self.distance_strategy == "dot":
                # For similarity measures, normalize to [0, 1] where 1 is most similar
                normalized_score = (score - min_score) / (max_score - min_score)
            else:
                # For distance measures, invert so that 1 is most similar
                normalized_score = 1.0 - (score / max_score)
                
            normalized_results.append((doc, normalized_score))
            
        return normalized_results
    
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
        Perform maximum marginal relevance search using the vector store.
        
        Args:
            query: Query text
            k: Number of results to return
            fetch_k: Number of initial results to fetch
            lambda_mult: Balance between relevance and diversity (0-1)
            filter: Optional filter for metadata
            
        Returns:
            List of Document objects
        """
        # Generate embedding for query
        embedding = self.embedding.embed_query(query)
        
        # Convert filter dictionary to SQL WHERE clause if provided
        filter_query = None
        if filter:
            filter_query = self._filter_dict_to_sql(filter)
        
        # Perform similarity search with vectors using Arrow Flight
        results = self.client.similarity_search(
            table_name=self.table_name,
            query_vector=embedding,
            k=fetch_k,
            filter_query=filter_query,
            include_metadata=True,
            include_vectors=True,  # Need vectors for MMR
            distance_strategy=self.distance_strategy
        )
        
        # Extract vectors and metadata
        vectors = [result['vector'] for result in results]
        metadatas = [result['metadata'] for result in results]
        texts = [result['text'] for result in results]
        
        # Use GPU-accelerated MMR reranking if available
        try:
            # Convert to numpy arrays for MMR
            query_vector = np.array(embedding, dtype=np.float32)
            result_vectors = np.array(vectors, dtype=np.float32)
            
            if hasattr(self, 'memory_manager') and self.memory_manager.cuda_available:
                # Use GPU-accelerated MMR
                mmr_indices = self.memory_manager.mmr_rerank(
                    query_vector=query_vector,
                    vectors=result_vectors,
                    indices=list(range(len(vectors))),
                    k=k,
                    lambda_mult=lambda_mult
                )
            else:
                # Fall back to CPU MMR
                mmr_indices = maximal_marginal_relevance(
                    query_vector, result_vectors, k=k, lambda_mult=lambda_mult
                )
                
            # Rerank documents based on MMR indices
            mmr_results = [
                Document(page_content=texts[i], metadata=metadatas[i])
                for i in mmr_indices
            ]
            
            return mmr_results
            
        except Exception as e:
            logger.warning(f"Error in MMR reranking: {str(e)}")
            
            # Fall back to standard similarity search
            docs_with_scores = self.similarity_search_with_score(query, k, filter, **kwargs)
            return [doc for doc, _ in docs_with_scores]
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete
            filter: Optional filter for metadata
        """
        if ids is None and filter is None:
            raise ValueError("Either ids or filter must be provided")
            
        # Build delete query
        delete_query = f"DELETE FROM {self.table_name}"
        params = []
        
        if ids is not None:
            # Delete by IDs
            id_list = ", ".join(["?" for _ in ids])
            delete_query += f" WHERE {self.id_column} IN ({id_list})"
            params.extend(ids)
        elif filter is not None:
            # Delete by metadata filter
            filter_clause = self._filter_dict_to_sql(filter)
            if filter_clause:
                delete_query += f" WHERE {filter_clause}"
        
        # Execute delete command
        self._execute_command(delete_query, params)
    
    def get(
        self,
        ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Document]:
        """
        Get documents by ID.
        
        Args:
            ids: List of document IDs to get
            limit: Optional limit on number of documents to return
            
        Returns:
            Dictionary of ID -> Document
        """
        if ids is None:
            raise ValueError("ids must be provided")
            
        # Build select query
        select_query = f"""
        SELECT {self.id_column}, {self.text_column}, {self.metadata_column}
        FROM {self.table_name}
        WHERE {self.id_column} IN ({', '.join(['?' for _ in ids])})
        """
        
        if limit is not None:
            select_query += f" LIMIT {limit}"
            
        # Execute query
        results = self._execute_command(select_query, ids)
        
        # Convert results to Documents
        docs_by_id = {}
        for id_val, text, metadata_json in results:
            metadata = self._json_to_metadata(metadata_json)
            doc = Document(page_content=text, metadata=metadata)
            docs_by_id[id_val] = doc
            
        return docs_by_id
    
    def update_document(
        self,
        document_id: str,
        document: Document,
        **kwargs: Any,
    ) -> None:
        """
        Update a document in the vector store.
        
        Args:
            document_id: ID of document to update
            document: New document
        """
        # Generate embedding for the document
        embedding = self.embedding.embed_documents([document.page_content])[0]
        
        # Convert metadata to JSON
        metadata_json = self._metadata_to_json(document.metadata)
        
        # Upload using Arrow Flight
        self.client.upload_vectors(
            table_name=self.table_name,
            vectors=[embedding],
            texts=[document.page_content],
            metadata=[document.metadata],
            ids=[document_id],
            batch_size=1
        )
    
    def update_documents(
        self,
        ids: List[str],
        documents: List[Document],
        **kwargs: Any,
    ) -> None:
        """
        Update multiple documents in the vector store.
        
        Args:
            ids: List of document IDs to update
            documents: List of new documents
        """
        if len(ids) != len(documents):
            raise ValueError("Number of IDs must match number of documents")
            
        # Generate embeddings for documents
        embeddings = self.embedding.embed_documents([doc.page_content for doc in documents])
        
        # Extract texts and metadata
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Upload using Arrow Flight
        self.client.upload_vectors(
            table_name=self.table_name,
            vectors=embeddings,
            texts=texts,
            metadata=metadatas,
            ids=ids,
            batch_size=self.batch_size
        )
    
    def search(
        self,
        query: Union[str, List[float]],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Flexible search interface supporting text or vector queries.
        
        Args:
            query: Query text or vector
            k: Number of results to return
            filter: Optional filter for metadata
            
        Returns:
            List of Document objects
        """
        if isinstance(query, str):
            # Text query - use standard similarity search
            return self.similarity_search(query, k, filter, **kwargs)
        else:
            # Vector query - use vector directly
            filter_query = None
            if filter:
                filter_query = self._filter_dict_to_sql(filter)
            
            # Perform similarity search using Arrow Flight
            results = self.client.similarity_search(
                table_name=self.table_name,
                query_vector=query,
                k=k,
                filter_query=filter_query,
                include_metadata=True,
                include_vectors=False,
                distance_strategy=self.distance_strategy
            )
            
            # Convert results to Documents
            documents = []
            for result in results:
                if 'text' in result and 'metadata' in result:
                    doc = Document(
                        page_content=result['text'],
                        metadata=result['metadata']
                    )
                    documents.append(doc)
            
            return documents
    
    def search_with_score(
        self,
        query: Union[str, List[float]],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Flexible search interface with scores, supporting text or vector queries.
        
        Args:
            query: Query text or vector
            k: Number of results to return
            filter: Optional filter for metadata
            
        Returns:
            List of (Document, score) tuples
        """
        if isinstance(query, str):
            # Text query - use standard similarity search with score
            return self.similarity_search_with_score(query, k, filter, **kwargs)
        else:
            # Vector query - use vector directly
            filter_query = None
            if filter:
                filter_query = self._filter_dict_to_sql(filter)
            
            # Perform similarity search using Arrow Flight
            results = self.client.similarity_search(
                table_name=self.table_name,
                query_vector=query,
                k=k,
                filter_query=filter_query,
                include_metadata=True,
                include_vectors=False,
                distance_strategy=self.distance_strategy
            )
            
            # Convert results to Documents with scores
            docs_with_scores = []
            for result in results:
                if 'text' in result and 'metadata' in result and 'score' in result:
                    doc = Document(
                        page_content=result['text'],
                        metadata=result['metadata']
                    )
                    score = result['score']
                    docs_with_scores.append((doc, score))
            
            return docs_with_scores
    
    def aclose(self) -> None:
        """Close all resources and connections."""
        if hasattr(self, 'client') and self.client is not None:
            self.client.close()
            
        if hasattr(self, 'memory_manager') and self.memory_manager is not None:
            self.memory_manager.cleanup()