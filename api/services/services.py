"""Services for the API."""

import logging
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_hana import HanaDB, HanaInternalEmbeddings
from langchain_hana.utils import DistanceStrategy

from api.config import config
from api.models.models import DocumentModel, DocumentResponse
from api.gpu import gpu_utils
from api.embeddings.embeddings import GPUAcceleratedEmbeddings, GPUHybridEmbeddings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for vector store operations."""
    
    def __init__(
        self,
        connection: dbapi.Connection,
        embedding: Embeddings,
        table_name: Optional[str] = None,
    ):
        """
        Initialize the vector store service.
        
        Args:
            connection: Database connection.
            embedding: Embedding model.
            table_name: Name of the table to use. Defaults to None.
        """
        self.connection = connection
        self.embedding = embedding
        self.table_name = table_name or config.vectorstore.table_name
        
        # Initialize the vector store
        self.vectorstore = HanaDB(
            connection=connection,
            embedding=embedding,
            distance_strategy=DistanceStrategy.COSINE,
            table_name=self.table_name,
            content_column=config.vectorstore.content_column,
            metadata_column=config.vectorstore.metadata_column,
            vector_column=config.vectorstore.vector_column,
            vector_column_length=config.vectorstore.vector_column_length,
            vector_column_type=config.vectorstore.vector_column_type,
        )
        
        # Log GPU availability
        if gpu_utils.is_gpu_available():
            gpu_info = gpu_utils.get_gpu_info()
            logger.info(f"GPU acceleration is available with {gpu_info.get('device_count', 0)} device(s)")
        else:
            logger.info("GPU acceleration is not available, using CPU")
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Add texts to the vector store.
        
        Args:
            texts: List of texts to add.
            metadatas: Optional list of metadata dictionaries.
        """
        logger.info(f"Adding {len(texts)} texts to vector store table {self.table_name}")
        
        # Batch processing for efficiency
        batch_size = config.gpu.batch_size if hasattr(config, 'gpu') else 32
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size] if metadatas else None
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            self.vectorstore.add_texts(batch_texts, batch_metadatas)
        
        logger.info(f"Successfully added {len(texts)} texts to vector store")
    
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentResponse]:
        """
        Perform similarity search.
        
        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional filter.
            
        Returns:
            List of documents with scores.
        """
        logger.info(f"Performing similarity search with query: {query}, k: {k}")
        docs_and_scores = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
        )
        
        return [
            DocumentResponse(
                document=DocumentModel(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                ),
                score=score,
            )
            for doc, score in docs_and_scores
        ]
    
    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentResponse]:
        """
        Perform similarity search by vector.
        
        Args:
            embedding: Query embedding.
            k: Number of results to return.
            filter: Optional filter.
            
        Returns:
            List of documents with scores.
        """
        logger.info(f"Performing similarity search by vector, k: {k}")
        docs_and_scores = self.vectorstore.similarity_search_with_score_and_vector_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
        )
        
        return [
            DocumentResponse(
                document=DocumentModel(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                ),
                score=score,
            )
            for doc, score, _ in docs_and_scores
        ]
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentResponse]:
        """
        Perform max marginal relevance search.
        
        Args:
            query: Query text.
            k: Number of results to return.
            fetch_k: Number of results to fetch initially.
            lambda_mult: Lambda multiplier.
            filter: Optional filter.
            
        Returns:
            List of documents.
        """
        logger.info(f"Performing MMR search with query: {query}, k: {k}")
        
        # Use the standard HanaDB MMR search for internal embeddings
        if isinstance(self.embedding, HanaInternalEmbeddings) or (
            isinstance(self.embedding, GPUHybridEmbeddings) and self.embedding.use_internal
        ):
            docs = self.vectorstore.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter,
            )
        else:
            # Use custom GPU-accelerated MMR for external embeddings
            # 1. Get the query embedding
            query_embedding = self.embedding.embed_query(query)
            
            # 2. Get the initial results
            whole_result = self.vectorstore.similarity_search_with_score_and_vector_by_vector(
                embedding=query_embedding,
                k=fetch_k,
                filter=filter,
            )
            
            # 3. Extract document embeddings
            embeddings = [result_item[2] for result_item in whole_result]
            
            # 4. Use GPU MMR
            mmr_doc_indexes = gpu_utils.gpu_maximal_marginal_relevance(
                query_embedding=query_embedding,
                embedding_list=embeddings,
                lambda_mult=lambda_mult,
                k=k,
            )
            
            # 5. Create document list
            docs = [whole_result[i][0] for i in mmr_doc_indexes]
        
        return [
            DocumentResponse(
                document=DocumentModel(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                ),
            )
            for doc in docs
        ]
    
    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentResponse]:
        """
        Perform max marginal relevance search by vector.
        
        Args:
            embedding: Query embedding.
            k: Number of results to return.
            fetch_k: Number of results to fetch initially.
            lambda_mult: Lambda multiplier.
            filter: Optional filter.
            
        Returns:
            List of documents.
        """
        logger.info(f"Performing MMR search by vector, k: {k}")
        
        # Get the initial results
        whole_result = self.vectorstore.similarity_search_with_score_and_vector_by_vector(
            embedding=embedding,
            k=fetch_k,
            filter=filter,
        )
        
        # Extract document embeddings
        embeddings = [result_item[2] for result_item in whole_result]
        
        # Use GPU MMR if available
        mmr_doc_indexes = gpu_utils.gpu_maximal_marginal_relevance(
            query_embedding=embedding,
            embedding_list=embeddings,
            lambda_mult=lambda_mult,
            k=k,
        )
        
        # Create document list
        docs = [whole_result[i][0] for i in mmr_doc_indexes]
        
        return [
            DocumentResponse(
                document=DocumentModel(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                ),
            )
            for doc in docs
        ]
    
    def delete(
        self,
        filter: Dict[str, Any],
    ) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            filter: Filter to use for deletion.
            
        Returns:
            True if successful.
        """
        logger.info(f"Deleting documents with filter: {filter}")
        result = self.vectorstore.delete(filter=filter)
        logger.info("Successfully deleted documents")
        return result


# Add aliases to match what's expected in the import statements
VectorService = VectorStoreService


class APIService:
    """Service class for API operations."""
    
    def __init__(self):
        """Initialize the API service."""
        logger.info("Initializing APIService in CPU-only mode")


class EmbeddingService:
    """Service class for embedding operations."""
    
    def __init__(self, embedding_model=None):
        """Initialize the embedding service."""
        logger.info("Initializing EmbeddingService in CPU-only mode")
        self.embedding_model = embedding_model