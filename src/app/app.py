#!/usr/bin/env python
"""
LangChain Integration for SAP HANA Cloud - Working System

This application provides a functional integration between LangChain and 
SAP HANA Cloud's vector database capabilities for document storage and retrieval.
"""

import os
import json
import logging
from typing import List, Dict, Any

from hdbcli import dbapi
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from langchain_hana.vectorstores import HanaDB
from langchain_hana.utils import DistanceStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SAP_HANA_Langchain_Integration:
    """Main class for SAP HANA Cloud and LangChain integration."""
    
    def __init__(self, connection_params=None, table_name="LANGCHAIN_DOCUMENTS"):
        """
        Initialize the integration with SAP HANA Cloud.
        
        Args:
            connection_params: Dictionary with connection parameters or None to use env vars
            table_name: Name of the table to store document vectors
        """
        self.connection = self._get_connection(connection_params)
        self.table_name = table_name
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize vector store
        self.vector_store = HanaDB(
            connection=self.connection,
            embedding=self.embedding_model,
            table_name=self.table_name,
            distance_strategy=DistanceStrategy.COSINE,
        )
        
        logger.info(f"Initialized SAP HANA Langchain integration with table: {table_name}")
    
    def _get_connection(self, connection_params=None):
        """Create a connection to SAP HANA Cloud."""
        # If no connection parameters provided, try to get from environment variables
        if connection_params is None:
            connection_params = {
                "address": os.environ.get("HANA_HOST"),
                "port": int(os.environ.get("HANA_PORT", "443")),
                "user": os.environ.get("HANA_USER"),
                "password": os.environ.get("HANA_PASSWORD"),
            }
        
        # Check for required parameters
        required_params = ["address", "port", "user", "password"]
        missing_params = [param for param in required_params if not connection_params.get(param)]
        if missing_params:
            raise ValueError(f"Missing required connection parameters: {', '.join(missing_params)}")
        
        # Connect to SAP HANA
        logger.info(f"Connecting to SAP HANA at {connection_params['address']}:{connection_params['port']}...")
        connection = dbapi.connect(
            address=connection_params["address"],
            port=connection_params["port"],
            user=connection_params["user"],
            password=connection_params["password"],
            encrypt=True,
            sslValidateCertificate=False,
        )
        logger.info("Connected successfully to SAP HANA Cloud.")
        return connection
    
    def load_connection_config(self, config_path=None):
        """Load connection configuration from a JSON file."""
        if config_path is None:
            # Check common locations for the configuration file
            possible_paths = [
                "connection.json",
                "config/connection.json",
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path and os.path.exists(config_path):
            logger.info(f"Loading connection configuration from {config_path}")
            with open(config_path, 'r') as f:
                return json.load(f)
        
        logger.warning("No connection configuration file found.")
        return None
    
    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """
        Add documents to the vector store.
        
        Args:
            texts: List of text documents to add
            metadatas: Optional list of metadata dictionaries, one per document
            
        Returns:
            Success status
        """
        try:
            if metadatas and len(texts) != len(metadatas):
                raise ValueError("Number of texts and metadatas must match")
            
            # Add documents to vector store
            self.vector_store.add_texts(texts, metadatas)
            
            # Optionally create an index if table has significant data
            if len(texts) > 100:
                self._create_index()
                
            logger.info(f"Added {len(texts)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def _create_index(self):
        """Create an HNSW index on the vector store for faster searches."""
        try:
            self.vector_store.create_hnsw_index()
            logger.info("Created HNSW index for vector search")
            return True
        except Exception as e:
            logger.warning(f"Could not create index: {e}")
            return False
    
    def search(self, query: str, k: int = 5, filter: Dict[str, Any] = None):
        """
        Search for documents similar to the query.
        
        Args:
            query: The search query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of documents with similarity to the query
        """
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Found {len(results)} results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def search_with_score(self, query: str, k: int = 5, filter: Dict[str, Any] = None):
        """
        Search for documents and return with similarity scores.
        
        Args:
            query: The search query text
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        try:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter
            )
            logger.info(f"Found {len(results)} scored results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error during scored search: {e}")
            return []
    
    def diverse_search(self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5):
        """
        Perform diverse search using Maximal Marginal Relevance.
        
        Args:
            query: The search query text
            k: Number of results to return
            fetch_k: Number of results to consider for diversity
            lambda_mult: Diversity factor (0=max diversity, 1=max relevance)
            
        Returns:
            List of diverse documents
        """
        try:
            results = self.vector_store.max_marginal_relevance_search(
                query=query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult
            )
            logger.info(f"Found {len(results)} diverse results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error during diverse search: {e}")
            return []
    
    def update_documents(self, texts: List[str], filter: Dict[str, Any], metadatas: List[Dict[str, Any]] = None):
        """
        Update existing documents matching the filter.
        
        Args:
            texts: New text content
            filter: Filter to identify documents to update
            metadatas: Optional new metadata
            
        Returns:
            Success status
        """
        try:
            result = self.vector_store.update_texts(
                texts=texts,
                filter=filter,
                metadatas=metadatas
            )
            logger.info(f"Updated documents with filter: {filter}")
            return result
        except Exception as e:
            logger.error(f"Error updating documents: {e}")
            return False
    
    def delete_documents(self, filter: Dict[str, Any]):
        """
        Delete documents matching the filter.
        
        Args:
            filter: Filter to identify documents to delete
            
        Returns:
            Success status
        """
        try:
            result = self.vector_store.delete(filter=filter)
            logger.info(f"Deleted documents with filter: {filter}")
            return result
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def close(self):
        """Close the database connection and cleanup resources."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
            logger.info("Connection to SAP HANA Cloud closed")

# Example usage
if __name__ == "__main__":
    # Initialize the integration
    try:
        # Try to load config from file first
        integration = SAP_HANA_Langchain_Integration()
        
        # Add sample documents
        sample_docs = [
            "SAP HANA Cloud is a cloud-based database management system that enables real-time analytics.",
            "LangChain is a framework for developing applications powered by large language models.",
            "Vector databases store data as high-dimensional vectors for efficient similarity search.",
            "Embedding models convert text into numerical representations that capture semantic meaning.",
            "SAP HANA's vector engine supports AI and machine learning workloads directly in the database."
        ]
        
        sample_metadata = [
            {"source": "SAP Documentation", "category": "database"},
            {"source": "LangChain Documentation", "category": "framework"},
            {"source": "AI Engineering Blog", "category": "database"},
            {"source": "ML Research Paper", "category": "machine learning"},
            {"source": "SAP Blog", "category": "database"}
        ]
        
        # Add the documents
        integration.add_documents(sample_docs, sample_metadata)
        
        # Perform a simple search
        results = integration.search("What is SAP HANA Cloud?", k=2)
        
        print("\nSearch Results:")
        for i, doc in enumerate(results):
            print(f"Result {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print()
        
        # Perform a filtered search
        filtered_results = integration.search(
            "database capabilities",
            k=2,
            filter={"category": "database"}
        )
        
        print("\nFiltered Results:")
        for i, doc in enumerate(filtered_results):
            print(f"Result {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
            print()
        
        # Clean up
        integration.close()
        
    except Exception as e:
        logger.error(f"Error in example: {e}")
        import traceback
        logger.error(traceback.format_exc())