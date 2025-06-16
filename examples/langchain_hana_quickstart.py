#!/usr/bin/env python
"""
LangChain Integration for SAP HANA Cloud - Quickstart Example

This example demonstrates how to use LangChain with SAP HANA Cloud for 
vector search capabilities. It shows the basic setup, document insertion,
and similarity search operations.

Usage:
    python langchain_hana_quickstart.py

Requirements:
    - SAP HANA Cloud instance with credentials
    - Python packages: langchain, langchain_hana, sentence-transformers
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

def get_connection(connection_params: Dict[str, Any] = None):
    """
    Create a connection to SAP HANA Cloud.
    
    Args:
        connection_params: Dictionary with connection parameters.
                           If None, tries to read from environment variables.
    
    Returns:
        HANA database connection
    """
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
    missing_params = [param for param in required_params if param not in connection_params]
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
    logger.info("Connected successfully.")
    return connection

def create_sample_documents() -> List[Dict[str, Any]]:
    """
    Create sample documents for the example.
    
    Returns:
        List of dictionaries containing document text and metadata
    """
    documents = [
        {
            "text": "SAP HANA Cloud is a cloud-based database management system that enables organizations to analyze large volumes of data in real-time.",
            "metadata": {
                "source": "SAP Documentation",
                "category": "database",
                "topic": "cloud",
                "year": 2023
            }
        },
        {
            "text": "LangChain is a framework for developing applications powered by large language models. It provides abstractions for working with different types of models, data sources, and tools.",
            "metadata": {
                "source": "LangChain Documentation",
                "category": "framework",
                "topic": "ai",
                "year": 2023
            }
        },
        {
            "text": "Vector databases store data as high-dimensional vectors and enable efficient similarity search, making them ideal for AI applications.",
            "metadata": {
                "source": "AI Engineering Blog",
                "category": "database",
                "topic": "vectors",
                "year": 2023
            }
        },
        {
            "text": "Embedding models convert text into numerical vector representations that capture semantic meaning, allowing machines to understand relationships between concepts.",
            "metadata": {
                "source": "ML Research Paper",
                "category": "machine learning",
                "topic": "embeddings",
                "year": 2022
            }
        },
        {
            "text": "SAP HANA's vector engine provides advanced capabilities for AI and machine learning workloads, supporting vector operations directly in the database.",
            "metadata": {
                "source": "SAP Blog",
                "category": "database",
                "topic": "vectors",
                "year": 2023
            }
        }
    ]
    return documents

def load_connection_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load connection configuration from a JSON file.
    
    Args:
        config_path: Path to the connection configuration file.
                    If None, looks for 'connection.json' in the current directory
                    or in the config directory.
    
    Returns:
        Dictionary with connection parameters
    """
    if config_path is None:
        # Check common locations for the configuration file
        possible_paths = [
            "connection.json",
            "config/connection.json",
            "../config/connection.json"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading connection configuration from {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    logger.warning("No connection configuration file found. Using environment variables.")
    return None

def main():
    """Main function to run the example."""
    # Load connection configuration
    connection_params = load_connection_config()
    
    try:
        # Create database connection
        connection = get_connection(connection_params)
        
        # Initialize embedding model
        logger.info("Initializing embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = HanaDB(
            connection=connection,
            embedding=embedding_model,
            table_name="LANGCHAIN_QUICKSTART",
            distance_strategy=DistanceStrategy.COSINE,
        )
        
        # Create sample documents
        documents = create_sample_documents()
        logger.info(f"Created {len(documents)} sample documents")
        
        # Add documents to vector store
        logger.info("Adding documents to vector store...")
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        vector_store.add_texts(texts, metadatas)
        
        # Optionally create an HNSW index for faster searches
        logger.info("Creating HNSW index for faster similarity search...")
        vector_store.create_hnsw_index()
        
        # Perform similarity search
        logger.info("Performing similarity search...")
        query = "How does SAP HANA support AI applications?"
        results = vector_store.similarity_search(query, k=2)
        
        logger.info(f"Top 2 results for query: '{query}'")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Content: {doc.page_content}")
            logger.info(f"  Metadata: {doc.metadata}")
            logger.info("")
        
        # Perform filtered search
        logger.info("Performing filtered search...")
        filter_query = {"category": "database"}
        filtered_results = vector_store.similarity_search(
            query, 
            k=2, 
            filter=filter_query
        )
        
        logger.info(f"Top 2 results for query: '{query}' with filter: {filter_query}")
        for i, doc in enumerate(filtered_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Content: {doc.page_content}")
            logger.info(f"  Metadata: {doc.metadata}")
            logger.info("")
        
        # Perform MMR search for diversity
        logger.info("Performing MMR search for diverse results...")
        mmr_results = vector_store.max_marginal_relevance_search(
            query,
            k=2,
            fetch_k=3,
            lambda_mult=0.5  # Balance between relevance and diversity
        )
        
        logger.info(f"Top 2 diverse results for query: '{query}'")
        for i, doc in enumerate(mmr_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Content: {doc.page_content}")
            logger.info(f"  Metadata: {doc.metadata}")
            logger.info("")
        
        # Update a document
        logger.info("Updating a document...")
        update_text = "SAP HANA Cloud is a cloud-native database management system with advanced vector capabilities for AI workloads and real-time analytics."
        update_metadata = {
            "source": "SAP Documentation",
            "category": "database",
            "topic": "cloud",
            "year": 2024,  # Updated year
            "updated": True
        }
        
        vector_store.update_texts(
            texts=[update_text],
            filter={"source": "SAP Documentation", "topic": "cloud"},
            metadatas=[update_metadata]
        )
        
        # Search again to see updated document
        logger.info("Searching again to see updated document...")
        updated_results = vector_store.similarity_search("SAP HANA Cloud capabilities", k=1)
        
        logger.info("Updated document:")
        logger.info(f"  Content: {updated_results[0].page_content}")
        logger.info(f"  Metadata: {updated_results[0].metadata}")
        
        # Clean up - drop the table if you don't want to keep the data
        logger.info("Cleaning up...")
        cursor = connection.cursor()
        cursor.execute('DROP TABLE "LANGCHAIN_QUICKSTART"')
        cursor.close()
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        # Close the connection
        if 'connection' in locals():
            connection.close()
            logger.info("Connection closed.")

if __name__ == "__main__":
    main()