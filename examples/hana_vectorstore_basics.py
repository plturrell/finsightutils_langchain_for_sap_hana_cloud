#!/usr/bin/env python3
"""
SAP HANA Cloud Vector Store Basics Example

This example demonstrates the fundamental operations of the SAP HANA Cloud Vector Store
implementation for LangChain. It covers:

1. Connecting to SAP HANA Cloud
2. Creating a vector store
3. Adding documents with metadata
4. Performing similarity search
5. Using metadata filters
6. Creating and using HNSW indexes for performance
7. Using Maximal Marginal Relevance (MMR) search

Prerequisites:
- SAP HANA Cloud instance
- Python 3.8+
- Required packages: langchain, langchain_hana, sentence-transformers, hdbcli

Usage:
    python hana_vectorstore_basics.py --config_file config/connection.json
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional

from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from hdbcli import dbapi

from langchain_hana.vectorstores import HanaDB
from langchain_hana.utils import DistanceStrategy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_connection_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load connection configuration from file or environment variables."""
    # Try to load from file
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading connection configuration from {config_path}")
        with open(config_path, 'r') as f:
            return json.load(f)
    
    # Try environment variables
    logger.info("Loading connection configuration from environment variables")
    return {
        "address": os.environ.get("HANA_HOST"),
        "port": int(os.environ.get("HANA_PORT", "443")),
        "user": os.environ.get("HANA_USER"),
        "password": os.environ.get("HANA_PASSWORD"),
    }


def create_connection(connection_params: Dict[str, Any]) -> dbapi.Connection:
    """Create a connection to SAP HANA Cloud."""
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
    """Create sample documents with metadata for the example."""
    return [
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
            "text": "Vector databases store data as high-dimensional vectors, enabling efficient similarity search based on meaning rather than exact matches.",
            "metadata": {
                "source": "AI Engineering Blog",
                "category": "database",
                "topic": "vectors",
                "year": 2023
            }
        },
        {
            "text": "LangChain is a framework for developing applications powered by language models, providing tools for combining LLMs with other sources of computation or knowledge.",
            "metadata": {
                "source": "LangChain Documentation",
                "category": "framework",
                "topic": "llm",
                "year": 2023
            }
        },
        {
            "text": "SAP HANA's vector engine provides advanced capabilities for AI and machine learning workloads, supporting vector operations directly in the database.",
            "metadata": {
                "source": "SAP Blog",
                "category": "database",
                "topic": "ai",
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
            "text": "Retrieval-Augmented Generation (RAG) combines the strengths of retrieval-based and generation-based approaches to enhance the accuracy and factuality of language model outputs.",
            "metadata": {
                "source": "AI Research Paper",
                "category": "technique",
                "topic": "rag",
                "year": 2023
            }
        },
        {
            "text": "The HNSW (Hierarchical Navigable Small World) algorithm is a graph-based method for approximate nearest neighbor search, offering excellent performance for high-dimensional vector spaces.",
            "metadata": {
                "source": "Algorithm Documentation",
                "category": "algorithm",
                "topic": "search",
                "year": 2022
            }
        },
        {
            "text": "SAP HANA Cloud can integrate with various AI services and tools, enabling enterprises to build intelligent applications leveraging both structured and unstructured data.",
            "metadata": {
                "source": "SAP Documentation",
                "category": "integration",
                "topic": "ai",
                "year": 2023
            }
        },
    ]


def initialize_embedding_model():
    """Initialize a sentence transformer embedding model."""
    logger.info("Initializing embedding model (Sentence Transformers)...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def demonstrate_basic_operations(connection):
    """Demonstrate basic vector store operations."""
    # Step 1: Create sample documents
    documents = create_sample_documents()
    logger.info(f"Created {len(documents)} sample documents")
    
    # Step 2: Initialize embedding model
    embedding_model = initialize_embedding_model()
    
    # Step 3: Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = HanaDB(
        connection=connection,
        embedding=embedding_model,
        table_name="VECTORSTORE_BASICS",
        distance_strategy=DistanceStrategy.COSINE,
    )
    
    # Step 4: Add documents to vector store
    logger.info("Adding documents to vector store...")
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    
    start_time = time.time()
    vector_store.add_texts(texts, metadatas)
    logger.info(f"Added documents in {time.time() - start_time:.2f} seconds")
    
    # Step 5: Basic similarity search
    logger.info("\nPerforming basic similarity search...")
    query = "How does SAP HANA Cloud support AI applications?"
    
    start_time = time.time()
    results = vector_store.similarity_search(query, k=2)
    search_time = time.time() - start_time
    
    logger.info(f"Search completed in {search_time:.4f} seconds")
    logger.info(f"Top 2 results for query: '{query}'")
    for i, doc in enumerate(results):
        logger.info(f"Result {i+1}:")
        logger.info(f"  Content: {doc.page_content}")
        logger.info(f"  Metadata: {doc.metadata}")
        logger.info("")
    
    # Step 6: Create HNSW index for better performance
    logger.info("\nCreating HNSW index for faster similarity search...")
    try:
        start_time = time.time()
        vector_store.create_hnsw_index(
            m=16,                  # Number of connections per node
            ef_construction=128,   # Index building quality parameter
            ef_search=64           # Search quality parameter
        )
        logger.info(f"HNSW index created in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        logger.warning(f"Failed to create HNSW index: {str(e)}")
    
    # Step 7: Search with HNSW index (should be faster)
    logger.info("\nPerforming similarity search with HNSW index...")
    start_time = time.time()
    results_with_index = vector_store.similarity_search(query, k=2)
    indexed_search_time = time.time() - start_time
    
    logger.info(f"Indexed search completed in {indexed_search_time:.4f} seconds")
    if indexed_search_time < search_time:
        improvement = (1 - indexed_search_time/search_time) * 100
        logger.info(f"Search is {improvement:.1f}% faster with HNSW index")
    
    # Step 8: Demonstrate metadata filtering
    logger.info("\nPerforming filtered search...")
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
    
    # Step 9: Complex filtering
    logger.info("\nPerforming search with complex filter...")
    complex_filter = {
        "$and": [
            {"year": 2023},
            {"$or": [
                {"topic": "ai"},
                {"topic": "cloud"}
            ]}
        ]
    }
    complex_results = vector_store.similarity_search(
        query, 
        k=3, 
        filter=complex_filter
    )
    
    logger.info(f"Results for query: '{query}' with complex filter")
    for i, doc in enumerate(complex_results):
        logger.info(f"Result {i+1}:")
        logger.info(f"  Content: {doc.page_content}")
        logger.info(f"  Metadata: {doc.metadata}")
        logger.info("")
    
    # Step 10: Demonstrate MMR search for diversity
    logger.info("\nPerforming MMR search for diverse results...")
    mmr_results = vector_store.max_marginal_relevance_search(
        query,
        k=3,
        fetch_k=5,
        lambda_mult=0.5  # Balance between relevance and diversity
    )
    
    logger.info(f"Top 3 diverse results for query: '{query}'")
    for i, doc in enumerate(mmr_results):
        logger.info(f"Result {i+1}:")
        logger.info(f"  Content: {doc.page_content}")
        logger.info(f"  Metadata: {doc.metadata}")
        logger.info("")
    
    # Step 11: Demonstrate document update
    logger.info("\nUpdating a document...")
    update_text = "SAP HANA Cloud is a cloud-native database with advanced vector capabilities for AI workloads and real-time analytics."
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
    logger.info("\nSearching again to see updated document...")
    updated_results = vector_store.similarity_search("SAP HANA Cloud capabilities", k=1)
    
    logger.info("Updated document:")
    logger.info(f"  Content: {updated_results[0].page_content}")
    logger.info(f"  Metadata: {updated_results[0].metadata}")
    
    return vector_store


def cleanup(vector_store):
    """Clean up by dropping the table."""
    logger.info("\nCleaning up...")
    try:
        cursor = vector_store.connection.cursor()
        cursor.execute(f'DROP TABLE "{vector_store.table_name}"')
        cursor.close()
        logger.info(f"Dropped table {vector_store.table_name}")
    except Exception as e:
        logger.warning(f"Cleanup failed: {str(e)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SAP HANA Vector Store Basics Example")
    parser.add_argument("--config_file", help="Path to connection configuration file")
    parser.add_argument("--no_cleanup", action="store_true", help="Don't drop the table at the end")
    return parser.parse_args()


def main():
    """Main function to run the vector store basics example."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Load connection configuration
        connection_params = load_connection_config(args.config_file)
        
        # Create connection to SAP HANA Cloud
        connection = create_connection(connection_params)
        
        # Run demonstration
        vector_store = demonstrate_basic_operations(connection)
        
        # Cleanup
        if not args.no_cleanup:
            cleanup(vector_store)
        
        logger.info("Example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Close the connection
        if 'connection' in locals():
            connection.close()
            logger.info("Connection closed.")


if __name__ == "__main__":
    main()