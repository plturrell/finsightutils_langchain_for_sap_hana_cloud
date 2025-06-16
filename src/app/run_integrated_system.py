#\!/usr/bin/env python
"""
LangChain Integration for SAP HANA Cloud - Production Demo

This script demonstrates the production-grade integration between
LangChain and SAP HANA Cloud for vector search capabilities.
"""

import os
import json
import logging
import time
from typing import List, Dict, Any, Optional

from langchain_hana_integration import SAP_HANA_VectorStore, HanaOptimizedEmbeddings
from langchain_hana_integration.connection import create_connection_pool
from langchain_hana_integration.utils.distance import DistanceStrategy
from langchain_hana_integration.exceptions import ConnectionError, VectorOperationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_connection_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load connection configuration from file or environment variables."""
    # Try to load from file
    if config_path is None:
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
    
    # Try environment variables
    logger.info("Loading connection configuration from environment variables")
    return {
        "address": os.environ.get("HANA_HOST"),
        "port": int(os.environ.get("HANA_PORT", "443")),
        "user": os.environ.get("HANA_USER"),
        "password": os.environ.get("HANA_PASSWORD"),
    }


def create_sample_documents() -> List[Dict[str, Any]]:
    """Create sample documents for the demo."""
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


def run_demo():
    """Run the production demo."""
    try:
        # Step 1: Load configuration
        connection_params = load_connection_config()
        
        # Validate required parameters
        required_params = ["address", "port", "user", "password"]
        missing_params = [param for param in required_params if not connection_params.get(param)]
        if missing_params:
            logger.error(f"Missing required connection parameters: {', '.join(missing_params)}")
            logger.error("Please check your connection.json file or environment variables.")
            return
        
        logger.info(f"Using connection parameters: host={connection_params['address']}, port={connection_params['port']}, user={connection_params['user']}")
        
        # Step 2: Create connection pool
        logger.info("Creating connection pool...")
        create_connection_pool(
            connection_params=connection_params,
            pool_name="demo_pool",
            min_connections=1,
            max_connections=5
        )
        
        # Step 3: Initialize embedding model
        logger.info("Initializing embedding model...")
        embedding_model = HanaOptimizedEmbeddings(
            model_name="all-MiniLM-L6-v2",
            enable_caching=True,
            cache_dir="./cache",
            memory_cache_size=1000,
            normalize_embeddings=True
        )
        
        # Step 4: Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = SAP_HANA_VectorStore(
            embedding=embedding_model,
            pool_name="demo_pool",
            table_name="LANGCHAIN_HANA_DEMO",
            distance_strategy=DistanceStrategy.COSINE,
            auto_create_index=True,
            batch_size=50,
            enable_logging=True
        )
        
        # Step 5: Create sample documents
        documents = create_sample_documents()
        logger.info(f"Created {len(documents)} sample documents")
        
        # Step 6: Add documents to vector store
        logger.info("Adding documents to vector store...")
        start_time = time.time()
        
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        vector_store.add_texts(texts, metadatas)
        
        add_time = time.time() - start_time
        logger.info(f"Added {len(documents)} documents in {add_time:.2f} seconds")
        
        # Step 7: Perform similarity search
        logger.info("Performing similarity search...")
        query = "How does SAP HANA support AI applications?"
        start_time = time.time()
        
        results = vector_store.similarity_search(
            query=query,
            k=2
        )
        
        search_time = time.time() - start_time
        logger.info(f"Completed search in {search_time:.4f} seconds")
        logger.info(f"Top 2 results for query: '{query}'")
        
        for i, doc in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
        
        # Step 8: Perform filtered search
        logger.info("\nPerforming filtered search...")
        filter_query = {"category": "database", "topic": "vectors"}
        start_time = time.time()
        
        filtered_results = vector_store.similarity_search(
            query=query,
            k=2,
            filter=filter_query
        )
        
        filter_time = time.time() - start_time
        logger.info(f"Completed filtered search in {filter_time:.4f} seconds")
        logger.info(f"Top 2 results for query: '{query}' with filter: {filter_query}")
        
        for i, doc in enumerate(filtered_results):
            print(f"\nFiltered Result {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
        
        # Step 9: Perform diverse search with MMR
        logger.info("\nPerforming diverse search with MMR...")
        start_time = time.time()
        
        mmr_results = vector_store.max_marginal_relevance_search(
            query=query,
            k=2,
            fetch_k=4,
            lambda_mult=0.7  # Balance between relevance and diversity
        )
        
        mmr_time = time.time() - start_time
        logger.info(f"Completed MMR search in {mmr_time:.4f} seconds")
        logger.info(f"Top 2 diverse results for query: '{query}'")
        
        for i, doc in enumerate(mmr_results):
            print(f"\nMMR Result {i+1}:")
            print(f"Content: {doc.page_content}")
            print(f"Metadata: {doc.metadata}")
        
        # Step 10: Update a document
        logger.info("\nUpdating a document...")
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
        
        logger.info("Document updated successfully")
        
        # Step 11: Search for the updated document
        logger.info("Searching for the updated document...")
        updated_results = vector_store.similarity_search(
            query="What is SAP HANA Cloud?",
            k=1,
            filter={"updated": True}
        )
        
        print("\nUpdated Document:")
        print(f"Content: {updated_results[0].page_content}")
        print(f"Metadata: {updated_results[0].metadata}")
        
        # Step 12: Display performance metrics
        logger.info("\nPerformance Metrics:")
        metrics = vector_store.get_metrics()
        
        print("\nVector Store Metrics:")
        print(f"Total documents added: {metrics.get('total_documents_added', 0)}")
        print(f"Total search calls: {metrics.get('search_calls', 0)}")
        print(f"Average search time: {metrics.get('avg_search_time', 0):.4f} seconds")
        
        embedding_metrics = embedding_model.get_metrics()
        print("\nEmbedding Metrics:")
        print(f"Total embedding calls: {embedding_metrics.get('total_embedding_calls', 0)}")
        print(f"Cache hit rate: {embedding_metrics.get('cache_hit_rate', 0):.2f}")
        print(f"Total tokens processed: {embedding_metrics.get('total_tokens_processed', 0)}")
        
        # Step 13: Clean up (optional - comment this out to keep the data)
        logger.info("\nCleaning up...")
        vector_store.delete(filter={})  # Delete all documents
        
        logger.info("Demo completed successfully\!")
    
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        if hasattr(e, 'details') and e.details:
            logger.error(f"Details: {e.details}")
    
    except VectorOperationError as e:
        logger.error(f"Vector operation error: {e}")
        if hasattr(e, 'details') and e.details:
            logger.error(f"Details: {e.details}")
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    run_demo()
EOF < /dev/null