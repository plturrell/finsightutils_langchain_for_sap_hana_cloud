#!/usr/bin/env python
"""
SAP HANA Cloud Integration with LangChain Example

This example demonstrates how to use LangChain with SAP HANA Cloud vector capabilities,
including:

1. Connecting to SAP HANA Cloud
2. Creating a vector store with embeddings
3. Adding documents to the vector store
4. Performing similarity search
5. Using metadata filters
6. Running maximal marginal relevance search
7. Using internal and external embeddings

Usage:
    python hana_langchain_example.py --host <hana-host> --port <hana-port> --user <username> --password <password>
"""

import argparse
import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hana-langchain-example")

# Import HDBcli
try:
    from hdbcli import dbapi
except ImportError:
    logger.error("HDBcli not found. Please install it with 'pip install hdbcli'")
    sys.exit(1)

# Import LangChain components
try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
except ImportError:
    logger.error("LangChain not found. Please install it with 'pip install langchain-core'")
    sys.exit(1)

# Import HuggingFace embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    logger.error("LangChain Community not found. Please install it with 'pip install langchain-community'")
    sys.exit(1)

# Import our HANA Cloud integration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from langchain_hana.connection import create_connection, test_connection
from langchain_hana.embeddings import HanaInternalEmbeddings, HanaEmbeddingsCache
from langchain_hana.vectorstore import HanaVectorStore, DistanceStrategy


def get_sample_documents() -> List[Dict[str, Any]]:
    """
    Generate sample documents for demonstration.
    
    Returns:
        List of documents with text and metadata
    """
    return [
        {
            "text": "SAP HANA Cloud is a cloud-based database management system that "
                    "combines in-memory processing with columnar storage, enabling "
                    "high-speed analytics and transactions in real-time.",
            "metadata": {
                "source": "documentation",
                "category": "database",
                "tags": ["sap", "cloud", "in-memory"],
                "year": 2023,
            }
        },
        {
            "text": "LangChain is a framework for developing applications powered by "
                    "language models, providing tools for chains, data augmentation, "
                    "agents, memory, and evaluation.",
            "metadata": {
                "source": "documentation",
                "category": "framework",
                "tags": ["llm", "agents", "python"],
                "year": 2023,
            }
        },
        {
            "text": "Vector databases are specialized database systems designed to store, "
                    "manage, and search vector embeddings for machine learning applications "
                    "like semantic search, recommendation systems, and anomaly detection.",
            "metadata": {
                "source": "documentation",
                "category": "database",
                "tags": ["vector", "embeddings", "search"],
                "year": 2023,
            }
        },
        {
            "text": "Embeddings are numerical representations of data, like text or images, "
                    "that capture semantic meaning in a high-dimensional vector space, "
                    "enabling machines to understand relationships between different items.",
            "metadata": {
                "source": "documentation",
                "category": "ml",
                "tags": ["embeddings", "nlp", "vector"],
                "year": 2022,
            }
        },
        {
            "text": "SAP Business Technology Platform (BTP) is a platform-as-a-service "
                    "offering that provides tools, technologies, and services for "
                    "application development, integration, and extension.",
            "metadata": {
                "source": "documentation",
                "category": "platform",
                "tags": ["sap", "cloud", "paas"],
                "year": 2023,
            }
        },
        {
            "text": "Machine learning is a subset of artificial intelligence that enables "
                    "systems to learn from data, identify patterns, and make decisions "
                    "with minimal human intervention.",
            "metadata": {
                "source": "documentation",
                "category": "ml",
                "tags": ["ai", "algorithms", "data"],
                "year": 2022,
            }
        },
        {
            "text": "Natural Language Processing (NLP) is a field of AI that focuses on "
                    "the interaction between computers and human language, enabling "
                    "machines to understand, interpret, and generate human language.",
            "metadata": {
                "source": "documentation",
                "category": "ml",
                "tags": ["nlp", "ai", "language"],
                "year": 2022,
            }
        },
        {
            "text": "SAP S/4HANA is an intelligent ERP system that uses in-memory computing "
                    "to process vast amounts of data and support real-time business processes, "
                    "analytics, and reporting.",
            "metadata": {
                "source": "documentation",
                "category": "erp",
                "tags": ["sap", "enterprise", "hana"],
                "year": 2023,
            }
        },
        {
            "text": "Data analytics is the process of examining data sets to find trends, "
                    "draw conclusions, and make informed decisions using specialized systems "
                    "and software.",
            "metadata": {
                "source": "documentation",
                "category": "analytics",
                "tags": ["data", "insights", "business"],
                "year": 2023,
            }
        },
        {
            "text": "Cloud computing is the delivery of computing services—including servers, "
                    "storage, databases, networking, software, analytics, and intelligence—over "
                    "the internet to offer faster innovation, flexible resources, and economies of scale.",
            "metadata": {
                "source": "documentation",
                "category": "cloud",
                "tags": ["infrastructure", "services", "internet"],
                "year": 2023,
            }
        },
    ]


def run_example(args: argparse.Namespace) -> None:
    """
    Run the SAP HANA Cloud LangChain integration example.
    
    Args:
        args: Command line arguments with connection parameters
    """
    logger.info("Starting SAP HANA Cloud LangChain integration example")
    
    try:
        # Create connection to SAP HANA Cloud
        logger.info(f"Connecting to SAP HANA Cloud at {args.host}:{args.port}")
        connection = create_connection(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
        )
        
        # Test connection
        logger.info("Testing connection...")
        connection_valid, info = test_connection(connection)
        if not connection_valid:
            logger.error(f"Connection test failed: {info.get('error', 'Unknown error')}")
            return
        
        logger.info(f"Connected to SAP HANA Cloud {info.get('version', 'Unknown version')}")
        logger.info(f"Current schema: {info.get('current_schema', 'Unknown')}")
        
        # Set up variables
        schema_name = info.get("current_schema")
        table_name = args.table
        full_table_name = f"{schema_name}.{table_name}"
        
        # Get sample documents
        sample_docs = get_sample_documents()
        texts = [doc["text"] for doc in sample_docs]
        metadatas = [doc["metadata"] for doc in sample_docs]
        
        # Section 1: Using External Embeddings (HuggingFace)
        logger.info("\n=== Example 1: Using External Embeddings (HuggingFace) ===")
        
        # Initialize HuggingFace embeddings
        logger.info("Initializing HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",  # Small model for quick demo
        )
        
        # Create vector store with external embeddings
        logger.info(f"Creating vector store with table {full_table_name}_EXT...")
        vector_store_ext = HanaVectorStore(
            connection=connection,
            embedding=embeddings,
            schema_name=schema_name,
            table_name=f"{table_name}_EXT",
            create_table=True,
            create_hnsw_index=True,
        )
        
        # Add documents
        logger.info(f"Adding {len(sample_docs)} documents to vector store...")
        start_time = time.time()
        vector_store_ext.add_texts(texts, metadatas)
        elapsed_time = time.time() - start_time
        logger.info(f"Added documents in {elapsed_time:.2f} seconds")
        
        # Perform similarity search
        logger.info("Performing similarity search...")
        query = "What is SAP HANA Cloud?"
        start_time = time.time()
        results = vector_store_ext.similarity_search(query, k=3)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Search completed in {elapsed_time:.2f} seconds")
        logger.info(f"Query: \"{query}\"")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc.page_content[:100]}...")
            logger.info(f"Metadata: {doc.metadata}\n")
        
        # Perform search with metadata filter
        logger.info("Performing search with metadata filter...")
        filter_dict = {"category": "database"}
        results = vector_store_ext.similarity_search(
            query, k=3, filter=filter_dict
        )
        
        logger.info(f"Query: \"{query}\" with filter: {filter_dict}")
        for i, doc in enumerate(results):
            logger.info(f"Result {i+1}: {doc.page_content[:100]}...")
            logger.info(f"Metadata: {doc.metadata}\n")
        
        # Perform MMR search
        logger.info("Performing maximal marginal relevance search...")
        mmr_results = vector_store_ext.max_marginal_relevance_search(
            query, k=3, fetch_k=5, lambda_mult=0.7
        )
        
        logger.info(f"MMR Query: \"{query}\"")
        for i, doc in enumerate(mmr_results):
            logger.info(f"Result {i+1}: {doc.page_content[:100]}...")
            logger.info(f"Metadata: {doc.metadata}\n")
        
        # Section 2: Using Cached Embeddings
        logger.info("\n=== Example 2: Using Cached Embeddings ===")
        
        # Create cached embeddings
        logger.info("Creating cached embeddings...")
        cached_embeddings = HanaEmbeddingsCache(
            base_embeddings=embeddings,
            ttl_seconds=3600,  # 1 hour cache lifetime
            max_size=1000,
        )
        
        # Create vector store with cached embeddings
        logger.info(f"Creating vector store with table {full_table_name}_CACHED...")
        vector_store_cached = HanaVectorStore(
            connection=connection,
            embedding=cached_embeddings,
            schema_name=schema_name,
            table_name=f"{table_name}_CACHED",
            create_table=True,
            create_hnsw_index=True,
        )
        
        # Add documents
        logger.info(f"Adding {len(sample_docs)} documents to vector store...")
        vector_store_cached.add_texts(texts, metadatas)
        
        # Perform two searches to demonstrate caching
        logger.info("Performing first search (cache miss)...")
        start_time = time.time()
        vector_store_cached.similarity_search(query, k=3)
        first_search_time = time.time() - start_time
        
        logger.info("Performing second search (cache hit)...")
        start_time = time.time()
        vector_store_cached.similarity_search(query, k=3)
        second_search_time = time.time() - start_time
        
        logger.info(f"First search time: {first_search_time:.4f} seconds")
        logger.info(f"Second search time: {second_search_time:.4f} seconds")
        logger.info(f"Speedup: {first_search_time / second_search_time:.2f}x")
        
        # Get cache statistics
        cache_stats = cached_embeddings.get_stats()
        logger.info(f"Cache statistics: {cache_stats}")
        
        # Section 3: Using Different Distance Strategies
        logger.info("\n=== Example 3: Using Different Distance Strategies ===")
        
        # Create vector store with cosine similarity
        logger.info("Creating vector store with Cosine similarity...")
        vector_store_cosine = HanaVectorStore(
            connection=connection,
            embedding=embeddings,
            schema_name=schema_name,
            table_name=f"{table_name}_COSINE",
            create_table=True,
            create_hnsw_index=True,
            distance_strategy=DistanceStrategy.COSINE,
        )
        
        # Create vector store with Euclidean distance
        logger.info("Creating vector store with Euclidean distance...")
        vector_store_euclidean = HanaVectorStore(
            connection=connection,
            embedding=embeddings,
            schema_name=schema_name,
            table_name=f"{table_name}_EUCLIDEAN",
            create_table=True,
            create_hnsw_index=True,
            distance_strategy=DistanceStrategy.EUCLIDEAN,
        )
        
        # Add documents to both vector stores
        logger.info("Adding documents to both vector stores...")
        vector_store_cosine.add_texts(texts, metadatas)
        vector_store_euclidean.add_texts(texts, metadatas)
        
        # Perform similarity search with both distance strategies
        logger.info("Performing search with Cosine similarity...")
        cosine_results = vector_store_cosine.similarity_search(query, k=3)
        
        logger.info("Performing search with Euclidean distance...")
        euclidean_results = vector_store_euclidean.similarity_search(query, k=3)
        
        logger.info("Cosine similarity results:")
        for i, doc in enumerate(cosine_results):
            logger.info(f"Result {i+1}: {doc.page_content[:100]}...")
            logger.info(f"Score: {doc.metadata.get('similarity_score', 'N/A')}")
        
        logger.info("\nEuclidean distance results:")
        for i, doc in enumerate(euclidean_results):
            logger.info(f"Result {i+1}: {doc.page_content[:100]}...")
            logger.info(f"Score: {doc.metadata.get('similarity_score', 'N/A')}")
        
        # Section 4: Using Internal Embeddings (if available)
        if args.internal_model:
            logger.info("\n=== Example 4: Using Internal Embeddings ===")
            
            try:
                # Initialize internal embeddings
                logger.info(f"Initializing internal embeddings with model {args.internal_model}...")
                internal_embeddings = HanaInternalEmbeddings(
                    model_id=args.internal_model,
                )
                
                # Create vector store with internal embeddings
                logger.info(f"Creating vector store with table {full_table_name}_INT...")
                vector_store_int = HanaVectorStore(
                    connection=connection,
                    embedding=internal_embeddings,
                    schema_name=schema_name,
                    table_name=f"{table_name}_INT",
                    create_table=True,
                    create_hnsw_index=True,
                )
                
                # Add documents
                logger.info(f"Adding {len(sample_docs)} documents to vector store...")
                start_time = time.time()
                vector_store_int.add_texts(texts, metadatas)
                elapsed_time = time.time() - start_time
                logger.info(f"Added documents in {elapsed_time:.2f} seconds")
                
                # Perform similarity search
                logger.info("Performing similarity search with internal embeddings...")
                start_time = time.time()
                results = vector_store_int.similarity_search(query, k=3)
                elapsed_time = time.time() - start_time
                
                logger.info(f"Search completed in {elapsed_time:.2f} seconds")
                logger.info(f"Query: \"{query}\"")
                for i, doc in enumerate(results):
                    logger.info(f"Result {i+1}: {doc.page_content[:100]}...")
                    logger.info(f"Metadata: {doc.metadata}\n")
                
            except Exception as e:
                logger.error(f"Error using internal embeddings: {str(e)}")
                logger.error("Internal embeddings might not be available in your SAP HANA Cloud instance")
        
        logger.info("\nSAP HANA Cloud LangChain integration example completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running example: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close connection
        if 'connection' in locals() and connection:
            connection.close()
            logger.info("Connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAP HANA Cloud LangChain Integration Example")
    
    # Connection parameters
    parser.add_argument("--host", required=True, help="SAP HANA host")
    parser.add_argument("--port", type=int, required=True, help="SAP HANA port")
    parser.add_argument("--user", required=True, help="SAP HANA username")
    parser.add_argument("--password", required=True, help="SAP HANA password")
    
    # Optional parameters
    parser.add_argument("--table", default="LANGCHAIN_EXAMPLE", help="Base table name for examples")
    parser.add_argument("--internal-model", help="SAP HANA internal embedding model ID (if available)")
    
    args = parser.parse_args()
    
    # Run the example
    run_example(args)