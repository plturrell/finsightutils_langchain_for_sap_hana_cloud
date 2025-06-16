#!/usr/bin/env python
"""
Production-Ready LangChain Integration for SAP HANA Cloud

This example demonstrates how to use the LangChain integration with SAP HANA Cloud
in a production environment, including:

1. Connection pooling for improved performance and reliability
2. Automatic retry logic for handling transient errors
3. Robust error handling
4. Advanced logging for monitoring and troubleshooting
5. Performance benchmarking

Prerequisites:
- SAP HANA Cloud instance with vector capabilities
- Python 3.8+
- Required packages: langchain, langchain_hana, sentence-transformers

Usage:
    python langchain_hana_production.py --host your-hana-host.hanacloud.ondemand.com --port 443 --user your-username --password your-password
"""

import argparse
import json
import logging
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from langchain_hana.vectorstore import HanaVectorStore
from langchain_hana.connection import (
    create_connection_pool,
    close_all_connection_pools,
    get_connection_pool
)
from langchain_hana.embeddings import HanaEmbeddingsCache
from langchain_hana.utils import DistanceStrategy

# Configure logging with detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "text": "SAP HANA Cloud is a cloud-based in-memory database that combines OLTP and OLAP workloads on a single platform, enabling real-time analytics on live transactional data.",
        "metadata": {"source": "SAP Documentation", "category": "database", "topic": "cloud", "year": 2023}
    },
    {
        "text": "LangChain is a framework for developing applications powered by language models. It provides tools and components for creating context-aware, reasoning applications using LLMs.",
        "metadata": {"source": "LangChain Documentation", "category": "framework", "topic": "ai", "year": 2023}
    },
    {
        "text": "Vector databases store and query data as high-dimensional vectors, enabling semantic search based on meaning rather than keywords. They're ideal for machine learning and AI applications.",
        "metadata": {"source": "Database Guide", "category": "database", "topic": "vector", "year": 2023}
    },
    {
        "text": "Retrieval-Augmented Generation (RAG) is a technique that enhances large language models by retrieving relevant information from external sources before generating responses.",
        "metadata": {"source": "AI Research Paper", "category": "technique", "topic": "ai", "year": 2022}
    },
    {
        "text": "SAP HANA Cloud offers vector capabilities that allow storing and searching high-dimensional vectors efficiently. This makes it suitable for AI applications like semantic search.",
        "metadata": {"source": "SAP Blog", "category": "database", "topic": "vector", "year": 2023}
    },
    {
        "text": "Connection pooling improves application performance by maintaining a pool of database connections that can be reused, reducing the overhead of creating new connections.",
        "metadata": {"source": "Database Best Practices", "category": "performance", "topic": "connection", "year": 2023}
    },
    {
        "text": "HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search that offers excellent performance for vector search operations.",
        "metadata": {"source": "Research Paper", "category": "algorithm", "topic": "search", "year": 2020}
    },
    {
        "text": "Embedding models convert text into numerical vector representations that capture semantic meaning, allowing machines to understand relationships between concepts.",
        "metadata": {"source": "ML Guide", "category": "machine learning", "topic": "embeddings", "year": 2022}
    },
    {
        "text": "Error handling and retry logic are essential components of production-ready applications, helping to handle transient errors and improve reliability.",
        "metadata": {"source": "Software Engineering Guide", "category": "reliability", "topic": "error handling", "year": 2023}
    },
    {
        "text": "Proper monitoring and logging are critical for production applications, providing visibility into system behavior and facilitating troubleshooting.",
        "metadata": {"source": "DevOps Handbook", "category": "operations", "topic": "monitoring", "year": 2023}
    }
]

# Sample queries for testing
SAMPLE_QUERIES = [
    "What is SAP HANA Cloud?",
    "How does vector search work?",
    "Explain connection pooling benefits",
    "What is HNSW algorithm?",
    "How do embedding models work?",
]


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Production-Ready LangChain for SAP HANA Cloud")
    
    # Connection parameters
    parser.add_argument("--host", help="SAP HANA Cloud host")
    parser.add_argument("--port", type=int, default=443, help="SAP HANA Cloud port (default: 443)")
    parser.add_argument("--user", help="SAP HANA Cloud username")
    parser.add_argument("--password", help="SAP HANA Cloud password")
    parser.add_argument("--config", help="Path to connection configuration file")
    
    # Connection pool parameters
    parser.add_argument("--pool-min", type=int, default=2, help="Minimum connections in pool (default: 2)")
    parser.add_argument("--pool-max", type=int, default=10, help="Maximum connections in pool (default: 10)")
    
    # Test parameters
    parser.add_argument("--table", default="LANGCHAIN_PRODUCTION", help="Table name for vector store")
    parser.add_argument("--test-concurrency", type=int, default=4, help="Number of concurrent test operations")
    parser.add_argument("--benchmark-time", type=int, default=30, help="Time in seconds to run benchmark")
    parser.add_argument("--cleanup", action="store_true", help="Clean up the test table after running")
    
    return parser.parse_args()


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
    
    logger.warning("No connection configuration file found. Using command line arguments.")
    return None


def setup_connection_pool(args):
    """
    Set up a connection pool for SAP HANA Cloud.
    
    Args:
        args: Command line arguments
        
    Returns:
        Name of the created connection pool
    """
    # Load connection parameters
    connection_params = {}
    
    if args.config:
        # Load from config file
        config = load_connection_config(args.config)
        if config:
            connection_params.update(config)
    
    # Override with command line arguments if provided
    if args.host:
        connection_params["host"] = args.host
    if args.port:
        connection_params["port"] = args.port
    if args.user:
        connection_params["user"] = args.user
    if args.password:
        connection_params["password"] = args.password
    
    # Check for required parameters
    required_params = ["host", "port", "user", "password"]
    missing_params = [param for param in required_params if param not in connection_params]
    if missing_params:
        raise ValueError(
            f"Missing required connection parameters: {', '.join(missing_params)}. "
            f"Please provide them as command line arguments or in a config file."
        )
    
    # Create the connection pool
    pool_name = "production_test"
    logger.info(f"Creating connection pool '{pool_name}' with {args.pool_min}-{args.pool_max} connections")
    
    pool = create_connection_pool(
        pool_name=pool_name,
        min_connections=args.pool_min,
        max_connections=args.pool_max,
        **connection_params
    )
    
    return pool_name


def setup_vector_store(pool_name: str, args):
    """
    Set up the vector store using connection pooling.
    
    Args:
        pool_name: Name of the connection pool to use
        args: Command line arguments
        
    Returns:
        HanaVectorStore instance
    """
    logger.info("Setting up embedding model")
    
    # Initialize embedding model with caching
    base_embeddings = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Create cached embeddings
    cached_embeddings = HanaEmbeddingsCache(
        base_embeddings=base_embeddings,
        ttl_seconds=3600,  # 1 hour cache lifetime
        max_size=1000,
        persist_path=f"{args.table}_cache.pkl"  # Persist cache to disk
    )
    
    logger.info(f"Setting up vector store with table: {args.table}")
    
    # Initialize the vector store with connection pooling
    vector_store = HanaVectorStore(
        embedding=cached_embeddings,
        use_connection_pool=True,
        connection_pool_name=pool_name,
        table_name=args.table,
        distance_strategy=DistanceStrategy.COSINE,
        create_table=True,
        create_hnsw_index=True,
        retry_attempts=3
    )
    
    return vector_store


def add_sample_documents(vector_store):
    """
    Add sample documents to the vector store.
    
    Args:
        vector_store: HanaVectorStore instance
        
    Returns:
        Number of documents added
    """
    logger.info("Adding sample documents to vector store")
    
    start_time = time.time()
    
    # Add documents to the vector store
    texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
    metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]
    
    try:
        vector_store.add_texts(texts, metadatas)
        
        elapsed_time = time.time() - start_time
        docs_per_second = len(texts) / elapsed_time
        
        logger.info(f"Added {len(texts)} documents in {elapsed_time:.2f}s ({docs_per_second:.2f} docs/s)")
        return len(texts)
    except Exception as e:
        logger.error(f"Error adding documents: {str(e)}")
        raise


def run_concurrent_searches(vector_store, num_workers: int, duration: int):
    """
    Run concurrent similarity searches to test performance and reliability.
    
    Args:
        vector_store: HanaVectorStore instance
        num_workers: Number of concurrent workers
        duration: Duration in seconds to run the test
        
    Returns:
        Dictionary with performance metrics
    """
    logger.info(f"Running concurrent searches with {num_workers} workers for {duration} seconds")
    
    # Define worker function
    def search_worker():
        query = random.choice(SAMPLE_QUERIES)
        filter_dict = None
        
        # Randomly apply a filter in 30% of searches
        if random.random() < 0.3:
            # Choose a random filter type
            filter_type = random.choice(["simple", "year", "category", "complex"])
            
            if filter_type == "simple":
                filter_dict = {"topic": random.choice(["vector", "ai", "cloud"])}
            elif filter_type == "year":
                filter_dict = {"year": {"$gte": 2022}}
            elif filter_type == "category":
                filter_dict = {"category": random.choice(["database", "machine learning", "performance"])}
            elif filter_type == "complex":
                filter_dict = {
                    "category": "database",
                    "year": {"$gte": 2022}
                }
        
        try:
            start_time = time.time()
            
            if random.random() < 0.2:
                # Use MMR search in 20% of cases
                results = vector_store.max_marginal_relevance_search(
                    query,
                    k=3,
                    fetch_k=5,
                    lambda_mult=0.5,
                    filter=filter_dict
                )
                search_type = "mmr"
            else:
                # Use regular search in 80% of cases
                results = vector_store.similarity_search(
                    query,
                    k=3,
                    filter=filter_dict
                )
                search_type = "standard"
            
            elapsed_time = time.time() - start_time
            return {
                "success": True,
                "query": query,
                "filter": filter_dict,
                "results_count": len(results),
                "elapsed_time": elapsed_time,
                "search_type": search_type
            }
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "filter": filter_dict,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    # Set up metrics collection
    results = []
    start_time = time.time()
    end_time = start_time + duration
    
    # Run concurrent searches
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        
        # Keep submitting tasks until time is up
        while time.time() < end_time:
            if len(futures) < num_workers:
                futures.append(executor.submit(search_worker))
            
            # Process completed futures
            for future in [f for f in futures if f.done()]:
                results.append(future.result())
                futures.remove(future)
            
            # Brief pause to avoid consuming too many resources
            time.sleep(0.01)
    
    # Wait for any remaining futures
    for future in futures:
        results.append(future.result())
    
    # Calculate metrics
    successful_searches = [r for r in results if r["success"]]
    failed_searches = [r for r in results if not r["success"]]
    
    if successful_searches:
        avg_time = sum(r["elapsed_time"] for r in successful_searches) / len(successful_searches)
        min_time = min(r["elapsed_time"] for r in successful_searches)
        max_time = max(r["elapsed_time"] for r in successful_searches)
        
        # Calculate percentiles
        times = sorted(r["elapsed_time"] for r in successful_searches)
        p50 = times[len(times) // 2]
        p95 = times[int(len(times) * 0.95)]
        p99 = times[int(len(times) * 0.99)]
        
        # Count by search type
        standard_searches = len([r for r in successful_searches if r["search_type"] == "standard"])
        mmr_searches = len([r for r in successful_searches if r["search_type"] == "mmr"])
        
        # Count by filter
        filtered_searches = len([r for r in successful_searches if r["filter"] is not None])
    else:
        avg_time = min_time = max_time = p50 = p95 = p99 = 0
        standard_searches = mmr_searches = filtered_searches = 0
    
    metrics = {
        "total_searches": len(results),
        "successful_searches": len(successful_searches),
        "failed_searches": len(failed_searches),
        "success_rate": len(successful_searches) / len(results) if results else 0,
        "searches_per_second": len(results) / duration,
        "avg_time": avg_time if successful_searches else 0,
        "min_time": min_time if successful_searches else 0,
        "max_time": max_time if successful_searches else 0,
        "p50_time": p50 if successful_searches else 0,
        "p95_time": p95 if successful_searches else 0,
        "p99_time": p99 if successful_searches else 0,
        "standard_searches": standard_searches,
        "mmr_searches": mmr_searches,
        "filtered_searches": filtered_searches,
    }
    
    return metrics


def print_performance_summary(metrics):
    """Print a summary of performance metrics."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print(f"\nSearch Operations:")
    print(f"  Total searches:       {metrics['total_searches']}")
    print(f"  Successful searches:  {metrics['successful_searches']}")
    print(f"  Failed searches:      {metrics['failed_searches']}")
    print(f"  Success rate:         {metrics['success_rate'] * 100:.2f}%")
    print(f"  Searches per second:  {metrics['searches_per_second']:.2f}")
    
    print(f"\nSearch Types:")
    print(f"  Standard searches:    {metrics['standard_searches']} ({metrics['standard_searches'] / metrics['successful_searches'] * 100:.1f}% of successful)")
    print(f"  MMR searches:         {metrics['mmr_searches']} ({metrics['mmr_searches'] / metrics['successful_searches'] * 100:.1f}% of successful)")
    print(f"  Filtered searches:    {metrics['filtered_searches']} ({metrics['filtered_searches'] / metrics['successful_searches'] * 100:.1f}% of successful)")
    
    print(f"\nLatency (seconds):")
    print(f"  Average:              {metrics['avg_time']:.6f}")
    print(f"  Minimum:              {metrics['min_time']:.6f}")
    print(f"  Maximum:              {metrics['max_time']:.6f}")
    print(f"  Median (P50):         {metrics['p50_time']:.6f}")
    print(f"  95th percentile:      {metrics['p95_time']:.6f}")
    print(f"  99th percentile:      {metrics['p99_time']:.6f}")
    
    print("\n" + "=" * 80)


def check_embedding_cache(vector_store):
    """
    Check the embedding cache statistics.
    
    Args:
        vector_store: HanaVectorStore instance
    """
    if hasattr(vector_store.embedding, 'get_stats'):
        stats = vector_store.embedding.get_stats()
        
        print("\n" + "-" * 80)
        print("EMBEDDING CACHE STATISTICS")
        print("-" * 80)
        
        print(f"Query cache size:      {stats['query_cache_size']}")
        print(f"Document cache size:   {stats['document_cache_size']}")
        print(f"Max cache size:        {stats['max_size']}")
        
        print(f"\nQuery cache hits:      {stats['query_hits']}")
        print(f"Query cache misses:    {stats['query_misses']}")
        print(f"Query hit rate:        {stats['query_hit_rate'] * 100:.2f}%")
        
        print(f"\nDocument cache hits:   {stats['document_hits']}")
        print(f"Document cache misses: {stats['document_misses']}")
        print(f"Document hit rate:     {stats['document_hit_rate'] * 100:.2f}%")
        
        print(f"\nPersistence enabled:   {stats['persistence_enabled']}")
        print("-" * 80)


def cleanup(vector_store):
    """
    Clean up resources after testing.
    
    Args:
        vector_store: HanaVectorStore instance
    """
    logger.info("Cleaning up...")
    
    try:
        # Delete all documents
        vector_store.delete(filter={})
        logger.info("Deleted all documents from the vector store")
        
        # Clear embedding cache
        if hasattr(vector_store.embedding, 'clear_cache'):
            vector_store.embedding.clear_cache()
            logger.info("Cleared embedding cache")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


def run_production_test(args):
    """
    Run a comprehensive production test of the SAP HANA vector store.
    
    Args:
        args: Command line arguments
        
    Returns:
        0 if successful, 1 otherwise
    """
    try:
        # Set up connection pool
        pool_name = setup_connection_pool(args)
        
        # Set up vector store
        vector_store = setup_vector_store(pool_name, args)
        
        # Add sample documents
        num_docs = add_sample_documents(vector_store)
        
        # Run concurrent searches to test performance and reliability
        metrics = run_concurrent_searches(
            vector_store=vector_store,
            num_workers=args.test_concurrency,
            duration=args.benchmark_time
        )
        
        # Print performance summary
        print_performance_summary(metrics)
        
        # Check embedding cache statistics
        check_embedding_cache(vector_store)
        
        # Clean up if requested
        if args.cleanup:
            cleanup(vector_store)
        
        # Close all connection pools
        close_all_connection_pools()
        
        logger.info("Production test completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error in production test: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Clean up connection pools
        close_all_connection_pools()
        
        return 1


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the production test
    return run_production_test(args)


if __name__ == "__main__":
    exit(main())