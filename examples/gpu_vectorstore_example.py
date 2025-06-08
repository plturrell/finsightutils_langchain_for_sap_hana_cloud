#!/usr/bin/env python
"""
GPU-Accelerated Vector Store Example

This example demonstrates how to use the GPU-accelerated vector store
for SAP HANA Cloud to achieve better performance for vector operations.

Usage:
    python gpu_vectorstore_example.py

Requirements:
    - SAP HANA Cloud instance with credentials
    - NVIDIA GPU with CUDA support (for GPU acceleration)
    - Python packages: langchain, langchain_hana, sentence-transformers
"""

import os
import time
import logging
import argparse
from typing import List, Dict, Any

from hdbcli import dbapi
from langchain_hana.gpu.hana_gpu_vectorstore import HanaGPUVectorStore
from langchain_hana.vectorstores import HanaDB
from langchain_hana.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_connection(host, port, user, password):
    """Create a connection to SAP HANA Cloud."""
    connection = dbapi.connect(
        address=host,
        port=port,
        user=user,
        password=password,
        encrypt=True,
        sslValidateCertificate=False,
    )
    return connection


def create_sample_data(num_docs=1000):
    """Create sample documents for testing."""
    documents = []
    metadata_list = []
    
    # Create sample documents
    for i in range(num_docs):
        category = "technology" if i % 3 == 0 else "business" if i % 3 == 1 else "science"
        priority = "high" if i % 5 == 0 else "medium" if i % 5 < 3 else "low"
        
        # Create document with different topics for diversity
        if i % 3 == 0:
            doc = f"Document {i}: This document discusses the latest advancements in artificial intelligence and machine learning technologies. It covers neural networks, deep learning, and applications in various industries."
        elif i % 3 == 1:
            doc = f"Document {i}: This document covers business strategies for digital transformation. It includes case studies on successful implementation of cloud computing, data analytics, and process automation."
        else:
            doc = f"Document {i}: This document explores scientific research in quantum computing. It discusses quantum bits, quantum gates, and potential applications in cryptography and optimization problems."
            
        documents.append(doc)
        
        # Create metadata
        metadata = {
            "id": str(i),
            "category": category,
            "priority": priority,
            "length": len(doc),
            "created_at": "2023-09-01"
        }
        metadata_list.append(metadata)
        
    return documents, metadata_list


def run_performance_comparison(connection, embedding_model, num_docs=1000):
    """
    Run performance comparison between CPU and GPU implementations.
    
    Args:
        connection: SAP HANA database connection
        embedding_model: Embedding model to use
        num_docs: Number of documents to use for testing
    """
    # Create sample data
    logger.info(f"Creating sample data with {num_docs} documents...")
    documents, metadata_list = create_sample_data(num_docs)
    
    # Create tables for CPU and GPU implementations
    cpu_table_name = "VECTORSTORE_CPU_TEST"
    gpu_table_name = "VECTORSTORE_GPU_TEST"
    
    # Initialize vectorstores
    logger.info("Initializing vector stores...")
    
    # Standard CPU-based implementation
    cpu_vectorstore = HanaDB(
        connection=connection,
        embedding=embedding_model,
        table_name=cpu_table_name,
        distance_strategy=DistanceStrategy.COSINE,
    )
    
    # GPU-accelerated implementation
    gpu_vectorstore = HanaGPUVectorStore(
        connection=connection,
        embedding=embedding_model,
        table_name=gpu_table_name,
        distance_strategy=DistanceStrategy.COSINE,
        gpu_acceleration_config={
            "use_gpu_batching": True,
            "embedding_batch_size": 32,
            "db_batch_size": 500,
            "build_index": True,
            "index_type": "hnsw",
            "prefetch_size": 10000,
        }
    )
    
    # Enable profiling for GPU vectorstore
    gpu_vectorstore.enable_profiling(True)
    
    # Test 1: Adding documents
    logger.info("\n=== Test 1: Adding Documents ===")
    
    # CPU implementation
    start_time = time.time()
    cpu_vectorstore.add_texts(documents, metadata_list)
    cpu_add_time = time.time() - start_time
    logger.info(f"CPU implementation - add_texts: {cpu_add_time:.2f} seconds")
    
    # GPU implementation
    start_time = time.time()
    gpu_vectorstore.add_texts(documents, metadata_list)
    gpu_add_time = time.time() - start_time
    logger.info(f"GPU implementation - add_texts: {gpu_add_time:.2f} seconds")
    logger.info(f"Speedup: {cpu_add_time / max(gpu_add_time, 0.001):.2f}x")
    
    # Test 2: Similarity search
    logger.info("\n=== Test 2: Similarity Search ===")
    query = "What are the latest advancements in artificial intelligence?"
    
    # CPU implementation
    start_time = time.time()
    cpu_results = cpu_vectorstore.similarity_search(query, k=5)
    cpu_search_time = time.time() - start_time
    logger.info(f"CPU implementation - similarity_search: {cpu_search_time:.2f} seconds")
    
    # GPU implementation
    start_time = time.time()
    gpu_results = gpu_vectorstore.similarity_search(query, k=5)
    gpu_search_time = time.time() - start_time
    logger.info(f"GPU implementation - similarity_search: {gpu_search_time:.2f} seconds")
    logger.info(f"Speedup: {cpu_search_time / max(gpu_search_time, 0.001):.2f}x")
    
    # Test 3: Filtered search
    logger.info("\n=== Test 3: Filtered Search ===")
    filter_query = {"category": "technology"}
    
    # CPU implementation
    start_time = time.time()
    cpu_results = cpu_vectorstore.similarity_search(query, k=5, filter=filter_query)
    cpu_filter_time = time.time() - start_time
    logger.info(f"CPU implementation - filtered search: {cpu_filter_time:.2f} seconds")
    
    # GPU implementation
    start_time = time.time()
    gpu_results = gpu_vectorstore.similarity_search(query, k=5, filter=filter_query)
    gpu_filter_time = time.time() - start_time
    logger.info(f"GPU implementation - filtered search: {gpu_filter_time:.2f} seconds")
    logger.info(f"Speedup: {cpu_filter_time / max(gpu_filter_time, 0.001):.2f}x")
    
    # Test 4: MMR search
    logger.info("\n=== Test 4: MMR Search ===")
    
    # CPU implementation
    start_time = time.time()
    cpu_results = cpu_vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20)
    cpu_mmr_time = time.time() - start_time
    logger.info(f"CPU implementation - MMR search: {cpu_mmr_time:.2f} seconds")
    
    # GPU implementation
    start_time = time.time()
    gpu_results = gpu_vectorstore.max_marginal_relevance_search(query, k=5, fetch_k=20)
    gpu_mmr_time = time.time() - start_time
    logger.info(f"GPU implementation - MMR search: {gpu_mmr_time:.2f} seconds")
    logger.info(f"Speedup: {cpu_mmr_time / max(gpu_mmr_time, 0.001):.2f}x")
    
    # Test 5: Upsert operation
    logger.info("\n=== Test 5: Upsert Operation ===")
    upsert_doc = "Updated document about artificial intelligence and machine learning applications in healthcare."
    upsert_metadata = {"category": "technology", "priority": "high"}
    
    # CPU implementation
    start_time = time.time()
    cpu_vectorstore.upsert_texts([upsert_doc], [upsert_metadata], filter={"category": "technology"})
    cpu_upsert_time = time.time() - start_time
    logger.info(f"CPU implementation - upsert: {cpu_upsert_time:.2f} seconds")
    
    # GPU implementation
    start_time = time.time()
    gpu_vectorstore.upsert_texts([upsert_doc], [upsert_metadata], filter={"category": "technology"})
    gpu_upsert_time = time.time() - start_time
    logger.info(f"GPU implementation - upsert: {gpu_upsert_time:.2f} seconds")
    logger.info(f"Speedup: {cpu_upsert_time / max(gpu_upsert_time, 0.001):.2f}x")
    
    # Print overall results
    logger.info("\n=== Overall Performance Summary ===")
    logger.info(f"CPU total time: {cpu_add_time + cpu_search_time + cpu_filter_time + cpu_mmr_time + cpu_upsert_time:.2f} seconds")
    logger.info(f"GPU total time: {gpu_add_time + gpu_search_time + gpu_filter_time + gpu_mmr_time + gpu_upsert_time:.2f} seconds")
    overall_speedup = (cpu_add_time + cpu_search_time + cpu_filter_time + cpu_mmr_time + cpu_upsert_time) / \
                      max(gpu_add_time + gpu_search_time + gpu_filter_time + gpu_mmr_time + gpu_upsert_time, 0.001)
    logger.info(f"Overall speedup: {overall_speedup:.2f}x")
    
    # Show GPU performance stats
    logger.info("\n=== GPU Performance Statistics ===")
    gpu_stats = gpu_vectorstore.get_performance_stats()
    for method, stats in gpu_stats.items():
        logger.info(f"{method}:")
        logger.info(f"  Count: {stats['count']}")
        logger.info(f"  Avg time: {stats['avg_time_ms']:.2f} ms")
        logger.info(f"  Min time: {stats['min_time_ms']:.2f} ms")
        logger.info(f"  Max time: {stats['max_time_ms']:.2f} ms")
    
    # Show GPU info
    logger.info("\n=== GPU Configuration ===")
    gpu_info = gpu_vectorstore.get_gpu_info()
    logger.info(f"GPU available: {gpu_info['gpu_available']}")
    logger.info(f"GPU IDs: {gpu_info.get('gpu_ids', [])}")
    if 'gpu_config' in gpu_info:
        for key, value in gpu_info['gpu_config'].items():
            logger.info(f"{key}: {value}")
    
    # Clean up tables
    try:
        cursor = connection.cursor()
        cursor.execute(f'DROP TABLE "{cpu_table_name}"')
        cursor.execute(f'DROP TABLE "{gpu_table_name}"')
        connection.commit()
        logger.info("Cleaned up test tables")
    except:
        logger.warning("Failed to clean up test tables")
    
    # Release resources
    gpu_vectorstore.release_resources()


def run_async_example(connection, embedding_model, num_docs=100):
    """
    Run an example of asynchronous operations with the GPU vectorstore.
    
    Args:
        connection: SAP HANA database connection
        embedding_model: Embedding model to use
        num_docs: Number of documents to use for testing
    """
    import asyncio
    
    # Create sample data
    logger.info(f"Creating sample data with {num_docs} documents...")
    documents, metadata_list = create_sample_data(num_docs)
    
    # Create table for async test
    async_table_name = "VECTORSTORE_ASYNC_TEST"
    
    # Initialize vectorstore
    logger.info("Initializing vector store...")
    gpu_vectorstore = HanaGPUVectorStore(
        connection=connection,
        embedding=embedding_model,
        table_name=async_table_name,
        distance_strategy=DistanceStrategy.COSINE,
        gpu_acceleration_config={
            "use_gpu_batching": True,
            "embedding_batch_size": 32,
            "db_batch_size": 500,
        }
    )
    
    # Enable profiling
    gpu_vectorstore.enable_profiling(True)
    
    async def run_async_operations():
        # Add documents asynchronously
        logger.info("Adding documents asynchronously...")
        await gpu_vectorstore.aadd_texts(documents, metadata_list)
        
        # Run multiple queries in parallel
        logger.info("Running multiple queries in parallel...")
        query1 = "What are the latest advancements in artificial intelligence?"
        query2 = "Discuss business strategies for digital transformation."
        query3 = "Explain quantum computing research."
        
        # Run queries concurrently
        results = await asyncio.gather(
            gpu_vectorstore.asimilarity_search(query1, k=3),
            gpu_vectorstore.asimilarity_search(query2, k=3),
            gpu_vectorstore.asimilarity_search(query3, k=3)
        )
        
        # Print results
        for i, (query, result) in enumerate(zip([query1, query2, query3], results)):
            logger.info(f"Query {i+1}: {query}")
            logger.info(f"Top result: {result[0].page_content[:100]}...\n")
        
        # Test async MMR search
        logger.info("Testing async MMR search...")
        mmr_results = await gpu_vectorstore.amax_marginal_relevance_search(
            query1, k=5, fetch_k=20, lambda_mult=0.7
        )
        
        logger.info("MMR results (diverse):")
        for i, doc in enumerate(mmr_results):
            logger.info(f"{i+1}. {doc.page_content[:100]}...")
            
        # Test async upsert
        logger.info("Testing async upsert...")
        upsert_doc = "Updated document about artificial intelligence and machine learning applications in healthcare."
        upsert_metadata = {"category": "technology", "priority": "high"}
        
        await gpu_vectorstore.aupsert_texts(
            [upsert_doc], [upsert_metadata], filter={"category": "technology"}
        )
        
        # Show performance stats
        logger.info("\n=== GPU Performance Statistics (Async) ===")
        gpu_stats = gpu_vectorstore.get_performance_stats()
        for method, stats in gpu_stats.items():
            if method.startswith('a'):  # Only show async methods
                logger.info(f"{method}:")
                logger.info(f"  Count: {stats['count']}")
                logger.info(f"  Avg time: {stats['avg_time_ms']:.2f} ms")
    
    # Run the async function
    asyncio.run(run_async_operations())
    
    # Clean up
    try:
        cursor = connection.cursor()
        cursor.execute(f'DROP TABLE "{async_table_name}"')
        connection.commit()
        logger.info("Cleaned up test table")
    except:
        logger.warning("Failed to clean up test table")
    
    # Release resources
    gpu_vectorstore.release_resources()


def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="GPU-Accelerated Vector Store Example")
    parser.add_argument("--host", type=str, help="SAP HANA host", default=os.environ.get("HANA_HOST"))
    parser.add_argument("--port", type=int, help="SAP HANA port", default=int(os.environ.get("HANA_PORT", "443")))
    parser.add_argument("--user", type=str, help="SAP HANA user", default=os.environ.get("HANA_USER"))
    parser.add_argument("--password", type=str, help="SAP HANA password", default=os.environ.get("HANA_PASSWORD"))
    parser.add_argument("--num-docs", type=int, help="Number of documents for testing", default=1000)
    parser.add_argument("--async-only", action="store_true", help="Run only the async example")
    args = parser.parse_args()
    
    # Check for required parameters
    if not (args.host and args.user and args.password):
        logger.error("Missing required parameters. Please provide host, user, and password.")
        parser.print_help()
        return
    
    # Connect to SAP HANA
    logger.info(f"Connecting to SAP HANA at {args.host}:{args.port}...")
    connection = get_connection(args.host, args.port, args.user, args.password)
    logger.info("Connected successfully.")
    
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    
    try:
        # Run async example if requested
        if args.async_only:
            run_async_example(connection, embedding_model, args.num_docs // 10)
        else:
            # Run performance comparison
            run_performance_comparison(connection, embedding_model, args.num_docs)
            
            # Also run async example with fewer documents
            logger.info("\n=== Running Async Example ===")
            run_async_example(connection, embedding_model, args.num_docs // 10)
    finally:
        # Close connection
        connection.close()
        logger.info("Connection closed.")


if __name__ == "__main__":
    main()