#!/usr/bin/env python
"""
LangChain Integration for SAP HANA Cloud with GPU Acceleration - Quickstart Example

This example demonstrates how to use GPU-accelerated embeddings with SAP HANA Cloud for 
vector search capabilities. It shows how to:

1. Connect to SAP HANA Cloud
2. Initialize GPU-accelerated embedding generation
3. Create a vector store with TensorRT optimized embeddings
4. Perform similarity search with performance metrics

Prerequisites:
- SAP HANA Cloud instance with vector capabilities
- Python 3.8+
- NVIDIA GPU with CUDA support
- Required packages: langchain, langchain_hana, sentence-transformers, torch, tensorrt

Usage:
    python langchain_hana_gpu_quickstart.py --host your-hana-host.hanacloud.ondemand.com --port 443 --user your-username --password your-password
"""

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Any

from langchain_hana.vectorstores import HanaDB
from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings
from langchain_hana.gpu.hana_tensorrt_vectorstore import HanaTensorRTVectorStore
from langchain_hana.connection import create_connection, test_connection
from langchain_hana.utils import DistanceStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "text": "SAP HANA Cloud is a cloud-based in-memory database that provides fast data processing and analytics capabilities.",
        "metadata": {"source": "SAP Documentation", "category": "database", "topic": "cloud"}
    },
    {
        "text": "GPU acceleration can significantly improve the performance of embedding generation and vector similarity search.",
        "metadata": {"source": "Technical Guide", "category": "hardware", "topic": "gpu"}
    },
    {
        "text": "TensorRT is a high-performance deep learning inference optimizer and runtime that delivers low latency and high throughput.",
        "metadata": {"source": "NVIDIA Documentation", "category": "software", "topic": "optimization"}
    },
    {
        "text": "Vector databases store and query data as high-dimensional vectors, enabling semantic search based on meaning rather than keywords.",
        "metadata": {"source": "Database Guide", "category": "database", "topic": "vector"}
    },
    {
        "text": "SAP HANA Cloud offers vector capabilities that allow storing and searching high-dimensional vectors efficiently.",
        "metadata": {"source": "SAP Blog", "category": "database", "topic": "vector"}
    },
    {
        "text": "Batched embedding generation can improve throughput by processing multiple inputs simultaneously on the GPU.",
        "metadata": {"source": "Performance Guide", "category": "optimization", "topic": "batch"}
    },
    {
        "text": "HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest neighbor search that offers excellent performance.",
        "metadata": {"source": "Research Paper", "category": "algorithm", "topic": "search"}
    },
    {
        "text": "Mixed precision computing uses both 16-bit and 32-bit floating-point types to reduce memory usage and increase processing speed.",
        "metadata": {"source": "NVIDIA Blog", "category": "optimization", "topic": "precision"}
    },
    {
        "text": "LangChain is a framework for developing applications powered by language models that provides tools for creating context-aware applications.",
        "metadata": {"source": "LangChain Documentation", "category": "framework", "topic": "ai"}
    },
    {
        "text": "Enterprise-grade vector search requires high performance, scalability, security, and integration with existing systems.",
        "metadata": {"source": "Enterprise Guide", "category": "solution", "topic": "enterprise"}
    }
]

# Sample queries for testing
SAMPLE_QUERIES = [
    "How does SAP HANA Cloud support vector search?",
    "What are the benefits of GPU acceleration for embeddings?",
    "Explain TensorRT optimization for deep learning models",
    "How does HNSW improve vector search performance?",
    "What is mixed precision computing?",
]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LangChain GPU-Accelerated Example with SAP HANA Cloud")
    
    # Connection parameters
    parser.add_argument("--host", help="SAP HANA Cloud host")
    parser.add_argument("--port", type=int, default=443, help="SAP HANA Cloud port (default: 443)")
    parser.add_argument("--user", help="SAP HANA Cloud username")
    parser.add_argument("--password", help="SAP HANA Cloud password")
    parser.add_argument("--config", help="Path to connection configuration file")
    
    # Vector store parameters
    parser.add_argument("--table", default="GPU_QUICKSTART", help="Table name for vector store")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="Model name for embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for embedding generation")
    parser.add_argument("--precision", choices=["float32", "float16"], default="float16", 
                        help="Precision for TensorRT optimization")
    
    return parser.parse_args()

def setup_connection(args):
    """Set up connection to SAP HANA Cloud."""
    logger.info("Setting up connection to SAP HANA Cloud...")
    
    # Load connection configuration from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
            connection = create_connection(**config)
    # Use command line arguments if provided
    elif args.host and args.user and args.password:
        connection = create_connection(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            encrypt=True,
            sslValidateCertificate=False
        )
    # Try to use environment variables
    else:
        connection = create_connection()
    
    # Test the connection
    valid, info = test_connection(connection)
    if not valid:
        raise ConnectionError(f"Failed to connect to SAP HANA Cloud: {info.get('error', 'Unknown error')}")
    
    logger.info(f"Connected to SAP HANA Cloud {info.get('version', 'Unknown version')}")
    logger.info(f"Current schema: {info.get('current_schema', 'Unknown schema')}")
    
    return connection

def setup_embeddings(args):
    """Set up GPU-accelerated embeddings."""
    logger.info(f"Setting up GPU-accelerated embeddings with model: {args.model}")
    
    # Create TensorRT optimized embeddings
    embeddings = HanaTensorRTEmbeddings(
        model_name=args.model,
        batch_size=args.batch_size,
        half_precision=(args.precision == "float16"),
        device="cuda"  # Use GPU
    )
    
    return embeddings

def setup_vector_store(connection, embeddings, args):
    """Set up the SAP HANA Cloud Vector Store with GPU acceleration."""
    logger.info(f"Setting up vector store: {args.table}")
    
    # Initialize the vector store
    vector_store = HanaTensorRTVectorStore(
        connection=connection,
        embedding=embeddings,
        table_name=args.table,
        distance_strategy=DistanceStrategy.COSINE,
        create_table=True,  # Create the table if it doesn't exist
    )
    
    # Create HNSW index for faster searches
    try:
        vector_store.create_hnsw_index()
        logger.info("HNSW index created successfully")
    except Exception as e:
        logger.warning(f"Could not create HNSW index: {str(e)}")
    
    return vector_store

def add_sample_documents(vector_store):
    """Add sample documents to the vector store."""
    logger.info("Adding sample documents to vector store...")
    
    start_time = time.time()
    
    # Add documents to the vector store
    texts = [doc["text"] for doc in SAMPLE_DOCUMENTS]
    metadatas = [doc["metadata"] for doc in SAMPLE_DOCUMENTS]
    
    vector_store.add_texts(texts, metadatas)
    
    elapsed_time = time.time() - start_time
    docs_per_second = len(texts) / elapsed_time
    
    logger.info(f"Added {len(texts)} documents in {elapsed_time:.2f}s ({docs_per_second:.2f} docs/s)")
    
    return len(texts)

def run_similarity_searches(vector_store):
    """Run sample similarity searches and measure performance."""
    logger.info("Running similarity searches...")
    
    total_time = 0
    total_queries = len(SAMPLE_QUERIES)
    
    for i, query in enumerate(SAMPLE_QUERIES):
        logger.info(f"Query {i+1}/{total_queries}: {query}")
        
        # Measure query time
        start_time = time.time()
        results = vector_store.similarity_search(query, k=3)
        query_time = time.time() - start_time
        total_time += query_time
        
        # Display results
        logger.info(f"  Found {len(results)} results in {query_time:.4f}s")
        for j, doc in enumerate(results):
            logger.info(f"  Result {j+1}: {doc.page_content[:100]}... | {doc.metadata}")
        
        logger.info("")
    
    # Calculate and display performance metrics
    avg_query_time = total_time / total_queries
    queries_per_second = total_queries / total_time
    
    logger.info(f"Performance Summary:")
    logger.info(f"  Total queries: {total_queries}")
    logger.info(f"  Average query time: {avg_query_time:.4f}s")
    logger.info(f"  Queries per second: {queries_per_second:.2f}")
    
    return {
        "total_queries": total_queries,
        "total_time": total_time,
        "avg_query_time": avg_query_time,
        "queries_per_second": queries_per_second
    }

def run_mmr_search(vector_store):
    """Run a Maximal Marginal Relevance search for diverse results."""
    logger.info("Running Maximal Marginal Relevance search for diverse results...")
    
    query = "What are the benefits of GPU acceleration for vector databases?"
    
    # Run MMR search
    start_time = time.time()
    results = vector_store.max_marginal_relevance_search(
        query,
        k=3,            # Number of results to return
        fetch_k=10,     # Number of results to fetch before applying MMR
        lambda_mult=0.5  # Diversity factor (0 = max diversity, 1 = max relevance)
    )
    query_time = time.time() - start_time
    
    logger.info(f"MMR search found {len(results)} diverse results in {query_time:.4f}s")
    for i, doc in enumerate(results):
        logger.info(f"  Result {i+1}: {doc.page_content[:100]}... | {doc.metadata}")
    
    logger.info("")

def run_filtered_search(vector_store):
    """Run a filtered similarity search."""
    logger.info("Running filtered similarity search...")
    
    query = "How can we optimize performance?"
    filter_dict = {"category": "optimization"}
    
    # Run filtered search
    start_time = time.time()
    results = vector_store.similarity_search(
        query,
        k=3,
        filter=filter_dict
    )
    query_time = time.time() - start_time
    
    logger.info(f"Filtered search with filter {filter_dict} found {len(results)} results in {query_time:.4f}s")
    for i, doc in enumerate(results):
        logger.info(f"  Result {i+1}: {doc.page_content[:100]}... | {doc.metadata}")
    
    logger.info("")

def main():
    """Main function to run the example."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Connect to SAP HANA Cloud
        connection = setup_connection(args)
        
        # Set up GPU-accelerated embeddings
        embeddings = setup_embeddings(args)
        
        # Set up vector store
        vector_store = setup_vector_store(connection, embeddings, args)
        
        # Add sample documents
        num_docs = add_sample_documents(vector_store)
        
        # Run similarity searches
        search_metrics = run_similarity_searches(vector_store)
        
        # Run MMR search
        run_mmr_search(vector_store)
        
        # Run filtered search
        run_filtered_search(vector_store)
        
        # Print summary
        logger.info("GPU-Accelerated SAP HANA Cloud Example Summary:")
        logger.info(f"  Documents indexed: {num_docs}")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Precision: {args.precision}")
        logger.info(f"  Average query time: {search_metrics['avg_query_time']:.4f}s")
        logger.info(f"  Queries per second: {search_metrics['queries_per_second']:.2f}")
        
        logger.info("Example completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())