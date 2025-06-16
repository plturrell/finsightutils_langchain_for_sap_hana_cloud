#!/usr/bin/env python
"""
Benchmark for Apache Arrow Flight integration.

This script compares the performance of vector operations with and without
Apache Arrow Flight integration for the SAP HANA Cloud LangChain integration.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("arrow_flight_benchmark")

# Check for required dependencies
try:
    import pyarrow as pa
    import pyarrow.flight as flight
    HAS_ARROW_FLIGHT = True
except ImportError:
    HAS_ARROW_FLIGHT = False
    logger.warning(
        "The pyarrow and pyarrow.flight packages are required for Arrow Flight benchmarking. "
        "Install them with 'pip install pyarrow pyarrow.flight'."
    )

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    logger.warning(
        "LangChain is required for embedding generation. "
        "Install it with 'pip install langchain langchain-community'."
    )

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    logger.warning(
        "The sentence_transformers package is required for embedding generation. "
        "Install it with 'pip install sentence-transformers'."
    )

try:
    from hdbcli import dbapi
    HAS_HDBCLI = True
except ImportError:
    HAS_HDBCLI = False
    logger.warning(
        "The hdbcli package is required for SAP HANA connectivity. "
        "Install it with 'pip install hdbcli'."
    )

# Check if the Arrow Flight implementation is available
try:
    from langchain_hana.gpu import (
        ArrowFlightClient,
        HanaArrowFlightVectorStore,
        ArrowFlightMultiGPUManager,
        HAS_ARROW_FLIGHT as LIB_HAS_ARROW_FLIGHT
    )
    from langchain_hana.gpu.vector_serialization import (
        serialize_vector,
        deserialize_vector,
        vectors_to_arrow_batch,
        arrow_batch_to_vectors,
        serialize_arrow_batch,
        deserialize_arrow_batch
    )
    from langchain_hana.vectorstores import HanaDB
    HAS_LANGCHAIN_HANA = True
except ImportError:
    HAS_LANGCHAIN_HANA = False
    logger.warning(
        "The langchain_hana package is required for this benchmark. "
        "Make sure the package is installed."
    )


def check_prerequisites():
    """Check if all prerequisites are met."""
    if not HAS_ARROW_FLIGHT:
        logger.error("PyArrow and PyArrow.Flight are required for this benchmark.")
        return False
        
    if not HAS_LANGCHAIN:
        logger.error("LangChain is required for this benchmark.")
        return False
        
    if not HAS_SENTENCE_TRANSFORMERS:
        logger.error("SentenceTransformer is required for embedding generation.")
        return False
        
    if not HAS_HDBCLI:
        logger.error("The hdbcli package is required for SAP HANA connectivity.")
        return False
        
    if not HAS_LANGCHAIN_HANA:
        logger.error("The langchain_hana package is required for this benchmark.")
        return False
        
    return True


def get_connection(
    host: str,
    port: int,
    user: str,
    password: str
) -> dbapi.Connection:
    """
    Get a connection to the SAP HANA database.
    
    Args:
        host: SAP HANA host address
        port: SAP HANA port
        user: SAP HANA username
        password: SAP HANA password
        
    Returns:
        Database connection
    """
    try:
        connection = dbapi.connect(
            address=host,
            port=port,
            user=user,
            password=password
        )
        logger.info(f"Connected to SAP HANA at {host}:{port}")
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to SAP HANA: {str(e)}")
        raise


def load_or_generate_test_data(
    num_documents: int = 1000,
    embedding_dim: int = 768,
    file_path: Optional[str] = None
) -> Tuple[List[str], List[Dict[str, Any]], List[List[float]]]:
    """
    Load or generate test data for benchmarking.
    
    Args:
        num_documents: Number of documents to generate
        embedding_dim: Dimension of embedding vectors
        file_path: Optional path to JSON file with test data
        
    Returns:
        Tuple of (texts, metadatas, vectors)
    """
    if file_path and os.path.exists(file_path):
        # Load test data from file
        logger.info(f"Loading test data from {file_path}")
        with open(file_path, "r") as f:
            data = json.load(f)
            
        texts = data.get("texts", [])
        metadatas = data.get("metadatas", [])
        vectors = data.get("vectors", [])
        
        # Validate loaded data
        if len(texts) != num_documents or len(metadatas) != num_documents or len(vectors) != num_documents:
            logger.warning(
                f"Loaded data doesn't match requested size. "
                f"Generating {num_documents} documents instead."
            )
            return load_or_generate_test_data(num_documents, embedding_dim, None)
            
        return texts, metadatas, vectors
    
    # Generate test data
    logger.info(f"Generating {num_documents} test documents")
    
    # Generate texts
    texts = [
        f"This is test document {i} for benchmarking Arrow Flight integration."
        for i in range(num_documents)
    ]
    
    # Generate metadatas
    metadatas = [
        {
            "id": str(i),
            "source": "generated",
            "category": f"category_{i % 5}",
            "priority": random.choice(["high", "medium", "low"])
        }
        for i in range(num_documents)
    ]
    
    # Generate random vectors
    vectors = []
    for _ in range(num_documents):
        vector = np.random.randn(embedding_dim).astype(np.float32)
        vector = vector / np.linalg.norm(vector)  # Normalize
        vectors.append(vector.tolist())
    
    # Save generated data if file_path is provided
    if file_path:
        logger.info(f"Saving generated test data to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(
                {
                    "texts": texts,
                    "metadatas": metadatas,
                    "vectors": vectors
                },
                f
            )
    
    return texts, metadatas, vectors


def benchmark_traditional_approach(
    connection: dbapi.Connection,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    vectors: List[List[float]],
    table_name: str = "BENCHMARK_TRADITIONAL",
    batch_size: int = 100,
    num_queries: int = 10,
    k: int = 10
) -> Dict[str, Any]:
    """
    Benchmark the traditional approach using HanaDB.
    
    Args:
        connection: Database connection
        texts: List of text documents
        metadatas: List of metadata dictionaries
        vectors: List of embedding vectors
        table_name: Table name for the benchmark
        batch_size: Batch size for operations
        num_queries: Number of queries to perform
        k: Number of results to return for each query
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking traditional approach with {len(texts)} documents")
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create a new table for benchmarking
    vectorstore = HanaDB(
        connection=connection,
        embedding=embeddings,
        table_name=table_name,
        pre_delete_collection=True  # Start with a fresh table
    )
    
    # Measure insertion time
    logger.info(f"Inserting {len(texts)} documents in batches of {batch_size}")
    insert_times = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size] if metadatas else None
        
        # Custom embedding function to use provided vectors
        def get_embeddings(texts):
            return vectors[i:i+len(texts)]
        
        # Use the custom embedding function
        original_embed_documents = embeddings.embed_documents
        embeddings.embed_documents = get_embeddings
        
        # Measure insertion time
        start_time = time.time()
        vectorstore.add_texts(batch_texts, batch_metadatas)
        end_time = time.time()
        
        # Restore original embedding function
        embeddings.embed_documents = original_embed_documents
        
        insert_times.append(end_time - start_time)
    
    total_insert_time = sum(insert_times)
    avg_insert_time = total_insert_time / len(insert_times)
    docs_per_second = len(texts) / total_insert_time
    
    # Measure query time
    logger.info(f"Performing {num_queries} queries")
    query_times = []
    
    # Generate random query vectors
    query_vectors = vectors[:num_queries]
    
    for i in range(num_queries):
        # Custom embedding function to use provided vectors
        def get_query_embedding(text):
            return query_vectors[i]
        
        # Use the custom embedding function
        original_embed_query = embeddings.embed_query
        embeddings.embed_query = get_query_embedding
        
        # Measure query time
        start_time = time.time()
        results = vectorstore.similarity_search("dummy query", k=k)
        end_time = time.time()
        
        # Restore original embedding function
        embeddings.embed_query = original_embed_query
        
        query_times.append(end_time - start_time)
    
    avg_query_time = sum(query_times) / len(query_times)
    queries_per_second = num_queries / sum(query_times)
    
    # Return benchmark results
    return {
        "approach": "traditional",
        "table_name": table_name,
        "num_documents": len(texts),
        "batch_size": batch_size,
        "num_queries": num_queries,
        "k": k,
        "total_insert_time": total_insert_time,
        "avg_insert_time": avg_insert_time,
        "docs_per_second": docs_per_second,
        "avg_query_time": avg_query_time,
        "queries_per_second": queries_per_second,
        "insert_times": insert_times,
        "query_times": query_times
    }


def benchmark_arrow_flight_approach(
    host: str,
    port: int,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    vectors: List[List[float]],
    table_name: str = "BENCHMARK_ARROW_FLIGHT",
    batch_size: int = 100,
    num_queries: int = 10,
    k: int = 10,
    flight_port: int = 8815
) -> Dict[str, Any]:
    """
    Benchmark the Arrow Flight approach.
    
    Args:
        host: SAP HANA host address
        port: SAP HANA port
        texts: List of text documents
        metadatas: List of metadata dictionaries
        vectors: List of embedding vectors
        table_name: Table name for the benchmark
        batch_size: Batch size for operations
        num_queries: Number of queries to perform
        k: Number of results to return for each query
        flight_port: Arrow Flight server port
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking Arrow Flight approach with {len(texts)} documents")
    
    # Initialize Arrow Flight client
    client = ArrowFlightClient(
        host=host,
        port=flight_port
    )
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create a new table for benchmarking
    vectorstore = HanaArrowFlightVectorStore(
        embedding=embeddings,
        host=host,
        port=flight_port,
        table_name=table_name,
        pre_delete_collection=True  # Start with a fresh table
    )
    
    # Measure insertion time using Arrow Flight
    logger.info(f"Inserting {len(texts)} documents in batches of {batch_size}")
    insert_times = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_metadatas = metadatas[i:i+batch_size] if metadatas else None
        batch_vectors = vectors[i:i+batch_size]
        
        # Measure insertion time
        start_time = time.time()
        client.upload_vectors(
            table_name=table_name,
            vectors=batch_vectors,
            texts=batch_texts,
            metadata=batch_metadatas,
            batch_size=batch_size
        )
        end_time = time.time()
        
        insert_times.append(end_time - start_time)
    
    total_insert_time = sum(insert_times)
    avg_insert_time = total_insert_time / len(insert_times)
    docs_per_second = len(texts) / total_insert_time
    
    # Measure query time
    logger.info(f"Performing {num_queries} queries")
    query_times = []
    
    # Generate random query vectors
    query_vectors = vectors[:num_queries]
    
    for i in range(num_queries):
        query_vector = query_vectors[i]
        
        # Measure query time
        start_time = time.time()
        results = client.similarity_search(
            table_name=table_name,
            query_vector=query_vector,
            k=k,
            include_metadata=True
        )
        end_time = time.time()
        
        query_times.append(end_time - start_time)
    
    avg_query_time = sum(query_times) / len(query_times)
    queries_per_second = num_queries / sum(query_times)
    
    # Close client
    client.close()
    
    # Return benchmark results
    return {
        "approach": "arrow_flight",
        "table_name": table_name,
        "num_documents": len(texts),
        "batch_size": batch_size,
        "num_queries": num_queries,
        "k": k,
        "total_insert_time": total_insert_time,
        "avg_insert_time": avg_insert_time,
        "docs_per_second": docs_per_second,
        "avg_query_time": avg_query_time,
        "queries_per_second": queries_per_second,
        "insert_times": insert_times,
        "query_times": query_times
    }


def benchmark_arrow_flight_multi_gpu_approach(
    host: str,
    port: int,
    texts: List[str],
    metadatas: List[Dict[str, Any]],
    vectors: List[List[float]],
    table_name: str = "BENCHMARK_ARROW_FLIGHT_MULTI_GPU",
    batch_size: int = 100,
    num_queries: int = 10,
    k: int = 10,
    flight_port: int = 8815,
    gpu_ids: Optional[List[int]] = None,
    distribution_strategy: str = "round_robin"
) -> Dict[str, Any]:
    """
    Benchmark the Arrow Flight approach with multi-GPU support.
    
    Args:
        host: SAP HANA host address
        port: SAP HANA port
        texts: List of text documents
        metadatas: List of metadata dictionaries
        vectors: List of embedding vectors
        table_name: Table name for the benchmark
        batch_size: Batch size for operations
        num_queries: Number of queries to perform
        k: Number of results to return for each query
        flight_port: Arrow Flight server port
        gpu_ids: List of GPU device IDs to use
        distribution_strategy: Strategy for distributing work across GPUs
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking Arrow Flight multi-GPU approach with {len(texts)} documents")
    
    # Initialize Arrow Flight clients for each GPU
    flight_clients = [
        ArrowFlightClient(
            host=host,
            port=flight_port
        )
        for _ in range(len(gpu_ids) if gpu_ids else 1)
    ]
    
    # Initialize multi-GPU manager
    mgpu_manager = ArrowFlightMultiGPUManager(
        flight_clients=flight_clients,
        gpu_ids=gpu_ids,
        batch_size=batch_size,
        distribution_strategy=distribution_strategy
    )
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create a new table for benchmarking
    client = flight_clients[0]
    vectorstore = HanaArrowFlightVectorStore(
        embedding=embeddings,
        host=host,
        port=flight_port,
        table_name=table_name,
        pre_delete_collection=True  # Start with a fresh table
    )
    
    # Measure insertion time using multi-GPU Arrow Flight
    logger.info(f"Inserting {len(texts)} documents using multi-GPU approach")
    
    # Measure total insertion time
    start_time = time.time()
    ids = mgpu_manager.upload_vectors_multi_gpu(
        table_name=table_name,
        vectors=vectors,
        texts=texts,
        metadata=metadatas
    )
    end_time = time.time()
    
    total_insert_time = end_time - start_time
    docs_per_second = len(texts) / total_insert_time
    
    # Measure query time with multiple query vectors
    logger.info(f"Performing {num_queries} queries using multi-GPU approach")
    
    # Generate random query vectors
    query_vectors = vectors[:num_queries]
    
    # Measure total query time
    start_time = time.time()
    results = mgpu_manager.similarity_search_multi_gpu(
        table_name=table_name,
        query_vectors=query_vectors,
        k=k,
        include_metadata=True
    )
    end_time = time.time()
    
    total_query_time = end_time - start_time
    queries_per_second = num_queries / total_query_time
    
    # Close clients
    for client in flight_clients:
        client.close()
    
    # Return benchmark results
    return {
        "approach": "arrow_flight_multi_gpu",
        "table_name": table_name,
        "num_documents": len(texts),
        "batch_size": batch_size,
        "num_queries": num_queries,
        "k": k,
        "num_gpus": len(gpu_ids) if gpu_ids else 1,
        "gpu_ids": gpu_ids,
        "distribution_strategy": distribution_strategy,
        "total_insert_time": total_insert_time,
        "docs_per_second": docs_per_second,
        "total_query_time": total_query_time,
        "queries_per_second": queries_per_second
    }


def benchmark_serialization_performance(
    vectors: List[List[float]],
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    Benchmark serialization performance for different methods.
    
    Args:
        vectors: List of vectors to serialize
        num_iterations: Number of iterations for each method
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking serialization performance with {len(vectors)} vectors")
    
    # Prepare results dictionary
    results = {
        "approach": "serialization",
        "num_vectors": len(vectors),
        "num_iterations": num_iterations,
        "methods": {}
    }
    
    # Benchmark standard binary serialization
    binary_times = []
    binary_sizes = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        serialized = [serialize_vector(v) for v in vectors]
        end_time = time.time()
        
        binary_times.append(end_time - start_time)
        binary_sizes.append(sum(len(s) for s in serialized))
    
    results["methods"]["binary"] = {
        "avg_time": sum(binary_times) / num_iterations,
        "avg_size_bytes": sum(binary_sizes) / num_iterations,
        "throughput_vectors_per_second": len(vectors) / (sum(binary_times) / num_iterations)
    }
    
    # Benchmark compressed binary serialization
    compressed_times = []
    compressed_sizes = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        serialized = [serialize_vector(v, compression=True) for v in vectors]
        end_time = time.time()
        
        compressed_times.append(end_time - start_time)
        compressed_sizes.append(sum(len(s) for s in serialized))
    
    results["methods"]["compressed_binary"] = {
        "avg_time": sum(compressed_times) / num_iterations,
        "avg_size_bytes": sum(compressed_sizes) / num_iterations,
        "throughput_vectors_per_second": len(vectors) / (sum(compressed_times) / num_iterations)
    }
    
    # Benchmark Arrow serialization (batch)
    arrow_times = []
    arrow_sizes = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        batch = vectors_to_arrow_batch(vectors)
        serialized = serialize_arrow_batch(batch)
        end_time = time.time()
        
        arrow_times.append(end_time - start_time)
        arrow_sizes.append(len(serialized))
    
    results["methods"]["arrow_batch"] = {
        "avg_time": sum(arrow_times) / num_iterations,
        "avg_size_bytes": sum(arrow_sizes) / num_iterations,
        "throughput_vectors_per_second": len(vectors) / (sum(arrow_times) / num_iterations)
    }
    
    # Benchmark Arrow serialization (batch) with compression
    arrow_compressed_times = []
    arrow_compressed_sizes = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        batch = vectors_to_arrow_batch(vectors)
        serialized = serialize_arrow_batch(batch, compression=True)
        end_time = time.time()
        
        arrow_compressed_times.append(end_time - start_time)
        arrow_compressed_sizes.append(len(serialized))
    
    results["methods"]["arrow_batch_compressed"] = {
        "avg_time": sum(arrow_compressed_times) / num_iterations,
        "avg_size_bytes": sum(arrow_compressed_sizes) / num_iterations,
        "throughput_vectors_per_second": len(vectors) / (sum(arrow_compressed_times) / num_iterations)
    }
    
    return results


def plot_results(results: List[Dict[str, Any]], output_dir: str):
    """
    Plot benchmark results.
    
    Args:
        results: List of benchmark results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter results by approach
    traditional_results = [r for r in results if r.get("approach") == "traditional"]
    arrow_flight_results = [r for r in results if r.get("approach") == "arrow_flight"]
    multi_gpu_results = [r for r in results if r.get("approach") == "arrow_flight_multi_gpu"]
    serialization_results = [r for r in results if r.get("approach") == "serialization"]
    
    # Plot insertion throughput comparison
    if traditional_results and arrow_flight_results:
        plt.figure(figsize=(10, 6))
        
        approaches = ["Traditional", "Arrow Flight"]
        throughputs = [
            traditional_results[0].get("docs_per_second", 0),
            arrow_flight_results[0].get("docs_per_second", 0)
        ]
        
        if multi_gpu_results:
            approaches.append("Arrow Flight Multi-GPU")
            throughputs.append(multi_gpu_results[0].get("docs_per_second", 0))
        
        plt.bar(approaches, throughputs, color=["blue", "green", "orange"])
        plt.title("Document Insertion Throughput (documents/second)")
        plt.xlabel("Approach")
        plt.ylabel("Documents per Second")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        for i, v in enumerate(throughputs):
            plt.text(i, v + 5, f"{v:.2f}", ha="center")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "insertion_throughput.png"))
        
    # Plot query throughput comparison
    if traditional_results and arrow_flight_results:
        plt.figure(figsize=(10, 6))
        
        approaches = ["Traditional", "Arrow Flight"]
        throughputs = [
            traditional_results[0].get("queries_per_second", 0),
            arrow_flight_results[0].get("queries_per_second", 0)
        ]
        
        if multi_gpu_results:
            approaches.append("Arrow Flight Multi-GPU")
            throughputs.append(multi_gpu_results[0].get("queries_per_second", 0))
        
        plt.bar(approaches, throughputs, color=["blue", "green", "orange"])
        plt.title("Query Throughput (queries/second)")
        plt.xlabel("Approach")
        plt.ylabel("Queries per Second")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        
        for i, v in enumerate(throughputs):
            plt.text(i, v + 0.5, f"{v:.2f}", ha="center")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "query_throughput.png"))
    
    # Plot serialization performance
    if serialization_results:
        result = serialization_results[0]
        methods = result.get("methods", {})
        
        if methods:
            plt.figure(figsize=(10, 6))
            
            method_names = list(methods.keys())
            throughputs = [m.get("throughput_vectors_per_second", 0) for m in methods.values()]
            
            plt.bar(method_names, throughputs, color=["blue", "green", "red", "orange"])
            plt.title("Serialization Throughput (vectors/second)")
            plt.xlabel("Method")
            plt.ylabel("Vectors per Second")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            
            for i, v in enumerate(throughputs):
                plt.text(i, v + 500, f"{v:.2f}", ha="center")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "serialization_throughput.png"))
            
            # Plot serialization size comparison
            plt.figure(figsize=(10, 6))
            
            sizes = [m.get("avg_size_bytes", 0) / 1024 for m in methods.values()]  # Convert to KB
            
            plt.bar(method_names, sizes, color=["blue", "green", "red", "orange"])
            plt.title("Serialization Size (KB)")
            plt.xlabel("Method")
            plt.ylabel("Average Size (KB)")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            
            for i, v in enumerate(sizes):
                plt.text(i, v + 50, f"{v:.2f}", ha="center")
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "serialization_size.png"))
    
    # Generate summary report
    summary = {
        "traditional": traditional_results[0] if traditional_results else None,
        "arrow_flight": arrow_flight_results[0] if arrow_flight_results else None,
        "multi_gpu": multi_gpu_results[0] if multi_gpu_results else None,
        "serialization": serialization_results[0] if serialization_results else None
    }
    
    with open(os.path.join(output_dir, "benchmark_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate HTML report
    html_report = f"""
    <html>
    <head>
        <title>Arrow Flight Benchmark Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .chart {{ margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>Arrow Flight Benchmark Results</h1>
        
        <h2>Performance Comparison</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Traditional</th>
                <th>Arrow Flight</th>
                <th>Arrow Flight Multi-GPU</th>
                <th>Improvement</th>
            </tr>
    """
    
    # Add insertion throughput row
    trad_insert = traditional_results[0].get("docs_per_second", 0) if traditional_results else 0
    arrow_insert = arrow_flight_results[0].get("docs_per_second", 0) if arrow_flight_results else 0
    multi_insert = multi_gpu_results[0].get("docs_per_second", 0) if multi_gpu_results else 0
    
    improvement = ((arrow_insert / trad_insert) - 1) * 100 if trad_insert > 0 else 0
    multi_improvement = ((multi_insert / trad_insert) - 1) * 100 if trad_insert > 0 else 0
    
    html_report += f"""
            <tr>
                <td>Insertion Throughput (docs/sec)</td>
                <td>{trad_insert:.2f}</td>
                <td>{arrow_insert:.2f}</td>
                <td>{multi_insert:.2f}</td>
                <td>{improvement:.2f}% / {multi_improvement:.2f}%</td>
            </tr>
    """
    
    # Add query throughput row
    trad_query = traditional_results[0].get("queries_per_second", 0) if traditional_results else 0
    arrow_query = arrow_flight_results[0].get("queries_per_second", 0) if arrow_flight_results else 0
    multi_query = multi_gpu_results[0].get("queries_per_second", 0) if multi_gpu_results else 0
    
    improvement = ((arrow_query / trad_query) - 1) * 100 if trad_query > 0 else 0
    multi_improvement = ((multi_query / trad_query) - 1) * 100 if trad_query > 0 else 0
    
    html_report += f"""
            <tr>
                <td>Query Throughput (queries/sec)</td>
                <td>{trad_query:.2f}</td>
                <td>{arrow_query:.2f}</td>
                <td>{multi_query:.2f}</td>
                <td>{improvement:.2f}% / {multi_improvement:.2f}%</td>
            </tr>
        </table>
    """
    
    # Add serialization comparison if available
    if serialization_results:
        result = serialization_results[0]
        methods = result.get("methods", {})
        
        if methods:
            html_report += f"""
            <h2>Serialization Performance</h2>
            <table>
                <tr>
                    <th>Method</th>
                    <th>Throughput (vectors/sec)</th>
                    <th>Average Size (KB)</th>
                </tr>
            """
            
            for method, data in methods.items():
                throughput = data.get("throughput_vectors_per_second", 0)
                size = data.get("avg_size_bytes", 0) / 1024  # Convert to KB
                
                html_report += f"""
                <tr>
                    <td>{method}</td>
                    <td>{throughput:.2f}</td>
                    <td>{size:.2f}</td>
                </tr>
                """
            
            html_report += "</table>"
    
    # Add images
    html_report += f"""
        <h2>Charts</h2>
        <div class="chart">
            <h3>Insertion Throughput</h3>
            <img src="insertion_throughput.png" alt="Insertion Throughput">
        </div>
        
        <div class="chart">
            <h3>Query Throughput</h3>
            <img src="query_throughput.png" alt="Query Throughput">
        </div>
    """
    
    if serialization_results:
        html_report += f"""
        <div class="chart">
            <h3>Serialization Throughput</h3>
            <img src="serialization_throughput.png" alt="Serialization Throughput">
        </div>
        
        <div class="chart">
            <h3>Serialization Size</h3>
            <img src="serialization_size.png" alt="Serialization Size">
        </div>
        """
    
    html_report += """
    </body>
    </html>
    """
    
    with open(os.path.join(output_dir, "benchmark_report.html"), "w") as f:
        f.write(html_report)


def main():
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description="Arrow Flight Benchmark")
    
    # Connection parameters
    parser.add_argument("--host", type=str, required=True, help="SAP HANA host address")
    parser.add_argument("--port", type=int, default=30015, help="SAP HANA port")
    parser.add_argument("--user", type=str, required=True, help="SAP HANA username")
    parser.add_argument("--password", type=str, required=True, help="SAP HANA password")
    
    # Benchmark parameters
    parser.add_argument("--num-documents", type=int, default=1000, help="Number of test documents")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for operations")
    parser.add_argument("--num-queries", type=int, default=10, help="Number of queries to perform")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return for each query")
    parser.add_argument("--embedding-dim", type=int, default=768, help="Dimension of embedding vectors")
    
    # Arrow Flight parameters
    parser.add_argument("--flight-port", type=int, default=8815, help="Arrow Flight server port")
    
    # Multi-GPU parameters
    parser.add_argument("--gpu-ids", type=str, help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--distribution-strategy", type=str, default="round_robin", 
                        choices=["round_robin", "memory_based", "model_based"],
                        help="Strategy for distributing work across GPUs")
    
    # Output parameters
    parser.add_argument("--output-dir", type=str, default="benchmark_results", 
                        help="Directory to save results")
    parser.add_argument("--test-data-file", type=str, help="Path to JSON file with test data")
    
    # Benchmark selection
    parser.add_argument("--skip-traditional", action="store_true", help="Skip traditional approach")
    parser.add_argument("--skip-arrow-flight", action="store_true", help="Skip Arrow Flight approach")
    parser.add_argument("--skip-multi-gpu", action="store_true", help="Skip multi-GPU approach")
    parser.add_argument("--skip-serialization", action="store_true", help="Skip serialization benchmark")
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Parse GPU IDs
    gpu_ids = None
    if args.gpu_ids:
        try:
            gpu_ids = [int(gpu_id.strip()) for gpu_id in args.gpu_ids.split(",")]
        except ValueError:
            logger.error("Invalid GPU IDs. Must be comma-separated integers.")
            sys.exit(1)
    
    # Load or generate test data
    texts, metadatas, vectors = load_or_generate_test_data(
        num_documents=args.num_documents,
        embedding_dim=args.embedding_dim,
        file_path=args.test_data_file
    )
    
    # Initialize results list
    results = []
    
    # Run traditional approach benchmark
    if not args.skip_traditional:
        try:
            # Connect to database
            connection = get_connection(
                host=args.host,
                port=args.port,
                user=args.user,
                password=args.password
            )
            
            # Run benchmark
            traditional_results = benchmark_traditional_approach(
                connection=connection,
                texts=texts,
                metadatas=metadatas,
                vectors=vectors,
                table_name="BENCHMARK_TRADITIONAL",
                batch_size=args.batch_size,
                num_queries=args.num_queries,
                k=args.k
            )
            
            results.append(traditional_results)
            
            # Close connection
            connection.close()
            
        except Exception as e:
            logger.error(f"Error in traditional approach benchmark: {str(e)}")
    
    # Run Arrow Flight approach benchmark
    if not args.skip_arrow_flight:
        try:
            # Run benchmark
            arrow_flight_results = benchmark_arrow_flight_approach(
                host=args.host,
                port=args.port,
                texts=texts,
                metadatas=metadatas,
                vectors=vectors,
                table_name="BENCHMARK_ARROW_FLIGHT",
                batch_size=args.batch_size,
                num_queries=args.num_queries,
                k=args.k,
                flight_port=args.flight_port
            )
            
            results.append(arrow_flight_results)
            
        except Exception as e:
            logger.error(f"Error in Arrow Flight approach benchmark: {str(e)}")
    
    # Run multi-GPU approach benchmark
    if not args.skip_multi_gpu and gpu_ids:
        try:
            # Run benchmark
            multi_gpu_results = benchmark_arrow_flight_multi_gpu_approach(
                host=args.host,
                port=args.port,
                texts=texts,
                metadatas=metadatas,
                vectors=vectors,
                table_name="BENCHMARK_ARROW_FLIGHT_MULTI_GPU",
                batch_size=args.batch_size,
                num_queries=args.num_queries,
                k=args.k,
                flight_port=args.flight_port,
                gpu_ids=gpu_ids,
                distribution_strategy=args.distribution_strategy
            )
            
            results.append(multi_gpu_results)
            
        except Exception as e:
            logger.error(f"Error in multi-GPU approach benchmark: {str(e)}")
    
    # Run serialization benchmark
    if not args.skip_serialization:
        try:
            # Run benchmark
            serialization_results = benchmark_serialization_performance(
                vectors=vectors,
                num_iterations=10
            )
            
            results.append(serialization_results)
            
        except Exception as e:
            logger.error(f"Error in serialization benchmark: {str(e)}")
    
    # Plot results
    if results:
        plot_results(results, args.output_dir)
        
        logger.info(f"Benchmark results saved to {args.output_dir}")
    else:
        logger.error("No benchmark results to plot")


if __name__ == "__main__":
    main()