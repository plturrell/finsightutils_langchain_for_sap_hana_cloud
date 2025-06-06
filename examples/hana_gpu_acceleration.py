"""
GPU-accelerated SAP HANA Cloud Vector Search Example

This example demonstrates how to use GPU acceleration for embedding generation
and vector search with SAP HANA Cloud, leveraging NVIDIA GPUs (especially T4) 
for significant performance improvements.

Key features demonstrated:
1. GPU-accelerated embedding generation with TensorRT
2. Multi-GPU support for distributed processing
3. Memory-optimized vector serialization
4. Performance benchmarking across different precision modes
5. Integration with SAP HANA Cloud vectorstore

Usage:
    python hana_gpu_acceleration.py --host <hana_host> --port <hana_port> --user <username> --password <password> [--precision fp16]
"""

import os
import time
import argparse
import logging
from typing import List, Dict, Any, Optional
import json
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hana_gpu_example")

# Check for required dependencies
try:
    import torch
    from hdbcli import dbapi
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    logger.error("Please install required packages: pip install hdbcli torch")
    exit(1)

# Import LangChain components
from langchain_core.documents import Document

# Import our GPU-accelerated components
from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings
from langchain_hana.gpu.hana_tensorrt_vectorstore import HanaTensorRTVectorStore
from langchain_hana.gpu.vector_serialization import get_vector_memory_usage


def get_sample_documents(count: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate sample documents for demonstration purposes.
    
    Args:
        count: Number of documents to generate
        
    Returns:
        List of documents with text and metadata
    """
    # Sample content categories
    categories = ["finance", "technology", "business", "sap", "cloud", "database", "analytics"]
    
    # Sample document templates
    templates = [
        "An overview of {topic} and its applications in modern enterprises.",
        "How {topic} is transforming the way companies operate in the digital age.",
        "Understanding the fundamentals of {topic} for business innovation.",
        "The role of {topic} in enterprise digital transformation strategies.",
        "Best practices for implementing {topic} solutions in your organization.",
        "Challenges and opportunities in adopting {topic} technologies.",
        "{topic} trends that are reshaping the business landscape in {year}.",
        "A comprehensive guide to {topic} integration with existing systems.",
        "Key considerations for {topic} deployment in enterprise environments.",
        "How {topic} can drive operational efficiency and cost reduction.",
    ]
    
    # Sample topics
    topics = [
        "SAP HANA Cloud", "vector databases", "enterprise AI", "data analytics",
        "machine learning", "cloud computing", "business intelligence", "digital transformation",
        "data warehousing", "natural language processing", "predictive analytics",
        "real-time data processing", "intelligent ERP", "embedded analytics",
        "data integration", "data governance", "generative AI", "large language models",
        "semantic search", "vector embeddings", "multi-modal AI", "GPU acceleration",
    ]
    
    # Generate random documents
    documents = []
    for i in range(count):
        # Select random template and topic
        template = random.choice(templates)
        topic = random.choice(topics)
        
        # Generate text
        text = template.format(topic=topic, year=random.choice([2023, 2024, 2025]))
        
        # Generate metadata
        metadata = {
            "id": f"doc-{i:04d}",
            "category": random.choice(categories),
            "length": len(text),
            "priority": random.choice(["high", "medium", "low"]),
            "created_at": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        }
        
        documents.append({"text": text, "metadata": metadata})
    
    return documents


def create_hana_connection(args: argparse.Namespace) -> dbapi.Connection:
    """
    Create a connection to SAP HANA Cloud.
    
    Args:
        args: Command line arguments with connection parameters
        
    Returns:
        SAP HANA database connection
    """
    try:
        connection = dbapi.connect(
            address=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
        )
        logger.info(f"Connected to SAP HANA at {args.host}:{args.port}")
        return connection
    except dbapi.Error as e:
        logger.error(f"Failed to connect to SAP HANA: {e}")
        raise


def run_benchmark(
    embedding_model: HanaTensorRTEmbeddings,
    precision_modes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run benchmarks to compare performance across precision modes.
    
    Args:
        embedding_model: HanaTensorRTEmbeddings model to benchmark
        precision_modes: List of precision modes to benchmark
        
    Returns:
        Benchmark results
    """
    if precision_modes is None:
        precision_modes = ["fp32", "fp16", "int8"]
    
    # Filter precision modes based on GPU capabilities
    try:
        device_id = 0 if isinstance(embedding_model.device, int) else int(embedding_model.device.split(":")[-1])
        capabilities = torch.cuda.get_device_capability(device_id)
        major, minor = capabilities
        
        # Check tensor core support for each precision
        supported_modes = []
        if major >= 7:  # Volta or newer
            supported_modes.append("fp32")
            supported_modes.append("fp16")
            
            if (major > 7) or (major == 7 and minor >= 5):  # Turing (T4) or newer
                supported_modes.append("int8")
        else:
            supported_modes = ["fp32"]  # Older GPUs only support FP32
            
        # Filter to supported modes
        precision_modes = [mode for mode in precision_modes if mode in supported_modes]
    except Exception as e:
        logger.warning(f"Error checking GPU capabilities: {e}")
    
    logger.info(f"Running benchmarks for precision modes: {', '.join(precision_modes)}")
    
    # Generate benchmark results for each precision mode
    try:
        return embedding_model.benchmark_precision_comparison()
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {"error": str(e)}


def run_gpu_accelerated_example(args: argparse.Namespace) -> None:
    """
    Run the GPU-accelerated SAP HANA vectorstore example.
    
    This function demonstrates the following:
    1. Creating GPU-accelerated embeddings
    2. Benchmarking different precision modes
    3. Creating a vectorstore with GPU acceleration
    4. Adding documents with batched GPU processing
    5. Performing similarity search with GPU acceleration
    6. Viewing performance metrics
    
    Args:
        args: Command line arguments
    """
    logger.info("Starting GPU-accelerated SAP HANA vectorstore example")
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        logger.warning("No CUDA-compatible GPU found. This example requires a GPU.")
        return
    
    # Get GPU information
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    logger.info(f"Found {gpu_count} GPU(s): {', '.join(gpu_names)}")
    
    # Create connection to SAP HANA
    try:
        connection = create_hana_connection(args)
    except Exception as e:
        logger.error(f"Failed to connect to SAP HANA: {e}")
        return
    
    try:
        # Create embedding model with GPU acceleration
        logger.info("Initializing GPU-accelerated embeddings...")
        embedding_model = HanaTensorRTEmbeddings(
            model_name=args.model,
            precision=args.precision,
            multi_gpu=args.multi_gpu,
            max_batch_size=args.batch_size,
            enable_profiling=True
        )
        
        # Run benchmark if requested
        if args.benchmark:
            logger.info("Running embedding benchmark...")
            benchmark_results = run_benchmark(embedding_model)
            logger.info("Benchmark results:")
            print(json.dumps(benchmark_results, indent=2))
        
        # Create vectorstore with GPU acceleration
        logger.info("Creating GPU-accelerated vectorstore...")
        vectorstore = HanaTensorRTVectorStore(
            connection=connection,
            embedding=embedding_model,
            table_name=args.table,
            batch_size=args.batch_size,
            enable_performance_monitoring=True,
            # Use appropriate vector type based on precision
            vector_column_type="HALF_VECTOR" if args.precision == "fp16" else "REAL_VECTOR"
        )
        
        # Generate sample documents
        logger.info(f"Generating {args.num_docs} sample documents...")
        sample_docs = get_sample_documents(args.num_docs)
        
        # Extract texts and metadata
        texts = [doc["text"] for doc in sample_docs]
        metadatas = [doc["metadata"] for doc in sample_docs]
        
        # Add documents to vectorstore
        logger.info(f"Adding {len(texts)} documents to vectorstore...")
        start_time = time.time()
        vectorstore.add_texts(texts, metadatas)
        elapsed_time = time.time() - start_time
        
        logger.info(f"Added {len(texts)} documents in {elapsed_time:.2f}s "
                   f"({len(texts)/elapsed_time:.2f} docs/s)")
        
        # Perform similarity searches
        logger.info("Performing similarity searches...")
        queries = [
            "What are the best practices for SAP HANA Cloud?",
            "How can vector databases improve search capabilities?",
            "The impact of AI on business intelligence",
            "Digital transformation strategies for enterprises",
            "Real-time data processing in cloud environments",
        ]
        
        for i, query in enumerate(queries):
            logger.info(f"Search {i+1}: '{query}'")
            start_time = time.time()
            results = vectorstore.similarity_search(query, k=5)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Search completed in {elapsed_time:.4f}s, found {len(results)} results")
            
            # Display top results
            print(f"\nTop results for query: '{query}'")
            for j, doc in enumerate(results):
                print(f"{j+1}. {doc.page_content[:100]}... [Score: {doc.metadata.get('score', 'N/A')}]")
        
        # Get performance metrics
        if args.show_metrics:
            logger.info("Performance metrics:")
            metrics = vectorstore.get_performance_metrics()
            print(json.dumps(metrics, indent=2))
        
        # Calculate memory usage statistics
        if hasattr(embedding_model, "model_dim"):
            model_dim = embedding_model.model_dim
            memory_stats = get_vector_memory_usage(
                vector_dimension=model_dim,
                num_vectors=args.num_docs,
                precision=args.precision or "float32"
            )
            
            logger.info("Vector memory usage statistics:")
            print(f"Vector dimension: {memory_stats['vector_dimension']}")
            print(f"Number of vectors: {memory_stats['num_vectors']}")
            print(f"Current precision: {memory_stats['current_precision']}")
            print(f"Memory usage: {memory_stats['current_memory_mb']:.2f} MB")
            print(f"Memory usage with FP32: {memory_stats['memory_float32_mb']:.2f} MB")
            print(f"Memory usage with FP16: {memory_stats['memory_float16_mb']:.2f} MB")
            print(f"Memory usage with INT8: {memory_stats['memory_int8_mb']:.2f} MB")
            print(f"Savings (FP16 vs FP32): {memory_stats['savings_float16_vs_float32_percent']:.1f}%")
            print(f"Savings (INT8 vs FP32): {memory_stats['savings_int8_vs_float32_percent']:.1f}%")
    
    except Exception as e:
        logger.error(f"Error in GPU-accelerated example: {e}")
        raise
    finally:
        # Clean up
        if 'connection' in locals():
            connection.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU-accelerated SAP HANA Cloud Vector Search Example")
    
    # Connection parameters
    parser.add_argument("--host", required=True, help="SAP HANA host")
    parser.add_argument("--port", type=int, required=True, help="SAP HANA port")
    parser.add_argument("--user", required=True, help="SAP HANA username")
    parser.add_argument("--password", required=True, help="SAP HANA password")
    
    # Model parameters
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", 
                        help="Embedding model name")
    parser.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16",
                        help="Precision mode for GPU acceleration")
    parser.add_argument("--multi-gpu", action="store_true", help="Use multiple GPUs if available")
    
    # Vectorstore parameters
    parser.add_argument("--table", default="GPU_ACCELERATED_VECTORS", 
                        help="Name of the table to store vectors")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for processing")
    parser.add_argument("--num-docs", type=int, default=1000,
                        help="Number of sample documents to generate")
    
    # Other options
    parser.add_argument("--benchmark", action="store_true", 
                        help="Run benchmark before main example")
    parser.add_argument("--show-metrics", action="store_true",
                        help="Show detailed performance metrics")
    
    args = parser.parse_args()
    
    # Run the example
    run_gpu_accelerated_example(args)