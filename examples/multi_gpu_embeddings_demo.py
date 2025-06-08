#!/usr/bin/env python3
"""
Multi-GPU Embeddings Demo for SAP HANA Cloud LangChain Integration.

This example demonstrates how to use the multi-GPU embeddings functionality 
with SAP HANA Cloud to improve embedding generation performance by distributing
workloads across multiple NVIDIA GPUs.

Prerequisites:
- NVIDIA GPUs with CUDA support
- PyTorch with CUDA support
- SAP HANA Cloud connection
- Required Python packages: langchain, hdbcli, transformers, torch

Usage:
    python multi_gpu_embeddings_demo.py --connection-config config.json --num-docs 1000
"""

import argparse
import json
import time
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from hdbcli import dbapi

# Import langchain components
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Import SAP HANA Cloud integration
from langchain_hana import (
    HanaDB,
    MultiGPUEmbeddings,
    HanaTensorRTMultiGPUEmbeddings,
    CacheConfig
)
from langchain_hana.gpu.multi_gpu_manager import get_multi_gpu_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_connection(config_path: str) -> dbapi.Connection:
    """
    Create a connection to SAP HANA Cloud using the provided config.
    
    Args:
        config_path: Path to connection configuration JSON file
        
    Returns:
        SAP HANA Cloud connection
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    conn = dbapi.connect(
        address=config['address'],
        port=config['port'],
        user=config['user'],
        password=config['password']
    )
    
    return conn


def generate_sample_documents(n: int = 100, length: int = 200) -> List[Document]:
    """
    Generate sample documents for testing.
    
    Args:
        n: Number of documents to generate
        length: Approximate length of each document in words
        
    Returns:
        List of generated documents
    """
    documents = []
    
    # Sample words to use in generated documents
    words = [
        "data", "analytics", "cloud", "database", "vector", "embedding", "neural",
        "network", "enterprise", "business", "intelligence", "machine", "learning",
        "artificial", "intelligence", "digital", "transformation", "integration",
        "platform", "service", "application", "software", "hardware", "system",
        "computation", "algorithm", "performance", "scalability", "security",
        "hana", "sap", "azure", "aws", "gcp", "hybrid", "multi", "cloud", "saas",
        "paas", "iaas", "microservice", "architecture", "container", "kubernetes",
        "docker", "devops", "agile", "methodology", "implementation", "development"
    ]
    
    for i in range(n):
        # Generate a random document
        doc_words = np.random.choice(words, size=length)
        text = " ".join(doc_words)
        
        # Add some sentences for variety
        sentences = [
            "SAP HANA Cloud provides enterprise-grade capabilities for data management.",
            "Vector embeddings enable semantic search capabilities in databases.",
            "Multi-GPU processing accelerates AI workloads significantly.",
            "Large language models require efficient embedding generation.",
            "Enterprise data can be analyzed using modern AI techniques."
        ]
        
        # Insert 2-3 sentences randomly in the document
        for _ in range(np.random.randint(2, 4)):
            pos = np.random.randint(0, len(doc_words))
            sentence_idx = np.random.randint(0, len(sentences))
            text = text.split()
            text.insert(pos, sentences[sentence_idx])
            text = " ".join(text)
        
        # Create document with metadata
        doc = Document(
            page_content=text,
            metadata={
                "id": f"doc_{i}",
                "source": "generated",
                "length": len(text.split()),
                "category": np.random.choice(["analytics", "business", "technical", "educational"]),
                "complexity": np.random.choice(["low", "medium", "high"])
            }
        )
        
        documents.append(doc)
    
    return documents


def benchmark_embeddings(
    documents: List[Document],
    embeddings_list: List[Dict[str, Any]],
    queries: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark different embedding models and configurations.
    
    Args:
        documents: List of documents to embed
        embeddings_list: List of embedding models to benchmark
        queries: Optional list of queries to benchmark (if None, generates 5 random queries)
        
    Returns:
        Dictionary of benchmark results
    """
    if queries is None:
        # Generate 5 random queries
        queries = [
            "How does SAP HANA Cloud support vector search?",
            "What are the benefits of multi-GPU processing?",
            "Explain enterprise data management best practices.",
            "How can I improve database performance?",
            "What is the relationship between AI and business intelligence?"
        ]
    
    results = {}
    
    for config in embeddings_list:
        name = config["name"]
        embeddings = config["model"]
        logger.info(f"Benchmarking {name}...")
        
        # Extract document texts
        texts = [doc.page_content for doc in documents]
        
        # Benchmark document embedding
        doc_start_time = time.time()
        doc_embeddings = embeddings.embed_documents(texts)
        doc_end_time = time.time()
        doc_time = doc_end_time - doc_start_time
        
        # Benchmark query embedding
        query_times = []
        for query in queries:
            query_start_time = time.time()
            query_embedding = embeddings.embed_query(query)
            query_end_time = time.time()
            query_times.append(query_end_time - query_start_time)
        
        # Calculate statistics
        avg_query_time = sum(query_times) / len(query_times)
        
        # Get embedding stats if available
        stats = {}
        if hasattr(embeddings, "get_stats"):
            stats = embeddings.get_stats()
        
        # Record results
        results[name] = {
            "document_embedding_time": doc_time,
            "documents_per_second": len(texts) / doc_time,
            "avg_time_per_document": doc_time / len(texts),
            "avg_query_time": avg_query_time,
            "queries_per_second": 1 / avg_query_time,
            "embedding_dimension": len(doc_embeddings[0]),
            "stats": stats
        }
        
        logger.info(f"  Documents: {len(texts) / doc_time:.2f} docs/sec")
        logger.info(f"  Queries: {1 / avg_query_time:.2f} queries/sec")
    
    return results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Multi-GPU Embeddings Demo")
    parser.add_argument(
        "--connection-config", 
        type=str, 
        help="Path to SAP HANA Cloud connection config JSON"
    )
    parser.add_argument(
        "--num-docs", 
        type=int, 
        default=100, 
        help="Number of documents to generate for testing"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--enable-tensorrt", 
        action="store_true", 
        help="Enable TensorRT optimization"
    )
    parser.add_argument(
        "--model-name", 
        type=str, 
        default="sentence-transformers/all-mpnet-base-v2", 
        help="HuggingFace model name for embeddings"
    )
    args = parser.parse_args()
    
    # Generate sample documents
    logger.info(f"Generating {args.num_docs} sample documents...")
    documents = generate_sample_documents(n=args.num_docs)
    
    # Initialize GPU manager
    logger.info("Initializing GPU manager...")
    gpu_manager = get_multi_gpu_manager()
    logger.info(f"GPU manager status: {gpu_manager.get_status()}")
    
    # Setup base embeddings model
    logger.info(f"Creating base embeddings model: {args.model_name}")
    base_model = HuggingFaceEmbeddings(
        model_name=args.model_name,
        model_kwargs={"device": "cuda:0"}
    )
    
    # Create benchmark configurations
    logger.info("Creating embedding configurations for benchmark...")
    embeddings_list = [
        {
            "name": "Base Model (Single GPU)",
            "model": base_model
        },
        {
            "name": "Multi-GPU Embeddings",
            "model": MultiGPUEmbeddings(
                base_embeddings=base_model,
                batch_size=args.batch_size,
                enable_caching=True,
                gpu_manager=gpu_manager,
                normalize_embeddings=True
            )
        }
    ]
    
    # Add TensorRT if enabled
    if args.enable_tensorrt:
        try:
            embeddings_list.append({
                "name": "TensorRT Multi-GPU Embeddings",
                "model": HanaTensorRTMultiGPUEmbeddings(
                    model_name=args.model_name,
                    batch_size=args.batch_size,
                    use_fp16=True,
                    use_tensorrt=True,
                    enable_tensor_cores=True,
                    enable_caching=True,
                    gpu_manager=gpu_manager
                )
            })
        except ImportError as e:
            logger.warning(f"TensorRT not available: {e}")
    
    # Run benchmark
    logger.info("Running benchmark...")
    results = benchmark_embeddings(documents, embeddings_list)
    
    # Print results
    logger.info("\nBenchmark Results:")
    for name, result in results.items():
        logger.info(f"\n{name}:")
        logger.info(f"  Documents per second: {result['documents_per_second']:.2f}")
        logger.info(f"  Queries per second: {result['queries_per_second']:.2f}")
        logger.info(f"  Embedding dimension: {result['embedding_dimension']}")
    
    # Create vectorstore if connection config provided
    if args.connection_config:
        try:
            logger.info("Connecting to SAP HANA Cloud...")
            conn = get_connection(args.connection_config)
            
            # Use the best performing model from the benchmark
            best_model_name = max(results.keys(), key=lambda k: results[k]["documents_per_second"])
            best_model = next(config["model"] for config in embeddings_list if config["name"] == best_model_name)
            
            logger.info(f"Using best performing model: {best_model_name}")
            
            # Create vectorstore
            logger.info("Creating vector store...")
            vector_store = HanaDB(
                connection=conn,
                embedding=best_model,
                table_name="MULTI_GPU_DEMO_VECTORS",
                pre_delete_table=True
            )
            
            # Add documents
            logger.info("Adding documents to vector store...")
            vector_store.add_documents(documents)
            
            # Run a test query
            logger.info("Running test similarity search...")
            query = "What are the benefits of using SAP HANA Cloud for analytics?"
            results = vector_store.similarity_search(query, k=5)
            
            logger.info(f"Query: {query}")
            logger.info("Top 5 results:")
            for i, doc in enumerate(results):
                logger.info(f"  {i+1}. {doc.page_content[:100]}...")
            
        except Exception as e:
            logger.error(f"Error connecting to SAP HANA Cloud: {e}")
    
    # Cleanup
    logger.info("Shutting down GPU manager...")
    gpu_manager.stop()
    logger.info("Done!")


if __name__ == "__main__":
    main()