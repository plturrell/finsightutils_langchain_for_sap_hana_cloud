"""
Advanced optimization examples for SAP HANA Cloud LangChain integration.

This example demonstrates:
1. Data valuation with DVRL
2. Interpretable embeddings with Neural Additive Models
3. Optimized hyperparameters with opt_list
4. Model compression with state_of_sparsity
"""

import logging
import os
import json
import numpy as np
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangChain components
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_hana.vectorstores import HanaDB

# Import optimization components
from langchain_hana.optimization.data_valuation import DVRLDataValuation
from langchain_hana.optimization.interpretable_embeddings import NAMEmbeddings
from langchain_hana.optimization.hyperparameters import OptimizedHyperparameters
from langchain_hana.optimization.model_compression import SparseEmbeddingModel

# Check for required dependencies
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HAS_HF = True
except ImportError:
    logger.warning("HuggingFaceEmbeddings not available. Install langchain-community package.")
    HAS_HF = False


def create_sample_documents(num_docs: int = 100) -> List[Document]:
    """Create sample documents for demonstration."""
    documents = []
    categories = ["finance", "technology", "healthcare", "education", "energy"]
    
    for i in range(num_docs):
        category = categories[i % len(categories)]
        quality = "high" if i % 3 == 0 else "medium" if i % 3 == 1 else "low"
        
        # Generate document with different lengths based on quality
        if quality == "high":
            length = np.random.randint(500, 1000)
        elif quality == "medium":
            length = np.random.randint(200, 500)
        else:
            length = np.random.randint(50, 200)
        
        # Create document text
        words = [f"word{j}" for j in range(length)]
        text = f"This is a {quality} quality document about {category}. " + " ".join(words)
        
        # Create document with metadata
        doc = Document(
            page_content=text,
            metadata={
                "id": f"doc_{i}",
                "category": category,
                "quality": quality,
                "length": length,
            }
        )
        documents.append(doc)
    
    return documents


def example_data_valuation(documents: List[Document], embedding_model: Embeddings) -> None:
    """Demonstrate data valuation with DVRL."""
    logger.info("=== Data Valuation Example ===")
    
    # Create data valuation component
    data_valuation = DVRLDataValuation(
        embedding_dimension=768,  # Match embedding model dimension
        value_threshold=0.6,
        cache_file="data_values.json",
    )
    
    # Generate embeddings for documents
    embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])
    
    # Compute document values
    logger.info("Computing document values...")
    doc_values = data_valuation.compute_document_values(documents, embeddings)
    
    # Print top and bottom documents by value
    doc_with_values = list(zip(documents, doc_values))
    doc_with_values.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("Top 5 most valuable documents:")
    for i, (doc, value) in enumerate(doc_with_values[:5]):
        logger.info(f"{i+1}. Value: {value:.4f}, Category: {doc.metadata['category']}, "
                   f"Quality: {doc.metadata['quality']}, Length: {doc.metadata['length']}")
    
    logger.info("\nBottom 5 least valuable documents:")
    for i, (doc, value) in enumerate(doc_with_values[-5:]):
        logger.info(f"{i+1}. Value: {value:.4f}, Category: {doc.metadata['category']}, "
                   f"Quality: {doc.metadata['quality']}, Length: {doc.metadata['length']}")
    
    # Filter valuable documents
    valuable_docs = data_valuation.filter_valuable_documents(documents, embeddings=embeddings)
    logger.info(f"Filtered to {len(valuable_docs)} valuable documents out of {len(documents)} total")
    
    # Analyze valuable documents
    categories = {}
    qualities = {}
    
    for doc in valuable_docs:
        category = doc.metadata["category"]
        quality = doc.metadata["quality"]
        
        categories[category] = categories.get(category, 0) + 1
        qualities[quality] = qualities.get(quality, 0) + 1
    
    logger.info("Valuable documents by category:")
    for category, count in categories.items():
        logger.info(f"  {category}: {count}")
    
    logger.info("Valuable documents by quality:")
    for quality, count in qualities.items():
        logger.info(f"  {quality}: {count}")


def example_interpretable_embeddings(documents: List[Document], embedding_model: Embeddings) -> None:
    """Demonstrate interpretable embeddings with Neural Additive Models."""
    logger.info("\n=== Interpretable Embeddings Example ===")
    
    # Create NAM embeddings
    interpretable_embeddings = NAMEmbeddings(
        base_embeddings=embedding_model,
        dimension=768,  # Match base model dimension
        num_features=64,  # Number of interpretable features
        feature_names=[f"feature_{i}" for i in range(64)],
        cache_dir="nam_cache",
    )
    
    # Sample documents for explanation
    query = "Tell me about high quality finance documents"
    sample_docs = [doc for doc in documents 
                  if doc.metadata["category"] == "finance" and doc.metadata["quality"] == "high"]
    
    if sample_docs:
        sample_doc = sample_docs[0]
        
        # Get similarity explanation
        logger.info(f"Explaining similarity between query and document...")
        explanation = interpretable_embeddings.explain_similarity(
            query=query,
            document=sample_doc.page_content,
            top_k=5,
        )
        
        # Print explanation
        logger.info(f"Query: {query}")
        logger.info(f"Document category: {sample_doc.metadata['category']}, "
                   f"quality: {sample_doc.metadata['quality']}")
        logger.info(f"Similarity score: {explanation['similarity_score']:.4f}")
        
        logger.info("Top matching features:")
        for feature, score in explanation["top_matching_features"]:
            logger.info(f"  {feature}: {score:.4f}")
        
        logger.info("Least matching features:")
        for feature, score in explanation["least_matching_features"]:
            logger.info(f"  {feature}: {score:.4f}")
    else:
        logger.warning("No matching documents found for explanation")


def example_optimized_hyperparameters() -> None:
    """Demonstrate optimized hyperparameters with opt_list."""
    logger.info("\n=== Optimized Hyperparameters Example ===")
    
    # Create hyperparameter optimizer
    optimizer = OptimizedHyperparameters(
        cache_file="hyperparams.json",
    )
    
    # Example model sizes
    model_sizes = {
        "small": 1e6,    # 1M parameters
        "medium": 1e7,   # 10M parameters
        "large": 1e8,    # 100M parameters
        "xlarge": 1e9,   # 1B parameters
    }
    
    # Example batch sizes
    batch_sizes = [8, 16, 32, 64, 128]
    
    # Get optimized learning rates for different model sizes
    logger.info("Optimized learning rates for different model sizes (batch_size=32):")
    for name, size in model_sizes.items():
        lr = optimizer.get_learning_rate(
            model_size=size,
            batch_size=32,
        )
        logger.info(f"  {name} model ({size:.1e} params): {lr:.6f}")
    
    # Get optimized batch sizes for different model sizes
    logger.info("\nOptimized batch sizes for different model sizes:")
    for name, size in model_sizes.items():
        bs = optimizer.get_batch_size(
            model_size=size,
        )
        logger.info(f"  {name} model ({size:.1e} params): {bs}")
    
    # Get optimized embedding parameters
    logger.info("\nOptimized parameters for embedding model:")
    embedding_params = optimizer.get_embedding_parameters(
        embedding_dimension=768,
        vocabulary_size=50000,
        max_sequence_length=512,
    )
    
    for param, value in embedding_params.items():
        logger.info(f"  {param}: {value}")
    
    # Get training schedule
    logger.info("\nOptimized training schedule:")
    schedule = optimizer.get_training_schedule(
        model_size=model_sizes["medium"],
        dataset_size=100000,
        batch_size=32,
        target_epochs=10,
    )
    
    for param, value in schedule.items():
        logger.info(f"  {param}: {value}")


def example_model_compression(documents: List[Document], embedding_model: Embeddings) -> None:
    """Demonstrate model compression with state_of_sparsity."""
    logger.info("\n=== Model Compression Example ===")
    
    # Create compressed embedding model
    compressed_embeddings = SparseEmbeddingModel(
        base_embeddings=embedding_model,
        compression_ratio=0.7,  # Target 70% sparsity
        compression_strategy="magnitude",
        cache_dir="compressed_cache",
    )
    
    # Sample documents for comparison
    sample_docs = documents[:5]
    
    # Generate embeddings with both models
    logger.info("Generating embeddings with base model...")
    start_time = time.time()
    base_vectors = embedding_model.embed_documents([doc.page_content for doc in sample_docs])
    base_time = time.time() - start_time
    
    logger.info("Generating embeddings with compressed model...")
    start_time = time.time()
    compressed_vectors = compressed_embeddings.embed_documents([doc.page_content for doc in sample_docs])
    compressed_time = time.time() - start_time
    
    # Calculate statistics
    base_size = sum(len(vec) * 4 for vec in base_vectors)  # Assuming float32
    compressed_size = 0
    for vec in compressed_vectors:
        # Count non-zero elements
        non_zeros = sum(1 for v in vec if v != 0)
        compressed_size += non_zeros * 4  # Assuming float32 for non-zero values
    
    # Calculate compression ratio
    actual_ratio = compressed_size / base_size
    
    logger.info(f"Base model size: {base_size / 1024:.2f} KB")
    logger.info(f"Compressed model size: {compressed_size / 1024:.2f} KB")
    logger.info(f"Compression ratio: {actual_ratio:.2%}")
    logger.info(f"Base model time: {base_time:.4f} seconds")
    logger.info(f"Compressed model time: {compressed_time:.4f} seconds")
    logger.info(f"Speed improvement: {base_time / compressed_time:.2f}x")
    
    # Get detailed compression stats
    stats = compressed_embeddings.get_compression_stats()
    logger.info("\nCompression statistics:")
    logger.info(f"  Target compression ratio: {stats['compression_ratio']:.2%}")
    logger.info(f"  Actual sparsity achieved: {stats['total_sparsity']:.2%}")
    logger.info(f"  Compression strategy: {stats['compression_strategy']}")
    logger.info(f"  Cache size: {stats['cache_size']} embeddings")


def main():
    """Run all optimization examples."""
    import time
    
    # Create base embedding model
    if not HAS_HF:
        # Create a simple embedding model
        class DummyEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                return [np.random.normal(size=768).tolist() for _ in texts]
            
            def embed_query(self, text: str) -> List[float]:
                return np.random.normal(size=768).tolist()
        
        embedding_model = DummyEmbeddings()
        logger.warning("Using dummy embeddings for demonstration")
    else:
        # Use HuggingFace embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logger.info("Using HuggingFace embeddings")
    
    # Create sample documents
    logger.info("Creating sample documents...")
    documents = create_sample_documents(num_docs=100)
    logger.info(f"Created {len(documents)} sample documents")
    
    # Run examples
    example_data_valuation(documents, embedding_model)
    example_interpretable_embeddings(documents, embedding_model)
    example_optimized_hyperparameters()
    example_model_compression(documents, embedding_model)
    
    logger.info("\nAll optimization examples completed successfully!")


if __name__ == "__main__":
    main()