# Advanced Optimization Components Guide

This guide explains the advanced optimization components available in the SAP HANA Cloud LangChain Integration, which can significantly improve the quality, interpretability, and efficiency of your vector search applications.

## Overview

The optimization components provide four key capabilities:

1. **Data Valuation**: Identify the most valuable documents for retrieval quality
2. **Interpretable Embeddings**: Understand which features contribute to search results
3. **Optimized Hyperparameters**: Use data-driven learning rates and training schedules
4. **Model Compression**: Reduce memory footprint with minimal accuracy loss

## Installation

To use these advanced components, install the optional optimization dependencies:

```bash
pip install -U "langchain-hana[optimization]"
```

Or install from the requirements file:

```bash
pip install -r requirements-optimization.txt
```

## Component Details

### Data Valuation (DVRL)

Data Valuation using Reinforcement Learning (DVRL) helps identify which documents in your vector store are most valuable for retrieval quality. This allows you to:

- Filter out low-value documents to improve search precision
- Optimize storage by retaining only the most valuable content
- Understand which document characteristics contribute to retrieval value

```python
from langchain_hana.optimization.data_valuation import DVRLDataValuation
from langchain_core.documents import Document

# Create data valuation component
data_valuation = DVRLDataValuation(
    embedding_dimension=768,  # Match your embedding model dimension
    value_threshold=0.7,      # Higher threshold = more selective filtering
    cache_file="data_values.json",  # Optional cache for computed values
)

# Evaluate document importance
documents = [
    Document(page_content="High quality document", metadata={"category": "finance"}),
    Document(page_content="Low quality document", metadata={"category": "general"}),
]
doc_values = data_valuation.compute_document_values(documents)
print(f"Document values: {doc_values}")  # e.g. [0.85, 0.32]

# Filter valuable documents
valuable_docs = data_valuation.filter_valuable_documents(
    documents,
    threshold=0.6,  # Optional custom threshold
    top_k=10,       # Optional limit to top K documents
)

# Optimize existing vector store
from langchain_hana.vectorstores import HanaDB
vectorstore = HanaDB.from_documents(documents, embedding=embedding_model)
optimization_result = data_valuation.optimize_vectorstore(vectorstore)
```

#### REST API

Data valuation is also available through the API:

```bash
curl -X POST "http://localhost:8000/optimization/data-valuation" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      {"page_content": "High quality document", "metadata": {"category": "finance"}},
      {"page_content": "Low quality document", "metadata": {"category": "general"}}
    ],
    "threshold": 0.7,
    "top_k": 10
  }'
```

### Interpretable Embeddings (NAM)

Neural Additive Models (NAM) provide interpretable embeddings that help you understand which features contribute to search results. This allows you to:

- Explain why a document matches a query
- Identify which features are most important for similarity
- Generate human-understandable explanations of vector search results

```python
from langchain_hana.optimization.interpretable_embeddings import NAMEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create base embedding model
base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create interpretable embedding model
interpretable_embeddings = NAMEmbeddings(
    base_embeddings=base_embeddings,
    dimension=384,             # Match your base model dimension
    num_features=64,           # Number of interpretable features
    feature_names=None,        # Optional custom feature names
    cache_dir="nam_cache",     # Optional cache directory
)

# Generate embeddings
texts = ["This is a document about finance", "This is a document about technology"]
embeddings = interpretable_embeddings.embed_documents(texts)

# Explain similarity between query and document
explanation = interpretable_embeddings.explain_similarity(
    query="Tell me about financial markets",
    document="This document discusses stock markets and investment strategies",
    top_k=5,  # Number of top features to include
)

print(f"Similarity score: {explanation['similarity_score']}")
print("Top matching features:")
for feature, score in explanation["top_matching_features"]:
    print(f"  {feature}: {score}")
```

#### REST API

Interpretable embeddings are also available through the API:

```bash
curl -X POST "http://localhost:8000/optimization/explain-similarity" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about financial markets",
    "document": "This document discusses stock markets and investment strategies",
    "top_k": 5
  }'
```

### Optimized Hyperparameters (opt_list)

Optimized hyperparameters provide data-driven learning rates, batch sizes, and training schedules based on large-scale optimization studies. This allows you to:

- Use optimal learning rates for different model sizes
- Determine the best batch size based on available hardware
- Generate complete training schedules with proper warmup and decay

```python
from langchain_hana.optimization.hyperparameters import OptimizedHyperparameters

# Create hyperparameter optimizer
optimizer = OptimizedHyperparameters(
    cache_file="hyperparams.json",  # Optional cache for computed values
    framework="tensorflow",         # Or "pytorch" or "auto"
)

# Get optimized learning rate
learning_rate = optimizer.get_learning_rate(
    model_size=10000000,  # Number of parameters
    batch_size=32,
    dataset_size=50000,   # Optional
)
print(f"Optimized learning rate: {learning_rate}")

# Get optimized batch size based on hardware
batch_size = optimizer.get_batch_size(
    model_size=10000000,
    max_memory=None,     # Auto-detect available memory
)
print(f"Optimized batch size: {batch_size}")

# Get embedding model parameters
embedding_params = optimizer.get_embedding_parameters(
    embedding_dimension=768,
    vocabulary_size=30000,
    max_sequence_length=512,
)
print(f"Embedding parameters: {embedding_params}")

# Get training schedule
schedule = optimizer.get_training_schedule(
    model_size=10000000,
    dataset_size=50000,
    batch_size=32,
)
print(f"Training schedule: {schedule}")
```

#### REST API

Optimized hyperparameters are also available through the API:

```bash
curl -X POST "http://localhost:8000/optimization/optimized-hyperparameters" \
  -H "Content-Type: application/json" \
  -d '{
    "model_size": 10000000,
    "batch_size": 32,
    "dataset_size": 50000,
    "embedding_dimension": 768,
    "vocabulary_size": 30000,
    "max_sequence_length": 512
  }'
```

### Model Compression (state_of_sparsity)

Model compression reduces the memory footprint of embedding models with minimal accuracy loss. This allows you to:

- Reduce memory usage for embedding vectors
- Speed up similarity calculations
- Deploy on resource-constrained environments

```python
from langchain_hana.optimization.model_compression import SparseEmbeddingModel
from langchain_community.embeddings import HuggingFaceEmbeddings

# Create base embedding model
base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create compressed embedding model
compressed_embeddings = SparseEmbeddingModel(
    base_embeddings=base_embeddings,
    compression_ratio=0.7,           # Target sparsity (0.0-1.0)
    compression_strategy="magnitude", # "magnitude", "random", or "structured"
    cache_dir="compressed_cache",     # Optional cache directory
)

# Generate compressed embeddings
texts = ["This is a document about finance", "This is a document about technology"]
embeddings = compressed_embeddings.embed_documents(texts)

# Get compression statistics
stats = compressed_embeddings.get_compression_stats()
print(f"Compression ratio: {stats['compression_ratio']}")
print(f"Actual sparsity: {stats['total_sparsity']}")
```

#### REST API

Model compression is also available through the API:

```bash
curl -X POST "http://localhost:8000/optimization/compressed-embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "This is a document about finance", 
      "This is a document about technology"
    ],
    "compression_ratio": 0.7,
    "compression_strategy": "magnitude"
  }'
```

## Example Application

See the complete example in `examples/advanced_optimization.py` which demonstrates all four optimization components working together:

```python
# Import components
from langchain_hana.optimization.data_valuation import DVRLDataValuation
from langchain_hana.optimization.interpretable_embeddings import NAMEmbeddings
from langchain_hana.optimization.hyperparameters import OptimizedHyperparameters
from langchain_hana.optimization.model_compression import SparseEmbeddingModel

# Run the example
python examples/advanced_optimization.py
```

## Performance Considerations

- **GPU Acceleration**: All optimization components can leverage GPU acceleration when available
- **Caching**: Use the cache options to avoid recomputing values for the same inputs
- **Resource Usage**: Data valuation and interpretable embeddings are more computationally intensive than standard embeddings
- **Compression Tradeoffs**: Higher compression ratios provide more memory savings but may impact accuracy

## Fallback Mechanisms

All optimization components include fallback mechanisms that activate automatically when required dependencies are not available. This ensures your application will continue to function even without the optional packages, though with reduced functionality.