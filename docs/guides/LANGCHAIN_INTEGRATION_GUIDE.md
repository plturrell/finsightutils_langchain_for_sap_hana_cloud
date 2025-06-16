# LangChain Integration Guide for SAP HANA Cloud

This guide provides a comprehensive overview of using LangChain with SAP HANA Cloud for building vector search and retrieval applications. It explains key concepts, implementation patterns, and optimization techniques.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
- [Core Components](#core-components)
- [Implementation Patterns](#implementation-patterns)
- [Domain-Specific Embeddings](#domain-specific-embeddings)
- [Performance Optimization](#performance-optimization)
- [Example Applications](#example-applications)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Overview

The LangChain integration for SAP HANA Cloud allows you to leverage SAP HANA's vector database capabilities within the LangChain framework. This enables you to build sophisticated AI applications using large language models (LLMs) with SAP HANA Cloud as the vector store backend.

Key benefits include:

- **Enterprise-grade storage** for embeddings in SAP HANA Cloud
- **Fast vector search** with HNSW indexing
- **Scalable architecture** for large document collections
- **Domain-specific embeddings** with FinE5 models for financial applications
- **GPU acceleration** options for high-throughput scenarios
- **Comprehensive filtering** capabilities with metadata

## Getting Started

### Prerequisites

- SAP HANA Cloud instance with vector capabilities
- Python 3.8+
- Required packages: `langchain`, `langchain_hana`, `hdbcli`

### Installation

```bash
pip install langchain-hana langchain-core langchain-openai hdbcli
```

### Basic Usage

```python
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_hana.vectorstore import HanaVectorStore
from langchain_hana.connection import create_connection

# Connect to SAP HANA Cloud
connection = create_connection(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password"
)

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vector_store = HanaVectorStore(
    connection=connection,
    embedding=embeddings,
    table_name="MY_VECTORS",
    create_table=True  # Create table if it doesn't exist
)

# Add documents
docs = ["Document 1", "Document 2", "Document 3"]
metadata = [{"source": "wiki"}, {"source": "web"}, {"source": "book"}]
vector_store.add_texts(docs, metadata)

# Search
results = vector_store.similarity_search("your query", k=3)
for doc in results:
    print(doc.page_content, doc.metadata)
```

## Core Components

### 1. HanaVectorStore

The central component is the `HanaVectorStore` class, which implements the LangChain `VectorStore` interface. This provides:

- Vector similarity search (`similarity_search`)
- Maximal Marginal Relevance search (`max_marginal_relevance_search`)
- Metadata filtering capabilities
- HNSW indexing for fast similarity search

### 2. Embedding Models

The integration supports various embedding models:

- **External embeddings**: Any LangChain-compatible embeddings (HuggingFace, OpenAI, etc.)
- **Internal embeddings**: SAP HANA's internal embedding functionality via `HanaInternalEmbeddings`
- **Financial embeddings**: Domain-specific embeddings with `FinE5Embeddings`

### 3. Connection Management

The `connection` module provides utilities for connecting to SAP HANA Cloud:

- `create_connection`: Create a new connection with specified parameters
- `test_connection`: Verify connection status and retrieve version information

## Implementation Patterns

### Basic RAG Pattern

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Create a retriever from vector store
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Define prompt template
template = """
You are an assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Define document formatter
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Create RAG chain
llm = ChatOpenAI(temperature=0)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Run the chain
response = rag_chain.invoke("What is SAP HANA Cloud?")
print(response)
```

### Filtered Search Pattern

```python
# Metadata-filtered search
results = vector_store.similarity_search(
    "cloud database capabilities",
    filter={"category": "database"}
)

# Complex filtering with logical operators
results = vector_store.similarity_search(
    "cloud database capabilities",
    filter={
        "$and": [
            {"category": "database"},
            {"year": {"$gt": 2020}},
            {"$or": [
                {"topic": {"$contains": "cloud"}},
                {"topic": {"$contains": "vector"}}
            ]}
        ]
    }
)
```

### MMR Search for Diverse Results

```python
# Maximal Marginal Relevance search for diverse results
results = vector_store.max_marginal_relevance_search(
    "cloud database features",
    k=5,                  # Number of results to return
    fetch_k=20,           # Number of results to fetch before applying MMR
    lambda_mult=0.5       # Diversity parameter (0.0 = max diversity, 1.0 = max relevance)
)
```

## Domain-Specific Embeddings

### Financial Embeddings with FinE5

```python
from langchain_hana.financial import create_financial_embeddings, FinE5Embeddings

# Create financial embeddings with default settings
embeddings = create_financial_embeddings(
    model_type="default",         # Options: default, high_quality, efficient, tone, financial_bert, finance_base
    use_gpu=True,                 # Use GPU if available
    add_financial_prefix=True,    # Add financial context prefix for better results
    enable_caching=True           # Enable caching for improved performance
)

# Create vector store with financial embeddings
vector_store = HanaVectorStore(
    connection=connection,
    embedding=embeddings,
    table_name="FINANCIAL_DOCUMENTS"
)

# Add financial documents
financial_texts = [
    "Q1 2025 Financial Results: Company XYZ reported revenue of $1.2 billion, up 15% year-over-year.",
    "Market volatility remains elevated due to geopolitical tensions and inflationary pressures.",
    "The board approved a quarterly dividend of $0.45 per share."
]

metadata = [
    {"type": "earnings_report", "quarter": "Q1", "year": 2025},
    {"type": "risk_report", "date": "2025-04-15"},
    {"type": "dividend_announcement", "date": "2025-04-15"}
]

vector_store.add_texts(financial_texts, metadatas=metadata)
```

### GPU Acceleration with TensorRT

For high-throughput scenarios, you can use TensorRT acceleration:

```python
from langchain_hana.financial import FinE5TensorRTEmbeddings

# Create TensorRT-accelerated financial embeddings
embeddings = FinE5TensorRTEmbeddings(
    model_type="default",
    precision="fp16",        # Options: fp32, fp16, int8
    multi_gpu=True,          # Use multiple GPUs if available
    add_financial_prefix=True,
    enable_caching=True
)

# Run performance benchmark
benchmark_results = embeddings.benchmark(batch_sizes=[16, 32, 64])
print(f"Performance: {benchmark_results['documents_per_second']} docs/sec")
```

## Performance Optimization

### 1. HNSW Indexing

Create an HNSW index for faster vector searches:

```python
# Create HNSW index with custom parameters
vector_store.create_hnsw_index(
    m=16,                  # Number of connections per node
    ef_construction=128,   # Index building quality parameter
    ef_search=64           # Search quality parameter
)
```

### 2. Embedding Caching

Use embedding caching to improve performance:

```python
from langchain_hana.embeddings import HanaEmbeddingsCache

# Create base embeddings model
base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create cached embeddings with 1-hour TTL and disk persistence
cached_embeddings = HanaEmbeddingsCache(
    base_embeddings=base_embeddings,
    ttl_seconds=3600,  # 1 hour cache lifetime
    max_size=10000,
    persist_path="/path/to/cache.pkl"  # Optional: persist cache to disk
)

# Use like any other embeddings model
vector_store = HanaVectorStore(
    connection=connection,
    embedding=cached_embeddings,
    table_name="CACHED_VECTORS"
)
```

### 3. Batch Processing

For large document sets, use batch processing:

```python
# Process documents in batches
batch_size = 100
all_texts = [...]  # Large list of documents
all_metadata = [...]  # Corresponding metadata

for i in range(0, len(all_texts), batch_size):
    batch_texts = all_texts[i:i+batch_size]
    batch_metadata = all_metadata[i:i+batch_size]
    vector_store.add_texts(batch_texts, metadatas=batch_metadata)
    print(f"Processed batch {i//batch_size + 1}/{(len(all_texts)-1)//batch_size + 1}")
```

## Example Applications

### 1. Basic RAG Application

See the complete example in `examples/langchain_hana_rag_example.py`.

This example demonstrates:
- Setting up a vector store with SAP HANA Cloud
- Adding documents with metadata
- Creating a retrieval chain
- Running interactive Q&A

### 2. Financial Domain RAG

See the complete example in `examples/financial_rag_example.py`.

This example demonstrates:
- Using FinE5 financial domain-specific embeddings
- Building a RAG system for financial documents
- Evaluating retrieval quality with financial metrics
- Interactive financial Q&A

### 3. Advanced GPU Acceleration

For high-performance scenarios, see:
- `examples/gpu_vectorstore_example.py`: Using GPU-accelerated embeddings
- `examples/tensorrt_enhanced_embeddings_example.py`: TensorRT acceleration
- `examples/multi_gpu_embeddings_demo.py`: Multi-GPU parallelization

## Advanced Features

### Using SAP HANA's Internal Embeddings

```python
from langchain_hana.embeddings import HanaInternalEmbeddings

# Initialize with SAP HANA's internal embedding model ID
embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")

# Create vector store with internal embeddings
vector_store = HanaVectorStore(
    connection=connection,
    embedding=embeddings,
    table_name="INTERNAL_VECTORS"
)

# When similarity_search is called, embedding generation happens
# directly in the database for maximum performance
results = vector_store.similarity_search("What is SAP HANA?")
```

### Document Update Operations

```python
# Update existing documents by filter
vector_store.update_texts(
    texts=["Updated document content"],
    filter={"source": "wiki", "id": "doc123"},
    metadatas=[{"source": "wiki", "id": "doc123", "updated": True}]
)

# Upsert operation - update if exists, insert if not
vector_store.upsert_texts(
    texts=["Document content"],
    filter={"id": "doc456"},
    metadatas=[{"id": "doc456", "source": "wiki"}]
)

# Delete documents by filter
vector_store.delete(filter={"source": "wiki", "outdated": True})
```

### Lineage Tracking and Audit Logging

```python
# Enable lineage tracking and audit logging
vector_store = HanaVectorStore(
    connection=connection,
    embedding=embeddings,
    table_name="AUDITED_VECTORS",
    enable_lineage=True,
    enable_audit_logging=True,
    audit_log_to_database=True,
    current_user_id="user123",
    current_application="myapp"
)

# All operations will be tracked with user and application context
```

## Troubleshooting

### Connection Issues

If you encounter connection issues:

1. Verify your connection parameters (host, port, user, password)
2. Check network connectivity and firewall rules
3. Verify that your SAP HANA Cloud instance is running
4. Use the `test_connection` function to diagnose issues:

```python
from langchain_hana.connection import create_connection, test_connection

connection = create_connection(
    host="your-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password"
)

is_valid, info = test_connection(connection)
if not is_valid:
    print(f"Connection error: {info.get('error')}")
else:
    print(f"Connected to SAP HANA Cloud {info.get('version')}")
    print(f"Current schema: {info.get('current_schema')}")
```

### Vector Store Issues

Common issues with vector stores:

1. **Table creation failures**: Check permissions and schema names
2. **Embedding dimension mismatch**: Ensure your embeddings have consistent dimensions
3. **Performance issues**: Create HNSW indexes for large tables
4. **Out of memory errors**: Use batch processing for large document sets

### GPU Acceleration Issues

If TensorRT acceleration fails:

1. Check that TensorRT is properly installed
2. Verify GPU compatibility and driver versions
3. Try falling back to standard PyTorch acceleration
4. Use the diagnostic functions:

```python
from langchain_hana.gpu.tensorrt_diagnostics import TensorRTDiagnostics

# Run diagnostics to check environment
diagnostics = TensorRTDiagnostics.run_diagnostics()
print(diagnostics.get_summary())

# Check if TensorRT is available
is_available, version, details = diagnostics.check_tensorrt_available()
print(f"TensorRT available: {is_available}, version: {version}")
```

## Additional Resources

- [README_LANGCHAIN.md](./README_LANGCHAIN.md): Overview of the LangChain integration
- [Examples directory](./examples): Complete code examples
- [SAP HANA Cloud Documentation](https://help.sap.com/docs/hana-cloud): Official SAP HANA Cloud documentation
- [LangChain Documentation](https://python.langchain.com/docs/get_started): LangChain framework documentation