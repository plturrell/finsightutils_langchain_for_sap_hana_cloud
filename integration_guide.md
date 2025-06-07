# LangChain Integration Guide for SAP HANA Cloud

This guide explains how to use the LangChain integration with SAP HANA Cloud for vector search and embedding operations.

## Overview

The `langchain-integration-for-sap-hana-cloud` package provides a seamless integration between LangChain and SAP HANA Cloud, enabling you to:

1. Store and retrieve vector embeddings in SAP HANA Cloud Vector Engine
2. Use SAP HANA's built-in embedding functions or external embedding models
3. Perform similarity searches with filtering capabilities
4. Leverage SAP HANA's HNSW vector indexing for improved performance
5. Use Maximal Marginal Relevance (MMR) for diverse search results

## Installation

```bash
pip install langchain-integration-for-sap-hana-cloud
```

This will install the package and its dependencies, including the required `hdbcli` package for connecting to SAP HANA Cloud.

## Quick Start

Here's a basic example of how to use the integration:

```python
from hdbcli import dbapi
from langchain_core.documents import Document
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_hana import HanaDB

# Connect to SAP HANA Cloud
connection = dbapi.connect(
    address="your-hana-host.com",
    port=443,
    user="YOUR_USERNAME",
    password="YOUR_PASSWORD",
    encrypt=True,
    sslValidateCertificate=False
)

# Initialize an embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vector store
vector_store = HanaDB(
    connection=connection,
    embedding=embedding_model,
    table_name="MY_EMBEDDINGS"
)

# Add documents to the vector store
documents = [
    "SAP HANA Cloud is an in-memory database as a service.",
    "Vector search capabilities enable semantic search in SAP HANA.",
    "LangChain provides tools for building LLM-powered applications."
]
metadata = [
    {"source": "SAP Documentation", "category": "database"},
    {"source": "Technical Blog", "category": "search"},
    {"source": "LangChain Docs", "category": "framework"}
]

vector_store.add_texts(documents, metadatas=metadata)

# Create an HNSW index for faster searches
vector_store.create_hnsw_index()

# Perform a similarity search
results = vector_store.similarity_search(
    query="How does semantic search work?",
    k=2,
    filter={"category": "search"}
)

# Display results
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()
```

## Using SAP HANA's Internal Embeddings

SAP HANA Cloud includes built-in embedding functionality through the `VECTOR_EMBEDDING` function. This approach offers several advantages:

1. Better performance as embeddings are generated directly in the database
2. Reduced data transfer between application and database
3. Leverage SAP HANA's optimized implementation

To use internal embeddings:

```python
from langchain_hana import HanaDB
from langchain_hana.embeddings import HanaInternalEmbeddings

# Create embedding instance using SAP HANA's internal embedding model
internal_embeddings = HanaInternalEmbeddings(
    internal_embedding_model_id="SAP_NEB.20240715"  # Use appropriate model ID for your instance
)

# Create vector store with internal embeddings
vector_store = HanaDB(
    connection=connection,
    embedding=internal_embeddings,
    table_name="MY_EMBEDDINGS"
)

# Add documents - embeddings will be generated in the database
vector_store.add_texts(documents, metadatas=metadata)

# Search - query embedding will also be generated in the database
results = vector_store.similarity_search("How does semantic search work?")
```

## Advanced Usage

### 1. Filtering

You can filter search results based on metadata:

```python
# Simple equality filter
results = vector_store.similarity_search(
    query="database capabilities",
    filter={"category": "database"}
)

# Complex filter with logical operators
results = vector_store.similarity_search(
    query="database capabilities",
    filter={
        "$and": [
            {"category": "database"},
            {"$or": [
                {"source": {"$contains": "Documentation"}},
                {"year": {"$gt": 2022}}
            ]}
        ]
    }
)
```

### 2. Maximal Marginal Relevance (MMR)

MMR helps you get more diverse search results by considering both relevance and diversity:

```python
results = vector_store.max_marginal_relevance_search(
    query="cloud database",
    k=5,                # Number of final results
    fetch_k=20,         # Initial fetch size before diversity calculation
    lambda_mult=0.5     # Balance between relevance (1.0) and diversity (0.0)
)
```

### 3. HNSW Indexing

Hierarchical Navigable Small World (HNSW) indexing significantly improves search performance:

```python
# Create HNSW index with custom parameters
vector_store.create_hnsw_index(
    m=16,                # Maximum number of connections per node (4-1000)
    ef_construction=100, # Search width during construction (1-100000)
    ef_search=50,        # Search width during queries (1-100000)
    index_name="my_custom_index"  # Optional custom index name
)
```

### 4. Specific Metadata Columns

For frequently filtered metadata fields, you can create dedicated columns for better performance:

```python
# Define specific metadata columns during initialization
vector_store = HanaDB(
    connection=connection,
    embedding=embedding_model,
    table_name="MY_EMBEDDINGS",
    specific_metadata_columns=["category", "source", "year"]
)

# Add documents with metadata that will be stored in dedicated columns
vector_store.add_texts(
    texts=["Document content here"],
    metadatas=[{
        "category": "database",
        "source": "Documentation",
        "year": 2023,
        "other_field": "This goes in the JSON metadata column"
    }]
)
```

### 5. Vector Column Types

SAP HANA Cloud supports different vector column types:

```python
# Using REAL_VECTOR (standard 32-bit precision)
vector_store = HanaDB(
    connection=connection,
    embedding=embedding_model,
    vector_column_type="REAL_VECTOR"  # Default, available in HANA Cloud QRC 1/2024+
)

# Using HALF_VECTOR (compressed 16-bit precision, reduced storage)
vector_store = HanaDB(
    connection=connection,
    embedding=embedding_model,
    vector_column_type="HALF_VECTOR"  # Available in HANA Cloud QRC 2/2025+
)
```

## Working with Large Document Sets

For large document sets, batch processing is recommended:

```python
# Process documents in batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch_docs = documents[i:i+batch_size]
    batch_metadata = metadata[i:i+batch_size] if metadata else None
    
    # Add batch to vector store
    vector_store.add_texts(batch_docs, metadatas=batch_metadata)
    
    print(f"Processed batch {i//batch_size + 1}, documents {i+1}-{min(i+batch_size, len(documents))}")
```

## GPU Acceleration

If you're running on a system with NVIDIA GPUs, this integration supports GPU acceleration for embedding generation:

```python
from langchain_hana.gpu.hana_tensorrt_embeddings import TensorRTEmbedding
from langchain_hana.gpu.hana_tensorrt_vectorstore import TensorRTVectorStore

# Initialize TensorRT-optimized embeddings
trt_embeddings = TensorRTEmbedding(
    model_name="all-MiniLM-L6-v2",
    cache_folder="/path/to/tensorrt/cache"
)

# Create GPU-accelerated vector store
vector_store = TensorRTVectorStore(
    connection=connection,
    embedding=trt_embeddings,
    table_name="GPU_EMBEDDINGS",
    batch_size=32  # Adjust based on your GPU memory
)

# Use as normal - embedding generation will be GPU-accelerated
vector_store.add_texts(documents, metadatas=metadata)
results = vector_store.similarity_search("your query here")
```

## Best Practices

1. **Connection Management**: Use connection pooling for production applications
2. **Indexing**: Always create an HNSW index for tables with more than a few hundred vectors
3. **Batch Processing**: Process large document sets in batches
4. **Error Handling**: The integration provides context-aware error messages; pay attention to them for quick troubleshooting
5. **Internal Embeddings**: Use SAP HANA's internal embeddings when possible for better performance
6. **Specific Metadata Columns**: Define frequently filtered metadata fields as specific columns

## Troubleshooting

### Common Issues

1. **Connection Problems**:
   - Verify your connection parameters (host, port, credentials)
   - Ensure network access to SAP HANA Cloud is available
   - Check if your HANA instance is running

2. **Missing Vector Types**:
   - Ensure your SAP HANA Cloud version supports vector types
   - REAL_VECTOR requires SAP HANA Cloud QRC 1/2024 or newer
   - HALF_VECTOR requires SAP HANA Cloud QRC 2/2025 or newer

3. **Internal Embedding Errors**:
   - Verify your internal_embedding_model_id is valid
   - Check if the VECTOR_EMBEDDING function is available in your instance
   - Ensure you have permissions to execute the function

4. **Performance Issues**:
   - Create an HNSW index for large vector tables
   - Use specific_metadata_columns for frequently filtered fields
   - Process large document sets in batches
   - Consider GPU acceleration for embedding generation

## Additional Resources

- [SAP HANA Cloud Documentation](https://help.sap.com/docs/hana-cloud)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Vector Search in SAP HANA Cloud](https://help.sap.com/docs/hana-cloud/sap-hana-cloud-administration/vector-search)