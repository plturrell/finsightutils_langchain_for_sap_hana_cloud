# LangChain Integration for SAP HANA Cloud - Implementation Summary

This document provides a technical summary of the LangChain integration for SAP HANA Cloud, including its components, features, and usage examples.

## Components

The integration consists of the following key components:

### 1. Vector Store

The core of the integration is the `HanaVectorStore` class, which implements LangChain's `VectorStore` interface:

```python
from langchain_hana.vectorstore import HanaVectorStore
```

Key features:
- Store and retrieve embeddings in SAP HANA Cloud
- Multiple distance metrics (cosine, euclidean, etc.)
- Rich metadata filtering
- Maximal Marginal Relevance (MMR) search for diverse results
- HNSW indexing for performance optimization

### 2. Embeddings

Two embedding classes are provided:

```python
from langchain_hana.embeddings import HanaInternalEmbeddings, HanaEmbeddingsCache
```

- `HanaInternalEmbeddings`: Uses SAP HANA's internal embedding functionality
- `HanaEmbeddingsCache`: Caching layer for embedding models to improve performance

### 3. Connection Management

Utilities for connecting to SAP HANA Cloud:

```python
from langchain_hana.connection import create_connection, test_connection, get_connection
```

Features:
- Multiple connection methods (parameters, environment variables, config file)
- Connection testing and validation
- Connection pooling and management

### 4. Utilities

Various utility functions for working with vectors and SAP HANA:

```python
from langchain_hana.utils import (
    DistanceStrategy,
    serialize_vector,
    deserialize_vector,
    create_vector_table,
    create_hnsw_index
)
```

## Usage Examples

### Basic Usage

```python
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_hana.vectorstore import HanaVectorStore

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vector_store = HanaVectorStore(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password",
    embedding=embeddings,
    table_name="MY_VECTORS",
    create_table=True
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

### Using SAP HANA's Internal Embeddings

```python
from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.vectorstore import HanaVectorStore

# Initialize with SAP HANA's internal embedding model ID
embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")

# Create vector store with internal embeddings
vector_store = HanaVectorStore(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password",
    embedding=embeddings,
    table_name="MY_VECTORS",
    create_table=True
)

# Embeddings are generated directly in the database
results = vector_store.similarity_search("What is SAP HANA?")
```

### RAG Application

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_hana.vectorstore import HanaVectorStore

# Set up vector store
embeddings = OpenAIEmbeddings()
vector_store = HanaVectorStore(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password",
    embedding=embeddings,
    table_name="RAG_STORE"
)

# Add documents
vector_store.add_texts([
    "SAP HANA Cloud is a cloud-based in-memory database",
    "Vector databases store data as high-dimensional vectors"
])

# Create a retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Define the prompt template
template = """
You are an assistant that answers questions based on the provided context.

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

# Define the processing function for formatting context
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Create the RAG chain
llm = ChatOpenAI()
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Get answer
answer = rag_chain.invoke("What is SAP HANA Cloud?")
print(answer)
```

## Example Scripts

The integration includes several example scripts:

1. `examples/langchain_hana_quickstart.py`: Basic usage example
2. `examples/langchain_hana_rag_example.py`: Retrieval-Augmented Generation example
3. `examples/langchain_hana_gpu_quickstart.py`: GPU-accelerated embeddings example

## Advanced Features

### Metadata Filtering

```python
# Simple filter
results = vector_store.similarity_search(
    "your query",
    filter={"category": "database"}
)

# Complex filter
results = vector_store.similarity_search(
    "your query",
    filter={
        "category": "database",
        "year": {"$gt": 2020},
        "$or": [
            {"tags": {"$contains": "cloud"}},
            {"tags": {"$contains": "vector"}}
        ]
    }
)
```

### Maximal Marginal Relevance (MMR) Search

```python
# Search with MMR to ensure diverse results
results = vector_store.max_marginal_relevance_search(
    "your query",
    k=5,  # Number of results to return
    fetch_k=20,  # Number of results to fetch before applying MMR
    lambda_mult=0.5  # Diversity factor (0.0 = max diversity, 1.0 = max relevance)
)
```

### Embedding Caching

```python
from langchain_core.embeddings import HuggingFaceEmbeddings
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
embedding = cached_embeddings.embed_query("Hello, world!")
```

## Performance Optimization

1. **HNSW Indexing**: Create an HNSW index for faster similarity search

```python
vector_store.create_hnsw_index()
```

2. **Batch Processing**: Use batch operations for efficient document insertion

```python
vector_store.add_texts(large_document_list, metadata_list)
```

3. **GPU Acceleration**: Use GPU-accelerated embeddings for improved performance

```python
from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings

embeddings = HanaTensorRTEmbeddings(
    model_name="all-MiniLM-L6-v2",
    batch_size=32,
    half_precision=True,
    device="cuda"
)
```

## Testing

The integration includes comprehensive unit tests in the `tests/test_vectorstore.py` file, covering:

1. Connection management
2. Vector serialization/deserialization
3. Basic vectorstore operations (add, search, delete)
4. Advanced search (MMR, filtering)
5. Distance strategies
6. Embedding models and caching

## Next Steps

1. **Explore Examples**: Check out the example scripts in the `examples/` directory
2. **Try with Your Data**: Adapt the examples to your own use case
3. **Optimize Performance**: Experiment with different embedding models and indexing strategies
4. **Integrate with Applications**: Use the integration in your LangChain applications