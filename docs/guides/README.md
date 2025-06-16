# LangChain Integration for SAP HANA Cloud

A production-grade integration between LangChain and SAP HANA Cloud's vector database capabilities for building robust AI applications.

## Features

- **High-Performance Vector Operations**: Optimized for production workloads
- **Advanced Connection Management**: Connection pooling, automatic reconnection
- **Comprehensive Error Handling**: Detailed error information and recovery
- **Flexible Query Capabilities**: Complex filtering, MMR search, async operations
- **GPU Acceleration Support**: TensorRT integration for high-throughput embedding
- **Production-Ready Architecture**: Monitoring, logging, reliability features
- **Optimized for SAP HANA Cloud**: Leverages native vector capabilities
- **Financial Domain Embeddings**: Specialized models for financial applications
- **Model Fine-Tuning**: Customize embedding models for your specific domain

## Installation

```bash
pip install langchain-hana-integration
```

## Quickstart

```python
from langchain_hana_integration import SAP_HANA_VectorStore, HanaOptimizedEmbeddings
from langchain_hana_integration.connection import create_connection_pool

# Configure connection
connection_params = {
    "address": "your-hana-instance.hanacloud.ondemand.com",
    "port": 443,
    "user": "DBADMIN",
    "password": "your-password"
}

# Create connection pool
create_connection_pool(connection_params=connection_params)

# Initialize embedding model
embedding_model = HanaOptimizedEmbeddings(
    model_name="all-MiniLM-L6-v2",
    enable_caching=True
)

# Create vector store
vector_store = SAP_HANA_VectorStore(
    embedding=embedding_model,
    table_name="LANGCHAIN_VECTORS",
    auto_create_index=True
)

# Add documents
documents = [
    "SAP HANA Cloud is a cloud-based database management system.",
    "LangChain is a framework for developing applications powered by LLMs."
]

metadata = [
    {"source": "SAP Documentation", "category": "database"},
    {"source": "LangChain Documentation", "category": "framework"}
]

vector_store.add_texts(documents, metadata)

# Search for similar documents
results = vector_store.similarity_search(
    "What is SAP HANA Cloud?",
    k=2
)

# Print results
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print()
```

## Advanced Usage

### Connection Management

```python
from langchain_hana_integration.connection import create_connection_pool, get_connection

# Create connection pool with advanced options
create_connection_pool(
    connection_params=connection_params,
    pool_name="my_pool",
    min_connections=2,
    max_connections=10
)

# Use connection context manager
with get_connection("my_pool") as connection:
    # Use connection...
    pass
```

### Advanced Search Features

```python
# Filtered search
filtered_results = vector_store.similarity_search(
    "cloud database",
    k=5,
    filter={"category": "database", "year": {"$gte": 2023}}
)

# Maximal Marginal Relevance (diverse results)
diverse_results = vector_store.max_marginal_relevance_search(
    "cloud database",
    k=5,
    fetch_k=20,
    lambda_mult=0.7  # Higher values prioritize relevance over diversity
)

# Asynchronous operations
import asyncio

async def search_multiple_queries():
    results = await asyncio.gather(
        vector_store.asimilarity_search("What is SAP HANA?"),
        vector_store.asimilarity_search("What is LangChain?"),
        vector_store.asimilarity_search("How do vector databases work?")
    )
    return results

# Run async function
asyncio.run(search_multiple_queries())
```

### Document Management

```python
# Update documents
vector_store.update_texts(
    ["Updated description of SAP HANA Cloud"],
    filter={"source": "SAP Documentation"},
    metadatas=[{"source": "SAP Documentation", "category": "database", "updated": True}]
)

# Delete documents
vector_store.delete(
    filter={"category": "outdated"}
)
```

### Optimized Embeddings

```python
from langchain_hana_integration import HanaOptimizedEmbeddings

# Create optimized embeddings with caching
embeddings = HanaOptimizedEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_dir="./embedding_cache",
    enable_caching=True,
    memory_cache_size=10000,
    batch_size=32
)

# Get performance metrics
metrics = embeddings.get_metrics()
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2f}")
print(f"Average time per call: {metrics['avg_time_per_call']:.4f}s")
```

## SAP HANA Internal Embeddings

```python
from langchain_hana_integration import HanaOptimizedEmbeddings, SAP_HANA_VectorStore

# Use SAP HANA's internal embedding function
embeddings = HanaOptimizedEmbeddings(
    internal_embedding_model_id="SAP_NEB.20240715"
)

vector_store = SAP_HANA_VectorStore(
    embedding=embeddings,
    table_name="INTERNAL_EMBEDDINGS"
)

# Embeddings will be generated directly in the database
vector_store.add_texts(documents, metadata)
```

## Performance Monitoring

```python
# Get vector store performance metrics
metrics = vector_store.get_metrics()
print(f"Total documents added: {metrics['total_documents_added']}")
print(f"Average search time: {metrics['avg_search_time']:.4f}s")
```

## Error Handling

```python
from langchain_hana_integration.exceptions import ConnectionError, DatabaseError, VectorOperationError

try:
    results = vector_store.similarity_search("cloud database")
except ConnectionError as e:
    print(f"Connection error: {e}")
    # Handle connection issues
except VectorOperationError as e:
    print(f"Vector operation error: {e}")
    print(f"Details: {e.details}")
    # Handle vector operation issues
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

## Configuration

The integration can be configured through various options:

- Connection parameters
- Table and column names
- Distance strategies
- Batch sizes and timeouts
- Logging and monitoring
- Vector column types

## Financial Domain Models

For financial applications, we provide specialized embedding models:

```python
from langchain_hana.financial import create_financial_system

# Create financial embedding system
system = create_financial_system(
    host="your-hana-host.ondemand.com",
    port=443,
    user="your-user",
    password="your-password",
    model_name="FinMTEB/Fin-E5",  # Financial domain model
    table_name="FINANCIAL_DOCUMENTS"
)

# Add financial documents
system.add_documents(financial_documents)

# Search with financial domain understanding
results = system.similarity_search(
    "What market risks are mentioned in the quarterly report?",
    filter={"type": "quarterly_report"}
)
```

## ✨ Essence: The Foundation of Understanding

Experience a fundamental reimagining of how we work with financial language models:

```bash
# Absorb financial understanding
./essence.py absorb documents.json

# Transform the model's understanding
./essence.py enlighten

# Apply understanding to a question
./essence.py contemplate "What market risks are mentioned?"

# Reflect on the transformation
./essence.py reflect
```

Essence isn't just an interface; it's a philosophy that creates true harmony between language and implementation. Each concept in the code embodies its name - Understanding objects hold concepts and their relationships, Enlightenment transforms a model's perspective, and Contemplation finds meaning in financial questions.

See `ESSENCE.md` for the philosophy behind this approach.

For technical details on customizing financial models, see `FINE_TUNING_GUIDE.md`.

## Requirements

- Python 3.8+
- SAP HANA Cloud instance
- `hdbcli` Python client for SAP HANA
- `langchain` and `langchain-core`
- `sentence-transformers` (for embedding generation)
- `numpy` (for vector operations)
- `torch` (for GPU acceleration and fine-tuning)

## License

Apache 2.0
EOF < /dev/null