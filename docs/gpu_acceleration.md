# GPU Acceleration for SAP HANA Cloud Vector Store

This document provides information about the GPU acceleration capabilities for the SAP HANA Cloud LangChain integration.

## Overview

The LangChain integration for SAP HANA Cloud includes GPU acceleration capabilities that significantly improve performance for vector operations, especially for large collections of documents. The implementation leverages NVIDIA GPUs to accelerate vector operations directly in the data layer, minimizing data transfer between the database and application servers.

## Key Features

- **GPU-accelerated similarity search**: Faster query processing using GPU parallel computing
- **GPU-accelerated MMR search**: Diverse results with optimized performance
- **Memory-efficient vector operations**: Efficient handling of large vector collections
- **Multiple execution modes**: Full GPU, hybrid, or database fallback
- **Async support**: Non-blocking operations for high-throughput applications
- **Complete CRUD operations**: Full create, read, update, and delete capabilities
- **Performance profiling**: Built-in profiling and monitoring tools
- **Automatic batch processing**: Optimized handling of large document collections

## Requirements

- SAP HANA Cloud instance
- NVIDIA GPU with CUDA support
- Python packages:
  - `langchain-hana`
  - `torch` (with CUDA support)
  - `faiss-gpu` (optional, for index acceleration)

## Usage

### Basic Usage

```python
from hdbcli import dbapi
from langchain_hana.gpu.hana_gpu_vectorstore import HanaGPUVectorStore
from langchain_hana.utils import DistanceStrategy
from sentence_transformers import SentenceTransformer

# Connect to SAP HANA Cloud
connection = dbapi.connect(
    address="your-hana-host",
    port=443,
    user="your-user",
    password="your-password",
    encrypt=True,
    sslValidateCertificate=False,
)

# Initialize embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize GPU-accelerated vector store
gpu_vectorstore = HanaGPUVectorStore(
    connection=connection,
    embedding=embedding_model,
    table_name="EMBEDDINGS",
    distance_strategy=DistanceStrategy.COSINE,
    gpu_acceleration_config={
        "use_gpu_batching": True,
        "embedding_batch_size": 32,
        "build_index": True,
    }
)

# Add documents
documents = ["Document 1", "Document 2", "Document 3"]
metadata = [{"source": "source1"}, {"source": "source2"}, {"source": "source3"}]
gpu_vectorstore.add_texts(documents, metadata)

# Search for similar documents
results = gpu_vectorstore.similarity_search("Query text", k=3)

# Get diverse results with MMR
diverse_results = gpu_vectorstore.max_marginal_relevance_search("Query text", k=3, fetch_k=10)
```

### Async Operations

```python
import asyncio

async def main():
    # Initialize GPU vector store (as above)
    
    # Add documents asynchronously
    await gpu_vectorstore.aadd_texts(documents, metadata)
    
    # Search asynchronously
    results = await gpu_vectorstore.asimilarity_search("Query text", k=3)
    
    # Run multiple queries in parallel
    results = await asyncio.gather(
        gpu_vectorstore.asimilarity_search("Query 1", k=3),
        gpu_vectorstore.asimilarity_search("Query 2", k=3),
        gpu_vectorstore.asimilarity_search("Query 3", k=3)
    )

asyncio.run(main())
```

## Configuration Options

The `gpu_acceleration_config` parameter accepts the following options:

| Option | Description | Default |
|--------|-------------|---------|
| `gpu_ids` | List of GPU IDs to use | `None` (use all available) |
| `memory_limit_gb` | Maximum GPU memory to use in GB | `4.0` |
| `precision` | Computation precision (`float32`, `float16`, `int8`) | `float32` |
| `enable_tensor_cores` | Use Tensor Cores if available | `True` |
| `use_gpu_batching` | Process large document collections in batches | `True` |
| `embedding_batch_size` | Batch size for embedding generation | `32` |
| `db_batch_size` | Batch size for database operations | `1000` |
| `build_index` | Build an index on initialization | `False` |
| `index_type` | Type of index to build (`hnsw` or `flat`) | `hnsw` |
| `rebuild_index_on_add` | Rebuild index after adding documents | `False` |
| `rebuild_index_on_update` | Rebuild index after updating documents | `False` |
| `rebuild_index_on_delete` | Rebuild index after deleting documents | `False` |
| `prefetch_size` | Number of vectors to prefetch | `100000` |

## Execution Modes

The GPU-accelerated vector store supports multiple execution modes:

### Full GPU Mode

In this mode, all vector operations are performed on the GPU:

```python
# Example of full GPU mode
results = gpu_vectorstore.similarity_search(
    query="Query text",
    k=3,
    kwargs={"fetch_all_vectors": True}
)
```

### Hybrid Mode (Default)

In hybrid mode, the database is used for filtering, and the GPU is used for similarity calculations:

```python
# Example of hybrid mode (default)
results = gpu_vectorstore.similarity_search(
    query="Query text",
    k=3,
    filter={"category": "technology"}
)
```

### Database Fallback Mode

If GPU acceleration is not available, operations automatically fall back to the database:

```python
# If no GPU is available, this will use the database
results = gpu_vectorstore.similarity_search(
    query="Query text",
    k=3
)
```

## Performance Monitoring

The GPU-accelerated vector store includes built-in performance monitoring:

```python
# Enable performance profiling
gpu_vectorstore.enable_profiling(True)

# Perform operations...

# Get performance statistics
stats = gpu_vectorstore.get_performance_stats()
print(stats)

# Get GPU information
gpu_info = gpu_vectorstore.get_gpu_info()
print(gpu_info)

# Reset performance statistics
gpu_vectorstore.reset_performance_stats()
```

## Examples

Check the `examples` directory for complete examples:

- `gpu_vectorstore_example.py`: Performance comparison between CPU and GPU implementations
- `gpu_vectorstore_api.py`: FastAPI web service using GPU-accelerated vector store

## Best Practices

- **Memory Management**: Set appropriate memory limits for your GPU to avoid out-of-memory errors
- **Batch Processing**: Use batch processing for large document collections
- **Index Building**: Build an index for large collections to improve search performance
- **Async Operations**: Use async methods for high-throughput applications
- **Monitoring**: Use the built-in performance monitoring to identify bottlenecks

## Troubleshooting

### GPU Not Available

If GPU acceleration is not available, check the following:

- Ensure that PyTorch is installed with CUDA support
- Check that your GPU drivers are installed and up-to-date
- Verify that the GPU is recognized by the system

### Out of Memory Errors

If you encounter out-of-memory errors:

- Reduce the `memory_limit_gb` value
- Decrease the batch size for processing
- Use hybrid mode instead of full GPU mode for large collections

### Performance Issues

If performance is not as expected:

- Enable profiling to identify bottlenecks
- Adjust batch sizes for optimal performance
- Consider building an index for large collections
- Use appropriate precision settings based on your workload

## FAQ

### Can I use multiple GPUs?

Yes, you can specify the GPUs to use in the `gpu_ids` parameter:

```python
gpu_vectorstore = HanaGPUVectorStore(
    # ... other parameters ...
    gpu_acceleration_config={
        "gpu_ids": [0, 1, 2],  # Use GPUs 0, 1, and 2
    }
)
```

### Is GPU acceleration required?

No, the vector store will automatically fall back to CPU/database operations if GPU acceleration is not available.

### How do I check if GPU acceleration is being used?

You can check the `gpu_available` property of the GPU information:

```python
gpu_info = gpu_vectorstore.get_gpu_info()
print(f"GPU available: {gpu_info['gpu_available']}")
```

### Can I use GPU acceleration in a production environment?

Yes, the GPU acceleration is designed for production use, with:

- Automatic fallback mechanisms
- Memory management
- Performance monitoring
- Batch processing for large collections
- Async operations for high-throughput applications
EOL < /dev/null