# Apache Arrow Flight Integration for SAP HANA Cloud

This document describes the integration of Apache Arrow Flight with the SAP HANA Cloud LangChain integration, which enables high-performance vector search and data transfer.

## Overview

Apache Arrow Flight is a protocol for high-performance data transfer that uses the Arrow columnar memory format. This integration enables:

1. Zero-copy data transfer between the client and server
2. GPU-accelerated vector search without CPU-GPU data transfer overhead
3. Efficient batch processing of vector operations
4. Multi-GPU support for distributed workloads

## Architecture

The integration consists of the following components:

1. **Arrow Flight Server**: Implements the Flight protocol server that accepts requests for vector operations
2. **Arrow Flight Client**: Client-side library for interacting with the Flight server
3. **GPU-Aware Memory Manager**: Manages GPU memory allocation and zero-copy transfers
4. **Vector Serialization**: Handles efficient serialization and deserialization of vector data
5. **Arrow Flight VectorStore**: LangChain compatible VectorStore implementation using Arrow Flight

## Deployment

### Docker Deployment

We provide multiple Docker deployment options:

1. **Standalone Arrow Flight API**:
   ```
   docker-compose -f docker-compose.arrow-flight.yml up -d
   ```

2. **Unified Application (API + Frontend)**:
   ```
   docker-compose -f docker-compose.unified.yml up -d
   ```

### Configuration

Configure the Arrow Flight server using these environment variables:

- `FLIGHT_AUTO_START`: Set to "true" to automatically start the Flight server with the API
- `FLIGHT_HOST`: Host address to bind the Flight server (default: "0.0.0.0")
- `FLIGHT_PORT`: Port for the Flight server (default: 8815)
- `FLIGHT_LOCATION`: Optional URI for the Flight server location
- `FLIGHT_GPU_MEMORY_FRACTION`: Fraction of GPU memory to use (default: 0.8)

## Usage

### Python Client

```python
from langchain_hana.gpu.arrow_flight_client import ArrowFlightClient
from langchain_hana.gpu.arrow_flight_vectorstore import ArrowFlightVectorStore

# Connect to Flight server
client = ArrowFlightClient(host="localhost", port=8815)

# Create a vector store
vectorstore = ArrowFlightVectorStore(
    client=client,
    embedding_dimension=384,
    table_name="my_vectors"
)

# Add documents
vectorstore.add_texts(
    texts=["Document 1", "Document 2"],
    metadatas=[{"source": "file1"}, {"source": "file2"}]
)

# Search
results = vectorstore.similarity_search("query", k=2)
```

### Performance Considerations

- Enable batch processing for optimal performance
- Configure GPU memory fraction based on your workload
- Use the multi-GPU manager for distributing workloads across multiple GPUs

## Integration with NVIDIA Blueprint

This integration is designed to work with NVIDIA Blueprint deployments. The unified Docker container includes:

1. GPU-accelerated API with Arrow Flight support
2. React-based frontend for user interaction
3. Optimized TensorRT models for inference

## Monitoring and Management

- API endpoint: `/health` - Check API health
- API endpoint: `/flight/info` - Get Flight server information
- API endpoint: `/gpu/info` - Get GPU information and utilization

## Troubleshooting

1. **Connection Issues**: Ensure the Flight server port (8815) is accessible
2. **Memory Issues**: Adjust `FLIGHT_GPU_MEMORY_FRACTION` to control GPU memory usage
3. **Performance Issues**: Enable batch processing and check GPU utilization

## References

- [Apache Arrow Flight Documentation](https://arrow.apache.org/docs/python/flight.html)
- [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [LangChain Vector Stores](https://python.langchain.com/docs/modules/data_connection/vectorstores/)