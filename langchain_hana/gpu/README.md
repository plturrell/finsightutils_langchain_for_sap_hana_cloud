# GPU Acceleration for SAP HANA Cloud LangChain Integration

This module provides GPU-accelerated components for SAP HANA Cloud LangChain integration, optimized for NVIDIA GPUs (especially T4 GPUs) to deliver significant performance improvements for embedding generation and vector operations.

## Key Features

- **TensorRT-accelerated embedding generation**: High-performance embedding generation using NVIDIA TensorRT
- **T4 GPU Tensor Core optimizations**: Specialized optimizations for NVIDIA T4 GPUs
- **Multi-GPU support**: Intelligent workload distribution across multiple GPUs
- **Mixed-precision operations**: Support for FP32, FP16, and INT8 precision modes
- **Memory-optimized vector serialization**: Efficient vector storage and transfer
- **Dynamic batch processing**: Automatic batch size optimization based on GPU memory
- **Integration with SAP HANA vectorstore**: Seamless integration with existing components
- **Performance monitoring**: Comprehensive metrics for optimization

## Components

- **hana_tensorrt_embeddings.py**: GPU-accelerated embedding generation for SAP HANA Cloud
- **hana_tensorrt_vectorstore.py**: GPU-optimized vectorstore implementation
- **tensorrt_embeddings.py**: Base TensorRT implementation for embedding generation
- **vector_serialization.py**: Memory-efficient vector serialization utilities
- **tensor_core_optimizer.py**: Tensor Core optimizations for T4 GPUs
- **multi_gpu_manager.py**: Multi-GPU work distribution and management
- **batch_processor.py**: Dynamic batch sizing for optimal performance
- **calibration_datasets.py**: Domain-specific calibration datasets for INT8 quantization
- **accelerator.py**: Unified interface for GPU acceleration frameworks
- **utils.py**: General GPU utilities and helper functions

## Installation Requirements

- NVIDIA GPU with CUDA support (T4 recommended for optimal performance)
- CUDA Toolkit 11.8+
- PyTorch 2.0+ with CUDA support
- TensorRT 8.0+
- SAP HANA Cloud account

## Usage

### Basic Usage

```python
# Import components
from langchain_hana.gpu import HanaTensorRTEmbeddings, HanaTensorRTVectorStore

# Initialize GPU-accelerated embeddings
embeddings = HanaTensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    precision="fp16",
    multi_gpu=True
)

# Create connection to SAP HANA Cloud
from hdbcli import dbapi
conn = dbapi.connect(address='your-hana-host', port=443, user='username', password='password')

# Initialize GPU-accelerated vectorstore
vectorstore = HanaTensorRTVectorStore(
    connection=conn,
    embedding=embeddings,
    table_name="GPU_ACCELERATED_VECTORS"
)

# Add documents
documents = ["Document 1", "Document 2", "Document 3"]
metadatas = [{"source": "source1"}, {"source": "source2"}, {"source": "source3"}]
vectorstore.add_texts(documents, metadatas)

# Search for similar documents
results = vectorstore.similarity_search("What is SAP HANA Cloud?", k=5)
```

### Advanced Configuration

For advanced configuration options including multi-GPU setup, precision modes, Tensor Core optimization, and more, please refer to the [GPU Acceleration Documentation](/docs/gpu_acceleration.md).

## Performance Considerations

- **T4 GPU-Specific Optimizations**: Enables efficient use of Tensor Cores with optimized memory layouts
- **Precision Modes**: Different precision modes offer varying trade-offs between speed and accuracy
- **Batch Size Optimization**: Important for maximizing throughput and GPU utilization
- **Memory Management**: Critical for handling large embedding models and document collections

## Development and Testing

- Unit tests are available in `/tests/unit_tests/test_hana_tensorrt_components.py`
- Performance benchmarks are available in `/tests/test_tensorrt_t4.py`
- Automated testing framework in `/run_automated_tests.py` includes GPU-specific test suites

## Production Deployment

For production deployment considerations, including containerization, resource allocation, high availability, and security, please refer to the [Production Deployment Best Practices](/docs/gpu_acceleration.md#production-deployment-best-practices) section in the documentation.

## References

- [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [SAP HANA Cloud Vector Engine](https://www.sap.com/products/technology-platform/hana.html)
- [LangChain Vectorstores](https://js.langchain.com/docs/modules/data_connection/vectorstores/)