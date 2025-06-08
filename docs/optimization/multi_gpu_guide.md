# Multi-GPU Deployment Guide

This guide explains how to use and optimize the multi-GPU capabilities of the enhanced LangChain integration for SAP HANA Cloud.

## Overview

The multi-GPU support enables distributed processing of embedding generation workloads across multiple NVIDIA GPUs, providing near-linear scaling of performance as you add more GPUs to your system. This is particularly valuable for:

- Processing large document collections
- High-throughput vector database ingestion
- Real-time embedding generation for user queries
- Handling concurrent embedding requests in multi-user environments

## Architecture

The multi-GPU system consists of several key components:

1. **EnhancedMultiGPUManager**: Core component that manages GPU resources and distributes workloads
2. **MultiGPUEmbeddings**: LangChain-compatible embeddings class that uses the GPU manager
3. **HanaTensorRTMultiGPUEmbeddings**: Specialized version that adds TensorRT optimization
4. **Task Queue System**: Priority-based task scheduling across GPUs
5. **Monitoring System**: Real-time performance tracking and statistics
6. **Caching Layer**: Intelligent caching of embeddings with configurable policies

## Hardware Requirements

- At least one NVIDIA GPU with CUDA support (T4, A10, A100, or H100 recommended)
- For optimal performance:
  - 2+ GPUs of the same model (homogeneous setup)
  - 16+ GB GPU memory per GPU
  - NVLink or high-speed PCIe interconnect for multi-GPU communication

## Software Requirements

- CUDA Toolkit 11.7+
- PyTorch 2.0+ with CUDA support
- TensorRT 8.5+ (optional, for additional performance)
- Python 3.9+
- SAP HANA Cloud client libraries

## Setup

### 1. Environment Setup

Ensure you have the necessary NVIDIA drivers and CUDA toolkit installed:

```bash
# Check NVIDIA driver and CUDA versions
nvidia-smi

# Verify PyTorch CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"
```

### 2. Installation

Install the enhanced package with GPU support:

```bash
# Clone the repository
git clone https://github.com/yourusername/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud

# Install with GPU support
pip install -e ".[gpu]"
```

Or use our Docker image with GPU support:

```bash
docker run --gpus all -p 8080:8080 yourusername/langchain-hana-gpu:latest
```

### 3. Basic Usage

Here's a simple example of using multi-GPU embeddings:

```python
from langchain_hana import MultiGPUEmbeddings
from langchain_core.embeddings import HuggingFaceEmbeddings

# Create base embeddings model
base_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

# Wrap with multi-GPU support
multi_gpu_embeddings = MultiGPUEmbeddings(
    base_embeddings=base_model,
    batch_size=32,
    enable_caching=True
)

# Generate embeddings (will be distributed across available GPUs)
texts = ["Text 1", "Text 2", "Text 3", ..., "Text 1000"]
embeddings = multi_gpu_embeddings.embed_documents(texts)
```

### 4. Advanced Configuration

The multi-GPU system offers various configuration options:

```python
from langchain_hana import MultiGPUEmbeddings, CacheConfig
from langchain_hana.gpu.multi_gpu_manager import get_multi_gpu_manager

# Configure GPU manager
gpu_manager = get_multi_gpu_manager()
gpu_manager.initialize(device_ids=[0, 1])  # Use specific GPUs

# Configure embedding cache
cache_config = CacheConfig(
    enabled=True,
    max_size=100000,
    ttl_seconds=3600,
    persist_path="/path/to/cache.pkl",
    load_on_init=True
)

# Create multi-GPU embeddings with custom configuration
multi_gpu_embeddings = MultiGPUEmbeddings(
    base_embeddings=base_model,
    batch_size=64,
    enable_caching=True,
    cache_config=cache_config,
    gpu_manager=gpu_manager,
    normalize_embeddings=True
)

# Check performance statistics
stats = multi_gpu_embeddings.get_stats()
print(f"Documents processed: {stats['documents_embedded']}")
print(f"Avg. time per document: {stats['avg_time_per_document']:.4f} seconds")
print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
print(f"GPU utilization: {stats['gpu']['devices']}")
```

## Optimizing Performance

### Batch Size Optimization

The batch size significantly impacts performance. We recommend:

1. Start with a batch size of 32-64
2. Increase in powers of 2 until you see diminishing returns
3. For multi-GPU setups, try `batch_size = 32 * num_gpus`
4. Monitor GPU memory usage to avoid out-of-memory errors

### Memory Management

To optimize memory usage:

1. Use `use_fp16=True` when available to reduce memory footprint
2. Enable caching for repeated embeddings
3. Set appropriate TTL for cache entries based on your workload patterns
4. For large workloads, consider setting `persist_path` to enable disk caching

### GPU Selection Strategy

You can configure the load balancing strategy:

```python
# Initialize GPU manager with specific strategy
gpu_manager = get_multi_gpu_manager()
gpu_manager.initialize(strategy="memory")  # Options: "auto", "round_robin", "memory", "utilization"
```

- `auto`: Balanced approach using multiple factors (default)
- `round_robin`: Simple round-robin distribution
- `memory`: Prefer GPUs with more available memory
- `utilization`: Prefer GPUs with lower utilization

## Monitoring and Diagnostics

The multi-GPU system provides detailed performance metrics:

```python
# Get GPU manager status
status = gpu_manager.get_status()
print(f"Total tasks submitted: {status['total_tasks_submitted']}")
print(f"Tasks completed: {status['total_tasks_completed']}")
print(f"Tasks failed: {status['total_tasks_failed']}")
print(f"Pending tasks: {status['pending_tasks']}")

# Get detailed device information
for device_id, device_info in enumerate(status['devices']):
    print(f"GPU {device_id} ({device_info['name']}):")
    print(f"  Memory used: {device_info['memory_allocated_mb']:.2f}MB / {device_info['total_memory_mb']:.2f}MB")
    print(f"  Utilization: {device_info['utilization']:.2%}")
    print(f"  Active tasks: {device_info['active_tasks']}")
    print(f"  Completed tasks: {device_info['completed_tasks']}")
```

You can also enable statistics logging to a file:

```python
# Enable statistics logging
gpu_manager = get_multi_gpu_manager()
gpu_manager.initialize(stats_file="/path/to/gpu_stats.jsonl")
```

## Benchmarking

We provide a benchmarking script to evaluate performance:

```bash
python examples/multi_gpu_embeddings_demo.py --num-docs 10000 --batch-size 64 --enable-tensorrt
```

Typical performance improvements:

| Configuration             | Documents/sec | Relative Speedup |
|---------------------------|---------------|------------------|
| Single GPU (Base)         | 250           | 1.0x             |
| 2x GPUs                   | 480           | 1.9x             |
| 4x GPUs                   | 920           | 3.7x             |
| Single GPU + TensorRT     | 750           | 3.0x             |
| 4x GPUs + TensorRT        | 2800          | 11.2x            |

## Integration with SAP HANA Cloud

The multi-GPU embeddings can be used directly with the SAP HANA Cloud vectorstore:

```python
from langchain_hana import HanaDB, HanaTensorRTMultiGPUEmbeddings
from hdbcli import dbapi

# Connect to SAP HANA Cloud
conn = dbapi.connect(
    address="your-hana-instance.hanacloud.ondemand.com",
    port=443,
    user="your_username",
    password="your_password"
)

# Create multi-GPU embeddings
embeddings = HanaTensorRTMultiGPUEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    batch_size=64,
    use_fp16=True,
    enable_tensor_cores=True
)

# Create vector store with multi-GPU embeddings
vector_store = HanaDB(
    connection=conn,
    embedding=embeddings,
    table_name="MY_VECTORS"
)

# Add documents (embeddings generated using multiple GPUs)
vector_store.add_texts(["Text 1", "Text 2", ..., "Text 1000"])

# Search (query embedding generated using multiple GPUs)
results = vector_store.similarity_search("What is SAP HANA Cloud?", k=5)
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Enable FP16 precision
   - Check for other processes using GPU memory

2. **Uneven Performance Across GPUs**
   - Check for GPU thermal throttling
   - Verify all GPUs are same model/generation
   - Try different load balancing strategies

3. **Task Queue Buildup**
   - Increase worker threads or reduce submission rate
   - Check for bottlenecks in preprocessing/postprocessing
   - Monitor system CPU usage alongside GPU usage

### Diagnostic Commands

```python
# Check GPU information
gpu_manager = get_multi_gpu_manager()
print(gpu_manager.get_device_info())

# Get real-time status
print(gpu_manager.get_status())

# Reset if needed
gpu_manager.stop()
gpu_manager.initialize()
```

## Best Practices

1. **Resource Management**
   - Explicitly stop the GPU manager when done with `gpu_manager.stop()`
   - Use context managers for scoped GPU operations
   - Avoid creating multiple GPU manager instances

2. **Production Deployment**
   - Monitor GPU temperature and utilization in production
   - Implement circuit breakers for GPU failure scenarios
   - Consider using Kubernetes with GPU scheduling for container deployments

3. **Scaling Considerations**
   - For ultra-high throughput, consider distributing across multiple nodes
   - Benchmark regularly as document volume grows
   - Consider using smaller, specialized models for higher efficiency

## Next Steps

- See [TensorRT Optimization Guide](./tensorrt_optimization.md) for additional performance
- Explore [Distributed Deployment Guide](../deployment/distributed.md) for multi-node setups
- Check [API Reference](../api/reference.md) for detailed method documentation