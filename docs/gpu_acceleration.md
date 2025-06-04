# GPU Acceleration in SAP HANA Cloud LangChain Integration

This document describes the GPU acceleration capabilities provided by the SAP HANA Cloud LangChain integration, including dynamic batch processing for efficient embedding generation.

## Overview

The integration provides several components for GPU acceleration:

1. **TensorRT-optimized Embeddings**: High-performance embedding generation using NVIDIA TensorRT
2. **Dynamic Batch Processing**: Memory-aware batch sizing for optimal throughput
3. **GPU Accelerator Interface**: Unified interface for GPU operations across frameworks
4. **Mixed Precision Support**: FP32, FP16, and INT8 precision modes for optimal performance/accuracy tradeoffs

## Dynamic Batch Processing

The dynamic batch processor automatically determines and adjusts batch sizes during processing based on:

1. Available GPU memory at runtime
2. Model size and memory requirements
3. Current processing performance
4. OOM (out-of-memory) recovery and adaptation

### Key Features

- **Runtime GPU Memory Detection**: Automatically detects available GPU memory
- **Model-Aware Batch Sizing**: Calculates optimal batch size based on model characteristics
- **Dynamic Adjustment**: Adapts batch size during processing for maximum throughput
- **Safety Margins**: Includes configurable safety margins to prevent OOM errors
- **Batch Splitting**: Automatically handles large requests by splitting into optimal batches
- **OOM Recovery**: Recovers gracefully from OOM errors by reducing batch size
- **Performance Monitoring**: Tracks processing statistics for optimization

### Usage

The dynamic batch processor is integrated into the `TensorRTEmbeddings` class and is used automatically when generating embeddings:

```python
from langchain_hana.gpu import TensorRTEmbeddings

# Create TensorRT embeddings with dynamic batch processing
embeddings = TensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    max_batch_size=32,  # Initial batch size (will be dynamically adjusted)
    precision="fp16"    # Use FP16 for better performance
)

# Generate embeddings (dynamic batch processing happens automatically)
texts = ["Text 1", "Text 2", ..., "Text 1000"]
embedding_vectors = embeddings.embed_documents(texts)
```

### Custom Batch Processing

For advanced use cases, you can also use the batch processor directly:

```python
from langchain_hana.gpu import EmbeddingBatchProcessor, ModelMemoryProfile

# Define your embedding function
def embed_batch(batch_texts):
    # Your embedding generation code
    return embeddings

# Create model memory profile
profile = ModelMemoryProfile(
    model_name="your-model-name",
    embedding_dim=768,
    dtype="float16"
)

# Create batch processor
processor = EmbeddingBatchProcessor(
    embedding_fn=embed_batch,
    model_name="your-model-name",
    embedding_dim=768,
    device_id=0,
    initial_batch_size=32,
    min_batch_size=1,
    max_batch_size=128,
    safety_factor=0.8,
    oom_recovery_factor=0.5,
    dtype="float16",
    enable_caching=True
)

# Process documents with dynamic batching
embeddings, stats = processor.embed_documents(texts)

# Print statistics
print(f"Total time: {stats.total_time:.2f}s")
print(f"Items per second: {stats.items_per_second:.2f}")
print(f"Batch size adjustment: {stats.initial_batch_size} → {stats.final_batch_size}")
```

## TensorRT Embeddings

The `TensorRTEmbeddings` class provides optimized embedding generation using NVIDIA TensorRT:

```python
from langchain_hana.gpu import TensorRTEmbeddings

# Create TensorRT embeddings with INT8 precision for maximum throughput
embeddings = TensorRTEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_dir="./trt_engines",
    precision="int8"  # Use INT8 quantization for maximum throughput
)

# Generate embeddings for documents
documents = ["This is a sample document", "Another example text"]
doc_embeddings = embeddings.embed_documents(documents)

# Generate embedding for a query
query_embedding = embeddings.embed_query("Sample query text")
```

## Performance Benchmarking

The `TensorRTEmbeddings` class includes built-in benchmarking capabilities:

```python
# Benchmark different batch sizes
benchmark_results = embeddings.benchmark(
    batch_sizes=[1, 8, 16, 32, 64, 128],
    iterations=100,
    warmup=10
)

# Compare different precision modes (FP32, FP16, INT8)
precision_comparison = embeddings.benchmark_precision_comparison()
```

## Examples

For complete examples of GPU acceleration and dynamic batch processing, see the [examples directory](examples/):

- [Dynamic Batch Processing Example](examples/dynamic_batch_processing.py): Demonstrates dynamic batch sizing for embedding generation
- [TensorRT Optimization Example](examples/tensorrt_optimization.py): Shows how to optimize models with TensorRT for maximum performance

## Requirements

- NVIDIA GPU with CUDA support
- PyTorch
- TensorRT (for TensorRT optimization)
- CUDA Toolkit
- CuPy (for GPU-accelerated vector operations)

Optional dependencies:
- NVML or py3nvml (for advanced GPU monitoring)
- Triton Inference Server (for remote inference)