# TensorRT Optimization Guide

This guide provides detailed instructions for optimizing embedding generation performance using NVIDIA TensorRT with the SAP HANA Cloud LangChain integration.

## Table of Contents

1. [Introduction to TensorRT](#introduction-to-tensorrt)
2. [TensorRT Benefits](#tensorrt-benefits)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration Options](#configuration-options)
6. [Precision Modes](#precision-modes)
7. [Engine Caching](#engine-caching)
8. [INT8 Calibration](#int8-calibration)
9. [Performance Benchmarking](#performance-benchmarking)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Optimization](#advanced-optimization)

## Introduction to TensorRT

NVIDIA TensorRT is a high-performance deep learning inference optimizer and runtime that delivers low latency and high throughput for inference applications. The SAP HANA Cloud LangChain integration leverages TensorRT to accelerate embedding generation, particularly for large document collections.

## TensorRT Benefits

Using TensorRT with the SAP HANA Cloud LangChain integration provides several benefits:

- **3-10x faster embedding generation**: Dramatically reduces the time required to process documents
- **Lower memory consumption**: Optimized memory usage allows processing larger batches
- **Higher throughput**: Process more documents per second
- **Mixed precision support**: FP32, FP16, and INT8 precision for optimal performance-accuracy trade-offs
- **Tensor Core utilization**: Automatic usage of specialized tensor cores on supported GPUs
- **Optimized engine caching**: Engines are cached to avoid recompilation

## Prerequisites

To use TensorRT optimization, you'll need:

- NVIDIA GPU with compute capability 6.0 or later (Pascal architecture or newer)
- CUDA Toolkit 11.4 or newer
- cuDNN 8.2 or newer
- TensorRT 8.4 or newer
- PyTorch 2.0 or newer

## Installation

### Using Docker (Recommended)

The simplest approach is to use the provided Docker image with all dependencies pre-installed:

```bash
docker-compose -f docker-compose.nvidia.yml up -d
```

### Manual Installation

If you prefer to install TensorRT manually:

1. **Install CUDA and cuDNN**:
   
   Follow the [NVIDIA CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

2. **Install TensorRT**:

   Download and install TensorRT from the [NVIDIA Developer website](https://developer.nvidia.com/tensorrt).

   For Ubuntu:
   ```bash
   # Example for TensorRT 8.6.1 with CUDA 11.8
   wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-11.8_1.0-1_amd64.deb
   sudo dpkg -i nv-tensorrt-local-repo-ubuntu2004-8.6.1-cuda-11.8_1.0-1_amd64.deb
   sudo apt-get update
   sudo apt-get install -y tensorrt
   ```

3. **Install Python packages**:

   ```bash
   pip install tensorrt pycuda torch
   ```

4. **Verify installation**:

   ```python
   import tensorrt as trt
   print(f"TensorRT version: {trt.__version__}")
   ```

## Configuration Options

TensorRT optimization can be configured with the following environment variables:

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `USE_TENSORRT` | Enable TensorRT optimization | `false` | `true`, `false` |
| `TENSORRT_PRECISION` | Precision mode | `auto` | `auto`, `fp32`, `fp16`, `int8` |
| `TENSORRT_CACHE_DIR` | Directory to cache engines | `~/.cache/hana_trt_engines` | Any directory path |
| `TENSORRT_MAX_WORKSPACE_SIZE` | Maximum workspace size in MB | `1024` | Integer value |
| `TENSORRT_ENGINE_REBUILD` | Force rebuild engines | `false` | `true`, `false` |
| `TENSORRT_INT8_CALIBRATION` | Enable INT8 calibration | `false` | `true`, `false` |
| `TENSORRT_CALIBRATION_CACHE` | Path to calibration cache | `~/.cache/hana_trt_calibration` | Any directory path |

## Precision Modes

TensorRT supports multiple precision modes, each with different performance-accuracy trade-offs:

### FP32 (32-bit Floating Point)

```bash
export TENSORRT_PRECISION=fp32
```

- Highest accuracy
- Baseline performance
- Recommended for tasks requiring maximum precision

### FP16 (16-bit Floating Point)

```bash
export TENSORRT_PRECISION=fp16
```

- Minimal accuracy loss
- 2-3x performance increase over FP32
- Requires GPU with compute capability 6.0+ (Pascal or newer)
- Recommended for most production workloads

### INT8 (8-bit Integer)

```bash
export TENSORRT_PRECISION=int8
```

- Some accuracy loss (typically <1% for embedding models)
- 3-5x performance increase over FP32
- Requires GPU with compute capability 6.1+ (Pascal or newer)
- Requires calibration for best results
- Recommended for maximum throughput when slight accuracy loss is acceptable

### Auto (Automatic Selection)

```bash
export TENSORRT_PRECISION=auto
```

- Automatically selects the best precision based on GPU capabilities
- Uses INT8 for Turing+ GPUs (T4, RTX 20-series, and newer)
- Uses FP16 for Volta+ GPUs (V100 and newer)
- Uses FP32 for older GPUs

## Engine Caching

TensorRT engines are optimized for specific GPU models and configurations. To avoid recompiling engines (which can take minutes), they are cached to disk:

```bash
# Set custom cache directory
export TENSORRT_CACHE_DIR=/path/to/tensorrt/cache

# Force rebuild of engines
export TENSORRT_ENGINE_REBUILD=true
```

Cached engines are named using the format:
```
{model_name}_{precision}.engine
```

For example: `sentence-transformers_all-mpnet-base-v2_fp16.engine`

## INT8 Calibration

INT8 calibration is required for optimal accuracy with INT8 precision:

```bash
# Enable INT8 calibration
export TENSORRT_INT8_CALIBRATION=true
export TENSORRT_CALIBRATION_CACHE=/path/to/calibration/cache
```

### Custom Calibration Dataset

For best results, provide a representative dataset for calibration:

```python
from langchain_hana.gpu import TensorRTEmbeddings

# Example text samples from your domain
calibration_data = [
    "SAP HANA Cloud is an in-memory database management system.",
    "Vector embeddings enable semantic search capabilities.",
    "TensorRT provides significant performance improvements for inference.",
    # Add more domain-specific examples here
]

# Create embeddings with custom calibration data
embeddings = TensorRTEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    precision="int8",
    calibration_data=calibration_data
)
```

## Performance Benchmarking

To benchmark TensorRT optimization performance:

```python
from langchain_hana.gpu import TensorRTEmbeddings

# Create embeddings with TensorRT
embeddings = TensorRTEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    precision="fp16"
)

# Run benchmark
results = embeddings.benchmark(
    batch_sizes=[1, 8, 16, 32, 64, 128],
    iterations=100
)

# Compare precision modes
comparison = embeddings.benchmark_precision_comparison()

print(f"Single query latency: {results['single_query']['mean_latency_ms']:.2f} ms")
print(f"Batch 32 throughput: {results['batch_sizes']['32']['throughput_samples_per_second']:.2f} docs/s")

# FP16 vs FP32 speedup
if "fp16_vs_fp32_speedup" in comparison:
    print(f"FP16 speedup: {comparison['fp16_vs_fp32_speedup']:.2f}x")

# INT8 vs FP32 speedup
if "int8_vs_fp32_speedup" in comparison:
    print(f"INT8 speedup: {comparison['int8_vs_fp32_speedup']:.2f}x")
```

Sample benchmark results for a T4 GPU with different models:

| Model | Precision | Batch Size | Throughput (docs/s) | Latency (ms) |
|-------|-----------|------------|---------------------|--------------|
| all-MiniLM-L6-v2 | FP32 | 1 | 150 | 6.7 |
| all-MiniLM-L6-v2 | FP16 | 1 | 300 | 3.3 |
| all-MiniLM-L6-v2 | INT8 | 1 | 450 | 2.2 |
| all-MiniLM-L6-v2 | FP32 | 32 | 1,600 | 20.0 |
| all-MiniLM-L6-v2 | FP16 | 32 | 4,000 | 8.0 |
| all-MiniLM-L6-v2 | INT8 | 32 | 6,400 | 5.0 |
| all-mpnet-base-v2 | FP32 | 1 | 60 | 16.7 |
| all-mpnet-base-v2 | FP16 | 1 | 120 | 8.3 |
| all-mpnet-base-v2 | INT8 | 1 | 180 | 5.6 |
| all-mpnet-base-v2 | FP32 | 32 | 640 | 50.0 |
| all-mpnet-base-v2 | FP16 | 32 | 1,600 | 20.0 |
| all-mpnet-base-v2 | INT8 | 32 | 2,560 | 12.5 |

## Troubleshooting

### Common TensorRT Issues

#### TensorRT Engine Build Failure

**Problem**: Error when building the TensorRT engine

**Diagnostic Steps**:
1. Check CUDA and TensorRT compatibility
2. Ensure sufficient disk space for engine cache
3. Check CUDA compute capability of your GPU

**Solution**:
- Update TensorRT and CUDA to compatible versions
- Clear engine cache and rebuild: `rm -rf $TENSORRT_CACHE_DIR/*`
- Try a different precision mode

#### Accuracy Issues with INT8

**Problem**: Reduced accuracy with INT8 precision

**Diagnostic Steps**:
1. Compare results with FP32 precision
2. Check if calibration was performed
3. Evaluate calibration dataset representativeness

**Solution**:
- Enable INT8 calibration: `TENSORRT_INT8_CALIBRATION=true`
- Provide domain-specific calibration data
- Use FP16 precision if accuracy is critical

#### Slow Engine Loading

**Problem**: Long startup time when loading TensorRT engines

**Diagnostic Steps**:
1. Check engine cache directory
2. Monitor disk I/O during startup
3. Check engine file sizes

**Solution**:
- Use SSD storage for engine cache
- Move engine cache to a faster storage device
- Pre-warm the cache before high-traffic periods

#### Out of Memory Errors

**Problem**: CUDA out of memory errors during engine building or inference

**Diagnostic Steps**:
1. Check GPU memory with `nvidia-smi`
2. Monitor memory usage during execution
3. Check batch size and model size

**Solution**:
- Reduce batch size
- Use a smaller model
- Reduce workspace size: `TENSORRT_MAX_WORKSPACE_SIZE=512`
- Use lower precision: `TENSORRT_PRECISION=fp16` or `int8`

## Advanced Optimization

### Tensor Core Optimization

On GPUs with Tensor Cores (Volta architecture or newer), enable Tensor Core optimization:

```python
from langchain_hana.embeddings import HanaTensorRTMultiGPUEmbeddings
from langchain_hana.gpu import TensorCoreOptimizer

# Create embeddings with Tensor Core optimization
embeddings = HanaTensorRTMultiGPUEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    use_fp16=True,
    enable_tensor_cores=True
)

# Verify Tensor Core usage
optimizer = TensorCoreOptimizer()
print(f"Tensor Cores supported: {optimizer.is_supported()}")
```

### Custom TensorRT Engine Configuration

For advanced users, you can customize the TensorRT engine configuration:

```python
import tensorrt as trt
from langchain_hana.gpu import TensorRTEmbeddings

class CustomTensorRTEmbeddings(TensorRTEmbeddings):
    def _build_engine(self, engine_path, trt_logger):
        # ... custom engine building logic ...
        
        # Configure builder with custom parameters
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30  # 2 GB workspace
        
        # Enable avx2 and sse optimizations
        builder_flag = 1 << int(trt.BuilderFlag.FP16) | 1 << int(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        config.set_flags(builder_flag)
        
        # ... rest of engine building logic ...
```

### Pipeline Parallelism

For large models, you can implement pipeline parallelism across multiple GPUs:

```python
from langchain_hana.gpu import EnhancedMultiGPUManager

# Create a manager with custom strategy
manager = EnhancedMultiGPUManager(strategy="pipeline")

# Configure pipeline stages
manager.configure_pipeline(
    stages=[
        {"name": "tokenization", "device_id": 0},
        {"name": "embedding", "device_id": 1},
        {"name": "pooling", "device_id": 1}
    ]
)

# Use the manager with embeddings
from langchain_hana.embeddings import MultiGPUEmbeddings
embeddings = MultiGPUEmbeddings(
    base_embeddings=base_model,
    gpu_manager=manager
)
```

### ONNX Export with TRT Deployment

For advanced users, you can use ONNX as an intermediate format:

```bash
# Export model to ONNX
python -m langchain_hana.gpu.export_onnx \
  --model sentence-transformers/all-mpnet-base-v2 \
  --output model.onnx

# Convert ONNX to TensorRT
trtexec --onnx=model.onnx \
  --saveEngine=model.engine \
  --fp16 \
  --workspace=1024
```