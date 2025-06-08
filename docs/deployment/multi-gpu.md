# Multi-GPU Deployment Guide

This guide provides detailed instructions for deploying and optimizing the SAP HANA Cloud LangChain integration on multi-GPU systems. By leveraging multiple GPUs, you can significantly accelerate embedding generation, particularly for large document collections.

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Software Prerequisites](#software-prerequisites)
3. [Multi-GPU Architecture Overview](#multi-gpu-architecture-overview)
4. [Installation](#installation)
5. [Configuration Options](#configuration-options)
6. [Deployment Scenarios](#deployment-scenarios)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Management](#monitoring-and-management)
9. [Troubleshooting](#troubleshooting)

## Hardware Requirements

For optimal performance with multi-GPU deployment, the following hardware is recommended:

| Component | Minimum Requirement | Recommended Specification |
|-----------|---------------------|---------------------------|
| GPUs | 2x NVIDIA T4 | 2-8x NVIDIA A100, H100, or L4 |
| GPU Memory | 16 GB per GPU | 32-80 GB per GPU |
| CPU | 8 cores | 32+ cores |
| System RAM | 32 GB | 128+ GB |
| Storage | 100 GB SSD | 1+ TB NVMe SSD |
| PCIe | Gen3 x8 | Gen4 x16 with NVLink |

### Supported GPU Models

The multi-GPU functionality has been tested with the following NVIDIA GPU models:

- **Best Performance**: A100, H100, L40S
- **Great Performance**: A10, A30, L4
- **Good Performance**: T4, V100, RTX 4090, RTX 3090
- **Basic Support**: GTX 1080 Ti and newer

Older GPUs without Tensor Cores (pre-Volta architecture) will work but with significantly reduced performance.

## Software Prerequisites

Before deploying, ensure your system has the following software:

- **CUDA Toolkit**: 11.8 or newer
- **NVIDIA Driver**: 520.61.05 or newer
- **Docker**: 23.0 or newer
- **Docker Compose**: 2.17 or newer
- **NVIDIA Container Toolkit**: Latest version

### Installing Prerequisites

```bash
# Update package list
sudo apt-get update

# Install NVIDIA driver
sudo apt-get install -y nvidia-driver-525

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Multi-GPU Architecture Overview

The SAP HANA Cloud LangChain integration uses a distributed architecture for multi-GPU processing:

```
                     ┌───────────────────┐
                     │  FastAPI Backend  │
                     └─────────┬─────────┘
                               │
                     ┌─────────▼─────────┐
                     │  Multi-GPU Manager │
                     └─────────┬─────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
┌────────▼─────────┐  ┌────────▼─────────┐  ┌────────▼─────────┐
│   GPU Worker 0   │  │   GPU Worker 1   │  │   GPU Worker N   │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

Key components:

1. **FastAPI Backend**: Handles incoming requests and coordinates embedding operations
2. **Multi-GPU Manager**: Distributes workloads across available GPUs
3. **GPU Workers**: Execute embedding operations on individual GPUs
4. **Load Balancer**: Distributes tasks based on GPU capabilities and current workload

## Installation

### Docker Deployment (Recommended)

The simplest way to deploy with multi-GPU support is using Docker:

1. **Clone the repository**:

```bash
git clone https://github.com/yourusername/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud
```

2. **Configure environment variables**:

Create a `.env` file with the following variables:

```
# SAP HANA Cloud connection
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password

# Multi-GPU configuration
MULTI_GPU_ENABLED=true
MULTI_GPU_STRATEGY=auto  # Options: auto, round_robin, memory, utilization
MULTI_GPU_STATS_INTERVAL=60.0
MULTI_GPU_STATS_FILE=/app/logs/gpu_stats.jsonl

# TensorRT optimization
USE_TENSORRT=true
TENSORRT_CACHE_DIR=/app/cache/tensorrt
TENSORRT_PRECISION=auto  # Options: auto, fp32, fp16, int8
TENSORRT_INT8_CALIBRATION_CACHE=/app/cache/tensorrt/calibration

# Embedding model configuration
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_CACHE_ENABLED=true
EMBEDDING_CACHE_SIZE=100000
EMBEDDING_CACHE_TTL=3600
EMBEDDING_BATCH_SIZE=32
```

3. **Deploy with Docker Compose**:

```bash
# Start the multi-GPU optimized stack
docker-compose -f docker-compose.nvidia.yml up -d
```

4. **Verify deployment**:

```bash
# Check container status
docker-compose -f docker-compose.nvidia.yml ps

# Check GPU utilization
docker exec -it langchain-hana-api nvidia-smi

# View logs
docker-compose -f docker-compose.nvidia.yml logs -f api
```

### Manual Deployment

For manual deployment with multi-GPU support:

1. **Install Python dependencies**:

```bash
pip install -r requirements.txt

# Install GPU dependencies
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118
pip install tensorrt>=8.6.0 pycuda>=2022.2 nvidia-ml-py>=11.525.84
```

2. **Configure environment variables**:

```bash
export MULTI_GPU_ENABLED=true
export MULTI_GPU_STRATEGY=auto
export USE_TENSORRT=true
export EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
```

3. **Run the application**:

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Configuration Options

### Multi-GPU Manager Configuration

The multi-GPU manager can be configured through environment variables:

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `MULTI_GPU_ENABLED` | Enable multi-GPU support | `false` | `true`, `false` |
| `MULTI_GPU_STRATEGY` | Load balancing strategy | `auto` | `auto`, `round_robin`, `memory`, `utilization` |
| `MULTI_GPU_STATS_INTERVAL` | Statistics collection interval (seconds) | `60.0` | Any float value |
| `MULTI_GPU_STATS_FILE` | File to log GPU statistics | `None` | Path to log file |
| `MULTI_GPU_DEVICE_IDS` | Specific GPU devices to use | `None` | Comma-separated list of device IDs |

### TensorRT Optimization Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `USE_TENSORRT` | Enable TensorRT optimization | `false` | `true`, `false` |
| `TENSORRT_PRECISION` | TensorRT precision mode | `auto` | `auto`, `fp32`, `fp16`, `int8` |
| `TENSORRT_CACHE_DIR` | Directory to cache optimized engines | `~/.cache/hana_trt_engines` | Any directory path |
| `TENSORRT_INT8_CALIBRATION` | Enable INT8 calibration | `false` | `true`, `false` |

### Embedding Configuration

| Variable | Description | Default | Options |
|----------|-------------|---------|---------|
| `EMBEDDING_BATCH_SIZE` | Batch size for embedding generation | `32` | Any integer > 0 |
| `EMBEDDING_CACHE_ENABLED` | Enable embedding cache | `true` | `true`, `false` |
| `EMBEDDING_CACHE_SIZE` | Maximum cache size | `10000` | Any integer > 0 |
| `EMBEDDING_CACHE_TTL` | Cache time-to-live in seconds | `3600` | Any integer ≥ 0 |
| `EMBEDDING_CACHE_PERSISTENCE` | Cache persistence file | `None` | Path to cache file |
| `EMBEDDING_MODEL` | Hugging Face model name | `sentence-transformers/all-mpnet-base-v2` | Any supported model |
| `ENABLE_TENSOR_CORES` | Enable Tensor Core optimization | `true` | `true`, `false` |

## Deployment Scenarios

### Scenario 1: High-throughput Document Processing

For processing large document collections with maximum throughput:

```bash
# Environment variables
export MULTI_GPU_ENABLED=true
export MULTI_GPU_STRATEGY=memory
export EMBEDDING_BATCH_SIZE=64
export USE_TENSORRT=true
export TENSORRT_PRECISION=int8
export EMBEDDING_CACHE_ENABLED=true
export EMBEDDING_CACHE_SIZE=1000000
```

### Scenario 2: Low-latency Query Processing

For real-time query embedding with minimal latency:

```bash
# Environment variables
export MULTI_GPU_ENABLED=true
export MULTI_GPU_STRATEGY=utilization
export EMBEDDING_BATCH_SIZE=1
export USE_TENSORRT=true
export TENSORRT_PRECISION=fp16
export EMBEDDING_CACHE_ENABLED=true
export EMBEDDING_CACHE_TTL=86400
```

### Scenario 3: Balanced Processing

For a balanced approach suitable for most use cases:

```bash
# Environment variables
export MULTI_GPU_ENABLED=true
export MULTI_GPU_STRATEGY=auto
export EMBEDDING_BATCH_SIZE=32
export USE_TENSORRT=true
export TENSORRT_PRECISION=auto
export EMBEDDING_CACHE_ENABLED=true
```

## Performance Optimization

### Batch Size Optimization

The optimal batch size depends on your GPU memory and model size:

1. **Large GPUs (A100, H100)**:
   - Document embedding: 128-256
   - Query embedding: 1-8

2. **Medium GPUs (A10, RTX 4090)**:
   - Document embedding: 64-128
   - Query embedding: 1-4

3. **Small GPUs (T4, V100)**:
   - Document embedding: 32-64
   - Query embedding: 1

### Precision Selection

Choose the right precision for your workload:

- **FP32**: Highest accuracy, slowest performance
- **FP16**: Good accuracy, 2-3x faster than FP32
- **INT8**: Slight accuracy decrease, 3-5x faster than FP32

For most applications, FP16 provides the best balance of accuracy and performance.

### Memory Management

To avoid out-of-memory errors:

1. **Gradual Batch Size Adjustment**: Start with a small batch size and gradually increase it
2. **Monitor Memory Usage**: Use the built-in monitoring to track GPU memory
3. **Enable Dynamic Batching**: Set `ENABLE_DYNAMIC_BATCHING=true` for automatic adjustment

## Monitoring and Management

### Real-time Monitoring

The multi-GPU manager provides real-time monitoring through:

1. **REST API**: Access GPU statistics via `GET /api/v1/gpu/stats`
2. **Logs**: Check logs for performance information
3. **Stats File**: Review the JSON stats file for historical data

Example API response:

```json
{
  "timestamp": 1682534987,
  "devices": [
    {
      "index": 0,
      "name": "NVIDIA A100-SXM4-40GB",
      "compute_capability": "8.0",
      "memory_total_mb": 40960,
      "memory_free_mb": 38912,
      "memory_usage_percent": 5.0,
      "utilization": 0.08,
      "temperature_c": 38,
      "active_tasks": 1,
      "completed_tasks": 1024
    },
    {
      "index": 1,
      "name": "NVIDIA A100-SXM4-40GB",
      "compute_capability": "8.0",
      "memory_total_mb": 40960,
      "memory_free_mb": 39936,
      "memory_usage_percent": 2.5,
      "utilization": 0.03,
      "temperature_c": 36,
      "active_tasks": 0,
      "completed_tasks": 768
    }
  ],
  "total_tasks_completed": 1792,
  "total_tasks_pending": 64,
  "avg_time_per_batch_ms": 15.3
}
```

### Performance Benchmarking

To benchmark your multi-GPU deployment:

```bash
# Run the benchmark script
python -m langchain_hana.gpu.benchmark \
  --model sentence-transformers/all-mpnet-base-v2 \
  --batch-sizes 1,8,16,32,64,128 \
  --precision fp16,int8 \
  --iterations 100 \
  --output benchmark_results.json
```

The benchmark will test different batch sizes, precision modes, and report:

- Latency (min, max, mean, p95, p99)
- Throughput (documents per second)
- GPU utilization and memory usage

## Troubleshooting

### Common Issues

#### Out of Memory Errors

**Symptoms**: CUDA out of memory errors, process termination

**Solutions**:
1. Reduce batch size (`EMBEDDING_BATCH_SIZE`)
2. Use lower precision (`TENSORRT_PRECISION=fp16` or `int8`)
3. Enable dynamic batching (`ENABLE_DYNAMIC_BATCHING=true`)
4. Reduce model size or use a smaller embedding model

#### Slow Performance

**Symptoms**: Lower-than-expected throughput

**Solutions**:
1. Check GPU utilization with `nvidia-smi`
2. Increase batch size if utilization is low
3. Enable TensorRT optimization (`USE_TENSORRT=true`)
4. Check if Tensor Cores are being used (`ENABLE_TENSOR_CORES=true`)
5. Verify PCIe bandwidth is not bottlenecked

#### Uneven GPU Usage

**Symptoms**: One GPU is heavily utilized while others are idle

**Solutions**:
1. Set `MULTI_GPU_STRATEGY=utilization` or `memory`
2. Check for GPU-specific failures in logs
3. Verify all GPUs are of the same model for best results
4. Set environment variable `CUDA_VISIBLE_DEVICES` to control which GPUs are used

#### TensorRT Engine Build Failures

**Symptoms**: Errors during TensorRT engine building

**Solutions**:
1. Check TensorRT and CUDA compatibility
2. Ensure sufficient disk space for engine cache
3. Try a different precision mode
4. Clear the TensorRT cache and rebuild: `rm -rf $TENSORRT_CACHE_DIR/*`

### Diagnosing Issues

For detailed diagnostics:

1. **Enable debug logging**:
   
   ```bash
   export LOG_LEVEL=DEBUG
   ```

2. **Check GPU device information**:

   ```python
   from langchain_hana.gpu.imports import get_gpu_info
   print(get_gpu_info())
   ```

3. **Test individual components**:

   ```python
   from langchain_hana.gpu.imports import check_gpu_requirements
   print(check_gpu_requirements("tensorrt"))
   print(check_gpu_requirements("multi_gpu"))
   print(check_gpu_requirements("tensor_cores"))
   ```

4. **Verify CUDA installation**:

   ```bash
   nvidia-smi
   nvcc --version
   ```

## Advanced Configuration

### Custom Load Balancing Strategy

You can implement a custom load balancing strategy by extending the `EnhancedMultiGPUManager` class:

```python
from langchain_hana.gpu.multi_gpu_manager import EnhancedMultiGPUManager, Task

class CustomMultiGPUManager(EnhancedMultiGPUManager):
    def _select_device_for_task(self, task: Task) -> int:
        # Custom device selection logic
        # ...
        return selected_device_id
```

### Persistent Caching

For production environments, enable persistent caching to reduce startup times:

```bash
export EMBEDDING_CACHE_PERSISTENCE=/path/to/embedding_cache.pkl
export TENSORRT_CACHE_DIR=/path/to/trt_engines
```

### Automatic Precision Selection

The system can automatically select the optimal precision based on your hardware:

```bash
export TENSORRT_PRECISION=auto
```

This will use:
- INT8 for Turing+ GPUs (RTX 20-series, T4, and newer)
- FP16 for Volta+ GPUs (V100 and newer)
- FP32 for older GPUs

## Conclusion

This multi-GPU deployment guide should help you maximize the performance of the SAP HANA Cloud LangChain integration. By properly configuring the system for your specific hardware and workload, you can achieve significant speedups for embedding generation and vector operations.

For assistance with specific deployment scenarios or custom optimizations, please refer to the [GitHub issues](https://github.com/yourusername/langchain-integration-for-sap-hana-cloud/issues) or contact the project maintainers.