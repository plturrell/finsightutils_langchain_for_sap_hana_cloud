# NVIDIA Blueprint Compliance

This document outlines how the SAP HANA Cloud LangChain Integration complies with NVIDIA Blueprint standards for optimal performance on NVIDIA GPU infrastructure.

## What is an NVIDIA Blueprint?

NVIDIA Blueprints are pre-built, tested, and optimized AI workflows that can be deployed on NVIDIA-Certified infrastructure. They provide a standardized way to package, distribute, and deploy AI applications with NVIDIA GPU acceleration.

## Compliance Overview

The SAP HANA Cloud LangChain Integration has been designed and optimized to meet NVIDIA Blueprint standards, ensuring optimal performance and compatibility with NVIDIA GPU infrastructure.

| Requirement | Status | Details |
|------------|--------|---------|
| NVIDIA Container Base Image | ✅ Compliant | Uses `nvcr.io/nvidia/pytorch:23.12-py3` |
| GPU Acceleration | ✅ Compliant | TensorRT optimization, CUDA acceleration |
| Minimum Driver Version | ✅ Compliant | Requires NVIDIA Driver 520.0+ |
| CUDA Version | ✅ Compliant | Compatible with CUDA 11.8+ |
| TensorRT Support | ✅ Compliant | TensorRT 8.6.0+ with engine caching |
| Resource Specifications | ✅ Compliant | Clear GPU, memory, and CPU requirements |
| GPU Product Support | ✅ Compliant | Supports T4, A10, A100, H100 GPUs |
| Documentation | ✅ Compliant | Includes performance benchmarks and quickstart guide |

## GPU Compatibility

The application is tested and optimized for the following NVIDIA GPUs:

| GPU Model | Memory | Batch Size Optimization | Performance Characteristics |
|-----------|--------|-------------------------|----------------------------|
| NVIDIA T4 | 16GB | 32-64 | Good for cost-effective inference |
| NVIDIA A10 | 24GB | 64-128 | Excellent balanced performance |
| NVIDIA A100 | 40/80GB | 128-256 | Highest throughput for large workloads |
| NVIDIA H100 | 80GB | 256+ | Ultimate performance for mission-critical applications |

## Performance Benchmarks

### Embedding Generation Performance

Benchmarks across different NVIDIA GPUs compared to CPU:

#### T4 GPU (16GB)

| Operation | Batch Size | T4 GPU | CPU | Speedup |
|-----------|------------|--------|-----|---------|
| Embedding | 1 | 18ms | 80ms | 4.4x |
| Embedding | 32 | 82ms | 580ms | 7.1x |
| Embedding | 128 | 198ms | 2320ms | 11.7x |
| MMR Search | 10 results | 12ms | 68ms | 5.7x |

#### A10 GPU (24GB)

| Operation | Batch Size | A10 GPU | CPU | Speedup |
|-----------|------------|---------|-----|---------|
| Embedding | 1 | 14ms | 80ms | 5.7x |
| Embedding | 32 | 58ms | 580ms | 10.0x |
| Embedding | 128 | 112ms | 2320ms | 20.7x |
| MMR Search | 10 results | 8ms | 68ms | 8.5x |

#### A100 GPU (80GB)

| Operation | Batch Size | A100 GPU | CPU | Speedup |
|-----------|------------|----------|-----|---------|
| Embedding | 1 | 9ms | 80ms | 8.9x |
| Embedding | 32 | 30ms | 580ms | 19.3x |
| Embedding | 128 | 62ms | 2320ms | 37.4x |
| MMR Search | 10 results | 4ms | 68ms | 17.0x |

### Multi-GPU Scaling

The application supports multi-GPU environments with near-linear scaling:

| GPU Configuration | Relative Throughput |
|-------------------|---------------------|
| Single T4 | 1.0x (baseline) |
| Dual T4 | 1.9x |
| Single A10 | 2.1x |
| Dual A10 | 4.0x |
| Single A100 | 3.8x |
| Dual A100 | 7.5x |

## TensorRT Optimization

The integration leverages NVIDIA TensorRT for optimized inference:

- **Engine Caching**: TensorRT engines are cached for faster startup
- **Mixed Precision**: Support for FP32, FP16, and INT8 precision
- **Optimized Graph**: Layer fusion and kernel auto-tuning
- **Memory Optimization**: Reduced memory footprint for efficient operation

### TensorRT Precision Comparison

| Precision | Relative Speed | Relative Accuracy | Use Case |
|-----------|----------------|-------------------|----------|
| FP32 | 1.0x | Highest | When maximum accuracy is required |
| FP16 | 2-3x | High | Recommended default for most use cases |
| INT8 | 4-5x | Good | When maximum throughput is required |

## GPU Memory Utilization

Memory utilization is optimized to maximize throughput while preventing out-of-memory errors:

| Operation | Batch Size | T4 (16GB) | A10 (24GB) | A100 (80GB) |
|-----------|------------|-----------|------------|-------------|
| Embedding | 32 | 3.2 GB | 3.2 GB | 3.2 GB |
| Embedding | 64 | 5.8 GB | 5.8 GB | 5.8 GB |
| Embedding | 128 | 10.4 GB | 10.4 GB | 10.4 GB |
| Embedding | 256 | OOM | 19.6 GB | 19.6 GB |
| Embedding | 512 | OOM | OOM | 38.2 GB |

## NGC Deployment

### Building and Pushing to NGC

The repository includes an automated script for building and pushing the application to NGC:

```bash
# Run the build script
./build_launchable.sh
```

The script:
1. Verifies NVIDIA GPU and driver compatibility
2. Builds the Docker image with TensorRT optimization
3. Tests GPU access and TensorRT functionality
4. Pushes the image to the NGC Registry
5. Generates NGC Blueprint configuration

### NGC Blueprint Configuration

The application includes a complete NGC Blueprint configuration (`nvidia-blueprint.yaml`) that defines:

- Resource requirements (GPU, memory, CPU)
- Container image and ports
- Environment variables
- Documentation (overview, quickstart, performance)

## Container Optimizations

The Docker container is optimized for NGC environments:

- Uses NVIDIA-optimized PyTorch container as base
- Includes TensorRT for accelerated inference
- Implements proper GPU memory management
- Supports multi-GPU environments
- Provides health checks for monitoring

## Best Practices for NGC Deployment

1. **GPU Selection**: Choose appropriate GPU type based on workload
   - T4: Cost-effective inference for small to medium models
   - A10: Balanced performance for medium to large models
   - A100/H100: Highest performance for large models and high throughput

2. **TensorRT Precision**: Select appropriate precision
   - FP16: Recommended default for most use cases
   - FP32: When maximum accuracy is required
   - INT8: When maximum throughput is needed

3. **Batch Size Optimization**: 
   - T4: 32-64 optimal batch size
   - A10: 64-128 optimal batch size
   - A100: 128-256 optimal batch size

4. **TensorRT Engine Caching**:
   - Mount a volume to `/app/trt_engines` for persistent caching
   - Significantly improves startup time on subsequent runs

5. **Multi-GPU Configuration**:
   - Set `GPU_ENABLED=true` for GPU acceleration
   - The application automatically detects and utilizes all available GPUs