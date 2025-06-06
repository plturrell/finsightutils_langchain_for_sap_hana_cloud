# Enhanced LangChain Integration for SAP HANA Cloud - NVIDIA NGC Blueprint

This README provides instructions for deploying the Enhanced LangChain Integration for SAP HANA Cloud as an NVIDIA NGC Blueprint.

## Overview

The Enhanced LangChain Integration for SAP HANA Cloud is an NVIDIA-optimized solution for vector store operations that leverages GPU acceleration, TensorRT optimization, and interactive visualizations for maximum performance and usability.

### Key Enhancements

- **TensorRT Optimization**: 3-10x faster inference with engine caching
- **Multi-GPU Load Balancing**: Automatic workload distribution across GPUs
- **Dynamic Batch Sizing**: Memory-aware processing for optimal throughput
- **Mixed Precision Support**: FP16, FP32, and INT8 precision options
- **Interactive 3D Visualization**: Visual exploration of vector embeddings
- **Mobile-Responsive UI**: Accessibility-focused frontend that works on all devices
- **Context-Aware Error Handling**: Detailed diagnostics with remediation suggestions

## Requirements

- **NVIDIA GPU** (T4, A10, A100, or H100 recommended)
- **NVIDIA Driver** 520.0 or later
- **CUDA** 11.8 or later
- **TensorRT** 8.6.0 or later
- **Docker** 20.10.0 or later
- **NVIDIA Container Toolkit** 1.14.0 or later

## Quick Start

### Using NGC CLI

1. Pull the container from NGC:

```bash
ngc registry image pull nvcr.io/nvidia/sap-enhanced/langchain-hana-gpu:latest
```

2. Create an environment file:

```bash
cat > .env << EOL
# SAP HANA Cloud Connection
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password
DEFAULT_TABLE_NAME=EMBEDDINGS

# GPU Settings
TENSORRT_PRECISION=fp16
BATCH_SIZE=64
MAX_BATCH_SIZE=256
EOL
```

3. Run the container:

```bash
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/trt_engines:/app/trt_engines \
  nvcr.io/nvidia/sap-enhanced/langchain-hana-gpu:latest
```

### Using Docker Compose

1. Clone this repository:

```bash
git clone https://github.com/plturrell/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud
```

2. Create an environment file with your SAP HANA Cloud credentials.

3. Deploy using Docker Compose:

```bash
docker-compose -f docker-compose.ngc-blueprint.yml up -d
```

## Docker Compose Files

This repository includes several Docker Compose files for different deployment scenarios:

- `docker-compose.ngc-blueprint.yml`: Optimized for NVIDIA NGC Blueprint deployment
- `docker-compose.nvidia-launchable.yml`: Full stack with frontend for NVIDIA Launchables
- `docker-compose.api.yml`: Backend API only, suitable for integration with external frontends

## Environment Variables

### Required Variables

- `HANA_HOST`: SAP HANA Cloud host
- `HANA_USER`: SAP HANA Cloud username
- `HANA_PASSWORD`: SAP HANA Cloud password

### Optional Variables

- `HANA_PORT`: SAP HANA Cloud port (default: 443)
- `DEFAULT_TABLE_NAME`: Default vector table name (default: EMBEDDINGS)
- `TENSORRT_PRECISION`: TensorRT precision - fp16, fp32, int8 (default: fp16)
- `BATCH_SIZE`: Default batch size for embedding operations (default: 64)
- `MAX_BATCH_SIZE`: Maximum batch size for embedding operations (default: 256)
- `ENABLE_MULTI_GPU`: Enable multi-GPU support (default: true)
- `ENABLE_VECTOR_VISUALIZATION`: Enable 3D visualization (default: true)

## Performance Benchmarks

| GPU Model | Batch Size | Embedding Time | vs. CPU | Memory Usage | Recommended For |
|-----------|------------|----------------|---------|--------------|-----------------|
| NVIDIA T4 | 32 | 82ms | 7.1x | 3.2 GB | Cost-effective inference |
| NVIDIA T4 | 128 | 198ms | 11.7x | 10.4 GB | - |
| NVIDIA A10 | 32 | 58ms | 10.0x | 3.2 GB | Balanced performance |
| NVIDIA A10 | 128 | 112ms | 20.7x | 10.4 GB | - |
| NVIDIA A100 | 32 | 30ms | 19.3x | 3.2 GB | High-throughput workloads |
| NVIDIA A100 | 128 | 62ms | 37.4x | 10.4 GB | - |
| NVIDIA H100 | 128 | ~40ms | ~58x | 10.4 GB | Maximum performance |
| NVIDIA H100 | 512 | ~120ms | ~77x | 38.2 GB | - |

## Deployment Architectures

### Backend-Only Deployment

For integrating with existing frontends or systems:

```bash
docker-compose -f docker-compose.ngc-blueprint.yml up -d
```

### Full-Stack Deployment

For a complete solution with frontend:

```bash
docker-compose -f docker-compose.nvidia-launchable.yml up -d
```

### Hybrid Deployment

For maximum flexibility, deploy the backend on GPU-accelerated infrastructure and the frontend on Vercel:

1. Deploy backend:
```bash
./setup_nvidia_backend.sh
./start_nvidia_backend.sh
```

2. Deploy frontend to Vercel:
```bash
./setup_vercel_frontend.sh
```

## TensorRT Engine Caching

The container automatically caches TensorRT optimized engines for faster startup on subsequent runs. Mount a volume to `/app/trt_engines` to preserve these caches:

```bash
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/trt_engines:/app/trt_engines \
  nvcr.io/nvidia/sap-enhanced/langchain-hana-gpu:latest
```

## Building Custom Images

To build a custom image based on this blueprint:

```bash
./build_launchable.sh
```

This script will:
1. Check for NVIDIA GPU and driver compatibility
2. Build the Docker image with TensorRT optimization
3. Test GPU access and TensorRT functionality
4. Generate NGC Blueprint configuration

## Additional Resources

- [NVIDIA Blueprint Compliance Documentation](docs/nvidia_blueprint_compliance.md)
- [Configuration Guide](docs/updated_configuration_guide.md)
- [API Documentation](docs/updated_api_documentation.md)
- [NVIDIA NGC Catalog](https://ngc.nvidia.com/catalog)
- [SAP HANA Cloud Documentation](https://help.sap.com/docs/hana-cloud)
- [LangChain Documentation](https://python.langchain.com/docs/)