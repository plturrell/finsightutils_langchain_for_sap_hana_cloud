# SAP HANA Cloud LangChain Integration - NVIDIA NGC Blueprint

This README provides instructions for deploying the SAP HANA Cloud LangChain Integration as an NVIDIA NGC Blueprint.

## Overview

The SAP HANA Cloud LangChain Integration with GPU Acceleration is an enterprise-ready solution for SAP HANA Cloud vector store operations that leverages NVIDIA GPUs for high-performance embedding generation and vector operations.

### Key Features

- TensorRT optimization for maximum inference speed
- Multi-GPU load balancing for high throughput
- Dynamic batch sizing based on GPU memory
- Mixed precision support (FP16, FP32, INT8)
- Context-aware error handling with intelligent suggestions
- Interactive 3D vector visualization

## Requirements

- NVIDIA GPU (T4, A10, A100, or H100 recommended)
- NVIDIA Driver 520.0 or later
- CUDA 11.8 or later
- TensorRT 8.6.0 or later
- Docker 20.10.0 or later
- NVIDIA Container Toolkit 1.14.0 or later

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
git clone https://github.com/your-org/sap-hana-langchain-ngc-blueprint.git
cd sap-hana-langchain-ngc-blueprint
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

## GPU Optimization

The container is optimized for different NVIDIA GPU models:

| GPU Model | Recommended Batch Size | Memory Usage | Performance |
|-----------|------------------------|--------------|-------------|
| NVIDIA T4 | 32-64 | 3.2-10.4 GB | 4-12x CPU |
| NVIDIA A10 | 64-128 | 5.8-19.6 GB | 6-21x CPU |
| NVIDIA A100 | 128-256 | 10.4-38.2 GB | 9-37x CPU |
| NVIDIA H100 | 256+ | 19.6-70+ GB | 12-50x CPU |

## Access

Once deployed, access the following endpoints:

- API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- GPU Information: http://localhost:8000/gpu-info
- Health Check: http://localhost:8000/health/ping

## TensorRT Engine Caching

The container caches TensorRT optimized engines for faster startup on subsequent runs. Mount a volume to `/app/trt_engines` to preserve these caches.

## NVIDIA NGC Blueprint Configuration

This repository includes a complete NGC Blueprint configuration (`nvidia-blueprint.yaml`) that defines:

- Resource requirements (GPU, memory, CPU)
- Container image and ports
- Environment variables
- Documentation (overview, quickstart, performance)

## Building Custom Images

To build a custom image based on this blueprint:

```bash
./build_launchable.sh
```

This script will:
1. Verify NVIDIA GPU and driver compatibility
2. Build the Docker image with TensorRT optimization
3. Test GPU access and TensorRT functionality
4. Generate NGC Blueprint configuration

## Additional Resources

- [Documentation](docs/nvidia_blueprint_compliance.md)
- [NVIDIA NGC Catalog](https://ngc.nvidia.com/catalog)
- [SAP HANA Cloud Documentation](https://help.sap.com/docs/hana-cloud)
- [LangChain Documentation](https://python.langchain.com/docs/)