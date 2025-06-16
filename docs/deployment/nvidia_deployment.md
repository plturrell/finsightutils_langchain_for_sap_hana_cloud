# NVIDIA GPU Deployment Guide

This guide provides instructions for deploying the SAP HANA Cloud langchain integration with NVIDIA GPU acceleration.

## Prerequisites

- NVIDIA GPU with CUDA support (Compute Capability 5.0+)
- NVIDIA driver compatible with CUDA 12.0+
- Docker with NVIDIA Container Toolkit installed
- Access to NVIDIA NGC registry (for pre-built containers)

## Deployment Options

### 1. Using NGC Pre-built Container

The easiest way to deploy is using our pre-built NGC container:

```bash
# Pull the container from NGC
docker pull nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest

# Run with GPU support
docker run --gpus all -p 8000:8000 \
  -e GPU_ENABLED=true \
  -e HANA_HOST=your-hana-host \
  -e HANA_PORT=your-hana-port \
  -e HANA_USER=your-hana-user \
  -e HANA_PASSWORD=your-hana-password \
  nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest
```

### 2. Using Docker Compose

For a more configurable setup, use our docker-compose configuration:

```bash
# Clone the repository
git clone https://github.com/plturrell/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud

# Configure environment variables
cp api/.env.example api/.env
# Edit api/.env with your SAP HANA Cloud credentials

# Start with GPU support
docker-compose -f api/docker-compose.yml -f api/docker-compose.gpu.yml up -d
```

### 3. Kubernetes Deployment

For production deployments, we recommend using Kubernetes with the NVIDIA device plugin:

```bash
# Install NVIDIA device plugin
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Deploy using our Helm chart
helm repo add langchain-hana https://plturrell.github.io/langchain-integration-for-sap-hana-cloud/charts
helm install langchain-hana langchain-hana/langchain-hana-gpu --set hana.host=your-hana-host,hana.port=your-hana-port
```

## GPU Configuration Options

### Multi-GPU Support

The service automatically detects and utilizes all available GPUs. To limit to specific GPUs:

```bash
# Using Docker, limit to first GPU
docker run --gpus '"device=0"' -p 8000:8000 [...]

# Using environment variables
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0,1 -p 8000:8000 [...]
```

### Memory Optimization

Configure memory optimization settings:

| Environment Variable | Description | Default |
|----------------------|-------------|---------|
| `MAX_BATCH_SIZE` | Maximum batch size for embedding operations | `512` |
| `DYNAMIC_BATCHING` | Enable dynamic batch sizing | `true` |
| `MIN_MEMORY_GB` | Minimum free GPU memory required (GB) | `2` |
| `TENSOR_PARALLEL_SIZE` | Number of GPUs for tensor parallelism | `1` |

Example:

```bash
docker run --gpus all \
  -e MAX_BATCH_SIZE=1024 \
  -e DYNAMIC_BATCHING=true \
  -e MIN_MEMORY_GB=4 \
  -e TENSOR_PARALLEL_SIZE=2 \
  -p 8000:8000 [...]
```

## Performance Tuning

### Benchmarking

The API includes benchmark endpoints to measure performance:

```bash
# Run embedding benchmark
curl -X POST "http://localhost:8000/benchmark/embedding" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["sample text"], "count": 1000, "batch_size": 32}'

# Run vector search benchmark
curl -X POST "http://localhost:8000/benchmark/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "sample query", "k": 10, "iterations": 100}'
```

### Recommended Instance Types

| Workload | NVIDIA GPU | Cloud Instance |
|----------|------------|----------------|
| Development | T4 (16GB) | AWS g4dn.xlarge, GCP n1-standard-4 + T4 |
| Medium Production | A10 (24GB) | AWS g5.xlarge, Azure NC A10 v4 |
| Large Production | A100 (40/80GB) | AWS p4d.24xlarge, GCP a2-highgpu-1g |

## Monitoring

### GPU Metrics

The API exposes Prometheus metrics at `/metrics` including:

- GPU utilization
- GPU memory usage
- Batch processing throughput
- Embedding latency

To view these metrics:

```bash
# Install NVIDIA DCGM Exporter in Kubernetes
helm install --name dcgm-exporter \
  nvidia/dcgm-exporter \
  -n gpu-monitoring

# Configure Prometheus to scrape metrics
# See api/prometheus.yml for example configuration
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size using `MAX_BATCH_SIZE` environment variable
2. **Slow first request**: This is normal due to CUDA initialization; subsequent requests will be faster
3. **GPU not detected**: Ensure NVIDIA drivers are installed and visible to Docker

### Logs

To view detailed GPU operations in logs:

```bash
# Enable debug logging
docker run -e LOG_LEVEL=DEBUG -e GPU_DEBUG=true --gpus all [...]
```

### Support

For additional support with NVIDIA GPU deployment:
- Visit our [GitHub Issues](https://github.com/plturrell/langchain-integration-for-sap-hana-cloud/issues)
- Contact us at [support@example.com](mailto:support@example.com)
- Check the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/)