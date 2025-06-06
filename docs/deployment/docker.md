# Docker Deployment Options

This project provides two Docker Compose configurations for different deployment scenarios.

## 1. CPU-Only Deployment (Local Testing)

For local testing without GPU requirements:

```bash
docker-compose -f docker-compose.cpu.yml up
```

This configuration:
- Uses Python 3.9 base image
- Mounts the application files as volumes (no rebuild needed for code changes)
- Runs in CPU-only mode
- Available at http://localhost:8000

### Testing the CPU Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check system info
curl http://localhost:8000/system-info

# Test embeddings
curl -X POST "http://localhost:8000/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["This is a test sentence for embeddings"]}'
```

## 2. NVIDIA GPU Deployment

For deployment on NVIDIA GPU environments (including NVIDIA Launchables):

```bash
docker-compose -f docker-compose.nvidia.yml up
```

This configuration:
- Uses NVIDIA PyTorch base image
- Builds from Dockerfile.nvidia
- Enables GPU acceleration with CUDA
- Optimizes with TensorRT
- Uses multiple workers for better performance
- Available at http://localhost:8000

### Testing the GPU Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check GPU info
curl http://localhost:8000/gpu-info

# Run tensor operation test (GPU performance)
curl http://localhost:8000/tensor-test

# Test embeddings
curl -X POST "http://localhost:8000/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["This is a test sentence for embeddings"]}'
```

## Automated Build Scripts

For advanced deployment options, use the provided scripts:

- `./build_nvidia_local.sh` - Build and test locally with or without GPU
- `./build_launchable.sh` - Prepare and push for NVIDIA Launchables deployment

## Troubleshooting

### Missing dependencies
If you see errors about missing dependencies in the CPU version:

```bash
docker exec -it $(docker ps -q --filter name=langchain) pip install missing-package-name
```

### Container not starting
Check logs with:

```bash
docker-compose -f docker-compose.cpu.yml logs
# or
docker-compose -f docker-compose.nvidia.yml logs
```

### Health check failing
Ensure curl is installed in the container:

```bash
docker exec -it $(docker ps -q --filter name=langchain) apt-get update && apt-get install -y curl
```