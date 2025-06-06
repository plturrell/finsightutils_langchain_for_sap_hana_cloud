# Docker Configuration Files

This directory contains Docker Compose configuration files for different deployment scenarios.

## Available Docker Compose Files

| File | Purpose | Use Case |
|------|---------|----------|
| `docker-compose.yml` | Standard CPU deployment | Basic deployment without GPU acceleration |
| `docker-compose.nvidia.yml` | NVIDIA GPU deployment | Enhanced deployment with NVIDIA GPU acceleration |
| `docker-compose.ngc-blueprint.yml` | NGC Blueprint deployment | Optimized for NVIDIA NGC platform |
| `docker-compose.vercel-backend.yml` | Vercel backend | Backend for Vercel frontend deployment |

## Usage Examples

### Standard Deployment

```bash
# Deploy with standard CPU configuration
docker-compose -f config/docker/docker-compose.yml up -d
```

### NVIDIA GPU Deployment

```bash
# Deploy with NVIDIA GPU acceleration
docker-compose -f config/docker/docker-compose.nvidia.yml up -d
```

### NGC Blueprint Deployment

```bash
# Deploy with NGC Blueprint configuration
docker-compose -f config/docker/docker-compose.ngc-blueprint.yml up -d
```

### Vercel Backend Deployment

```bash
# Deploy backend for Vercel frontend
docker-compose -f config/docker/docker-compose.vercel-backend.yml up -d
```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```
# SAP HANA Cloud Connection
HANA_HOST=your-hana-hostname.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your-hana-user
HANA_PASSWORD=your-hana-password
DEFAULT_TABLE_NAME=EMBEDDINGS

# Deployment Settings
TENSORRT_PRECISION=fp16  # Options: fp32, fp16, int8
BATCH_SIZE=32
MAX_BATCH_SIZE=128
ENABLE_MULTI_GPU=true
GPU_COUNT=1  # For multi-GPU setups

# Authentication
JWT_SECRET=your-secret-key
REQUIRE_AUTH=false

# Vercel Integration
VERCEL_URL=your-vercel-url.vercel.app
FRONTEND_URL=https://your-frontend-url.com
```

## Docker Compose File Details

### docker-compose.yml

Standard deployment with CPU processing:
- FastAPI backend
- Static frontend
- No GPU acceleration

### docker-compose.nvidia.yml

NVIDIA GPU-accelerated deployment:
- NVIDIA GPU-enabled container
- TensorRT optimization
- Multi-GPU support
- Engine caching

### docker-compose.ngc-blueprint.yml

NGC Blueprint-compatible deployment:
- Built on NGC PyTorch container
- Optimized for NGC platform
- Advanced GPU acceleration
- Performance tuning

### docker-compose.vercel-backend.yml

Backend for Vercel frontend:
- FastAPI backend with GPU acceleration
- Configured for Vercel frontend integration
- Authentication support
- CORS configured for Vercel domains