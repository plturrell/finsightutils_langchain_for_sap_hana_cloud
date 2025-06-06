# Docker Deployment for SAP HANA Cloud LangChain Integration

This directory contains Docker configurations for deploying the SAP HANA Cloud LangChain Integration in various environments.

## Directory Structure

```
docker/
├── config/                # Configuration files for monitoring and services
│   ├── grafana/           # Grafana dashboard configurations
│   └── prometheus/        # Prometheus monitoring configurations
├── healthcheck/           # Blue-green deployment health checker
├── Dockerfile             # Standard CPU-only Dockerfile
├── Dockerfile.dev         # Development Dockerfile with hot reload
├── Dockerfile.frontend    # Frontend production Dockerfile
├── Dockerfile.frontend.dev # Frontend development Dockerfile
├── Dockerfile.nvidia      # NVIDIA GPU-accelerated Dockerfile
├── docker-compose.yml     # Standard deployment with CPU
├── docker-compose.dev.yml # Development environment
├── docker-compose.nvidia.yml # NVIDIA GPU-accelerated deployment
└── docker-compose.blue-green.yml # Blue-green deployment for zero downtime
```

## Deployment Options

### Standard Deployment (CPU)

For standard deployment without GPU acceleration:

```bash
docker-compose -f docker/docker-compose.yml up -d
```

### Development Environment

For local development with hot reload:

```bash
docker-compose -f docker/docker-compose.dev.yml up -d
```

### NVIDIA GPU Deployment

For GPU-accelerated deployment:

```bash
docker-compose -f docker/docker-compose.nvidia.yml up -d
```

### Blue-Green Deployment (Zero Downtime)

For production environments with zero-downtime updates:

```bash
docker-compose -f docker/docker-compose.blue-green.yml up -d
```

## Environment Variables

Create a `.env` file with the following variables:

```bash
# SAP HANA Cloud Connection
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password
DEFAULT_TABLE_NAME=EMBEDDINGS

# API Configuration
PORT=8000
LOG_LEVEL=INFO
ENABLE_CORS=true
CORS_ORIGINS=*
JWT_SECRET=your-secret-key
DB_MAX_CONNECTIONS=5
DB_CONNECTION_TIMEOUT=600

# GPU Acceleration (for NVIDIA deployment)
GPU_ENABLED=true
USE_TENSORRT=true
TENSORRT_PRECISION=fp16
BATCH_SIZE=32
MAX_BATCH_SIZE=128
ENABLE_MULTI_GPU=true
GPU_MEMORY_FRACTION=0.9

# Blue-Green Deployment (if using docker-compose.blue-green.yml)
BLUE_VERSION=1.0.0
GREEN_VERSION=1.0.0
CHECK_INTERVAL=30
SWITCH_THRESHOLD=3
GRAFANA_ADMIN_PASSWORD=admin
```

## Blue-Green Deployment

The blue-green deployment setup includes:

1. **Two identical environments**:
   - Blue deployment (initially active)
   - Green deployment (for updates)

2. **Traffic Management**:
   - Traefik reverse proxy routes traffic to the active deployment
   - Health checker monitors both deployments and switches traffic when needed

3. **Monitoring**:
   - Prometheus collects metrics
   - Grafana provides dashboards for monitoring

### Switching Deployments

To switch between blue and green deployments:

```bash
./scripts/switch-deployment.sh [blue|green]
```

## Troubleshooting

### Checking Logs

```bash
# View API logs
docker logs sap-hana-langchain-api

# View frontend logs
docker logs sap-hana-langchain-frontend

# View blue-green deployment logs
docker logs sap-hana-langchain-healthcheck
```

### Health Checks

```bash
# Check API health
curl http://localhost:8000/health/ping

# Check detailed status
curl http://localhost:8000/health/status
```

## Further Documentation

For more details, refer to the main [README_DOCKER.md](../README_DOCKER.md) file.