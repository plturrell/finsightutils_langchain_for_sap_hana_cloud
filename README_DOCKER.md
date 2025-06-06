# Docker Deployment Guide for SAP HANA LangChain Integration

This guide provides detailed instructions for deploying the SAP HANA LangChain integration with Docker, including standard deployment, NVIDIA GPU acceleration, and blue-green deployment for zero-downtime updates.

> **Important Update**: All Docker files have been consolidated in the [docker/](./docker/) directory for better organization. Please use the configurations from this directory for deployment.

## Table of Contents

- [Standard Deployment](#standard-deployment)
- [NVIDIA GPU Deployment](#nvidia-gpu-deployment)
- [Blue-Green Deployment](#blue-green-deployment)
- [Environment Variables](#environment-variables)
- [Volume Management](#volume-management)
- [Monitoring](#monitoring)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Standard Deployment

### Prerequisites

- Docker and Docker Compose installed
- SAP HANA Cloud account and connection details
- Sufficient memory and CPU resources

### Deployment Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/langchain-integration-for-sap-hana-cloud.git
   cd langchain-integration-for-sap-hana-cloud
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your SAP HANA Cloud connection details
   ```

3. Start the containers:
   ```bash
   docker-compose -f docker/docker-compose.yml up -d
   ```

4. Verify deployment:
   ```bash
   docker-compose -f docker/docker-compose.yml ps
   curl http://localhost:8000/health/ping
   ```

## NVIDIA GPU Deployment

### Prerequisites

- NVIDIA GPU with CUDA support (T4 recommended)
- NVIDIA Container Toolkit installed
- Docker and Docker Compose installed
- SAP HANA Cloud account and connection details

### Deployment Steps

1. Verify NVIDIA Container Toolkit installation:
   ```bash
   docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env.nvidia
   # Edit .env.nvidia file with your SAP HANA Cloud connection details
   # Add GPU-specific environment variables
   ```

3. Start the containers with NVIDIA GPU support:
   ```bash
   docker-compose -f docker/docker-compose.nvidia.yml --env-file .env.nvidia up -d
   ```

4. Verify deployment:
   ```bash
   docker-compose -f docker/docker-compose.nvidia.yml ps
   curl http://localhost:8000/health/ping
   ```

5. Verify GPU utilization:
   ```bash
   docker exec -it sap-hana-langchain-api-nvidia nvidia-smi
   ```

## Blue-Green Deployment

Blue-green deployment enables zero-downtime updates by running two identical environments ("blue" and "green") and switching traffic between them when deploying new versions.

### Prerequisites

- All NVIDIA GPU deployment prerequisites
- Traefik as a load balancer (included in the configuration)

### Architecture

The blue-green deployment consists of:

1. **Blue environment**: The initially active deployment
2. **Green environment**: A parallel deployment for updates
3. **Traefik**: Load balancer that directs traffic to the active deployment
4. **Health checker**: Service that monitors health and manages traffic switching
5. **Prometheus & Grafana**: For monitoring deployment health and performance

### Initial Deployment

1. Set up environment variables:
   ```bash
   cp .env.example .env.blue-green
   # Edit .env.blue-green with your configuration
   
   # Set version environment variables
   echo "BLUE_VERSION=1.0.0" >> .env.blue-green
   echo "GREEN_VERSION=1.0.0" >> .env.blue-green
   ```

2. Start the blue-green deployment:
   ```bash
   docker-compose -f docker/docker-compose.blue-green.yml --env-file .env.blue-green up -d
   ```

3. Verify the deployment:
   ```bash
   docker-compose -f docker/docker-compose.blue-green.yml ps
   
   # Check which deployment is active
   curl http://localhost/api/deployment/status
   ```

### Performing a Blue-Green Update

1. Update the idle deployment (assuming blue is active):
   ```bash
   # Update GREEN_VERSION in .env.blue-green
   sed -i 's/GREEN_VERSION=.*/GREEN_VERSION=1.0.1/g' .env.blue-green
   
   # Apply the update to only the green deployment
   docker-compose -f config/docker/docker-compose.blue-green.yml --env-file .env.blue-green up -d api-green
   ```

2. Monitor the health of the green deployment:
   ```bash
   # Check green deployment health directly
   curl http://localhost:8001/health/status
   
   # Check health checker logs
   docker logs sap-hana-langchain-healthcheck
   ```

3. Once green is healthy, the health checker will automatically switch traffic. Alternatively, force the switch:
   ```bash
   # Using the provided script
   ./scripts/switch-deployment.sh green
   
   # Or manually update Traefik labels
   docker-compose -f config/docker/docker-compose.blue-green.yml exec healthcheck python -c "from healthcheck import switch_traffic; switch_traffic('green')"
   ```

4. Verify the switch:
   ```bash
   # Verify active deployment
   curl http://localhost/api/deployment/status
   
   # Verify your application is working
   curl http://localhost/api/health/ping
   ```

5. Once confirmed, update the previously active deployment (blue):
   ```bash
   # Update BLUE_VERSION in .env.blue-green
   sed -i 's/BLUE_VERSION=.*/BLUE_VERSION=1.0.1/g' .env.blue-green
   
   # Apply the update to the blue deployment
   docker-compose -f config/docker/docker-compose.blue-green.yml --env-file .env.blue-green up -d api-blue
   ```

### Rollback Procedure

If issues are detected with the new deployment:

```bash
# Roll back to the previous deployment (assuming switched to green)
./scripts/switch-deployment.sh blue

# Verify rollback
curl http://localhost/api/deployment/status
```

## Environment Variables

### Common Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HANA_HOST` | SAP HANA Cloud host | (required) |
| `HANA_PORT` | SAP HANA Cloud port | 443 |
| `HANA_USER` | SAP HANA Cloud username | (required) |
| `HANA_PASSWORD` | SAP HANA Cloud password | (required) |
| `DEFAULT_TABLE_NAME` | Default table name for vectors | EMBEDDINGS |

### GPU-Specific Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GPU_ENABLED` | Enable GPU acceleration | true |
| `USE_TENSORRT` | Enable TensorRT optimization | true |
| `TENSORRT_PRECISION` | Precision for TensorRT (fp32, fp16, int8) | fp16 |
| `BATCH_SIZE` | Default batch size for processing | 32 |
| `MAX_BATCH_SIZE` | Maximum batch size | 128 |
| `ENABLE_MULTI_GPU` | Enable multi-GPU support | true |
| `GPU_MEMORY_FRACTION` | Fraction of GPU memory to use | 0.9 |

### Blue-Green Deployment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `BLUE_VERSION` | Version of the blue deployment | 1.0.0 |
| `GREEN_VERSION` | Version of the green deployment | 1.0.0 |
| `CHECK_INTERVAL` | Health check interval in seconds | 30 |
| `SWITCH_THRESHOLD` | Required consecutive successful checks before switching | 3 |

## Volume Management

The deployment uses Docker volumes to persist data:

### Standard Deployment
- `trt_engines`: TensorRT engine cache
- `api-data`: Application data
- `api-logs`: Application logs

### NVIDIA GPU Deployment
- `nvidia-trt-engines`: TensorRT engine cache
- `nvidia-api-data`: Application data
- `nvidia-api-logs`: Application logs

### Blue-Green Deployment
- Separate volumes for blue and green deployments
- `prometheus-data`: Prometheus time-series data
- `grafana-data`: Grafana dashboards and configuration

## Monitoring

The blue-green deployment includes a Prometheus and Grafana monitoring stack:

1. **Metrics**: Available at http://localhost:9090
2. **Dashboards**: Available at http://localhost:3001 (admin/admin)

Key metrics to monitor:
- GPU utilization and memory usage
- API response times
- Error rates
- Health check status

## Security

Security considerations for Docker deployment:

1. **Secrets Management**:
   - Never store sensitive information in Docker Compose files
   - Use environment variables or Docker secrets
   
2. **Network Isolation**:
   - Use Docker networks to isolate services
   - Only expose necessary ports
   
3. **Container Hardening**:
   - Use non-root users inside containers
   - Apply principle of least privilege
   
4. **Resource Limits**:
   - Set memory and CPU limits for containers
   - Configure GPU memory limits

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```bash
   # Check NVIDIA Container Toolkit installation
   docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   
   # Verify GPU availability in container
   docker exec -it sap-hana-langchain-api-nvidia nvidia-smi
   ```

2. **Connection Issues to SAP HANA**:
   ```bash
   # Check connection from container
   docker exec -it sap-hana-langchain-api-nvidia python -c "from hdbcli import dbapi; conn = dbapi.connect(address='$HANA_HOST', port=$HANA_PORT, user='$HANA_USER', password='$HANA_PASSWORD'); print(conn.isconnected())"
   ```

3. **Blue-Green Switching Issues**:
   ```bash
   # Check health checker logs
   docker logs sap-hana-langchain-healthcheck
   
   # Check Traefik routing configuration
   curl http://localhost:8080/api/http/routers
   ```

### Logs

Access container logs for troubleshooting:

```bash
# Standard deployment
docker-compose logs -f api

# NVIDIA GPU deployment
docker-compose -f config/docker/docker-compose.nvidia.yml logs -f api

# Blue-Green deployment
docker-compose -f config/docker/docker-compose.blue-green.yml logs -f api-blue api-green traefik healthcheck
```

For more assistance, please refer to the [troubleshooting documentation](/docs/troubleshooting.md).