#\!/bin/bash
#
# NVIDIA Deployment Script for SAP HANA Cloud LangChain Integration
# 
# This script deploys the application on NVIDIA GPU-enabled environment
# with comprehensive monitoring and optimization

set -e  # Exit on error

# Configuration
CONFIG_FILE=".env.nvidia"
DOCKER_COMPOSE_FILE="docker/docker-compose.nvidia.yml"
DEFAULT_JWT_SECRET="$(openssl rand -hex 32)"

# Print banner
echo "======================================================================"
echo "    SAP HANA Cloud LangChain Integration - NVIDIA GPU Deployment"
echo "======================================================================"
echo

# Check requirements
echo "Checking requirements..."

# Check if Docker is installed
if \! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if \! docker compose version &> /dev/null; then
    echo "ERROR: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed
if \! docker info | grep -q "Runtimes:.*nvidia"; then
    echo "WARNING: NVIDIA Container Toolkit might not be properly installed."
    echo "This is required for GPU support. Continue anyway? (y/n)"
    read -r continue_anyway
    if [[ "$continue_anyway" \!= "y" ]]; then
        exit 1
    fi
fi

# Check for NVIDIA GPU
if \! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. This might indicate NVIDIA drivers are not properly installed."
    echo "This is required for GPU support. Continue anyway? (y/n)"
    read -r continue_anyway
    if [[ "$continue_anyway" \!= "y" ]]; then
        exit 1
    fi
else
    echo "NVIDIA GPU information:"
    nvidia-smi
fi

# Check for config file or create it
if [[ \! -f "$CONFIG_FILE" ]]; then
    echo "Configuration file $CONFIG_FILE not found. Creating it..."
    
    # Get required info from user
    read -p "SAP HANA Host: " HANA_HOST
    read -p "SAP HANA Port [443]: " HANA_PORT
    HANA_PORT=${HANA_PORT:-443}
    read -p "SAP HANA User: " HANA_USER
    read -sp "SAP HANA Password: " HANA_PASSWORD
    echo
    read -p "JWT Secret [auto-generated]: " JWT_SECRET
    JWT_SECRET=${JWT_SECRET:-$DEFAULT_JWT_SECRET}
    
    # Create config file
    cat > "$CONFIG_FILE" << EOFCONFIG
# SAP HANA Cloud Connection
HANA_HOST=$HANA_HOST
HANA_PORT=$HANA_PORT
HANA_USER=$HANA_USER
HANA_PASSWORD=$HANA_PASSWORD
DEFAULT_TABLE_NAME=EMBEDDINGS

# API Configuration
LOG_LEVEL=INFO
ENABLE_CORS=true
CORS_ORIGINS=https://example.com,http://localhost:3000
JWT_SECRET=$JWT_SECRET
DB_MAX_CONNECTIONS=5
DB_CONNECTION_TIMEOUT=600

# GPU Acceleration
USE_TENSORRT=true
TENSORRT_PRECISION=fp16
BATCH_SIZE=32
MAX_BATCH_SIZE=128
ENABLE_MULTI_GPU=true
GPU_MEMORY_FRACTION=0.9

# Advanced GPU Optimization
DALI_ENABLED=true
USE_TRANSFORMER_ENGINE=true
NVTX_PROFILING_ENABLED=false
AUTO_TUNE_ENABLED=true
AUTO_TUNE_DURATION_MINUTES=60

# Error Handling
ENABLE_CONTEXT_AWARE_ERRORS=true
ERROR_VERBOSITY=standard
ENABLE_ERROR_TELEMETRY=true

# Vector Operations
ENABLE_PRECISE_SIMILARITY=true
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENABLE_VECTOR_VISUALIZATION=true

# Monitoring
ENABLE_PROMETHEUS=true
ENABLE_OPENTELEMETRY=true
DCGM_ENABLED=true

# Application Version
APP_VERSION=1.0.0
EOFCONFIG
    
    echo "Configuration file created at $CONFIG_FILE"
else
    echo "Using existing configuration file: $CONFIG_FILE"
fi

# Check if .env.nvidia exists and load it
if [[ -f "$CONFIG_FILE" ]]; then
    export $(grep -v '^#' "$CONFIG_FILE" | xargs)
fi

# Setup directories for monitoring configuration
echo "Setting up monitoring configuration..."

# Create directories
mkdir -p prometheus dcgm

# Create Prometheus configuration
cat > prometheus/prometheus.yml << EOFPROM
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']
  
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
  
  - job_name: 'triton'
    static_configs:
      - targets: ['triton-server:8002']
EOFPROM

# Create DCGM Exporter configuration
cat > dcgm/default-counters.csv << EOFDCGM
# Format,,
# DCGM FI Field ID, Prometheus metric type, help message

# GPU utilization
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %)

# Memory utilization
DCGM_FI_DEV_FB_USED, gauge, GPU framebuffer memory used (in MiB)
DCGM_FI_DEV_FB_FREE, gauge, GPU framebuffer memory free (in MiB)
DCGM_FI_DEV_FB_TOTAL, gauge, GPU framebuffer memory total (in MiB)

# SM clocks
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)

# Memory clocks
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)

# Power usage
DCGM_FI_DEV_POWER_USAGE, gauge, Power usage (in W)

# Temperature
DCGM_FI_DEV_GPU_TEMP, gauge, GPU temperature (in C)

# PCIe throughput
DCGM_FI_DEV_PCIE_TX_THROUGHPUT, gauge, PCIe transmit throughput (in KB/s)
DCGM_FI_DEV_PCIE_RX_THROUGHPUT, gauge, PCIe receive throughput (in KB/s)
EOFDCGM

# Create continuous learning configuration
mkdir -p config
cat > config/continuous_learning_config.json << EOFCONFIG1
{
  "parameters": {
    "batch_size": {
      "default": 32,
      "min": 1,
      "max": 128,
      "step": 1
    },
    "gpu_memory_fraction": {
      "default": 0.8,
      "min": 0.1,
      "max": 0.95,
      "step": 0.05
    },
    "worker_count": {
      "default": 4,
      "min": 1,
      "max": 16,
      "step": 1
    }
  },
  "metrics": {
    "throughput": {"weight": 0.5},
    "latency": {"weight": 0.3},
    "memory_usage": {"weight": 0.2}
  }
}
EOFCONFIG1

# Create shared configuration for auto-tuning
cat > config/auto_tuned_config.json << EOFCONFIG2
{
  "batch_sizes": {
    "default": 32,
    "embedding_generation": 64,
    "vector_search": 16,
    "max_batch_size": 128
  },
  "precision": "fp16",
  "worker_counts": {
    "api_workers": 4,
    "gpu_workers": 1,
    "db_pool_size": 8,
    "thread_count": 4
  },
  "memory_allocation": {
    "memory_fraction": 0.8,
    "cache_size_mb": 2048,
    "max_workspace_size_mb": 1024
  },
  "hnsw_parameters": {
    "m": 16,
    "ef_construction": 100,
    "ef_search": 50
  }
}
EOFCONFIG2

# Deploy with Docker Compose
echo "Deploying with Docker Compose..."
docker compose -f "$DOCKER_COMPOSE_FILE" --env-file "$CONFIG_FILE" up -d

# Wait for services to start
echo "Waiting for services to start..."
sleep 5

# Check service health
echo "Checking service health..."
docker compose -f "$DOCKER_COMPOSE_FILE" ps

# Show logs
echo "Showing initial logs..."
docker compose -f "$DOCKER_COMPOSE_FILE" logs --tail=20

echo
echo "======================================================================"
echo "    Deployment Complete\! Services are starting up..."
echo "======================================================================"
echo
echo "You can access the services at:"
echo "- API: http://localhost:8000"
echo "- API Documentation: http://localhost:8000/docs"
echo "- Frontend: http://localhost:3000"
echo "- Prometheus: http://localhost:9090"
echo "- DCGM Metrics: http://localhost:9400/metrics"
echo
echo "Check container logs with:"
echo "  docker compose -f $DOCKER_COMPOSE_FILE logs -f"
echo
echo "To stop the deployment:"
echo "  docker compose -f $DOCKER_COMPOSE_FILE down"
echo
echo "To stop and remove volumes:"
echo "  docker compose -f $DOCKER_COMPOSE_FILE down -v"
echo
echo "======================================================================"
