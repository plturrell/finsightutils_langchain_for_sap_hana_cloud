#!/bin/bash
# NVIDIA Deployment script for SAP HANA Cloud LangChain Integration

set -e

# Display banner
echo "======================================================================"
echo "  NVIDIA GPU Deployment for SAP HANA Cloud LangChain Integration"
echo "======================================================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is installed
if ! docker info | grep -q "Runtimes.*nvidia"; then
    echo "Warning: NVIDIA Docker runtime doesn't appear to be installed."
    echo "You may need to install the NVIDIA Container Toolkit first."
    echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    read -p "Continue anyway? (y/n): " continue_answer
    if [[ "$continue_answer" != "y" ]]; then
        echo "Deployment aborted."
        exit 1
    fi
fi

# Check if environment file exists
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    echo "Environment file $ENV_FILE not found."
    echo "Creating example environment file..."
    cat > "$ENV_FILE" << EOF
# SAP HANA Cloud Connection
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password
DEFAULT_TABLE_NAME=EMBEDDINGS

# API Configuration
LOG_LEVEL=INFO
ENABLE_CORS=true
CORS_ORIGINS=*
JWT_SECRET=your-secret-key
DB_MAX_CONNECTIONS=5
DB_CONNECTION_TIMEOUT=600

# GPU Acceleration
TENSORRT_PRECISION=fp16
BATCH_SIZE=32
MAX_BATCH_SIZE=128
ENABLE_MULTI_GPU=true
GPU_MEMORY_FRACTION=0.9

# Error Handling
ENABLE_CONTEXT_AWARE_ERRORS=true
ERROR_VERBOSITY=standard
ENABLE_ERROR_TELEMETRY=true

# Vector Operations
ENABLE_PRECISE_SIMILARITY=true
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EOF
    echo "Please edit the $ENV_FILE file with your SAP HANA Cloud credentials."
    read -p "Continue with deployment after editing? (y/n): " edit_answer
    if [[ "$edit_answer" != "y" ]]; then
        echo "Deployment aborted. Please edit the $ENV_FILE file and run the script again."
        exit 1
    fi
fi

# Build and start the containers
echo "Building and starting the containers..."
docker-compose -f docker/docker-compose.nvidia.yml up -d --build

# Check if the containers are running
echo "Checking if the containers are running..."
if docker ps | grep -q "sap-hana-langchain-api-nvidia"; then
    echo "API container is running."
else
    echo "Error: API container failed to start."
    echo "Checking container logs..."
    docker logs sap-hana-langchain-api-nvidia
    exit 1
fi

if docker ps | grep -q "sap-hana-langchain-frontend"; then
    echo "Frontend container is running."
else
    echo "Warning: Frontend container failed to start."
    echo "Checking container logs..."
    docker logs sap-hana-langchain-frontend
fi

# Display success message
echo "======================================================================"
echo "  Deployment completed successfully!"
echo "======================================================================"
echo "API is available at: http://localhost:8000"
echo "Frontend is available at: http://localhost:3000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "To check GPU utilization, run: nvidia-smi"
echo "To view container logs, run: docker logs sap-hana-langchain-api-nvidia"
echo "To stop the deployment, run: docker-compose -f docker/docker-compose.nvidia.yml down"
echo "======================================================================"

exit 0