#!/bin/bash
# Script to deploy the SAP HANA Cloud LangChain Integration to a VM with GPU support

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Deploying SAP HANA Cloud LangChain Integration to VM${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}WARNING: NVIDIA drivers not detected.${NC}"
    echo -e "${YELLOW}For GPU support, ensure NVIDIA drivers and NVIDIA Container Toolkit are installed.${NC}"
    echo -e "${YELLOW}See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
    USE_GPU=false
else
    echo -e "${GREEN}NVIDIA drivers detected. GPU support will be enabled.${NC}"
    USE_GPU=true
    echo -e "${BLUE}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Navigate to the project directory
cd "$(dirname "$0")"

# Update .env file in api directory
echo -e "${BLUE}Updating API configuration...${NC}"
cat > api/.env << EOF
# GPU Configuration
GPU_ENABLED=${USE_GPU}
USE_TENSORRT=${USE_GPU}
TENSORRT_PRECISION=fp16

# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG
ENABLE_CORS=true

# Performance
DEFAULT_TIMEOUT=30
HEALTH_CHECK_TIMEOUT=10
EMBEDDING_TIMEOUT=60
SEARCH_TIMEOUT=45
CONNECTION_TEST_TIMEOUT=5

# Authentication
JWT_SECRET=sap-hana-langchain-t4-integration-secret-key-2025
REQUIRE_AUTH=false

# Error Handling
ENABLE_ERROR_CONTEXT=true
ENABLE_DETAILED_LOGGING=true
ENABLE_MEMORY_TRACKING=true
MAX_RETRY_COUNT=3
RETRY_DELAY_MS=1000
ENABLE_SSL_VERIFICATION=true
EOF

# Copy environment file for the frontend
echo -e "${BLUE}Updating frontend configuration...${NC}"
cat > frontend/.env << EOF
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENABLE_AUTH=false
EOF

# Build and start the services
echo -e "${BLUE}Building and starting services...${NC}"
if [ "$USE_GPU" = true ]; then
    echo -e "${GREEN}Starting with GPU support...${NC}"
    docker-compose -f api/docker-compose.yml -f api/docker-compose.gpu.yml up -d --build
else
    echo -e "${YELLOW}Starting without GPU support...${NC}"
    docker-compose -f api/docker-compose.yml up -d --build
fi

# Wait for services to start
echo -e "${BLUE}Waiting for services to start...${NC}"
sleep 10

# Check if services are running
if docker-compose -f api/docker-compose.yml ps | grep -q "Up"; then
    echo -e "${GREEN}Services are running!${NC}"
    
    # Get container IPs
    API_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker-compose -f api/docker-compose.yml ps -q api))
    FRONTEND_IP=$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker-compose -f api/docker-compose.yml ps -q frontend))
    
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${BLUE}API URL:${NC} http://localhost:8000"
    echo -e "${BLUE}API Documentation:${NC} http://localhost:8000/docs"
    echo -e "${BLUE}Frontend URL:${NC} http://localhost:3000"
    
    if [ "$USE_GPU" = true ]; then
        echo -e "${BLUE}GPU Info:${NC} http://localhost:8000/benchmark/gpu_info"
    fi
    
    echo -e "\n${BLUE}Container Information:${NC}"
    echo -e "${BLUE}API Container IP:${NC} $API_IP"
    echo -e "${BLUE}Frontend Container IP:${NC} $FRONTEND_IP"
    
    echo -e "\n${BLUE}To view logs:${NC}"
    echo -e "docker-compose -f api/docker-compose.yml logs -f"
    
    echo -e "\n${BLUE}To stop the services:${NC}"
    echo -e "docker-compose -f api/docker-compose.yml down"
else
    echo -e "${RED}Service startup failed. Check the logs:${NC}"
    docker-compose -f api/docker-compose.yml logs
    exit 1
fi