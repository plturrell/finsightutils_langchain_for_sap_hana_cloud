#!/bin/bash
# Simple script to deploy the frontend and backend on a T4 GPU VM

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Deploying SAP HANA Cloud LangChain Integration to T4 GPU${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Navigate to the project directory
cd "$(dirname "$0")"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Installing Docker...${NC}"
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    echo -e "${GREEN}Docker installed successfully!${NC}"
    echo -e "${YELLOW}You may need to log out and log back in for Docker permissions to take effect.${NC}"
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Installing Docker Compose...${NC}"
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    echo -e "${GREEN}Docker Compose installed successfully!${NC}"
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA drivers not detected.${NC}"
    echo -e "${YELLOW}For proper T4 GPU support, please install NVIDIA drivers and NVIDIA Container Toolkit.${NC}"
    echo -e "${YELLOW}See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
    exit 1
else
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi

# Check for NVIDIA Container Toolkit
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${RED}NVIDIA Container Toolkit not detected.${NC}"
    echo -e "${YELLOW}Installing NVIDIA Container Toolkit...${NC}"
    
    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo systemctl restart docker
    
    echo -e "${GREEN}NVIDIA Container Toolkit installed successfully!${NC}"
else
    echo -e "${GREEN}NVIDIA Container Toolkit detected.${NC}"
fi

# Create backend configuration
echo -e "${BLUE}Configuring backend...${NC}"
cat > api/.env << EOF
# GPU Configuration
GPU_ENABLED=true
USE_TENSORRT=true
TENSORRT_PRECISION=fp16
TENSORRT_CACHE_DIR=/app/tensorrt_cache

# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG
ENABLE_CORS=true
APP_NAME=langchain-hana-integration

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

# Embedding configuration
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
USE_INTERNAL_EMBEDDING=false
CACHE_EMBEDDINGS=true
EMBEDDING_CACHE_DIR=/tmp/embedding_cache
EOF

# Configure frontend
echo -e "${BLUE}Configuring frontend...${NC}"
cat > frontend/.env << EOF
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENABLE_AUTH=false
EOF

# Set up Docker Compose with GPU configuration
echo -e "${BLUE}Setting up Docker Compose configuration...${NC}"
cat > docker-compose.yml << EOF
version: '3'

services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile.ngc
    ports:
      - "8000:8000"
    env_file:
      - ./api/.env
    volumes:
      - ./api:/app
      - tensorrt_cache:/app/tensorrt_cache
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - api
    restart: unless-stopped

volumes:
  tensorrt_cache:
EOF

# Make API start script executable
chmod +x api/start.sh
chmod +x frontend/entrypoint.sh

# Build and start the services
echo -e "${BLUE}Building and starting services...${NC}"
docker-compose up -d --build

# Wait for services to start
echo -e "${BLUE}Waiting for services to start...${NC}"
sleep 20

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}Services are running!${NC}"
    
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}Deployment completed successfully!${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${BLUE}API URL:${NC} http://localhost:8000"
    echo -e "${BLUE}API Documentation:${NC} http://localhost:8000/docs"
    echo -e "${BLUE}Frontend URL:${NC} http://localhost:3000"
    echo -e "${BLUE}GPU Info:${NC} http://localhost:8000/benchmark/gpu_info"
    
    # Get external IP for remote access
    EXTERNAL_IP=$(curl -s ifconfig.me || curl -s ipinfo.io/ip)
    if [ -n "$EXTERNAL_IP" ]; then
        echo -e "\n${BLUE}For external access (if your VM is publicly accessible):${NC}"
        echo -e "${BLUE}API URL:${NC} http://$EXTERNAL_IP:8000"
        echo -e "${BLUE}Frontend URL:${NC} http://$EXTERNAL_IP:3000"
    fi
    
    echo -e "\n${BLUE}To view logs:${NC}"
    echo -e "docker-compose logs -f"
    
    echo -e "\n${BLUE}To stop the services:${NC}"
    echo -e "docker-compose down"
else
    echo -e "${RED}Service startup failed. Check the logs:${NC}"
    docker-compose logs
    exit 1
fi