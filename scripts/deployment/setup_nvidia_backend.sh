#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Setup NVIDIA Backend for SAP HANA Cloud LangChain Integration${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if NVIDIA drivers are installed
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA drivers are installed.${NC}"
    nvidia-smi
else
    echo -e "${YELLOW}WARNING: NVIDIA drivers not detected. This script is intended for NVIDIA GPU deployments.${NC}"
    echo -e "${YELLOW}You can still continue, but GPU acceleration will not be available.${NC}"
    
    read -p "Continue anyway? (y/n) " continue_decision
    if [[ ! "$continue_decision" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Exiting.${NC}"
        exit 1
    fi
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${YELLOW}WARNING: NVIDIA Container Toolkit not detected.${NC}"
    echo -e "${YELLOW}This script is intended for NVIDIA GPU deployments.${NC}"
    echo -e "${YELLOW}See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
    
    read -p "Continue anyway? (y/n) " continue_decision
    if [[ ! "$continue_decision" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Exiting.${NC}"
        exit 1
    fi
fi

# Create directories for TensorRT engines and data
echo -e "${BLUE}Creating directories for TensorRT engines and data...${NC}"
mkdir -p trt_engines
mkdir -p api/data
mkdir -p api/logs

# Set environment variables
echo -e "${BLUE}Setting up environment variables...${NC}"
cat > .env.backend << EOL
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
JWT_SECRET=sap-hana-langchain-integration-secret-key

# GPU Acceleration
GPU_ENABLED=true
USE_TENSORRT=true
TENSORRT_PRECISION=fp16
TENSORRT_ENGINE_CACHE_DIR=/app/trt_engines
BATCH_SIZE=32
MAX_BATCH_SIZE=128
ENABLE_MULTI_GPU=true

# Error Handling
ENABLE_CONTEXT_AWARE_ERRORS=true
ERROR_VERBOSITY=standard
ENABLE_ERROR_TELEMETRY=true

# Vector Operations
ENABLE_PRECISE_SIMILARITY=true
DEFAULT_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EOL

echo -e "${GREEN}Environment file created: .env.backend${NC}"
echo -e "${YELLOW}Please edit .env.backend with your SAP HANA Cloud credentials.${NC}"

# Create docker-compose file for backend only
echo -e "${BLUE}Creating docker-compose file for backend...${NC}"
cat > docker-compose.nvidia-backend.yml << EOL
version: '3.8'

name: sap-hana-langchain-api-nvidia

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.nvidia
    image: sap-hana-langchain-api-nvidia:latest
    container_name: sap-hana-langchain-api-nvidia
    ports:
      - "8000:8000"
    env_file:
      - .env.backend
    volumes:
      - ./trt_engines:/app/trt_engines
      - ./api/data:/app/data
      - ./api/logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
EOL

echo -e "${GREEN}Docker Compose file created: docker-compose.nvidia-backend.yml${NC}"

# Build the Docker image
echo -e "${BLUE}Building the Docker image...${NC}"
echo -e "${YELLOW}This may take a while as it downloads the NVIDIA PyTorch container and installs dependencies.${NC}"

docker-compose -f docker-compose.nvidia-backend.yml build

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Docker image built successfully!${NC}"
else
    echo -e "${RED}Failed to build Docker image. Please check the error messages above.${NC}"
    exit 1
fi

# Create startup script
echo -e "${BLUE}Creating startup script...${NC}"
cat > start_nvidia_backend.sh << EOL
#!/bin/bash
set -e

# Start the NVIDIA-accelerated backend
docker-compose -f docker-compose.nvidia-backend.yml up -d

# Wait for the API to be ready
echo "Waiting for API to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health/ping &> /dev/null; then
        echo "API is ready!"
        echo ""
        echo "Access the API at: http://localhost:8000"
        echo "API documentation: http://localhost:8000/docs"
        echo "GPU information: http://localhost:8000/gpu-info"
        echo ""
        exit 0
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "WARNING: API did not respond within the expected time."
echo "Check the logs using: docker-compose -f docker-compose.nvidia-backend.yml logs"
EOL

chmod +x start_nvidia_backend.sh
echo -e "${GREEN}Startup script created: start_nvidia_backend.sh${NC}"

# Create shutdown script
echo -e "${BLUE}Creating shutdown script...${NC}"
cat > stop_nvidia_backend.sh << EOL
#!/bin/bash
set -e

# Stop the NVIDIA-accelerated backend
docker-compose -f docker-compose.nvidia-backend.yml down
EOL

chmod +x stop_nvidia_backend.sh
echo -e "${GREEN}Shutdown script created: stop_nvidia_backend.sh${NC}"

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo -e "To start the NVIDIA-accelerated backend:"
echo -e "  1. Edit ${YELLOW}.env.backend${NC} with your SAP HANA Cloud credentials"
echo -e "  2. Run ${YELLOW}./start_nvidia_backend.sh${NC}"
echo -e ""
echo -e "The API will be available at: ${CYAN}http://localhost:8000${NC}"
echo -e "API documentation: ${CYAN}http://localhost:8000/docs${NC}"
echo -e ""
echo -e "To stop the backend:"
echo -e "  Run ${YELLOW}./stop_nvidia_backend.sh${NC}"
echo -e "${BLUE}=========================================================${NC}"