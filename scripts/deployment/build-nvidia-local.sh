#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Building NVIDIA Docker Image for SAP HANA Cloud LangChain${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${YELLOW}WARNING: NVIDIA Container Toolkit not detected.${NC}"
    echo -e "${YELLOW}GPU support may not be available.${NC}"
    echo -e "${YELLOW}See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
fi

# Build the Docker image
echo -e "${BLUE}Building the Docker image...${NC}"
docker build -t langchain-nvidia:latest -f Dockerfile.nvidia .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Docker image built successfully!${NC}"
else
    echo -e "${RED}Failed to build Docker image.${NC}"
    exit 1
fi

# Check if user wants to run the container
read -p "Do you want to run the container locally? (y/n): " run_container
if [[ "$run_container" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Running the container...${NC}"
    
    # Stop any existing container with the same name
    if docker ps -a | grep -q "langchain-nvidia"; then
        echo -e "${YELLOW}Stopping and removing existing langchain-nvidia container...${NC}"
        docker stop langchain-nvidia >/dev/null 2>&1 || true
        docker rm langchain-nvidia >/dev/null 2>&1 || true
    fi
    
    # Check if NVIDIA Container Toolkit is available
    if docker info | grep -q "Runtimes:.*nvidia"; then
        echo -e "${GREEN}Running with NVIDIA GPU support...${NC}"
        docker run --gpus all -d -p 8000:8000 --name langchain-nvidia langchain-nvidia:latest
    else
        echo -e "${YELLOW}Running without GPU support...${NC}"
        docker run -d -p 8000:8000 --name langchain-nvidia langchain-nvidia:latest
    fi
    
    # Wait for container to start
    echo -e "${BLUE}Waiting for container to start...${NC}"
    sleep 5
    
    # Check container status
    if docker ps | grep -q "langchain-nvidia"; then
        echo -e "${GREEN}Container started successfully!${NC}"
        echo -e "${BLUE}Container ID:${NC} $(docker ps -q --filter name=langchain-nvidia)"
        
        # Test health endpoint
        echo -e "${BLUE}Testing health endpoint...${NC}"
        if curl -s http://localhost:8000/health | grep -q "UP"; then
            echo -e "${GREEN}Health check passed!${NC}"
            
            # Test GPU info endpoint if NVIDIA runtime is available
            if docker info | grep -q "Runtimes:.*nvidia"; then
                echo -e "${BLUE}Testing GPU capability...${NC}"
                gpu_info=$(curl -s http://localhost:8000/gpu-info)
                
                echo -e "${BLUE}GPU information:${NC}"
                echo $gpu_info | python -m json.tool
                
                # Check if CUDA is available
                if echo $gpu_info | grep -q '"cuda_available": true'; then
                    echo -e "${GREEN}CUDA is available in the container!${NC}"
                    
                    # Test tensor operations
                    echo -e "${BLUE}Running tensor test...${NC}"
                    curl -s http://localhost:8000/tensor-test | python -m json.tool
                else
                    echo -e "${YELLOW}CUDA is not available in the container.${NC}"
                    echo -e "${YELLOW}Check your NVIDIA driver and container configuration.${NC}"
                fi
            fi
        else
            echo -e "${RED}Health check failed.${NC}"
            echo -e "${YELLOW}Container logs:${NC}"
            docker logs langchain-nvidia
        fi
        
        echo -e "${BLUE}API is available at:${NC} http://localhost:8000"
        echo -e "${BLUE}To stop the container:${NC} docker stop langchain-nvidia"
        echo -e "${BLUE}To remove the container:${NC} docker rm langchain-nvidia"
    else
        echo -e "${RED}Container failed to start.${NC}"
        echo -e "${YELLOW}Container logs:${NC}"
        docker logs langchain-nvidia
    fi
fi

# Generate docker-compose file for local testing
echo -e "${BLUE}Generating docker-compose file for local testing...${NC}"
cat > docker-compose.nvidia.yml << EOL
version: '3'

services:
  langchain-api:
    build:
      context: .
      dockerfile: Dockerfile.nvidia
    image: langchain-nvidia:latest
    ports:
      - "8000:8000"
    environment:
      - GPU_ENABLED=true
      - USE_TENSORRT=true
      - TENSORRT_PRECISION=fp16
      - LOG_LEVEL=DEBUG
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
EOL

echo -e "${GREEN}Docker Compose file generated: docker-compose.nvidia.yml${NC}"
echo -e "${BLUE}To use Docker Compose:${NC} docker-compose -f docker-compose.nvidia.yml up -d"

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Build process completed!${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if user wants to proceed with NVIDIA Launchable preparation
read -p "Do you want to prepare for NVIDIA Launchable deployment? (y/n): " launchable_prep
if [[ "$launchable_prep" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}For NVIDIA Launchable deployment, see: ${YELLOW}./build_launchable.sh${NC}"
    echo -e "${BLUE}This script will guide you through the process of:${NC}"
    echo -e "1. Tagging the image for NGC"
    echo -e "2. Logging in to NGC Registry"
    echo -e "3. Pushing the image to NGC"
    echo -e "4. Generating a Launchable configuration file"
    
    # Offer to run the launchable script
    read -p "Do you want to run the Launchable preparation script now? (y/n): " run_launchable
    if [[ "$run_launchable" =~ ^[Yy]$ ]] && [ -f "./build_launchable.sh" ]; then
        chmod +x ./build_launchable.sh 2>/dev/null || true
        ./build_launchable.sh
    else
        # Print information for deploying to NVIDIA Launchable
        echo -e "${BLUE}To push to NVIDIA NGC Registry:${NC}"
        echo -e "1. Tag the image:"
        echo -e "   ${YELLOW}docker tag langchain-nvidia:latest nvcr.io/your-org/your-collection/langchain-nvidia:latest${NC}"
        echo -e "2. Login to NGC Registry:"
        echo -e "   ${YELLOW}docker login nvcr.io${NC}"
        echo -e "3. Push the image:"
        echo -e "   ${YELLOW}docker push nvcr.io/your-org/your-collection/langchain-nvidia:latest${NC}"
        echo -e "4. Deploy to NVIDIA Launchable using the image path."
    fi
fi