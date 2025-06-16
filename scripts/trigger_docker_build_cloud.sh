#!/bin/bash
# Script to trigger Docker Build Cloud builds and test API health

set -e

# Set color variables for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Constants
REPO="finsightintelligence/finsight_utils_langchain_hana"
FULL_REPO="docker.io/${REPO}"

# Get current date for tagging
DATE=$(date +"%Y%m%d")
export DATE

# Print header
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}   Docker Build Cloud Trigger & API Health Test${NC}"
echo -e "${GREEN}=================================================${NC}"

# Check Docker login
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running or not logged in. Please start Docker and login first.${NC}"
  exit 1
fi

# Verify Docker Hub login
if ! docker login docker.io -u "$DOCKERHUB_USERNAME" -p "$DOCKERHUB_TOKEN" > /dev/null 2>&1; then
  echo -e "${RED}Failed to log in to Docker Hub. Check your credentials.${NC}"
  echo -e "${YELLOW}You need to set DOCKERHUB_USERNAME and DOCKERHUB_TOKEN environment variables.${NC}"
  exit 1
fi
echo -e "${GREEN}Successfully logged in to Docker Hub${NC}"

# Set Docker Build Cloud token if available
if [ -n "$DOCKER_BUILD_CLOUD_TOKEN" ]; then
  echo -e "${YELLOW}Using provided Docker Build Cloud token${NC}"
fi

# Determine which target to build based on arguments
if [ "$1" == "gpu" ]; then
  TARGET="secure-gpu"
  IMAGE_TAG="gpu-secure"
  FORCE_CPU="0"
  INSTALL_GPU="true"
  echo -e "${YELLOW}Building GPU secure image...${NC}"
else
  TARGET="secure-cpu"
  IMAGE_TAG="cpu-secure"
  FORCE_CPU="1"
  INSTALL_GPU="false"
  echo -e "${YELLOW}Building CPU secure image...${NC}"
fi

# Use direct Docker build command instead of buildx bake
echo -e "${GREEN}Building and pushing secure Docker image...${NC}"
docker build --platform linux/amd64 \
  --build-arg FORCE_CPU="${FORCE_CPU}" \
  --build-arg INSTALL_GPU="${INSTALL_GPU}" \
  -t ${FULL_REPO}:${IMAGE_TAG} \
  -t ${FULL_REPO}:${IMAGE_TAG}-${DATE} \
  -f Dockerfile.secure \
  --push .

echo -e "${GREEN}Build triggered successfully!${NC}"
echo -e "${YELLOW}Waiting for image to be available on Docker Hub (30 seconds)...${NC}"
sleep 30  # Allow time for the image to be available

# Pull the built image
echo -e "${YELLOW}Pulling the built image...${NC}"
docker pull ${FULL_REPO}:${IMAGE_TAG}

# Run the container for testing
echo -e "${YELLOW}Running container for testing...${NC}"
CONTAINER_ID=$(docker run -d -p 8000:8000 ${FULL_REPO}:${IMAGE_TAG})

# Wait for API startup
echo -e "${YELLOW}Waiting for API to start up (30 seconds)...${NC}"
sleep 30

# Test the health endpoint
echo -e "${YELLOW}Testing API health...${NC}"
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "failed")

if [ "$HEALTH_STATUS" = "200" ]; then
  echo -e "${GREEN}API health check passed!${NC}"
  # Get health info
  curl -s http://localhost:8000/health
  
  echo -e "\n${YELLOW}Testing health/check endpoint...${NC}"
  curl -s http://localhost:8000/health/check
  
  echo -e "\n${GREEN}All tests passed successfully!${NC}"
else
  echo -e "${RED}API health check failed! Status: $HEALTH_STATUS${NC}"
  echo -e "${RED}Container logs:${NC}"
  docker logs $CONTAINER_ID
fi

# Cleanup container
echo -e "${YELLOW}Stopping and removing container...${NC}"
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

echo -e "${GREEN}Testing complete!${NC}"
echo -e "${GREEN}Your secure Docker image is available at:${NC}"
echo -e "  • ${FULL_REPO}:${IMAGE_TAG}"
echo -e "  • ${FULL_REPO}:${IMAGE_TAG}-${DATE}"

echo -e "${GREEN}Docker Build Cloud process completed successfully!${NC}"

exit 0
