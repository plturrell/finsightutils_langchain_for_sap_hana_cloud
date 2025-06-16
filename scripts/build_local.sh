#!/bin/bash

# Simple script to build Docker images locally using our consolidated Dockerfile

set -e

# Set color variables for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Constants
REPO="finsightintelligence/finsight_utils_langchain_hana"
DATE=$(date +"%Y%m%d")

# Print header
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}      Local Docker Build & API Health Test      ${NC}"
echo -e "${GREEN}=================================================${NC}"

# Determine which target to build based on arguments
if [ "$1" == "gpu" ]; then
  IMAGE_TAG="gpu-secure"
  FORCE_CPU="0"
  INSTALL_GPU="true"
  echo -e "${YELLOW}Building GPU secure image...${NC}"
else
  IMAGE_TAG="cpu-secure"
  FORCE_CPU="1"
  INSTALL_GPU="false"
  echo -e "${YELLOW}Building CPU secure image...${NC}"
fi

# Build locally
echo -e "${GREEN}Building Docker image locally...${NC}"
docker build --platform linux/amd64 \
  --build-arg FORCE_CPU="${FORCE_CPU}" \
  --build-arg INSTALL_GPU="${INSTALL_GPU}" \
  -t ${REPO}:${IMAGE_TAG} \
  -t ${REPO}:${IMAGE_TAG}-${DATE} \
  -f Dockerfile.secure .

echo -e "${GREEN}Build completed successfully!${NC}"

# Run the container for testing
echo -e "${YELLOW}Running container for testing...${NC}"
CONTAINER_ID=$(docker run -d -p 8000:8000 ${REPO}:${IMAGE_TAG})

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
echo -e "${GREEN}Your secure Docker image is available locally:${NC}"
echo -e "  • ${REPO}:${IMAGE_TAG}"
echo -e "  • ${REPO}:${IMAGE_TAG}-${DATE}"

exit 0
