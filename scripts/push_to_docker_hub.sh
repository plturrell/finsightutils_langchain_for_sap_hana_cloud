#!/bin/bash
# Script to build, push to Docker Hub, and test API health

set -e

# Variables
ORG_NAME="finsightintelligence"
REPO_NAME="finsight_utils_langchain_hana"
FULL_REPO="${ORG_NAME}/${REPO_NAME}"
TODAY=$(date +"%Y%m%d")
VERSION_TAG="cpu-secure-${TODAY}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building and pushing secure Docker image to Docker Hub...${NC}"

# Check if docker CLI is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker CLI not found. Please install Docker first.${NC}"
    exit 1
fi

# Check if logged in to Docker Hub
if ! docker info 2>/dev/null | grep -q "Username"; then
    echo -e "${YELLOW}You need to log in to Docker Hub:${NC}"
    docker login
fi

# Build the secure CPU Docker image
echo -e "${YELLOW}Building secure CPU image from Dockerfile.secure.cpu...${NC}"
docker build -t ${FULL_REPO}:${VERSION_TAG} -f Dockerfile.secure.cpu .

# Tag the image as secure
docker tag ${FULL_REPO}:${VERSION_TAG} ${FULL_REPO}:cpu-secure

# Push images to Docker Hub
echo -e "${YELLOW}Pushing images to Docker Hub...${NC}"
docker push ${FULL_REPO}:${VERSION_TAG}
docker push ${FULL_REPO}:cpu-secure

echo -e "${GREEN}Images successfully pushed to Docker Hub!${NC}"

# Wait a moment to ensure the image is available
echo -e "${YELLOW}Waiting for image to be available on Docker Hub...${NC}"
sleep 10

# Test API health
echo -e "${YELLOW}Testing API health on the pushed image...${NC}"
echo -e "${YELLOW}Running container for API testing...${NC}"

# Run the container in the background
CONTAINER_ID=$(docker run -d -p 8000:8000 ${FULL_REPO}:cpu-secure)

# Wait for container to initialize
echo -e "${YELLOW}Waiting for API to start up (30 seconds)...${NC}"
sleep 30

# Check API health
echo -e "${YELLOW}Checking API health...${NC}"
if curl -s http://localhost:8000/health | grep -q "status"; then
    echo -e "${GREEN}SUCCESS: API is healthy and responding correctly!${NC}"
    echo -e "${GREEN}Your secure Docker image is working properly on Docker Hub!${NC}"
else
    echo -e "${RED}API health check failed. Checking logs...${NC}"
    docker logs ${CONTAINER_ID}
fi

# Stop and remove the container
echo -e "${YELLOW}Stopping container...${NC}"
docker stop ${CONTAINER_ID}
docker rm ${CONTAINER_ID}

echo -e "${GREEN}Testing complete!${NC}"
echo -e "${YELLOW}Your secure Docker image is available at:${NC}"
echo -e "${GREEN}  • ${FULL_REPO}:${VERSION_TAG}${NC}"
echo -e "${GREEN}  • ${FULL_REPO}:cpu-secure${NC}"
