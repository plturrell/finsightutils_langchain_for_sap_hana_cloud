#!/bin/bash
# Script to trigger Docker Build Cloud and test API health

set -e

# Variables
ORG_NAME="finsightintelligence"
REPO_NAME="finsight_utils_langchain_hana"
FULL_REPO="${ORG_NAME}/${REPO_NAME}"
CONFIG_FILE="docker-build-secure.yml"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Triggering Docker Build Cloud for ${FULL_REPO}...${NC}"

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

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Docker Build Cloud config file ${CONFIG_FILE} not found.${NC}"
    exit 1
fi

# Use Docker CLI to create a build request to Docker Build Cloud
echo -e "${YELLOW}Sending build request to Docker Build Cloud for secure CPU image...${NC}"

# Create build request for just the secure CPU image
docker buildx bake -f ${CONFIG_FILE} cpu-secure --push

echo -e "${GREEN}Build request sent to Docker Build Cloud!${NC}"
echo -e "${YELLOW}You can monitor the build status on Docker Hub:${NC}"
echo -e "${GREEN}https://hub.docker.com/r/${FULL_REPO}/builds${NC}"

echo -e "${YELLOW}Once the build is complete, we'll test the API health.${NC}"
echo -e "${YELLOW}Waiting 5 minutes for build to complete...${NC}"

# Wait for build to complete - this is approximate
sleep 300

# Test API health
echo -e "${YELLOW}Testing API health on the newly built image...${NC}"
echo -e "${YELLOW}Pulling latest secure CPU image...${NC}"

# Pull the latest image from Docker Hub
docker pull ${FULL_REPO}:cpu-secure || {
    echo -e "${RED}Failed to pull image. Build may still be in progress or failed.${NC}"
    echo -e "${YELLOW}Please check build status at Docker Hub.${NC}"
    exit 1
}

# Run a container using the image
echo -e "${YELLOW}Running container for API testing...${NC}"
CONTAINER_ID=$(docker run -d -p 8000:8000 ${FULL_REPO}:cpu-secure)

# Wait for API to start
echo -e "${YELLOW}Waiting for API to initialize (30 seconds)...${NC}"
sleep 30

# Check API health
echo -e "${YELLOW}Testing API health endpoint...${NC}"
if curl -s http://localhost:8000/health | grep -q "status"; then
    echo -e "${GREEN}SUCCESS: API is healthy and responding correctly!${NC}"
else
    echo -e "${RED}FAILED: API health check failed. Checking logs...${NC}"
    docker logs ${CONTAINER_ID}
fi

# Stop and remove the container
echo -e "${YELLOW}Stopping and removing test container...${NC}"
docker stop ${CONTAINER_ID}
docker rm ${CONTAINER_ID}

echo -e "${GREEN}Docker Build Cloud testing complete!${NC}"
echo -e "${YELLOW}You can find your images at: https://hub.docker.com/r/${FULL_REPO}/tags${NC}"
