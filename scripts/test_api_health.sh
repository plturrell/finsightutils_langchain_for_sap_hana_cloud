#!/bin/bash
# Script to build Docker image locally and test API health

set -e

# Variables
ORG_NAME="finsightintelligence"
REPO_NAME="finsight_utils_langchain_hana"
FULL_REPO="${ORG_NAME}/${REPO_NAME}"
VERSION=$(date +"%Y%m%d")
TEST_TAG="local-test-${VERSION}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Building secure Docker image locally and testing API health...${NC}"

# Build the secure CPU Docker image locally
echo -e "${YELLOW}Building image with Dockerfile.secure.cpu...${NC}"
docker build -t ${FULL_REPO}:${TEST_TAG} -f Dockerfile.secure.cpu .

# Run the container in the background
echo -e "${YELLOW}Starting container for API health check...${NC}"
CONTAINER_ID=$(docker run -d -p 8000:8000 ${FULL_REPO}:${TEST_TAG})

# Wait for container to initialize
echo -e "${YELLOW}Waiting for API to start up (30 seconds)...${NC}"
sleep 30

# Check API health
echo -e "${YELLOW}Checking API health...${NC}"
if curl -s http://localhost:8000/health | grep -q "status.*ok"; then
    echo -e "${GREEN}API health check passed! API is healthy!${NC}"
    HEALTH_STATUS="PASSED"
else
    echo -e "${RED}API health check failed. Checking logs...${NC}"
    docker logs ${CONTAINER_ID}
    HEALTH_STATUS="FAILED"
fi

# Stop and remove the container
echo -e "${YELLOW}Stopping container...${NC}"
docker stop ${CONTAINER_ID}
docker rm ${CONTAINER_ID}

if [ "$HEALTH_STATUS" == "PASSED" ]; then
    # Ask if we want to push to Docker Hub
    echo -e "${YELLOW}API test passed. Would you like to push this image to Docker Hub? (y/n)${NC}"
    read -p "" PUSH_CHOICE
    if [[ "$PUSH_CHOICE" == "y" ]]; then
        echo -e "${YELLOW}Pushing image to Docker Hub...${NC}"
        
        # Tag with both version tag and secure tag
        docker tag ${FULL_REPO}:${TEST_TAG} ${FULL_REPO}:cpu-secure-${VERSION}
        docker tag ${FULL_REPO}:${TEST_TAG} ${FULL_REPO}:cpu-secure
        
        # Push both tags
        docker push ${FULL_REPO}:cpu-secure-${VERSION}
        docker push ${FULL_REPO}:cpu-secure
        
        echo -e "${GREEN}Image successfully pushed to Docker Hub!${NC}"
    else
        echo -e "${YELLOW}Skipping Docker Hub push.${NC}"
    fi
else
    echo -e "${RED}API health check failed. Not pushing to Docker Hub.${NC}"
fi

# Clean up local test image
docker rmi ${FULL_REPO}:${TEST_TAG}

echo -e "${GREEN}Testing complete!${NC}"
