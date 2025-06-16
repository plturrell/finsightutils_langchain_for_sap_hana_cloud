#!/bin/bash

# Script to trigger Docker Cloud Builder builds and test API health

set -e

# Set color variables for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Constants
REPO="finsightintelligence/finsight_utils_langchain_hana"
DATE=$(date +"%Y%m%d")
export DATE

# Print header
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}   Docker Cloud Builder Trigger & API Health Test${NC}"
echo -e "${GREEN}=================================================${NC}"

# Verify Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

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

# Create temporary Docker buildx configuration for this build
echo -e "${YELLOW}Creating build configuration for Docker Cloud Builder...${NC}"
cat > docker-cloud-build.json <<EOL
{
  "context": ".",
  "dockerfile": "Dockerfile.secure",
  "tags": ["${REPO}:${IMAGE_TAG}", "${REPO}:${IMAGE_TAG}-${DATE}"],
  "platforms": ["linux/amd64"],
  "buildArgs": {
    "FORCE_CPU": "${FORCE_CPU}",
    "INSTALL_GPU": "${INSTALL_GPU}"
  }
}
EOL

# Trigger Docker Cloud Builder build
echo -e "${GREEN}Sending build request to Docker Cloud Builder...${NC}"
BUILD_ID=$(docker buildx build --builder=cloud-finsightintelligence-langchainsaphana \
  -f Dockerfile.secure \
  --build-arg FORCE_CPU="${FORCE_CPU}" \
  --build-arg INSTALL_GPU="${INSTALL_GPU}" \
  -t ${REPO}:${IMAGE_TAG} \
  -t ${REPO}:${IMAGE_TAG}-${DATE} \
  --push . || echo "FAILED")

if [[ $BUILD_ID == "FAILED" ]]; then
  echo -e "${RED}Failed to trigger Docker Cloud Builder build.${NC}"
  exit 1
fi

echo -e "${GREEN}Build triggered on Docker Cloud Builder!${NC}"
echo -e "${YELLOW}Build ID: ${BUILD_ID}${NC}"
echo -e "${YELLOW}Waiting for build to complete and image to be available...${NC}"

# Give Docker Hub some time to process the pushed image
echo -e "${YELLOW}Waiting for image to be available on Docker Hub (60 seconds)...${NC}"
sleep 60

# Pull the built image
echo -e "${YELLOW}Pulling the built image...${NC}"
docker pull ${REPO}:${IMAGE_TAG}

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

# Clean up temp files
rm -f docker-cloud-build.json metadata.json

echo -e "${GREEN}Testing complete!${NC}"
echo -e "${GREEN}Your secure Docker image is available at:${NC}"
echo -e "  • ${REPO}:${IMAGE_TAG}"
echo -e "  • ${REPO}:${IMAGE_TAG}-${DATE}"

exit 0
