#!/bin/bash
# Script to test Docker Hub integration

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Testing Docker Hub Integration${NC}"
echo -e "================================"

# Test 1: Check Docker login status
echo -e "\n${YELLOW}Test 1: Checking Docker login status...${NC}"
if docker info 2>/dev/null | grep -q "Username"; then
  echo -e "${GREEN}✓ Logged in to Docker Hub${NC}"
  DOCKER_USERNAME=$(docker info 2>/dev/null | grep Username | awk '{print $2}')
  echo -e "   Username: ${DOCKER_USERNAME}"
else
  echo -e "${RED}✗ Not logged in to Docker Hub${NC}"
  echo -e "   Please run 'docker login' first"
  exit 1
fi

# Test 2: Check repository access
echo -e "\n${YELLOW}Test 2: Checking repository access...${NC}"
REPO="finsightintelligence/finsight_utils_langchain_hana"
if curl -s "https://hub.docker.com/v2/repositories/${REPO}/tags/?page_size=1" | grep -q "name"; then
  echo -e "${GREEN}✓ Repository exists and is accessible${NC}"
else
  echo -e "${RED}✗ Cannot access repository${NC}"
  echo -e "   Check if repository exists and you have access"
  exit 1
fi

# Test 3: Pull existing image
echo -e "\n${YELLOW}Test 3: Pulling existing image...${NC}"
if docker pull ${REPO}:cpu-secure >/dev/null 2>&1; then
  echo -e "${GREEN}✓ Successfully pulled image${NC}"
else
  echo -e "${RED}✗ Failed to pull image${NC}"
  echo -e "   Check network connection and repository permissions"
  exit 1
fi

# Test 4: Create and push a test tag
echo -e "\n${YELLOW}Test 4: Creating and pushing a test tag...${NC}"
TEST_TAG="test-$(date +%s)"
echo -e "   Creating tag: ${TEST_TAG}"
if docker tag ${REPO}:cpu-secure ${REPO}:${TEST_TAG} && \
   docker push ${REPO}:${TEST_TAG}; then
  echo -e "${GREEN}✓ Successfully pushed test tag${NC}"
else
  echo -e "${RED}✗ Failed to push test tag${NC}"
  echo -e "   Check Docker Hub permissions"
  exit 1
fi

# Test 5: Verify tag exists in Docker Hub
echo -e "\n${YELLOW}Test 5: Verifying tag exists in Docker Hub...${NC}"
MAX_RETRIES=10
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
  if curl -s "https://hub.docker.com/v2/repositories/${REPO}/tags/?page_size=20" | grep -q "\"name\":\"${TEST_TAG}\""; then
    echo -e "${GREEN}✓ Test tag found in Docker Hub${NC}"
    break
  else
    RETRY_COUNT=$((RETRY_COUNT+1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
      echo -e "   Tag not found yet, retrying in 5 seconds... (${RETRY_COUNT}/${MAX_RETRIES})"
      sleep 5
    else
      echo -e "${RED}✗ Test tag not found in Docker Hub after multiple retries${NC}"
      echo -e "   Check Docker Hub webhook integration"
      exit 1
    fi
  fi
done

# Test 6: Test buildx
echo -e "\n${YELLOW}Test 6: Testing Docker Buildx...${NC}"
if docker buildx version >/dev/null 2>&1; then
  echo -e "${GREEN}✓ Docker Buildx is installed${NC}"
  
  # List builders
  echo -e "   Available builders:"
  docker buildx ls
  
  # Check if cloud builder exists
  if docker buildx ls | grep -q "cloud"; then
    echo -e "${GREEN}✓ Cloud builder is available${NC}"
  else
    echo -e "${YELLOW}⚠ No cloud builder found${NC}"
    echo -e "   Creating a cloud builder for testing..."
    docker buildx create --name cloud-test --driver docker-container
    docker buildx use cloud-test
  fi
else
  echo -e "${RED}✗ Docker Buildx is not installed${NC}"
  echo -e "   Please install Docker Buildx"
  exit 1
fi

# Test 7: Small buildx test
echo -e "\n${YELLOW}Test 7: Testing small Docker Buildx build...${NC}"
echo -e 'FROM alpine:latest\nCMD ["echo", "Hello World"]' > test-dockerfile
if docker buildx build --load -t ${REPO}:buildx-test -f test-dockerfile .; then
  echo -e "${GREEN}✓ Docker Buildx build successful${NC}"
  rm test-dockerfile
else
  echo -e "${RED}✗ Docker Buildx build failed${NC}"
  rm test-dockerfile
  exit 1
fi

# Final summary
echo -e "\n${GREEN}All tests passed successfully!${NC}"
echo -e "Your Docker Hub integration is working correctly."
echo -e "\nNext steps:"
echo -e "1. Run './build_with_docker_cloud.sh cpu-secure' to build and push using Docker Build Cloud"
echo -e "2. Verify the image appears in Docker Hub"
echo -e "3. Use the GitHub Actions workflow for automated builds"

exit 0