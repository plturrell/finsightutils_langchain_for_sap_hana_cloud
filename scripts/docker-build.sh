#!/bin/bash
# Universal Docker build script for both local and cloud environments

set -e

# Colors for better output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE="cpu-secure"
USE_CLOUD=false
PUSH_IMAGE=false
TEST_IMAGE=false
REPOSITORY="finsightintelligence/finsight_utils_langchain_hana"
DATE_TAG=$(date +"%Y%m%d")

# Function to show usage
usage() {
  echo -e "${BLUE}Universal Docker Build Script${NC}"
  echo -e "Builds Docker images for SAP HANA LangChain integration"
  echo -e ""
  echo -e "${YELLOW}Usage:${NC}"
  echo -e "  $0 [options]"
  echo -e ""
  echo -e "${YELLOW}Options:${NC}"
  echo -e "  -t, --type TYPE     Build type: cpu-secure, gpu-secure, minimal-secure, arrow-flight (default: cpu-secure)"
  echo -e "  -c, --cloud         Use Docker Build Cloud (if available)"
  echo -e "  -p, --push          Push image to Docker Hub"
  echo -e "  -T, --test          Test image after build"
  echo -e "  -r, --repo REPO     Docker Hub repository (default: ${REPOSITORY})"
  echo -e "  -h, --help          Show this help message"
  echo -e ""
  echo -e "${YELLOW}Examples:${NC}"
  echo -e "  $0 --type cpu-secure --push    # Build CPU image and push to Docker Hub"
  echo -e "  $0 --type gpu-secure --cloud   # Build GPU image using Docker Build Cloud"
  echo -e "  $0 --test                      # Build default image and test it"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--type)
      BUILD_TYPE="$2"
      shift 2
      ;;
    -c|--cloud)
      USE_CLOUD=true
      shift
      ;;
    -p|--push)
      PUSH_IMAGE=true
      shift
      ;;
    -T|--test)
      TEST_IMAGE=true
      shift
      ;;
    -r|--repo)
      REPOSITORY="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      usage
      exit 1
      ;;
  esac
done

# Set variables based on build type
case $BUILD_TYPE in
  cpu-secure)
    DOCKERFILE="Dockerfile.secure"
    FORCE_CPU="1"
    INSTALL_GPU="false"
    ;;
  gpu-secure)
    DOCKERFILE="Dockerfile.secure"
    FORCE_CPU="0"
    INSTALL_GPU="true"
    ;;
  minimal-secure)
    DOCKERFILE="Dockerfile.minimal-secure"
    FORCE_CPU="1"
    INSTALL_GPU="false"
    ;;
  arrow-flight)
    DOCKERFILE="Dockerfile.arrow-flight"
    FORCE_CPU="1"
    INSTALL_GPU="false"
    ;;
  *)
    echo -e "${RED}Invalid build type: ${BUILD_TYPE}${NC}"
    usage
    exit 1
    ;;
esac

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

# Set image tags
IMAGE_TAG="${REPOSITORY}:${BUILD_TYPE}"
DATED_TAG="${REPOSITORY}:${BUILD_TYPE}-${DATE_TAG}"

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}     Universal Docker Build Script      ${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "${YELLOW}Build type:    ${NC}${BUILD_TYPE}"
echo -e "${YELLOW}Dockerfile:    ${NC}${DOCKERFILE}"
echo -e "${YELLOW}Force CPU:     ${NC}${FORCE_CPU}"
echo -e "${YELLOW}Install GPU:   ${NC}${INSTALL_GPU}"
echo -e "${YELLOW}Use cloud:     ${NC}${USE_CLOUD}"
echo -e "${YELLOW}Push image:    ${NC}${PUSH_IMAGE}"
echo -e "${YELLOW}Test image:    ${NC}${TEST_IMAGE}"
echo -e "${YELLOW}Repository:    ${NC}${REPOSITORY}"
echo -e "${YELLOW}Image tags:    ${NC}${IMAGE_TAG}, ${DATED_TAG}"
echo -e "${BLUE}=========================================${NC}"

# Check if we need to login to Docker Hub
if $PUSH_IMAGE; then
  echo -e "${YELLOW}Checking Docker Hub login status...${NC}"
  if ! docker info 2>/dev/null | grep -q "Username"; then
    echo -e "${YELLOW}Not logged in to Docker Hub. Please login:${NC}"
    docker login
  else
    echo -e "${GREEN}Already logged in to Docker Hub.${NC}"
  fi
fi

# Build the image
echo -e "${YELLOW}Building Docker image...${NC}"

# Determine build command based on environment
if $USE_CLOUD; then
  echo -e "${YELLOW}Using Docker Build Cloud...${NC}"
  
  # Check if buildx is available
  if ! docker buildx version > /dev/null 2>&1; then
    echo -e "${RED}Docker Buildx is not available. Cannot use cloud build.${NC}"
    exit 1
  fi
  
  # Check for cloud builder or create one
  if ! docker buildx ls | grep -q "cloud"; then
    echo -e "${YELLOW}Creating cloud builder...${NC}"
    docker buildx create --name cloud-builder --driver docker-container --bootstrap
    docker buildx use cloud-builder
  fi
  
  # Build with buildx
  BUILD_CMD="docker buildx build --platform linux/amd64"
  
  # Add push flag if needed
  if $PUSH_IMAGE; then
    BUILD_CMD+=" --push"
  else
    # Note: --load only works with single platform builds, which is our case
    BUILD_CMD+=" --load"
  fi
  
  # Add the rest of the options
  BUILD_CMD+=" -f ${DOCKERFILE} -t ${IMAGE_TAG} -t ${DATED_TAG} --build-arg FORCE_CPU=${FORCE_CPU} --build-arg INSTALL_GPU=${INSTALL_GPU} ."
else
  echo -e "${YELLOW}Using local Docker build...${NC}"
  
  # Build locally
  BUILD_CMD="docker build -f ${DOCKERFILE} -t ${IMAGE_TAG} -t ${DATED_TAG} --build-arg FORCE_CPU=${FORCE_CPU} --build-arg INSTALL_GPU=${INSTALL_GPU} ."
fi

# Execute the build command
echo -e "${YELLOW}Executing: ${BUILD_CMD}${NC}"
eval ${BUILD_CMD}

# Push the image if requested
if $PUSH_IMAGE && ! $USE_CLOUD; then
  echo -e "${YELLOW}Pushing images to Docker Hub...${NC}"
  docker push ${IMAGE_TAG}
  docker push ${DATED_TAG}
fi

# Test the image if requested
if $TEST_IMAGE; then
  echo -e "${YELLOW}Testing the image...${NC}"
  echo -e "${YELLOW}Starting container...${NC}"
  CONTAINER_ID=$(docker run -d -p 8000:8000 ${IMAGE_TAG})
  
  echo -e "${YELLOW}Waiting for container to start (30 seconds)...${NC}"
  sleep 30
  
  echo -e "${YELLOW}Testing API health...${NC}"
  if curl -s http://localhost:8000/health | grep -q "status.*ok"; then
    echo -e "${GREEN}✅ API health check passed!${NC}"
    
    # Get more health info
    echo -e "${YELLOW}Health info:${NC}"
    curl -s http://localhost:8000/health | python3 -m json.tool
  else
    echo -e "${RED}❌ API health check failed!${NC}"
    echo -e "${YELLOW}Container logs:${NC}"
    docker logs ${CONTAINER_ID}
  fi
  
  echo -e "${YELLOW}Stopping and removing container...${NC}"
  docker stop ${CONTAINER_ID}
  docker rm ${CONTAINER_ID}
fi

echo -e "${GREEN}Build process completed!${NC}"
if $PUSH_IMAGE; then
  echo -e "${GREEN}Images pushed to Docker Hub:${NC}"
  echo -e "  • ${IMAGE_TAG}"
  echo -e "  • ${DATED_TAG}"
fi

exit 0