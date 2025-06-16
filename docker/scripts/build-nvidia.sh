#!/bin/bash
# Script for building the NVIDIA Docker image using Docker Build Cloud

# Set variables
IMAGE_NAME="langchain-hana-nvidia"
TAG=${1:-latest}
DOCKER_USERNAME=${2:-plturrell}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Building $IMAGE_NAME:$TAG using Docker Build Cloud ===${NC}"

# Ensure we're using the cloud builder
docker buildx use cloud-plturrell-langchainsaphana || {
  echo -e "${YELLOW}Cloud builder not found. Creating new cloud builder...${NC}"
  docker buildx create --name cloud-plturrell-langchainsaphana --driver cloud --bootstrap
  docker buildx use cloud-plturrell-langchainsaphana
}

# Optional - login to NGC if needed
# echo -e "${YELLOW}Logging into NVIDIA NGC...${NC}"
# echo $NGC_API_KEY | docker login nvcr.io -u '$oauthtoken' --password-stdin

echo -e "${GREEN}Building and pushing image to Docker Hub...${NC}"
docker buildx build \
  --push \
  --platform linux/amd64 \
  --file Dockerfile.nvidia \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:23.12-py3 \
  --tag ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG} \
  --cache-from type=registry,ref=${DOCKER_USERNAME}/${IMAGE_NAME}:buildcache \
  --cache-to type=registry,ref=${DOCKER_USERNAME}/${IMAGE_NAME}:buildcache,mode=max \
  --output type=registry,compression=gzip,compression-level=9,force-compression=true \
  ..

echo -e "${BLUE}====================================${NC}"
echo -e "${GREEN}Build complete!${NC}"
echo -e "${YELLOW}To use this image with docker-compose:${NC}"
echo -e "1. Update your docker-compose.nvidia.yml to use ${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
echo -e "2. Run: docker-compose -f docker-compose.nvidia.yml --env-file ../.env.test up -d"
