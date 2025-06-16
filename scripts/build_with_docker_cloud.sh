#!/bin/bash
# Script to build and push using Docker Build Cloud

set -e

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker first."
  exit 1
fi

# Constants
REPO="finsightintelligence/finsight_utils_langchain_hana"
DATE=$(date +"%Y%m%d")
IMAGE_TAG=${1:-"cpu-secure"}

# Set build args based on tag
if [[ "$IMAGE_TAG" == *"gpu"* ]]; then
  FORCE_CPU="0"
  INSTALL_GPU="true"
  echo "Building GPU image: ${REPO}:${IMAGE_TAG}-${DATE}"
else
  FORCE_CPU="1"
  INSTALL_GPU="false"
  echo "Building CPU image: ${REPO}:${IMAGE_TAG}-${DATE}"
fi

# Ensure we're logged in
docker login

# Try with buildx directly
echo "Building and pushing with Docker Build Cloud..."
docker buildx build \
  --platform linux/amd64 \
  -t ${REPO}:${IMAGE_TAG} \
  -t ${REPO}:${IMAGE_TAG}-${DATE} \
  --build-arg FORCE_CPU=${FORCE_CPU} \
  --build-arg INSTALL_GPU=${INSTALL_GPU} \
  --push \
  -f Dockerfile.secure .

echo "Build and push completed!"
echo "Image should be available at:"
echo "• ${REPO}:${IMAGE_TAG}"
echo "• ${REPO}:${IMAGE_TAG}-${DATE}"