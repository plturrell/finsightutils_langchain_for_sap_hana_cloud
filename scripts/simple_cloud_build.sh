#!/bin/bash
set -e

# Set variables
REPO="finsightintelligence/finsight_utils_langchain_hana"
DATE=$(date +"%Y%m%d")
IMAGE_TAG="cpu-secure"
FORCE_CPU="1"
INSTALL_GPU="false"

# Verify Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Docker is not running. Please start Docker first."
  exit 1
fi

echo "Building ${REPO}:${IMAGE_TAG}-${DATE} image..."

# Run docker buildx build with cloud builder
docker buildx build \
  -f Dockerfile.secure \
  --build-arg FORCE_CPU="${FORCE_CPU}" \
  --build-arg INSTALL_GPU="${INSTALL_GPU}" \
  -t ${REPO}:${IMAGE_TAG} \
  -t ${REPO}:${IMAGE_TAG}-${DATE} \
  --platform linux/amd64 \
  --push .

echo "Build request sent to Docker. Check Docker Hub for the image."
echo "Image should be available at:"
echo "• ${REPO}:${IMAGE_TAG}"
echo "• ${REPO}:${IMAGE_TAG}-${DATE}"

exit 0