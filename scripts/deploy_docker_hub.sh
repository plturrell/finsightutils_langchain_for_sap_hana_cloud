#!/bin/bash
# Deploy to Docker Hub with the specified repository name
# finsightintelligence/finsight_utils_langchain_hana

set -e

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Variables
ORGANIZATION="finsightintelligence"
REPO_NAME="finsight_utils_langchain_hana"
TODAY=$(date +"%Y%m%d")
VERSION="${TODAY}"
CPU_TAG="${ORGANIZATION}/${REPO_NAME}:cpu-${VERSION}"
GPU_TAG="${ORGANIZATION}/${REPO_NAME}:gpu-${VERSION}"
LATEST_CPU_TAG="${ORGANIZATION}/${REPO_NAME}:cpu-latest"
LATEST_GPU_TAG="${ORGANIZATION}/${REPO_NAME}:gpu-latest"

# Set working directory
cd "$(dirname "$0")"

# Check Docker login status
if ! docker info 2>&1 | grep -q "Username"; then
  log "Please login to Docker Hub first"
  docker login
fi

# Clean up Docker system
log "Cleaning up Docker system to free space..."
docker system prune -af --volumes

# Build and push CPU image
log "Building CPU image: ${CPU_TAG}"
docker build -t "${CPU_TAG}" -t "${LATEST_CPU_TAG}" \
  --build-arg FORCE_CPU=1 \
  -f Dockerfile.cpu .

log "Pushing CPU images to Docker Hub..."
docker push "${CPU_TAG}"
docker push "${LATEST_CPU_TAG}"

# Clean up after CPU build to save space
docker image rm "${CPU_TAG}" "${LATEST_CPU_TAG}" || true
docker system prune -af --volumes

# Build and push GPU image
log "Building GPU image: ${GPU_TAG}"
docker build -t "${GPU_TAG}" -t "${LATEST_GPU_TAG}" \
  -f Dockerfile .

log "Pushing GPU images to Docker Hub..."
docker push "${GPU_TAG}"
docker push "${LATEST_GPU_TAG}"

log "Deployment complete!"
log "Images pushed to Docker Hub:"
log "- ${CPU_TAG}"
log "- ${LATEST_CPU_TAG}"
log "- ${GPU_TAG}"
log "- ${LATEST_GPU_TAG}"
