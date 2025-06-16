#!/bin/bash
# Script to rebuild Docker images with security fixes

set -e

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Variables
ORGANIZATION="finsightintelligence"
REPO_NAME="finsight_utils_langchain_hana"
TODAY=$(date +"%Y%m%d")
VERSION="${TODAY}-secure"
CPU_TAG="${ORGANIZATION}/${REPO_NAME}:cpu-${VERSION}"
GPU_TAG="${ORGANIZATION}/${REPO_NAME}:gpu-${VERSION}"
LATEST_CPU_TAG="${ORGANIZATION}/${REPO_NAME}:cpu-latest"
LATEST_GPU_TAG="${ORGANIZATION}/${REPO_NAME}:gpu-latest"

# Check Docker login status
if ! docker info 2>&1 | grep -q "Username"; then
  log "Please login to Docker Hub first"
  docker login
fi

# Clean up Docker system
log "Cleaning up Docker system to free space..."
docker system prune -af --volumes

# Build and push CPU image
log "Building secure CPU image: ${CPU_TAG}"
docker build -t "${CPU_TAG}" -t "${LATEST_CPU_TAG}" \
  -f Dockerfile.secure.cpu .

log "Pushing secure CPU images to Docker Hub..."
docker push "${CPU_TAG}"
docker push "${LATEST_CPU_TAG}"

# Clean up after CPU build to save space
docker image rm "${CPU_TAG}" "${LATEST_CPU_TAG}" || true
docker system prune -af --volumes

# Build and push GPU image
log "Building secure GPU image: ${GPU_TAG}"
docker build -t "${GPU_TAG}" -t "${LATEST_GPU_TAG}" \
  -f Dockerfile.secure.gpu .

log "Pushing secure GPU images to Docker Hub..."
docker push "${GPU_TAG}"
docker push "${LATEST_GPU_TAG}"

# Run Docker Scout on new images to verify fixes
log "Verifying security of new images..."
docker scout cves ${LATEST_CPU_TAG} --only-severity critical,high
docker scout cves ${LATEST_GPU_TAG} --only-severity critical,high

log "Secure deployment complete!"
log "Images pushed to Docker Hub:"
log "- ${CPU_TAG}"
log "- ${LATEST_CPU_TAG}"
log "- ${GPU_TAG}"
log "- ${LATEST_GPU_TAG}"
