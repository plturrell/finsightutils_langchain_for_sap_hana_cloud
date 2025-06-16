#!/bin/bash
# Script to deploy langchain-integration-for-sap-hana-cloud to Docker Hub
# Targets: finsightintelligence/langchainsaphana

set -e

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Variables
ORGANIZATION="finsightintelligence"
REPO_NAME="langchainsaphana"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$(date +%Y%m%d-%H%M%S)"
CPU_TAG="${ORGANIZATION}/${REPO_NAME}:cpu-${VERSION}"
GPU_TAG="${ORGANIZATION}/${REPO_NAME}:gpu-${VERSION}"
LATEST_CPU_TAG="${ORGANIZATION}/${REPO_NAME}:cpu-latest"
LATEST_GPU_TAG="${ORGANIZATION}/${REPO_NAME}:gpu-latest"

cd "$PROJECT_ROOT"
log "Project root: $PROJECT_ROOT"

# Check if Docker CLI is logged in
log "Checking Docker login status..."
if ! docker info 2>&1 | grep -q "Username"; then
  log "You are not logged in to Docker Hub. Please log in:"
  docker login
fi

# Build CPU image
log "Building CPU Docker image: $CPU_TAG"
docker build -t "$CPU_TAG" -t "$LATEST_CPU_TAG" \
  --build-arg INSTALL_GPU=false \
  -f Dockerfile.cpu .

# Build GPU image
log "Building GPU Docker image: $GPU_TAG"
docker build -t "$GPU_TAG" -t "$LATEST_GPU_TAG" \
  --build-arg INSTALL_GPU=true \
  -f Dockerfile .

# Run tests before pushing
log "Running quick validation tests on CPU image..."
docker run --rm "$CPU_TAG" python -c "from api.gpu import gpu_utils; print(f'GPU utils loaded, GPU available: {gpu_utils.is_gpu_available() if hasattr(gpu_utils, \"is_gpu_available\") else False}')"

log "Running quick validation tests on GPU image..."
docker run --rm "$GPU_TAG" python -c "from api.gpu import gpu_utils; print(f'GPU utils loaded, GPU available: {gpu_utils.is_gpu_available() if hasattr(gpu_utils, \"is_gpu_available\") else False}')"

# Push images to Docker Hub
log "Pushing CPU image to Docker Hub..."
docker push "$CPU_TAG"
docker push "$LATEST_CPU_TAG"

log "Pushing GPU image to Docker Hub..."
docker push "$GPU_TAG"
docker push "$LATEST_GPU_TAG"

log "Deployment complete!"
log "CPU Image: $CPU_TAG"
log "GPU Image: $GPU_TAG"
log "Latest CPU Image: $LATEST_CPU_TAG"
log "Latest GPU Image: $LATEST_GPU_TAG"

# Provide instructions for using the images
cat << EOF

------------------------------------------------------
Deployment Summary
------------------------------------------------------
The following Docker images have been pushed to Docker Hub:

CPU Image (latest): ${LATEST_CPU_TAG}
CPU Image (versioned): ${CPU_TAG}

GPU Image (latest): ${LATEST_GPU_TAG}
GPU Image (versioned): ${GPU_TAG}

To pull and run the CPU image:
  docker pull ${LATEST_CPU_TAG}
  docker run -p 8000:8000 ${LATEST_CPU_TAG}

To pull and run the GPU image with GPU access:
  docker pull ${LATEST_GPU_TAG}
  docker run --gpus all -p 8000:8000 ${LATEST_GPU_TAG}
------------------------------------------------------

EOF
