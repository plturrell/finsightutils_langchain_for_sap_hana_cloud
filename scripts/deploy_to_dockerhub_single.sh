#!/bin/bash
# Script to deploy langchain-integration-for-sap-hana-cloud to Docker Hub
# Targets: finsightintelligence/langchainsaphana
# This version builds and pushes one image at a time to avoid disk space issues

set -e

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling
handle_error() {
  log "ERROR: $1"
  exit 1
}

# Variables
ORGANIZATION="finsightintelligence"
REPO_NAME="langchainsaphana"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="$(date +%Y%m%d-%H%M%S)"
IMAGE_TYPE=${1:-"cpu"}  # Default to CPU if not specified

if [ "$IMAGE_TYPE" != "cpu" ] && [ "$IMAGE_TYPE" != "gpu" ]; then
  handle_error "Invalid image type. Use either 'cpu' or 'gpu'"
fi

IMAGE_TAG="${ORGANIZATION}/${REPO_NAME}:${IMAGE_TYPE}-${VERSION}"
LATEST_TAG="${ORGANIZATION}/${REPO_NAME}:${IMAGE_TYPE}-latest"

cd "$PROJECT_ROOT"
log "Project root: $PROJECT_ROOT"

# Check if Docker CLI is logged in
log "Checking Docker login status..."
if ! docker info 2>&1 | grep -q "Username"; then
  log "You are not logged in to Docker Hub. Please log in:"
  docker login
fi

# Clean up before building
log "Cleaning Docker system to free up space..."
docker system prune -f

# Build and push based on image type
if [ "$IMAGE_TYPE" == "cpu" ]; then
  log "Building CPU Docker image: $IMAGE_TAG"
  docker build -t "$IMAGE_TAG" -t "$LATEST_TAG" \
    --build-arg FORCE_CPU=1 \
    -f Dockerfile.cpu . || handle_error "CPU image build failed"
    
  log "Running validation test on CPU image..."
  docker run --rm "$IMAGE_TAG" python -c "from api.gpu import gpu_utils; print(f'GPU utils loaded, GPU available: {gpu_utils.is_gpu_available() if hasattr(gpu_utils, \"is_gpu_available\") else False}')" || log "Warning: Validation test failed, but continuing with push"
  
  log "Pushing CPU image to Docker Hub..."
  docker push "$IMAGE_TAG" || handle_error "Failed to push versioned CPU image"
  docker push "$LATEST_TAG" || handle_error "Failed to push latest CPU image"
else
  log "Building GPU Docker image: $IMAGE_TAG"
  docker build -t "$IMAGE_TAG" -t "$LATEST_TAG" \
    -f Dockerfile . || handle_error "GPU image build failed"
    
  log "Running validation test on GPU image..."
  docker run --rm "$IMAGE_TAG" python -c "from api.gpu import gpu_utils; print(f'GPU utils loaded, GPU available: {gpu_utils.is_gpu_available() if hasattr(gpu_utils, \"is_gpu_available\") else False}')" || log "Warning: Validation test failed, but continuing with push"
  
  log "Pushing GPU image to Docker Hub..."
  docker push "$IMAGE_TAG" || handle_error "Failed to push versioned GPU image"
  docker push "$LATEST_TAG" || handle_error "Failed to push latest GPU image"
fi

log "Deployment of ${IMAGE_TYPE} image complete!"
log "Image: $IMAGE_TAG"
log "Latest tag: $LATEST_TAG"

# Provide instructions for using the image
cat << EOF

------------------------------------------------------
Deployment Summary
------------------------------------------------------
The following Docker image has been pushed to Docker Hub:

${IMAGE_TYPE} Image (latest): ${LATEST_TAG}
${IMAGE_TYPE} Image (versioned): ${IMAGE_TAG}

To pull and run the ${IMAGE_TYPE} image:
  docker pull ${LATEST_TAG}
  docker run -p 8000:8000 ${LATEST_TAG}
EOF

if [ "$IMAGE_TYPE" == "gpu" ]; then
  echo "  # To use with GPU access:"
  echo "  docker run --gpus all -p 8000:8000 ${LATEST_TAG}"
fi

echo "------------------------------------------------------"
echo ""
echo "To deploy the other image type, run:"
echo "  ./$(basename "$0") $([ "$IMAGE_TYPE" == "cpu" ] && echo "gpu" || echo "cpu")"
echo ""
