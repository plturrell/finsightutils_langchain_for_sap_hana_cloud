#!/bin/bash
# Docker environment testing script for verifying embedding initialization
# Tests both CPU-only and GPU-enabled Docker containers

set -e

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
log "Project root: $PROJECT_ROOT"

# Docker image names
CPU_IMAGE="langchain-hana-integration:cpu"
GPU_IMAGE="langchain-hana-integration:gpu"

# Build Docker images
build_images() {
  log "Building CPU Docker image..."
  docker build -t "$CPU_IMAGE" \
    --build-arg INSTALL_GPU=false \
    -f docker/Dockerfile .
  
  log "Building GPU-compatible Docker image..."
  docker build -t "$GPU_IMAGE" \
    --build-arg INSTALL_GPU=true \
    -f docker/Dockerfile.gpu .
}

# Run embedding test in container
run_embedding_test() {
  local image="$1"
  local env_flags="$2"
  local container_name="langchain-hana-test-$(date +%s)"
  
  log "Running embedding test with image: $image"
  log "Environment flags: $env_flags"
  
  # Run the container with the embedding test
  docker run --rm --name "$container_name" \
    $env_flags \
    "$image" \
    python /app/tests/test_embedding_simple.py
  
  local exit_code=$?
  if [ $exit_code -eq 0 ]; then
    log "Test on $image succeeded"
  else
    log "Test on $image failed with exit code $exit_code"
    return 1
  fi
}

# Test CPU image
test_cpu_image() {
  log "Testing CPU-only Docker environment..."
  run_embedding_test "$CPU_IMAGE" "-e FORCE_CPU=1"
}

# Test GPU image with GPU access
test_gpu_image_with_gpu() {
  if command -v nvidia-smi &> /dev/null; then
    log "Testing GPU Docker environment with GPU access..."
    run_embedding_test "$GPU_IMAGE" "--gpus all"
  else
    log "No GPU detected on host, skipping GPU container with GPU access test"
  fi
}

# Test GPU image without GPU access (should fall back to CPU)
test_gpu_image_fallback() {
  log "Testing GPU Docker environment without GPU access (fallback behavior)..."
  run_embedding_test "$GPU_IMAGE" ""  # No GPU flags provided
}

# Main test execution
main() {
  log "Starting Docker environment tests for embedding initialization"
  
  # Build the Docker images
  build_images
  
  # Run tests
  test_cpu_image
  test_gpu_image_fallback
  test_gpu_image_with_gpu
  
  log "All Docker environment tests completed successfully"
}

main "$@"
