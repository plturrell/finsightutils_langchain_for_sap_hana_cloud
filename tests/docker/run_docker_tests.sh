#!/bin/bash
# Docker container testing script for embedding initialization
# Tests CPU-only and GPU-compatible containers with various configurations

set -e

# Log function with timestamps
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Define variables
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Docker image names
TEST_IMAGE="langchain-hana-test:latest"

log "Starting Docker container tests for embedding initialization"
log "Project root: $PROJECT_ROOT"

# Build the test Docker image
log "Building test Docker image..."
docker build -t "$TEST_IMAGE" -f tests/docker/Dockerfile.test .

# Test CPU-only mode
log "Running CPU-only test..."
docker run --rm \
  -e FORCE_CPU=1 \
  "$TEST_IMAGE" \
  python tests/test_embedding_simple.py

# Test with native environment (may use GPU if available)
log "Running test with native environment..."
docker run --rm \
  "$TEST_IMAGE" \
  python tests/test_embedding_simple.py

# Test with simulated GPU environment
log "Running test with simulated GPU..."
docker run --rm \
  -e SIMULATE_GPU=1 \
  -e CUDA_VISIBLE_DEVICES=0 \
  "$TEST_IMAGE" \
  python tests/test_embedding_simple.py

# Run performance benchmarks in container
log "Running embedding performance benchmarks..."
docker run --rm \
  "$TEST_IMAGE" \
  python tests/benchmark_embeddings.py --models cpu

# If NVIDIA Docker runtime is available, test with GPU
if command -v nvidia-smi &> /dev/null && docker info | grep -q "Runtimes:.*nvidia"; then
  log "NVIDIA GPU detected, running tests with GPU access..."
  
  # Run with GPU access
  docker run --rm \
    --gpus all \
    "$TEST_IMAGE" \
    python tests/test_embedding_simple.py
  
  # Run benchmarks with GPU
  docker run --rm \
    --gpus all \
    "$TEST_IMAGE" \
    python tests/benchmark_embeddings.py --models cpu gpu tensorrt
else
  log "NVIDIA GPU or Docker runtime not available, skipping GPU tests"
fi

log "All Docker container tests completed"
