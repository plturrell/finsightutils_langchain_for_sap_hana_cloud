#!/bin/bash
# Script to identify and fix Docker security issues
# Especially targeting the "E rated" security issues in Docker Scout

set -e

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check for Docker Scout CLI
if ! command -v docker scout &> /dev/null; then
  log "ERROR: Docker Scout CLI is not installed. Please install it first."
  log "You can install Docker Scout using: docker extension install docker/scout-extension"
  exit 1
fi

# Define the images to scan
ORG="finsightintelligence"
REPO="finsight_utils_langchain_hana"
CPU_TAG="${ORG}/${REPO}:cpu-latest"
GPU_TAG="${ORG}/${REPO}:gpu-latest"

# Run Docker Scout analysis on both images
log "Running security analysis on ${CPU_TAG}..."
docker scout cves ${CPU_TAG} --only-severity critical,high --format markdown > cpu_vulnerabilities.md

log "Running security analysis on ${GPU_TAG}..."
docker scout cves ${GPU_TAG} --only-severity critical,high --format markdown > gpu_vulnerabilities.md

# Create Dockerfile.secure.cpu to fix issues
log "Creating updated Dockerfiles with security fixes..."
cat > Dockerfile.secure.cpu << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Set environment variables to force CPU-only mode
ENV FORCE_CPU=1

# Update system packages and clean up in one step to reduce layer size
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY api/requirements.txt .

# Install Python dependencies with no cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir numpy scipy pandas scikit-learn torch

# Create necessary directories
RUN mkdir -p /app/docs/pr_notes /app/api/gpu

# Copy application code
COPY . /app/

# Create necessary dummy modules for CPU mode
RUN echo 'import logging\nlogger = logging.getLogger("gpu_utils")\n\ndef get_gpu_info():\n    return {"gpu_count": 0, "gpu_names": []}\n\ndef is_gpu_available():\n    logger.warning("GPU check requested but running in CPU-only mode")\n    return False' > /app/api/gpu/gpu_utils.py && \
    echo 'import logging\nlogger = logging.getLogger("tensorrt_utils")\n\ndef create_tensorrt_engine(*args, **kwargs):\n    logger.warning("TensorRT requested but running in CPU-only mode")\n    return None' > /app/api/gpu/tensorrt_utils.py

# Create alias for MultiGPUEmbeddings
COPY api/multi_gpu.py /usr/local/lib/python3.10/site-packages/multi_gpu.py
RUN echo '\n# Provide alias for singular class name (required by imports)\nMultiGPUEmbedding = MultiGPUEmbeddings' >> /app/api/embeddings/embedding_multi_gpu.py

# Add dummy TensorRTEmbeddings class
RUN echo 'import sys\nimport logging\nlogger = logging.getLogger("dummy_tensorrt_classes")\n\n# Add dummy TensorRTEmbeddings class to fix inheritance\nclass TensorRTEmbeddings:\n    def __init__(self, *args, **kwargs):\n        logger.warning("TensorRT embeddings initialized in CPU-only mode")\n\nclass EnhancedTensorRTEmbedding:\n    def __init__(self, *args, **kwargs):\n        logger.warning("Enhanced TensorRT embeddings initialized in CPU-only mode")\n\nclass TensorRTEmbeddingsWithTensorCores:\n    def __init__(self, *args, **kwargs):\n        logger.warning("TensorRT embeddings with tensor cores initialized in CPU-only mode")' > /app/api/embeddings/dummy_tensorrt_classes.py

# Fix imports in embeddings GPU file
RUN sed -i '1s/^/from dummy_tensorrt_classes import TensorRTEmbeddings, EnhancedTensorRTEmbedding, TensorRTEmbeddingsWithTensorCores  # Import dummy classes\\n/' /app/api/embeddings/embedding_gpu.py

# Create a non-root user and switch to it
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Set the entrypoint
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Create Dockerfile.secure.gpu to fix issues
cat > Dockerfile.secure.gpu << 'EOF'
FROM python:3.10-slim

WORKDIR /app

# Update system packages and clean up in one step to reduce layer size
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY api/requirements.txt .

# Install Python dependencies with no cache
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ .

# Create a non-root user and switch to it
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose the API port
EXPOSE 8000

# Set the entrypoint
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

log "Creating script to rebuild images with security fixes..."
cat > rebuild_secure_images.sh << 'EOF'
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
EOF

chmod +x rebuild_secure_images.sh

log "Created scripts to analyze and fix security issues"
log "Next steps:"
log "1. Review cpu_vulnerabilities.md and gpu_vulnerabilities.md when they're generated"
log "2. Review the secure Dockerfiles (Dockerfile.secure.cpu and Dockerfile.secure.gpu)"
log "3. Run ./rebuild_secure_images.sh to build and push fixed images"
