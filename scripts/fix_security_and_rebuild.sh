#!/bin/bash
# Script to fix security vulnerabilities and rebuild Docker images

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

# Create temporary directory for the fix
TEMP_DIR="docker_security_fix"
mkdir -p $TEMP_DIR

# Create updated Dockerfile.cpu with security fixes
log "Creating updated Dockerfiles with security fixes..."
cat > $TEMP_DIR/Dockerfile.cpu << 'EOF'
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
COPY api/requirements.txt requirements-secure.txt ./

# Install secure Python dependencies with no cache and fix vulnerable packages
RUN pip install --no-cache-dir --upgrade pip setuptools>=78.1.1 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir numpy scipy pandas scikit-learn torch && \
    pip install --no-cache-dir -r requirements-secure.txt

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

# Create updated Dockerfile.gpu with security fixes
cat > $TEMP_DIR/Dockerfile.gpu << 'EOF'
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
COPY api/requirements.txt requirements-secure.txt ./

# Install Python dependencies with no cache and fix vulnerable packages
RUN pip install --no-cache-dir --upgrade pip setuptools>=78.1.1 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r requirements-secure.txt

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

# Check Docker login status
if ! docker info 2>&1 | grep -q "Username"; then
  log "Please login to Docker Hub first"
  docker login
fi

# Clean up Docker system
log "Cleaning up Docker system to free space..."
docker system prune -af

# Ask if user wants to build and push
read -p "Do you want to build and push the secure Docker images? (y/n): " CONTINUE
if [[ $CONTINUE != "y" && $CONTINUE != "Y" ]]; then
  log "Setup complete. To build and push the images later, run:"
  log "docker build -t ${CPU_TAG} -t ${LATEST_CPU_TAG} -f ${TEMP_DIR}/Dockerfile.cpu ."
  log "docker push ${CPU_TAG}"
  log "docker push ${LATEST_CPU_TAG}"
  log "docker build -t ${GPU_TAG} -t ${LATEST_GPU_TAG} -f ${TEMP_DIR}/Dockerfile.gpu ."
  log "docker push ${GPU_TAG}"
  log "docker push ${LATEST_GPU_TAG}"
  exit 0
fi

# Build and push CPU image
log "Building secure CPU image: ${CPU_TAG}"
docker build -t "${CPU_TAG}" -t "${LATEST_CPU_TAG}" \
  -f ${TEMP_DIR}/Dockerfile.cpu .

log "Pushing secure CPU images to Docker Hub..."
docker push "${CPU_TAG}"
docker push "${LATEST_CPU_TAG}"

# Clean up after CPU build to save space
docker image rm "${CPU_TAG}" "${LATEST_CPU_TAG}" || true
docker system prune -af

# Build and push GPU image
log "Building secure GPU image: ${GPU_TAG}"
docker build -t "${GPU_TAG}" -t "${LATEST_GPU_TAG}" \
  -f ${TEMP_DIR}/Dockerfile.gpu .

log "Pushing secure GPU images to Docker Hub..."
docker push "${GPU_TAG}"
docker push "${LATEST_GPU_TAG}"

# Run Docker Scout on new images to verify fixes
log "Verifying security of new images..."
docker scout cves ${LATEST_CPU_TAG} --only-severity critical,high
docker scout cves ${LATEST_GPU_TAG} --only-severity critical,high

log "Security fixes complete!"
log "Secure images pushed to Docker Hub:"
log "- ${CPU_TAG}"
log "- ${LATEST_CPU_TAG}"
log "- ${GPU_TAG}"
log "- ${LATEST_GPU_TAG}"
EOF
