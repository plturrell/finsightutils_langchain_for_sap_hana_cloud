#!/bin/bash
# Script to test Docker build with security fixes and verify API health

set -e

# Variables
ORGANIZATION="finsightintelligence"
REPO_NAME="finsight_utils_langchain_hana"
TODAY=$(date +"%Y%m%d")
VERSION="${TODAY}-test"
CPU_TAG="${ORGANIZATION}/${REPO_NAME}:cpu-${VERSION}"

echo "Building test Docker image with security fixes..."

# Create temporary Dockerfile with security fixes
cat > Dockerfile.test << EOF
FROM python:3.10-slim

WORKDIR /app

# Set environment variables to force CPU-only mode
ENV FORCE_CPU=1

# Update system packages and clean up in one step to reduce layer size
RUN apt-get update && \\
    apt-get upgrade -y && \\
    apt-get install -y --no-install-recommends \\
    build-essential \\
    curl \\
    git \\
    && apt-get clean \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt /app/
COPY api/requirements.txt /app/api_requirements.txt
COPY security-requirements.txt /app/security_fixes.txt

# Install secure Python dependencies with no cache
RUN pip install --no-cache-dir --upgrade pip setuptools>=78.1.1 && \\
    pip install --no-cache-dir -r requirements.txt && \\
    pip install --no-cache-dir -r api_requirements.txt || true && \\
    pip install --no-cache-dir -r security_fixes.txt && \\
    pip install --no-cache-dir numpy scipy pandas scikit-learn torch

# Copy application code
COPY . /app/

# Create a non-root user and switch to it
RUN groupadd -r appuser && useradd -r -g appuser appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Add health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Expose the API port
EXPOSE 8000

# Set the entrypoint
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Build the test image
docker build -t ${CPU_TAG} -f Dockerfile.test .

# Run the container in the background
echo "Starting container for API health check..."
CONTAINER_ID=$(docker run -d -p 8000:8000 ${CPU_TAG})

# Wait for container to initialize
echo "Waiting for API to start up..."
sleep 10

# Check API health
echo "Checking API health..."
curl -f http://localhost:8000/health || { 
    echo "API health check failed!"; 
    docker logs ${CONTAINER_ID}; 
    docker stop ${CONTAINER_ID}; 
    exit 1; 
}

# If we get here, the health check passed
echo "API health check passed!"
echo "Stopping container..."
docker stop ${CONTAINER_ID}

# Ask if we want to push to Docker Hub
read -p "Push this image to Docker Hub? (y/n): " PUSH_CHOICE
if [[ "$PUSH_CHOICE" == "y" ]]; then
    echo "Pushing image to Docker Hub..."
    docker push ${CPU_TAG}
    
    # Also tag and push as latest
    echo "Tagging and pushing as latest..."
    LATEST_TAG="${ORGANIZATION}/${REPO_NAME}:cpu-latest"
    docker tag ${CPU_TAG} ${LATEST_TAG}
    docker push ${LATEST_TAG}
    echo "Image successfully pushed to Docker Hub!"
else
    echo "Skipping Docker Hub push."
fi

echo "Testing complete!"
