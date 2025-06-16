#!/bin/bash
# Script for building and pushing LangChain SAP HANA Cloud Integration Docker images to Docker Hub
# This script should be run from the root of the langchain-integration-for-sap-hana-cloud repository

set -e

# Configuration
DOCKER_HUB_USERNAME=${DOCKER_HUB_USERNAME:-"finsightdev"}
REGISTRY="${DOCKER_HUB_USERNAME}"
IMAGE_NAME="langchain-hana"
VERSION=$(cat VERSION 2>/dev/null || echo "0.1.0")
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "dev")
TAG="${VERSION}-${GIT_SHA}"
LATEST_TAG="latest"

# Build variant flags
BUILD_STANDARD=true
BUILD_GPU=false
BUILD_ALPINE=false
BUILD_MONITORING=false
PUSH_IMAGES=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --no-standard)
      BUILD_STANDARD=false
      shift
      ;;
    --gpu)
      BUILD_GPU=true
      shift
      ;;
    --alpine)
      BUILD_ALPINE=true
      shift
      ;;
    --monitoring)
      BUILD_MONITORING=true
      shift
      ;;
    --no-push)
      PUSH_IMAGES=false
      shift
      ;;
    --version)
      VERSION="$2"
      TAG="${VERSION}-${GIT_SHA}"
      shift 2
      ;;
    --registry)
      REGISTRY="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --no-standard     Skip building standard image"
      echo "  --gpu             Build GPU-enabled image"
      echo "  --alpine          Build using Alpine base images"
      echo "  --monitoring      Include monitoring tools"
      echo "  --no-push         Build images but don't push to registry"
      echo "  --version VERSION Set custom version (default: from VERSION file or 0.1.0)"
      echo "  --registry REG    Set custom registry (default: finsightdev)"
      echo "  --help            Display this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "===== LangChain HANA Docker Hub Deployment ====="
echo "Building images with tag: ${TAG}"
echo "Standard build: ${BUILD_STANDARD}"
echo "GPU build: ${BUILD_GPU}"
echo "Alpine variant: ${BUILD_ALPINE}"
echo "Monitoring included: ${BUILD_MONITORING}"
echo "Push images: ${PUSH_IMAGES}"
echo "================================================="

# Determine Dockerfile path and build args
DOCKERFILE="Dockerfile"
BUILD_ARGS=""

if [ "$BUILD_GPU" = true ] && [ "$BUILD_ALPINE" = true ]; then
  DOCKERFILE="Dockerfile.gpu.alpine"
  IMAGE_VARIANT="gpu-alpine"
  echo "Using GPU Alpine Dockerfile"
elif [ "$BUILD_GPU" = true ]; then
  DOCKERFILE="Dockerfile.gpu"
  IMAGE_VARIANT="gpu"
  echo "Using GPU Dockerfile"
elif [ "$BUILD_ALPINE" = true ]; then
  DOCKERFILE="Dockerfile.alpine"
  IMAGE_VARIANT="alpine"
  echo "Using Alpine Dockerfile"
else
  IMAGE_VARIANT="standard"
  echo "Using standard Dockerfile"
fi

if [ "$BUILD_MONITORING" = true ]; then
  BUILD_ARGS="--build-arg INCLUDE_MONITORING=true"
  IMAGE_VARIANT="${IMAGE_VARIANT}-monitoring"
fi

# Login to Docker Hub if pushing images
if [ "$PUSH_IMAGES" = true ]; then
  echo "Logging in to Docker Hub..."
  if [ -z "$DOCKER_HUB_TOKEN" ]; then
    echo "DOCKER_HUB_TOKEN environment variable not set. Please login manually:"
    docker login
  else
    echo "$DOCKER_HUB_TOKEN" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
  fi
fi

# Build and push standard image
if [ "$BUILD_STANDARD" = true ]; then
  FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${TAG}"
  LATEST_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${LATEST_TAG}"
  
  # Add variant suffix if not standard
  if [ "$IMAGE_VARIANT" != "standard" ]; then
    FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}-${IMAGE_VARIANT}:${TAG}"
    LATEST_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}-${IMAGE_VARIANT}:${LATEST_TAG}"
  fi
  
  echo "Building image: ${FULL_IMAGE_NAME}"
  
  docker build -t "${FULL_IMAGE_NAME}" \
    -t "${LATEST_IMAGE_NAME}" \
    ${BUILD_ARGS} \
    -f "${DOCKERFILE}" .
  
  if [ "$PUSH_IMAGES" = true ]; then
    echo "Pushing image to Docker Hub..."
    docker push "${FULL_IMAGE_NAME}"
    docker push "${LATEST_IMAGE_NAME}"
  fi
fi

# Generate docker-compose file with the new image versions
echo "Generating docker-compose.deploy.yml with the new image versions..."

# Determine image name for docker-compose
COMPOSE_IMAGE="${REGISTRY}/${IMAGE_NAME}:${TAG}"
if [ "$IMAGE_VARIANT" != "standard" ]; then
  COMPOSE_IMAGE="${REGISTRY}/${IMAGE_NAME}-${IMAGE_VARIANT}:${TAG}"
fi

cat > docker-compose.deploy.yml << EOF
version: '3.8'

services:
  langchain-hana:
    image: ${COMPOSE_IMAGE}
    restart: always
    ports:
      - "8000:8000"
    environment:
      - HANA_HOST=\${HANA_HOST}
      - HANA_PORT=\${HANA_PORT}
      - HANA_USER=\${HANA_USER}
      - HANA_PASSWORD=\${HANA_PASSWORD}
      - LOG_LEVEL=info
EOF

# Add GPU configuration if GPU variant
if [ "$BUILD_GPU" = true ]; then
  cat >> docker-compose.deploy.yml << EOF
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF
fi

# Add monitoring configuration if monitoring included
if [ "$BUILD_MONITORING" = true ]; then
  cat >> docker-compose.deploy.yml << EOF
  
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    depends_on:
      - langchain-hana
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  grafana-data:
EOF
fi

echo "Deployment files generated:"
echo "- docker-compose.deploy.yml (for Docker Compose deployment)"

# Create environment file template
cat > .env.template << EOF
# SAP HANA Cloud connection parameters
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=YOUR_USER
HANA_PASSWORD=YOUR_PASSWORD

# Optional settings
LOG_LEVEL=info
VECTOR_BATCH_SIZE=32
EOF

if [ "$BUILD_GPU" = true ]; then
  cat >> .env.template << EOF
# GPU settings
GPU_ENABLED=true
GPU_MEMORY_FRACTION=0.85
TENSORRT_ENABLED=true
EOF
fi

echo "- .env.template (environment variables template)"

if [ "$PUSH_IMAGES" = true ]; then
  echo ""
  echo "====== Deployment Instructions ======"
  echo "To deploy using Docker Compose:"
  echo "1. Copy docker-compose.deploy.yml and .env.template to your server"
  echo "2. Rename .env.template to .env and fill in your SAP HANA Cloud credentials"
  echo "3. Run: docker-compose -f docker-compose.deploy.yml up -d"
  echo ""
  echo "To deploy to Kubernetes using the GitOps approach:"
  echo "1. Update the image tags in your Kubernetes manifests"
  echo "2. Commit and push the changes to your GitOps repository"
  echo "3. Let your GitOps operator (Flux/ArgoCD) sync the changes"
  echo "======================================"
fi

echo "Done!"