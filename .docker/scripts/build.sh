#!/bin/bash
# Standardized Docker build script for LangChain SAP HANA Integration
# Based on the standardized template

set -e  # Exit immediately if a command fails

# Default values
VERSION="latest"
REGISTRY="ghcr.io"
ORGANIZATION="finsightdev"
PROJECT_NAME="langchain-hana"
PLATFORMS="linux/amd64"
PUSH=false
BUILD_ARGS=""
CACHE=true
SERVICES=()
SECURE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display help message
show_help() {
    echo -e "${BLUE}Docker Build Script for LangChain SAP HANA Integration${NC}"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -v, --version VERSION     Image version tag (default: latest)"
    echo "  -r, --registry REGISTRY   Container registry (default: ghcr.io)"
    echo "  -o, --org ORGANIZATION    Organization name (default: finsightdev)"
    echo "  --platforms PLATFORMS     Build platforms (default: linux/amd64)"
    echo "  --push                    Push images to registry after build"
    echo "  --no-cache                Disable build cache"
    echo "  -s, --service SERVICE     Build specific service(s) (can be specified multiple times)"
    echo "  -b, --build-arg ARG=VAL   Add build argument (can be specified multiple times)"
    echo "  --secure                  Build secure versions of images"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --version 1.0.0 --push                 # Build and push all services with version 1.0.0"
    echo "  $0 -s api -s frontend --push              # Build and push only api and frontend services"
    echo "  $0 --no-cache                             # Build without cache"
    echo "  $0 --secure                               # Build secure versions of images"
    echo ""
    exit 0
}

# Log message with timestamp
log() {
    local level=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - ${message}"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - ${message}"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}"
            ;;
        *)
            echo -e "${BLUE}[${level}]${NC} ${timestamp} - ${message}"
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -o|--org)
            ORGANIZATION="$2"
            shift 2
            ;;
        --platforms)
            PLATFORMS="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            CACHE=false
            shift
            ;;
        -s|--service)
            SERVICES+=("$2")
            shift 2
            ;;
        -b|--build-arg)
            BUILD_ARGS="${BUILD_ARGS} --build-arg $2"
            shift 2
            ;;
        --secure)
            SECURE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            show_help
            ;;
    esac
done

# Project root directory
PROJECT_ROOT=$(dirname $(dirname $(dirname $0)))
cd $PROJECT_ROOT

# Check if Docker and buildx are available
if ! command -v docker &> /dev/null; then
    log "ERROR" "Docker is not installed or not in PATH"
    exit 1
fi

if ! docker buildx version &> /dev/null; then
    log "ERROR" "Docker buildx plugin is not installed"
    exit 1
fi

# Create buildx builder if it doesn't exist
if ! docker buildx inspect langchain-hana-builder &> /dev/null; then
    log "INFO" "Creating buildx builder instance"
    docker buildx create --name langchain-hana-builder --driver docker-container --bootstrap
fi

# Use the builder
docker buildx use langchain-hana-builder

# Prepare build metadata
BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

# Determine services to build
if [ ${#SERVICES[@]} -eq 0 ]; then
    log "INFO" "No specific services specified, building default services"
    SERVICES=("api" "arrow-flight" "frontend")
fi

log "INFO" "Building services: ${SERVICES[*]}"
log "INFO" "Version: $VERSION"
log "INFO" "Registry: $REGISTRY"
log "INFO" "Organization: $ORGANIZATION"
log "INFO" "Project: $PROJECT_NAME"
log "INFO" "Platforms: $PLATFORMS"
log "INFO" "Push: $PUSH"
log "INFO" "Secure: $SECURE"

# Prepare cache settings
CACHE_SETTINGS=""
if [ "$CACHE" = false ]; then
    CACHE_SETTINGS="--no-cache"
    log "INFO" "Build cache disabled"
else
    CACHE_SETTINGS="--cache-from type=local,src=/tmp/.buildx-cache --cache-to type=local,dest=/tmp/.buildx-cache-new,mode=max"
    mkdir -p /tmp/.buildx-cache
fi

# Login to registry if pushing
if [ "$PUSH" = true ]; then
    log "INFO" "Logging in to registry: $REGISTRY"
    
    # Check for registry-specific login
    case "$REGISTRY" in
        "ghcr.io")
            if [ -z "$GITHUB_TOKEN" ]; then
                log "WARN" "GITHUB_TOKEN environment variable not set, using local credentials"
            else
                echo "$GITHUB_TOKEN" | docker login ghcr.io -u $ORGANIZATION --password-stdin
            fi
            ;;
        "docker.io")
            if [ -z "$DOCKER_USERNAME" ] || [ -z "$DOCKER_PASSWORD" ]; then
                log "WARN" "DOCKER_USERNAME or DOCKER_PASSWORD environment variables not set, using local credentials"
            else
                echo "$DOCKER_PASSWORD" | docker login -u $DOCKER_USERNAME --password-stdin
            fi
            ;;
        *)
            log "WARN" "Unknown registry: $REGISTRY, using local credentials"
            ;;
    esac
fi

# Build each service
for service in "${SERVICES[@]}"; do
    log "INFO" "Building service: $service"
    
    # Find Dockerfile
    dockerfile=""
    if [ "$SECURE" = true ] && [ -f ".docker/services/$service/Dockerfile.secure" ]; then
        dockerfile=".docker/services/$service/Dockerfile.secure"
        log "INFO" "Using secure Dockerfile: $dockerfile"
    elif [ -f ".docker/services/$service/Dockerfile" ]; then
        dockerfile=".docker/services/$service/Dockerfile"
    else
        log "ERROR" "No Dockerfile found for service: $service"
        continue
    fi
    
    # Build image
    image_name="$REGISTRY/$ORGANIZATION/${PROJECT_NAME}-$service:$VERSION"
    latest_tag="$REGISTRY/$ORGANIZATION/${PROJECT_NAME}-$service:latest"
    
    # Service-specific build args
    service_args=""
    if [ "$service" = "api" ]; then
        # Add any service-specific args here
        service_args="--build-arg ENABLE_GPU=false"
    fi
    
    # Prepare build command
    build_cmd="docker buildx build \
        --platform $PLATFORMS \
        -t $image_name \
        -t $latest_tag \
        --build-arg VERSION=$VERSION \
        --build-arg BUILD_DATE=$BUILD_DATE \
        --build-arg GIT_COMMIT=$GIT_COMMIT \
        $service_args \
        $BUILD_ARGS \
        $CACHE_SETTINGS \
        --provenance=true \
        --sbom=true"
    
    # Add push flag if needed
    if [ "$PUSH" = true ]; then
        build_cmd="$build_cmd --push"
    else
        build_cmd="$build_cmd --load"
    fi
    
    # Add Dockerfile path and context
    build_cmd="$build_cmd -f $dockerfile ."
    
    # Execute build command
    log "INFO" "Running build command: $build_cmd"
    eval $build_cmd
    
    # Check build result
    if [ $? -eq 0 ]; then
        log "INFO" "Successfully built image: $image_name"
    else
        log "ERROR" "Failed to build image: $image_name"
        exit 1
    fi
done

# Move cache if it exists
if [ "$CACHE" = true ] && [ -d "/tmp/.buildx-cache-new" ]; then
    rm -rf /tmp/.buildx-cache
    mv /tmp/.buildx-cache-new /tmp/.buildx-cache
fi

# Final message
if [ "$PUSH" = true ]; then
    log "INFO" "All images built and pushed successfully"
else
    log "INFO" "All images built successfully (not pushed)"
    log "INFO" "To push images, run: $0 --version $VERSION --push"
fi

exit 0