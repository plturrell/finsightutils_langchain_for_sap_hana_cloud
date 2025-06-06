#!/bin/bash
#
# NVIDIA LaunchPad Deployment Script
#
# This script handles building and deploying the application to NVIDIA LaunchPad
# or NGC container registry.
#
# Prerequisites:
# - NGC CLI installed and configured (https://ngc.nvidia.com/setup)
# - Docker installed and configured for NGC authentication
# - Proper NGC organization and team access

set -e

# Configuration
ORG_NAME="plturrell"
COLLECTION_NAME="sap-enhanced"
BACKEND_IMAGE_NAME="langchain-hana-gpu"
FRONTEND_IMAGE_NAME="langchain-hana-frontend"
VERSION=$(cat ../VERSION || echo "1.2.0")

# Project root directory (parent of the script directory)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Parse command line arguments
SKIP_BUILD=false
SKIP_PUSH=false
SKIP_FRONTEND=false
TAG="latest"

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --skip-push)
      SKIP_PUSH=true
      shift
      ;;
    --skip-frontend)
      SKIP_FRONTEND=true
      shift
      ;;
    --tag)
      TAG="$2"
      shift
      shift
      ;;
    --help)
      echo "NVIDIA LaunchPad Deployment Script"
      echo ""
      echo "Usage:"
      echo "  ./deploy_to_nvidia.sh [options]"
      echo ""
      echo "Options:"
      echo "  --skip-build     Skip the build step"
      echo "  --skip-push      Skip the push step"
      echo "  --skip-frontend  Skip frontend build and push"
      echo "  --tag <tag>      Specify tag (default: latest)"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check NGC CLI is installed
if ! command -v ngc &> /dev/null; then
    echo "Error: NGC CLI is not installed or not in PATH"
    echo "Please install NGC CLI from: https://ngc.nvidia.com/setup/installers/cli"
    exit 1
fi

# Verify NGC authentication
echo "Verifying NGC authentication..."
ngc config get > /dev/null || { echo "Error: NGC authentication failed. Please run 'ngc config set' to configure your API key."; exit 1; }

# Build backend image
if [ "$SKIP_BUILD" != true ]; then
    echo "Building backend Docker image..."
    cd "$PROJECT_ROOT/api"
    
    # Check if Dockerfile.ngc exists
    if [ ! -f "Dockerfile.ngc" ]; then
        echo "Error: Dockerfile.ngc not found in $PROJECT_ROOT/api"
        exit 1
    fi
    
    # Build the image
    docker build -t "$BACKEND_IMAGE_NAME:$TAG" -f Dockerfile.ngc .
    
    # Tag with NGC registry name
    docker tag "$BACKEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$TAG"
    docker tag "$BACKEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$VERSION"
    
    echo "Backend image built successfully."
    
    # Build frontend image (if not skipped)
    if [ "$SKIP_FRONTEND" != true ]; then
        echo "Building frontend Docker image..."
        cd "$PROJECT_ROOT/frontend"
        
        # Check if Dockerfile exists
        if [ ! -f "Dockerfile" ]; then
            echo "Error: Dockerfile not found in $PROJECT_ROOT/frontend"
            exit 1
        }
        
        # Build the frontend image
        docker build -t "$FRONTEND_IMAGE_NAME:$TAG" .
        
        # Tag with NGC registry name
        docker tag "$FRONTEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$FRONTEND_IMAGE_NAME:$TAG"
        docker tag "$FRONTEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$FRONTEND_IMAGE_NAME:$VERSION"
        
        echo "Frontend image built successfully."
    fi
fi

# Push images to NGC
if [ "$SKIP_PUSH" != true ]; then
    echo "Pushing backend image to NGC..."
    docker push "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$TAG"
    docker push "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$VERSION"
    
    echo "Backend image pushed successfully."
    
    # Push frontend image (if not skipped)
    if [ "$SKIP_FRONTEND" != true ]; then
        echo "Pushing frontend image to NGC..."
        docker push "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$FRONTEND_IMAGE_NAME:$TAG"
        docker push "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$FRONTEND_IMAGE_NAME:$VERSION"
        
        echo "Frontend image pushed successfully."
    fi
fi

# Update NVIDIA LaunchPad configuration
echo "Updating NVIDIA LaunchPad configuration..."
cd "$PROJECT_ROOT"

# Make sure nvidia-launchable.yaml exists
if [ ! -f "nvidia-launchable.yaml" ]; then
    echo "Error: nvidia-launchable.yaml not found in $PROJECT_ROOT"
    exit 1
fi

# Update version in nvidia-launchable.yaml if needed
sed -i.bak "s/^version:.*/version: $VERSION/" nvidia-launchable.yaml

# Update container image in nvidia-launchable.yaml if needed
sed -i.bak "s|^  image:.*|  image: nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$TAG|" nvidia-launchable.yaml

# Clean up backup file
rm -f nvidia-launchable.yaml.bak

echo "NVIDIA LaunchPad configuration updated successfully."

echo "NVIDIA LaunchPad deployment completed successfully!"
echo "Next steps:"
echo "1. Visit NGC to verify your images: https://ngc.nvidia.com/catalog/$ORG_NAME/$COLLECTION_NAME"
echo "2. Launch your application from NVIDIA LaunchPad"
echo "3. Connect to the application using the provided URL"
echo ""
echo "For more information, see the NVIDIA LaunchPad documentation in docs/nvidia_deployment.md"