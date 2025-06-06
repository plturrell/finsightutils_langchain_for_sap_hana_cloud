#!/bin/bash
# Script to build and publish container to NVIDIA NGC registry

set -e

# Check if NGC CLI is installed
if ! command -v ngc &> /dev/null; then
    echo "Error: NGC CLI is not installed. Please install it from https://ngc.nvidia.com/setup/installers/cli"
    exit 1
fi

# Check if user is logged in to NGC
if ! ngc config get &> /dev/null; then
    echo "Please log in to NGC using 'ngc config set'"
    exit 1
fi

# Get version from pyproject.toml or VERSION file
if [ -f "VERSION" ]; then
    VERSION=$(cat VERSION)
else
    VERSION=$(grep -m 1 version pyproject.toml | cut -d '"' -f 2)
fi

# Set organization, team and image name
ORG=${NGC_ORG:-"plturrell"}
TEAM=${NGC_TEAM:-"sap-enhanced"}
IMAGE_NAME="langchain-hana-gpu"
REGISTRY="nvcr.io/${ORG}/${TEAM}/${IMAGE_NAME}"

# Check if version argument is provided
if [ "$1" != "" ]; then
    VERSION="$1"
fi

echo "Building image version ${VERSION}..."

# Build the Docker image using NGC Dockerfile
docker build -f api/Dockerfile.ngc -t ${REGISTRY}:${VERSION} api/
docker tag ${REGISTRY}:${VERSION} ${REGISTRY}:latest

echo "Built images:"
echo "  ${REGISTRY}:${VERSION}"
echo "  ${REGISTRY}:latest"

# Push to NGC registry
echo "Pushing to NGC registry..."
docker push ${REGISTRY}:${VERSION}
docker push ${REGISTRY}:latest

echo "Successfully published to NGC:"
echo "  ${REGISTRY}:${VERSION}"
echo "  ${REGISTRY}:latest"

# Add NGC metadata
echo "Adding NGC metadata..."
ngc registry resource tag update ${ORG}/${TEAM}/${IMAGE_NAME}:${VERSION} \
    --tags "version=${VERSION}" "framework=pytorch" "accelerated=gpu" "sap-hana=cloud" "langchain=integration"

echo "NGC publication completed successfully!"