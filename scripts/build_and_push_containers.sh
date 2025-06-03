#!/bin/bash

# Build and push both API and frontend containers to NGC

# Check if NGC CLI is installed and configured
if ! command -v ngc &> /dev/null; then
    echo "NGC CLI is not installed. Please install it first."
    echo "Visit: https://ngc.nvidia.com/setup/installers/cli"
    exit 1
fi

# Check if logged in to NGC
ngc config get | grep "apikey" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Not logged in to NGC. Please run 'ngc config set' first."
    exit 1
fi

# Variables
ORG="plturrell"
COLLECTION="sap-enhanced"
API_IMAGE_NAME="langchain-hana-gpu"
FRONTEND_IMAGE_NAME="langchain-hana-frontend"
VERSION=$(cat VERSION || echo "1.0.0")
TAG="$VERSION"

# Build and push the API container
echo "Building API container..."
docker build -t "nvcr.io/$ORG/$COLLECTION/$API_IMAGE_NAME:$TAG" -f api/Dockerfile.ngc ./api
docker tag "nvcr.io/$ORG/$COLLECTION/$API_IMAGE_NAME:$TAG" "nvcr.io/$ORG/$COLLECTION/$API_IMAGE_NAME:latest"

echo "Pushing API container to NGC..."
docker push "nvcr.io/$ORG/$COLLECTION/$API_IMAGE_NAME:$TAG"
docker push "nvcr.io/$ORG/$COLLECTION/$API_IMAGE_NAME:latest"

# Build and push the frontend container
echo "Building frontend container..."
docker build -t "nvcr.io/$ORG/$COLLECTION/$FRONTEND_IMAGE_NAME:$TAG" -f frontend/Dockerfile ./frontend
docker tag "nvcr.io/$ORG/$COLLECTION/$FRONTEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG/$COLLECTION/$FRONTEND_IMAGE_NAME:latest"

echo "Pushing frontend container to NGC..."
docker push "nvcr.io/$ORG/$COLLECTION/$FRONTEND_IMAGE_NAME:$TAG"
docker push "nvcr.io/$ORG/$COLLECTION/$FRONTEND_IMAGE_NAME:latest"

echo "Container build and push complete!"
echo "API: nvcr.io/$ORG/$COLLECTION/$API_IMAGE_NAME:$TAG"
echo "Frontend: nvcr.io/$ORG/$COLLECTION/$FRONTEND_IMAGE_NAME:$TAG"