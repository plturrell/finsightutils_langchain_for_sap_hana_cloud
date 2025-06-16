#!/bin/bash
set -e

# Build the API container
echo "Building API container..."
docker build -t finsightintelligence/langchain-sap-hana:arrow-flight -f Dockerfile.arrow-flight .

# Push the API container
echo "Pushing API container to Docker Hub..."
docker push finsightintelligence/langchain-sap-hana:arrow-flight

echo "API build and push completed successfully!"