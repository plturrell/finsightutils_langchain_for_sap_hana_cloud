#!/bin/bash
set -e

# Build the API container
echo "Building API container..."
docker build -t finsightintelligence/langchain-sap-hana:arrow-flight -f Dockerfile.arrow-flight .

# Build the Frontend container
echo "Building Frontend container..."
docker build -t finsightintelligence/langchain-hana:frontend -f frontend/Dockerfile ./frontend

# Push the API container
echo "Pushing API container to Docker Hub..."
docker push finsightintelligence/langchain-sap-hana:arrow-flight

# Push the Frontend container
echo "Pushing Frontend container to Docker Hub..."
docker push finsightintelligence/langchain-hana:frontend

echo "Build and push completed successfully!"

# Start the containers
echo "Starting containers..."
docker-compose -f docker-compose.complete.yml up -d

# Check health
echo "Checking API health..."
sleep 10  # Give containers time to start
curl -f http://localhost:8000/health || echo "API health check failed"

echo "Checking Frontend availability..."
curl -f http://localhost:3000 || echo "Frontend check failed"

echo "Deployment complete!"