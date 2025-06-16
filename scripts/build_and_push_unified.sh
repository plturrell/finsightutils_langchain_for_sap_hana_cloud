#!/bin/bash
set -e

# Build the unified container
echo "Building unified container..."
docker build -t finsightintelligence/langchain-sap-hana:unified -f Dockerfile.unified .

# Push the container to Docker Hub
echo "Pushing unified container to Docker Hub..."
docker push finsightintelligence/langchain-sap-hana:unified

echo "Build and push completed successfully!"

# Start the container
echo "Starting unified container..."
docker-compose -f docker-compose.unified.yml up -d

# Check health
echo "Checking API health..."
sleep 10  # Give container time to start
curl -f http://localhost:8000/health || echo "API health check failed"

echo "Checking frontend availability..."
curl -f http://localhost:3000 || echo "Frontend check failed"

echo "Deployment complete! Access the application at:"
echo "- Frontend: http://localhost:3000"
echo "- API: http://localhost:8000"
echo "- Arrow Flight: localhost:8815 (gRPC)"