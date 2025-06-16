#!/bin/bash
set -e

# Build the Frontend container
echo "Building Frontend container..."
cd frontend
docker build -t finsightintelligence/langchain-hana:frontend .
cd ..

# Push the Frontend container
echo "Pushing Frontend container to Docker Hub..."
docker push finsightintelligence/langchain-hana:frontend

echo "Frontend build and push completed successfully!"