#!/bin/bash
# Script to build and push the Docker image with financial embeddings integration

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="finsightintelligence/langchain-sap-hana"
TAG="financial-embeddings"
DATE_TAG=$(date +"%Y%m%d")

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Building Docker Image with Financial Embeddings Integration ${NC}"
echo -e "${BLUE}============================================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

# Check if the financial visualization data exists
if [ ! -f "financial_visualization_data.json" ]; then
  echo -e "${YELLOW}Financial visualization data not found.${NC}"
  echo -e "${YELLOW}Extracting data from SAP HANA Cloud...${NC}"
  
  # Activate virtual environment and run extraction script
  source venv/bin/activate
  python extract_vector_data.py
  
  if [ ! -f "financial_visualization_data.json" ]; then
    echo -e "${RED}Failed to extract financial visualization data.${NC}"
    exit 1
  fi
  
  echo -e "${GREEN}Financial visualization data extracted successfully.${NC}"
fi

# Create a temporary Dockerfile for this build
echo -e "${YELLOW}Creating Dockerfile...${NC}"
cat > Dockerfile.financial << EOL
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
COPY api/requirements.txt api_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r api_requirements.txt

# Copy the code
COPY langchain_hana ./langchain_hana
COPY api ./api
COPY financial_visualization_data.json .

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO
ENV TEST_MODE=false

# Expose the API port
EXPOSE 8000

# Run the API
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
EOL

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t ${IMAGE_NAME}:${TAG} -t ${IMAGE_NAME}:${TAG}-${DATE_TAG} -f Dockerfile.financial .

# Ask user if they want to push the image
echo -e "${YELLOW}Do you want to push the image to Docker Hub? (y/n)${NC}"
read -r push_image

if [[ $push_image == "y" || $push_image == "Y" ]]; then
  # Check if user is logged in to Docker Hub
  if ! docker info | grep -q "Username"; then
    echo -e "${YELLOW}You are not logged in to Docker Hub. Please log in:${NC}"
    docker login
  fi

  # Push the Docker image
  echo -e "${YELLOW}Pushing Docker image to Docker Hub...${NC}"
  docker push ${IMAGE_NAME}:${TAG}
  docker push ${IMAGE_NAME}:${TAG}-${DATE_TAG}
  echo -e "${GREEN}Docker image pushed successfully!${NC}"
fi

# Run the container for testing
echo -e "${YELLOW}Running container for testing...${NC}"
CONTAINER_ID=$(docker run -d -p 8000:8000 ${IMAGE_NAME}:${TAG})

# Wait for API startup
echo -e "${YELLOW}Waiting for API to start up (15 seconds)...${NC}"
sleep 15

# Test the API endpoints
echo -e "${YELLOW}Testing API health...${NC}"
HEALTH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "failed")

if [ "$HEALTH_STATUS" = "200" ]; then
  echo -e "${GREEN}API health check passed!${NC}"
  
  # Get health info
  echo -e "${YELLOW}Health endpoint response:${NC}"
  curl -s http://localhost:8000/health | python -m json.tool
  
  # Test financial embeddings API
  echo -e "\n${YELLOW}Testing financial embeddings API...${NC}"
  FINANCIAL_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/financial-embeddings/status || echo "failed")
  
  if [ "$FINANCIAL_STATUS" = "200" ]; then
    echo -e "${GREEN}Financial embeddings API check passed!${NC}"
    echo -e "${YELLOW}Financial embeddings status:${NC}"
    curl -s http://localhost:8000/financial-embeddings/status | python -m json.tool
    
    # Test visualization data endpoint
    echo -e "\n${YELLOW}Testing visualization data endpoint...${NC}"
    VIZ_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/financial-embeddings/visualization-data || echo "failed")
    
    if [ "$VIZ_STATUS" = "200" ]; then
      echo -e "${GREEN}Visualization data endpoint check passed!${NC}"
      
      # Get a sample of the visualization data
      echo -e "${YELLOW}Sample of visualization data:${NC}"
      curl -s http://localhost:8000/financial-embeddings/visualization-data | python -c "import sys, json; data = json.load(sys.stdin); print(f'Total vectors: {data.get(\"total_vectors\", 0)}'); print(f'Sample points: {data.get(\"points\", [])[:1]}'); print(f'Sample metadata: {data.get(\"metadata\", [])[:1]}')"
      
      echo -e "\n${GREEN}All tests passed successfully!${NC}"
    else
      echo -e "${RED}Visualization data endpoint check failed! Status: $VIZ_STATUS${NC}"
    fi
  else
    echo -e "${RED}Financial embeddings API check failed! Status: $FINANCIAL_STATUS${NC}"
  fi
else
  echo -e "${RED}API health check failed! Status: $HEALTH_STATUS${NC}"
  echo -e "${RED}Container logs:${NC}"
  docker logs $CONTAINER_ID
fi

# Cleanup container
echo -e "${YELLOW}Stopping and removing container...${NC}"
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID

# Clean up temp files
rm -f Dockerfile.financial

echo -e "${GREEN}Testing complete!${NC}"
echo -e "${GREEN}Your Docker image is available at:${NC}"
echo -e "  • ${IMAGE_NAME}:${TAG}"
echo -e "  • ${IMAGE_NAME}:${TAG}-${DATE_TAG}"

exit 0