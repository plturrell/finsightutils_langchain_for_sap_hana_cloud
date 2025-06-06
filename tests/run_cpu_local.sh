#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Running CPU-only Test Container for SAP HANA Cloud LangChain${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Stop any existing container with the same name
if docker ps -a | grep -q "langchain-cpu-test"; then
    echo -e "${YELLOW}Stopping and removing existing langchain-cpu-test container...${NC}"
    docker stop langchain-cpu-test >/dev/null 2>&1 || true
    docker rm langchain-cpu-test >/dev/null 2>&1 || true
fi

echo -e "${BLUE}Creating test directory...${NC}"
mkdir -p /tmp/langchain-test

# Copy necessary files to the test directory
echo -e "${BLUE}Copying files to test directory...${NC}"
cp test_app_enhanced.py /tmp/langchain-test/
cp requirements.txt /tmp/langchain-test/

echo -e "${BLUE}Running the CPU-only container...${NC}"
docker run -d --name langchain-cpu-test \
    -p 8000:8000 \
    -v /tmp/langchain-test:/app \
    -e GPU_ENABLED=false \
    -e LOG_LEVEL=DEBUG \
    -e ENABLE_CORS=true \
    -w /app \
    python:3.9 \
    bash -c "apt-get update && apt-get install -y curl && pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir fastapi uvicorn pydantic python-multipart psutil torch && python -m uvicorn test_app_enhanced:app --host 0.0.0.0 --port 8000"

echo -e "${BLUE}Waiting for container to start...${NC}"
sleep 5

# Test the container
echo -e "${BLUE}Testing the container...${NC}"
max_retries=6
retry_count=0
success=false

while [ $retry_count -lt $max_retries ]; do
    if curl -s http://localhost:8000/health >/dev/null 2>&1; then
        success=true
        break
    fi
    echo -e "${YELLOW}Health check not yet available, retrying in 5 seconds...${NC}"
    sleep 5
    retry_count=$((retry_count + 1))
done

if [ "$success" = true ]; then
    echo -e "${GREEN}Container is running and health check passed!${NC}"
    echo -e "${BLUE}API is available at:${NC} http://localhost:8000"
    echo -e "${BLUE}Testing endpoints...${NC}"
    
    echo -e "\n${BLUE}Health check:${NC}"
    curl -s http://localhost:8000/health
    
    echo -e "\n${BLUE}System info:${NC}"
    curl -s http://localhost:8000/system-info | python -m json.tool
    
    echo -e "\n${BLUE}To view logs:${NC} docker logs langchain-cpu-test"
    echo -e "${BLUE}To stop the container:${NC} docker stop langchain-cpu-test"
    echo -e "${BLUE}To remove the container:${NC} docker rm langchain-cpu-test"
else
    echo -e "${RED}Container failed to start or health check failed.${NC}"
    echo -e "${YELLOW}Container logs:${NC}"
    docker logs langchain-cpu-test
fi

echo -e "${BLUE}=========================================================${NC}"