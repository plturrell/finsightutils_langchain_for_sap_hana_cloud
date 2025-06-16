#!/bin/bash
# Script to test the Docker build implementation

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}     Docker Build Implementation Test    ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

# Test 1: Build a minimal test image
echo -e "\n${YELLOW}Test 1: Building minimal CPU image...${NC}"
echo -e "${YELLOW}Running: ./docker-build.sh --type cpu-secure${NC}"
./docker-build.sh --type cpu-secure

# Test 2: Test the image
echo -e "\n${YELLOW}Test 2: Testing the built image...${NC}"
echo -e "${YELLOW}Running: ./docker-build.sh --type cpu-secure --test${NC}"
./docker-build.sh --type cpu-secure --test

# Test 3: Try Docker Compose
echo -e "\n${YELLOW}Test 3: Testing Docker Compose integration...${NC}"
echo -e "${YELLOW}Running: docker-compose -f docker-compose.integrated.yml up -d${NC}"
docker-compose -f docker-compose.integrated.yml up -d

# Check if containers are running
echo -e "${YELLOW}Checking container status...${NC}"
docker-compose -f docker-compose.integrated.yml ps

# Test API health
echo -e "${YELLOW}Testing API health...${NC}"
curl -s http://localhost:8000/health

# Clean up compose
echo -e "${YELLOW}Cleaning up Docker Compose services...${NC}"
docker-compose -f docker-compose.integrated.yml down

echo -e "\n${GREEN}All tests completed!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "Next steps:"
echo -e "1. Push your code to GitHub to test the GitHub Actions workflows"
echo -e "2. Run a full build with: ./docker-build.sh --type cpu-secure --push"
echo -e "${BLUE}=========================================${NC}"

exit 0