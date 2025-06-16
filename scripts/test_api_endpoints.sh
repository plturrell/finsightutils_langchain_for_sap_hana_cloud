#!/bin/bash

# Script to test all FastAPI endpoints in the backend
# This script will build and run the Docker container, then test all endpoints

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
  echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to test an endpoint
test_endpoint() {
  local method=$1
  local endpoint=$2
  local data=$3
  local expected_status=$4
  local description=$5
  
  echo -e "${YELLOW}Testing: ${description}${NC}"
  echo -e "Endpoint: ${method} ${endpoint}"
  
  if [ -n "$data" ]; then
    response=$(curl -s -X ${method} http://localhost:8001${endpoint} \
      -H "Content-Type: application/json" \
      -d "${data}" \
      -w "\n%{http_code}")
  else
    response=$(curl -s -X ${method} http://localhost:8001${endpoint} \
      -w "\n%{http_code}")
  fi
  
  # Extract status code from response
  status_code=$(echo "$response" | tail -n1)
  content=$(echo "$response" | sed '$d')
  
  # Check if status code matches expected
  if [ "$status_code" -eq "$expected_status" ]; then
    echo -e "${GREEN}✓ Success${NC} (Status: ${status_code})"
  else
    echo -e "${RED}✗ Failed${NC} (Status: ${status_code}, Expected: ${expected_status})"
    echo "Response: $content"
  fi
  
  echo
}

# Build and start the container
print_header "BUILDING AND STARTING CONTAINER"
docker-compose -f docker-compose.local.yml up -d --build
echo "Waiting for API to become available..."
sleep 10

# Health check
print_header "HEALTH ENDPOINTS"
test_endpoint "GET" "/health" "" 200 "Health Check"
test_endpoint "GET" "/health/ping" "" 200 "Health Ping"
test_endpoint "GET" "/health/status" "" 200 "Health Status"

# Basic API endpoints
print_header "BASIC API ENDPOINTS"
test_endpoint "GET" "/" "" 200 "Root Endpoint"
test_endpoint "GET" "/api/feature/error-handling" "" 200 "Error Handling Info"
test_endpoint "GET" "/api/feature/vector-similarity" "" 200 "Vector Similarity Info"
test_endpoint "GET" "/api/deployment/info" "" 200 "Deployment Info"

# GPU information
print_header "GPU ENDPOINTS"
test_endpoint "GET" "/gpu/info" "" 200 "GPU Information"

# Vector search endpoints
print_header "VECTOR SEARCH ENDPOINTS"
test_endpoint "POST" "/api/search" '{"query": "This is a test query", "k": 3}' 200 "Vector Search API"

# Skip vector operations endpoints - not available in minimal mode

# Documentation endpoints
print_header "DOCUMENTATION ENDPOINTS"
test_endpoint "GET" "/docs" "" 200 "Swagger UI"
test_endpoint "GET" "/redoc" "" 200 "ReDoc"
test_endpoint "GET" "/openapi.json" "" 200 "OpenAPI Schema"

# Stop the container
print_header "STOPPING CONTAINER"
docker-compose -f docker-compose.local.yml down
echo -e "${GREEN}All tests completed!${NC}"