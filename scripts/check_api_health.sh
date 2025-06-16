#!/bin/bash
# Helper script to check API health status

# Print colored messages
function echo_info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

function echo_success() {
  echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

function echo_error() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
}

function echo_warning() {
  echo -e "\033[1;33m[WARNING]\033[0m $1"
}

API_URL=${1:-"http://localhost:8000"}
echo_info "Checking API health at $API_URL"

# Function to check a specific endpoint
check_endpoint() {
  local endpoint=$1
  local description=$2
  local expect_json=${3:-true}
  
  echo_info "Checking $description ($endpoint)..."
  
  if $expect_json; then
    response=$(curl -s $API_URL$endpoint)
    if [ $? -eq 0 ] && [[ $response == {* ]]; then
      echo_success "$description is healthy"
      echo "Response: $response"
      return 0
    else
      echo_error "$description check failed"
      echo "Response: $response"
      return 1
    fi
  else
    response=$(curl -s $API_URL$endpoint)
    if [ $? -eq 0 ]; then
      echo_success "$description is healthy"
      echo "Response: $response"
      return 0
    else
      echo_error "$description check failed"
      echo "Response: $response"
      return 1
    fi
  fi
}

# Check basic health endpoints
check_endpoint "/" "Root endpoint"
check_endpoint "/health" "Health endpoint"
check_endpoint "/health/ping" "Ping endpoint" false
check_endpoint "/health/status" "Status endpoint"
check_endpoint "/gpu/info" "GPU information"
check_endpoint "/health/complete" "Complete health check"

# Print summary
echo_info "API health check complete"
echo_info "If any checks failed, check the logs for more details"
echo_info "To view complete health information, visit $API_URL/health/complete"