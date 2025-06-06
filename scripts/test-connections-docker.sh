#!/bin/bash
# Test connections to SAP HANA Cloud and DataSphere using Docker
# This script uses Docker to avoid installing dependencies locally

set -e

# Default values
TEST_HANA=true
TEST_DATASPHERE=true
USE_ENV_FILE=false
ENV_FILE=""

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --hana-only             Test only SAP HANA Cloud connection"
  echo "  --datasphere-only       Test only SAP DataSphere connection"
  echo "  --env-file <file>       Use environment file for credentials"
  echo "  -h, --help              Display this help message"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hana-only)
      TEST_HANA=true
      TEST_DATASPHERE=false
      shift
      ;;
    --datasphere-only)
      TEST_HANA=false
      TEST_DATASPHERE=true
      shift
      ;;
    --env-file)
      USE_ENV_FILE=true
      ENV_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: Unknown option $1"
      usage
      ;;
  esac
done

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create a temporary Dockerfile
TEMP_DIR=$(mktemp -d)
DOCKERFILE="${TEMP_DIR}/Dockerfile"

cat > "$DOCKERFILE" << 'EOF'
FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir hdbcli requests requests-oauthlib

COPY scripts/test_connections.py /app/

ENTRYPOINT ["python", "test_connections.py"]
EOF

# Build the Docker image
echo "Building Docker image for connection tests..."
docker build -t connection-tester -f "$DOCKERFILE" "$PROJECT_ROOT"

# Determine which connections to test
if [[ "$TEST_HANA" == "true" && "$TEST_DATASPHERE" == "true" ]]; then
  TEST_ARG="--all"
elif [[ "$TEST_HANA" == "true" ]]; then
  TEST_ARG="--test-hana"
elif [[ "$TEST_DATASPHERE" == "true" ]]; then
  TEST_ARG="--test-datasphere"
else
  echo "Error: No connection test selected"
  usage
fi

# Run the Docker container with appropriate environment variables
echo "Running connection tests in Docker container..."

if [[ "$USE_ENV_FILE" == "true" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: Environment file $ENV_FILE does not exist"
    exit 1
  fi
  
  # Run with environment file
  docker run --rm --env-file "$ENV_FILE" connection-tester $TEST_ARG
else
  # Run with current environment variables
  docker run --rm \
    -e HANA_HOST -e HANA_PORT -e HANA_USER -e HANA_PASSWORD \
    -e DATASPHERE_CLIENT_ID -e DATASPHERE_CLIENT_SECRET \
    -e DATASPHERE_AUTH_URL -e DATASPHERE_TOKEN_URL -e DATASPHERE_API_URL \
    connection-tester $TEST_ARG
fi

# Clean up
rm -rf "$TEMP_DIR"

echo "Connection tests completed!"