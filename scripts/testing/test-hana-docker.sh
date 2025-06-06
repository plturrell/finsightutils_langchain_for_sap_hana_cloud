#!/bin/bash
# Test SAP HANA Cloud connection using Docker

set -e

# Default values
USE_OAUTH=true
USE_BASIC=true
USE_ENV_FILE=false
ENV_FILE=""

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --oauth-only          Test only with OAuth authentication"
  echo "  --basic-only          Test only with basic authentication"
  echo "  --env-file <file>     Use environment file for credentials"
  echo "  -h, --help            Display this help message"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --oauth-only)
      USE_OAUTH=true
      USE_BASIC=false
      shift
      ;;
    --basic-only)
      USE_OAUTH=false
      USE_BASIC=true
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

RUN pip install --no-cache-dir hdbcli requests

COPY scripts/test_hana_oauth.py /app/

ENTRYPOINT ["python", "test_hana_oauth.py"]
EOF

# Build the Docker image
echo "Building Docker image for HANA connection test..."
docker build -t hana-tester -f "$DOCKERFILE" "$PROJECT_ROOT"

# Determine which authentication methods to test
CMD_ARGS=""
if [[ "$USE_OAUTH" == "true" && "$USE_BASIC" == "true" ]]; then
  # Both methods (default)
  CMD_ARGS=""
elif [[ "$USE_OAUTH" == "true" ]]; then
  CMD_ARGS="--oauth"
elif [[ "$USE_BASIC" == "true" ]]; then
  CMD_ARGS="--basic"
fi

# Run the Docker container with appropriate environment variables
echo "Running HANA connection test in Docker container..."

if [[ "$USE_ENV_FILE" == "true" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: Environment file $ENV_FILE does not exist"
    exit 1
  fi
  
  # Run with environment file
  docker run --rm \
    --env-file "$ENV_FILE" \
    hana-tester $CMD_ARGS
else
  # Run with current environment variables
  docker run --rm \
    -e HANA_HOST -e HANA_PORT -e HANA_USER -e HANA_PASSWORD \
    -e DATASPHERE_CLIENT_ID -e DATASPHERE_CLIENT_SECRET \
    -e DATASPHERE_TOKEN_URL \
    hana-tester $CMD_ARGS
fi

# Clean up
rm -rf "$TEMP_DIR"

echo "HANA connection test completed!"