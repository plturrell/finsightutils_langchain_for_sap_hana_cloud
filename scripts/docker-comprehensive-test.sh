#!/bin/bash
# Run comprehensive connection tests in Docker

set -e

# Default values
TEST_HANA=true
TEST_DATASPHERE=true
USE_ENV_FILE=false
ENV_FILE=""
OUTPUT_FILE="test_results.json"
SCHEMA=""
SPACE_ID=""
QUERY=""

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --hana-only             Test only SAP HANA Cloud connection"
  echo "  --datasphere-only       Test only SAP DataSphere connection"
  echo "  --env-file <file>       Use environment file for credentials"
  echo "  --output <file>         Output file for test results (default: test_results.json)"
  echo "  --schema <schema>       HANA schema to test"
  echo "  --space-id <id>         DataSphere space ID to test"
  echo "  --query <query>         Custom HANA query to execute"
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
    --output)
      OUTPUT_FILE="$2"
      shift 2
      ;;
    --schema)
      SCHEMA="$2"
      shift 2
      ;;
    --space-id)
      SPACE_ID="$2"
      shift 2
      ;;
    --query)
      QUERY="$2"
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

COPY scripts/comprehensive_test.py /app/

ENTRYPOINT ["python", "comprehensive_test.py"]
EOF

# Build the Docker image
echo "Building Docker image for comprehensive connection tests..."
docker build -t comprehensive-tester -f "$DOCKERFILE" "$PROJECT_ROOT"

# Determine which connections to test
CMD_ARGS=""
if [[ "$TEST_HANA" == "true" && "$TEST_DATASPHERE" == "true" ]]; then
  CMD_ARGS="--all"
elif [[ "$TEST_HANA" == "true" ]]; then
  CMD_ARGS="--test-hana"
elif [[ "$TEST_DATASPHERE" == "true" ]]; then
  CMD_ARGS="--test-datasphere"
else
  echo "Error: No connection test selected"
  usage
fi

# Add optional arguments
if [[ -n "$SCHEMA" ]]; then
  CMD_ARGS="$CMD_ARGS --schema $SCHEMA"
fi

if [[ -n "$SPACE_ID" ]]; then
  CMD_ARGS="$CMD_ARGS --space-id $SPACE_ID"
fi

if [[ -n "$QUERY" ]]; then
  CMD_ARGS="$CMD_ARGS --query \"$QUERY\""
fi

CMD_ARGS="$CMD_ARGS --output /app/$OUTPUT_FILE"

# Run the Docker container with appropriate environment variables
echo "Running comprehensive connection tests in Docker container..."

if [[ "$USE_ENV_FILE" == "true" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: Environment file $ENV_FILE does not exist"
    exit 1
  fi
  
  # Create output directory if it doesn't exist
  mkdir -p "$(dirname "$PROJECT_ROOT/$OUTPUT_FILE")"
  
  # Run with environment file and mount the parent directory
  OUTPUT_DIR=$(dirname "$PROJECT_ROOT/$OUTPUT_FILE")
  OUTPUT_FILENAME=$(basename "$OUTPUT_FILE")
  
  # Run with environment file
  docker run --rm \
    --env-file "$ENV_FILE" \
    -v "$OUTPUT_DIR:/app/output" \
    comprehensive-tester $CMD_ARGS --output "/app/output/$OUTPUT_FILENAME"
else
  # Create output directory if it doesn't exist
  mkdir -p "$(dirname "$PROJECT_ROOT/$OUTPUT_FILE")"
  
  # Run with current environment variables
  OUTPUT_DIR=$(dirname "$PROJECT_ROOT/$OUTPUT_FILE")
  OUTPUT_FILENAME=$(basename "$OUTPUT_FILE")
  
  docker run --rm \
    -e HANA_HOST -e HANA_PORT -e HANA_USER -e HANA_PASSWORD \
    -e DATASPHERE_CLIENT_ID -e DATASPHERE_CLIENT_SECRET \
    -e DATASPHERE_AUTH_URL -e DATASPHERE_TOKEN_URL -e DATASPHERE_API_URL \
    -v "$OUTPUT_DIR:/app/output" \
    comprehensive-tester $CMD_ARGS --output "/app/output/$OUTPUT_FILENAME"
fi

# Clean up
rm -rf "$TEMP_DIR"

echo "Connection tests completed!"
echo "Results saved to: $OUTPUT_FILE"

# Display summary of results
if [[ -f "$OUTPUT_DIR/$OUTPUT_FILENAME" ]]; then
  echo "Test results summary:"
  if command -v jq &> /dev/null; then
    jq '.hana.status, .datasphere.status' "$OUTPUT_DIR/$OUTPUT_FILENAME" 2>/dev/null || echo "Cannot parse JSON results."
  else
    grep -E '"status": "(success|failed)"' "$OUTPUT_DIR/$OUTPUT_FILENAME" 2>/dev/null || echo "Cannot display summary without jq tool."
  fi
  
  # Display result file path
  echo "Full test results available at: $OUTPUT_DIR/$OUTPUT_FILENAME"
else
  echo "No test results file was created."
fi