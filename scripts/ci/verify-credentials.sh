#!/bin/bash
# Script to verify SAP HANA Cloud and DataSphere credentials

set -e

# Default values
TEST_HANA=true
TEST_DATASPHERE=true

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --hana-only             Test only SAP HANA Cloud connection"
  echo "  --datasphere-only       Test only SAP DataSphere connection"
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

# Check for required environment variables
check_env_vars() {
  local missing=false
  
  if [[ "$TEST_HANA" == "true" ]]; then
    for var in HANA_HOST HANA_PORT HANA_USER HANA_PASSWORD; do
      if [[ -z "${!var}" ]]; then
        echo "Error: $var environment variable is not set"
        missing=true
      fi
    done
  fi
  
  if [[ "$TEST_DATASPHERE" == "true" ]]; then
    for var in DATASPHERE_CLIENT_ID DATASPHERE_CLIENT_SECRET DATASPHERE_AUTH_URL DATASPHERE_TOKEN_URL DATASPHERE_API_URL; do
      if [[ -z "${!var}" ]]; then
        echo "Error: $var environment variable is not set"
        missing=true
      fi
    done
  fi
  
  if [[ "$missing" == "true" ]]; then
    echo "Please set all required environment variables before running this script."
    exit 1
  fi
}

# Load environment variables from .env file if it exists
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  echo "Loading environment variables from .env file"
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

# Check if all required environment variables are set
check_env_vars

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

# Install required dependencies if needed
echo "Checking for required Python packages..."
pip list | grep -E 'hdbcli|requests|requests-oauthlib' >/dev/null 2>&1 || {
  echo "Installing required packages..."
  pip install hdbcli requests requests-oauthlib
}

# Run the connection test
echo "Running connection tests..."
python "$PROJECT_ROOT/scripts/test_connections.py" $TEST_ARG

# Check exit code
if [[ $? -eq 0 ]]; then
  echo "✅ All connection tests passed successfully!"
  exit 0
else
  echo "❌ One or more connection tests failed."
  exit 1
fi