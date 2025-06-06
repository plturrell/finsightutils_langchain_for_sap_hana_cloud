#!/bin/bash
# Test connections to SAP HANA Cloud and DataSphere from inside a Kubernetes pod
# This helps verify that the credentials work in the Kubernetes environment

set -e

# Default values
TEST_HANA=true
TEST_DATASPHERE=true
NAMESPACE="default"
SECRET_NAME="connection-test-secret"
POD_NAME="connection-test-pod"
DELETE_RESOURCES=true

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --hana-only             Test only SAP HANA Cloud connection"
  echo "  --datasphere-only       Test only SAP DataSphere connection"
  echo "  --namespace <namespace> Kubernetes namespace to use (default: default)"
  echo "  --keep-resources        Don't delete resources after test"
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
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --keep-resources)
      DELETE_RESOURCES=false
      shift
      ;;
    --env-file)
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

# Load environment variables from .env file if specified
if [[ -n "$ENV_FILE" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: Environment file $ENV_FILE does not exist"
    exit 1
  fi
  
  echo "Loading environment variables from $ENV_FILE"
  set -a
  source "$ENV_FILE"
  set +a
fi

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

# Check if all required environment variables are set
check_env_vars

# Clean up resources on exit if requested
cleanup() {
  if [[ "$DELETE_RESOURCES" == "true" ]]; then
    echo "Cleaning up Kubernetes resources..."
    kubectl delete pod "$POD_NAME" --namespace "$NAMESPACE" --ignore-not-found
    kubectl delete secret "$SECRET_NAME" --namespace "$NAMESPACE" --ignore-not-found
    echo "Cleanup completed."
  fi
}

# Set up trap to clean up on exit
trap cleanup EXIT

# Create Kubernetes secret with credentials
echo "Creating Kubernetes secret with credentials..."
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

if [[ "$TEST_HANA" == "true" && "$TEST_DATASPHERE" == "true" ]]; then
  kubectl create secret generic "$SECRET_NAME" \
    --namespace "$NAMESPACE" \
    --from-literal=HANA_HOST="$HANA_HOST" \
    --from-literal=HANA_PORT="$HANA_PORT" \
    --from-literal=HANA_USER="$HANA_USER" \
    --from-literal=HANA_PASSWORD="$HANA_PASSWORD" \
    --from-literal=DATASPHERE_CLIENT_ID="$DATASPHERE_CLIENT_ID" \
    --from-literal=DATASPHERE_CLIENT_SECRET="$DATASPHERE_CLIENT_SECRET" \
    --from-literal=DATASPHERE_AUTH_URL="$DATASPHERE_AUTH_URL" \
    --from-literal=DATASPHERE_TOKEN_URL="$DATASPHERE_TOKEN_URL" \
    --from-literal=DATASPHERE_API_URL="$DATASPHERE_API_URL" \
    --dry-run=client -o yaml | kubectl apply -f -
elif [[ "$TEST_HANA" == "true" ]]; then
  kubectl create secret generic "$SECRET_NAME" \
    --namespace "$NAMESPACE" \
    --from-literal=HANA_HOST="$HANA_HOST" \
    --from-literal=HANA_PORT="$HANA_PORT" \
    --from-literal=HANA_USER="$HANA_USER" \
    --from-literal=HANA_PASSWORD="$HANA_PASSWORD" \
    --dry-run=client -o yaml | kubectl apply -f -
elif [[ "$TEST_DATASPHERE" == "true" ]]; then
  kubectl create secret generic "$SECRET_NAME" \
    --namespace "$NAMESPACE" \
    --from-literal=DATASPHERE_CLIENT_ID="$DATASPHERE_CLIENT_ID" \
    --from-literal=DATASPHERE_CLIENT_SECRET="$DATASPHERE_CLIENT_SECRET" \
    --from-literal=DATASPHERE_AUTH_URL="$DATASPHERE_AUTH_URL" \
    --from-literal=DATASPHERE_TOKEN_URL="$DATASPHERE_TOKEN_URL" \
    --from-literal=DATASPHERE_API_URL="$DATASPHERE_API_URL" \
    --dry-run=client -o yaml | kubectl apply -f -
fi

# Create the test pod manifest
TEST_SCRIPT_PATH="$PROJECT_ROOT/scripts/test_connections.py"
TEST_SCRIPT=$(cat "$TEST_SCRIPT_PATH" | base64 -w 0)

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

# Create and apply the pod manifest
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: $POD_NAME
  namespace: $NAMESPACE
spec:
  containers:
  - name: connection-tester
    image: python:3.9-slim
    command: ["/bin/sh", "-c"]
    args:
      - |
        pip install --no-cache-dir hdbcli requests requests-oauthlib
        echo "$TEST_SCRIPT" | base64 -d > /app/test_connections.py
        python /app/test_connections.py $TEST_ARG
    envFrom:
    - secretRef:
        name: $SECRET_NAME
    resources:
      limits:
        memory: "512Mi"
        cpu: "500m"
      requests:
        memory: "256Mi"
        cpu: "100m"
  restartPolicy: Never
EOF

echo "Waiting for pod to start..."
kubectl wait --for=condition=Ready pod/$POD_NAME --namespace "$NAMESPACE" --timeout=60s || true

echo "Following pod logs..."
kubectl logs -f "$POD_NAME" --namespace "$NAMESPACE"

# Check pod status
POD_STATUS=$(kubectl get pod "$POD_NAME" --namespace "$NAMESPACE" -o jsonpath='{.status.phase}')
if [[ "$POD_STATUS" == "Succeeded" ]]; then
  echo "✅ Connection tests completed successfully!"
  exit 0
else
  echo "❌ Connection tests failed. Pod status: $POD_STATUS"
  exit 1
fi