#!/bin/bash
# Script to test the API locally

# Exit on error
set -e

# Change to project directory
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Navigate to API directory
cd "$PROJECT_ROOT/api"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed"
    exit 1
fi

# Check if required packages are installed
if ! python3 -c "import fastapi, uvicorn, requests, jwt, pydantic" &> /dev/null; then
    echo "Installing required packages..."
    pip install fastapi uvicorn requests pyjwt pydantic python-multipart
fi

# Set environment variables for local testing
export T4_GPU_BACKEND_URL="https://jupyter0-513syzm60.brevlab.com"
export ENVIRONMENT="development"
export JWT_SECRET="sap-hana-langchain-t4-integration-secret-key-2025"

# Run the API locally
echo "Starting API server on http://localhost:8000..."
echo "Press Ctrl+C to stop the server"
cd "$PROJECT_ROOT"
python3 -m uvicorn api.vercel_integration:app --reload --host 0.0.0.0 --port 8000