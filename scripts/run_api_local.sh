#!/bin/bash

# Script to run the API server locally for testing
# This script will set up the required environment variables and start the API server

# Set environment variables
export PYTHONPATH=$(pwd)
export TEST_MODE=true
export LOG_LEVEL=INFO
export ENABLE_ERROR_CONTEXT=true
export ERROR_DETAIL_LEVEL=verbose
export INCLUDE_SUGGESTIONS=true
export ENABLE_CORS=true
export CORS_ORIGINS=*
export JWT_SECRET=devsecrethanacloud
export EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
export MULTI_GPU_ENABLED=false
export USE_TENSORRT=false

# Create necessary directories
mkdir -p cache/tensorrt
mkdir -p cache/embeddings

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r api/requirements.txt
else
    source venv/bin/activate
fi

# Run the server
cd api
echo "Starting API server in test mode..."
echo "API will be available at http://localhost:8000"
echo "Press Ctrl+C to stop the server"

# Choose the appropriate module based on availability
if [ -f "app.py" ]; then
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
elif [ -f "index.py" ]; then
    uvicorn index:app --reload --host 0.0.0.0 --port 8000
else
    echo "Error: Could not find main application module (app.py or index.py)"
    exit 1
fi