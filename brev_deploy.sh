#!/bin/bash
# Direct API deployment script for SAP HANA Cloud LangChain Integration on NVIDIA Brev
# This script runs the API directly without Docker for debugging purposes

set -e

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

# Set up logging to file and console
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/deployment_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo_info "==================== DIRECT API DEPLOYMENT ===================="
echo_info "Deploying SAP HANA Cloud LangChain Integration with GPU Acceleration"
echo_info "Running API directly (no Docker) for debugging"
echo_info "==============================================================="

# Check if running in Brev environment
if [ -z "${BREV_ENV_ID}" ]; then
  echo_warning "Not running in a Brev environment. Some features may not work correctly."
fi

# Check for NVIDIA GPUs
echo_info "Checking for NVIDIA GPUs..."
if command -v nvidia-smi &> /dev/null; then
  echo_info "Detected NVIDIA GPUs:"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  
  # Check PyTorch CUDA compatibility
  echo_info "Checking PyTorch CUDA compatibility:"
  python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" || echo_warning "PyTorch CUDA check failed"
else
  echo_warning "NVIDIA driver not found. Running without GPU acceleration."
fi

# System information
echo_info "System information:"
echo_info "Hostname: $(hostname)"
echo_info "User: $(whoami)"
echo_info "Current directory: $(pwd)"
echo_info "Python version: $(python --version 2>&1)"
echo_info "Pip version: $(pip --version)"

# Create data directories
echo_info "Creating data directories..."
mkdir -p data/cache data/tensorrt api/logs api/data

# Set up Python environment
echo_info "Setting up Python environment..."
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export PYTHONUNBUFFERED=1

# Install dependencies
echo_info "Installing dependencies..."
if [ -f "requirements.txt" ]; then
  pip install --no-cache-dir -r requirements.txt
  echo_info "Installed root requirements.txt"
else
  echo_warning "Root requirements.txt not found"
fi

if [ -f "api/requirements.txt" ]; then
  pip install --no-cache-dir -r api/requirements.txt
  echo_info "Installed api/requirements.txt"
else
  echo_warning "api/requirements.txt not found"
fi

# Verify key packages
echo_info "Verifying key packages:"
pip list | grep -E 'fastapi|uvicorn|pydantic|langchain|torch|sentence-transformers|hdbcli'

# Set environment variables for test mode
echo_info "Setting up environment variables..."
export TEST_MODE=true
export ENABLE_CORS=true
export LOG_LEVEL=DEBUG  # Enable detailed logging for debugging
export PORT=8000

# Verify API files exist
echo_info "Verifying API files exist:"
find api -name "*.py" | sort

# Test importing test_mode module
echo_info "Testing test_mode module import:"
cd api
python -c "import logging; logging.basicConfig(level=logging.DEBUG); import test_mode; print('Test mode imported successfully')" || echo_warning "Failed to import test_mode"

# Start the API with comprehensive logging
echo_info "Starting API server with TEST_MODE=true and DEBUG logging..."
echo_info "API logs will be saved to $LOG_DIR and displayed in the console"

# Run the API
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level debug --reload