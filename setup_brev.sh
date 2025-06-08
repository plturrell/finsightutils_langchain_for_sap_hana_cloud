#!/bin/bash
# Setup script for SAP HANA Cloud LangChain Integration on NVIDIA Brev Environment

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

echo_info "Starting setup of SAP HANA Cloud LangChain Integration with GPU Acceleration"

# Check if running in Brev environment
if [ -z "${BREV_ENV_ID}" ]; then
  echo_warning "Not running in a Brev environment. Some features may not work correctly."
fi

# Check for NVIDIA GPUs
if ! command -v nvidia-smi &> /dev/null; then
  echo_error "NVIDIA driver not found. This setup requires NVIDIA GPUs."
  echo_info "Please ensure NVIDIA drivers are installed and GPUs are available."
  exit 1
fi

# Display NVIDIA GPU information
echo_info "Detected NVIDIA GPUs:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Check Docker installation
if ! command -v docker &> /dev/null; then
  echo_error "Docker not found. This setup requires Docker."
  echo_info "Please install Docker: https://docs.docker.com/get-docker/"
  exit 1
fi

# Check Docker Compose installation
if ! command -v docker-compose &> /dev/null; then
  echo_warning "Docker Compose not found. Installing docker-compose..."
  sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
fi

# Create directory structure
echo_info "Creating directory structure..."
mkdir -p ./data ./logs ./cache

# Copy configuration file
echo_info "Setting up docker-compose configuration..."
cp brev-compose.yaml docker-compose.yaml

# Check if .env file exists
if [ ! -f .env ]; then
  echo_info "Creating .env file from template..."
  cp .env.example .env
  echo_info "Please update the .env file with your SAP HANA Cloud credentials."
fi

# Pull Docker images
echo_info "Pulling required Docker images..."
docker pull nvcr.io/nvidia/pytorch:23.12-py3

# Check Docker GPU support
echo_info "Checking Docker GPU support..."
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
  echo_error "Docker GPU support is not working correctly."
  echo_info "Please ensure the NVIDIA Container Toolkit is installed: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  exit 1
fi

# Start services
echo_info "Starting services with GPU support..."
docker-compose up -d

# Check service health
echo_info "Checking service health..."
sleep 10

if docker-compose ps | grep -q "Exit"; then
  echo_error "One or more services failed to start. Checking logs..."
  docker-compose logs
  exit 1
else
  echo_success "Services started successfully!"
  echo_info "API is available at: http://localhost:8000"
  echo_info "API Documentation: http://localhost:8000/docs"
  echo_info "Frontend is available at: http://localhost:3000"
  echo_info "GPU Info: http://localhost:8000/gpu/info"
fi

echo_success "Setup completed successfully!"
echo_info "To view logs: docker-compose logs -f"
echo_info "To stop services: docker-compose down"