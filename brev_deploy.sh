#!/bin/bash
# Deployment script for SAP HANA Cloud LangChain Integration on NVIDIA Brev

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

echo_info "Deploying SAP HANA Cloud LangChain Integration with GPU Acceleration"

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

# Create data directories
echo_info "Creating data directories..."
mkdir -p data/cache data/tensorrt

# Check for .env file
if [ ! -f .env ]; then
  echo_warning "No .env file found. Checking for SAP HANA credentials in environment variables..."
  
  if [ -z "$HANA_HOST" ] || [ -z "$HANA_USER" ] || [ -z "$HANA_PASSWORD" ]; then
    echo_warning "SAP HANA credentials not found in environment variables."
    echo_info "Running in TEST_MODE=true without SAP HANA connection."
    echo_info "To connect to a real SAP HANA instance, run ./setup_hana_credentials.sh"
    
    # Set TEST_MODE to true
    export TEST_MODE=true
  else
    echo_info "Found SAP HANA credentials in environment variables."
    export TEST_MODE=false
  fi
else
  echo_info "Loading credentials from .env file..."
  set -a
  source .env
  set +a
fi

# Pull and build images
echo_info "Building services (this may take a while for the first run)..."
docker-compose build

# Start services
echo_info "Starting services with GPU support..."
docker-compose up -d

# Wait for services to be ready
echo_info "Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Exit"; then
  echo_error "One or more services failed to start. Checking logs..."
  docker-compose logs
  exit 1
fi

# Check API health
echo_info "Checking API health..."
if curl -s http://localhost:8000/health > /dev/null; then
  echo_success "API is healthy!"
else
  echo_warning "API health check failed. See logs for details:"
  docker-compose logs api
fi

echo_success "Deployment completed successfully!"
echo_info "API is available at: http://localhost:8000"
echo_info "API Documentation: http://localhost:8000/docs"
echo_info "Frontend is available at: http://localhost:3000"
echo_info "GPU Info: http://localhost:8000/gpu/info"
echo_info ""
echo_info "To view logs: docker-compose logs -f"
echo_info "To stop services: docker-compose down"