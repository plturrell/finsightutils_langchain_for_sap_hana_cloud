#!/bin/bash

# SAP HANA Cloud LangChain Integration Deployment Script
#
# This script deploys the backend on NVIDIA GPU using Docker Compose
# and provides instructions for deploying the frontend on Vercel.

set -e

# Text formatting
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
BLUE="\033[0;34m"
RED="\033[0;31m"
RESET="\033[0m"

# Function to print section headers
section() {
  echo -e "${BOLD}${BLUE}==== $1 ====${RESET}"
  echo
}

# Function to print success messages
success() {
  echo -e "${GREEN}✓ $1${RESET}"
}

# Function to print warning messages
warning() {
  echo -e "${YELLOW}! $1${RESET}"
}

# Function to print error messages
error() {
  echo -e "${RED}✗ $1${RESET}"
  exit 1
}

# Check for required tools
check_requirements() {
  section "Checking Requirements"

  # Check for Docker
  if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker and try again."
  fi
  success "Docker is installed"

  # Check for Docker Compose
  if ! command -v docker compose &> /dev/null; then
    error "Docker Compose is not installed. Please install Docker Compose and try again."
  fi
  success "Docker Compose is installed"

  # Check for NVIDIA Docker
  if ! docker info | grep -q "Runtimes:.*nvidia"; then
    warning "NVIDIA Docker runtime not detected. GPU acceleration may not work."
    warning "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  else
    success "NVIDIA Docker runtime detected"
  fi

  # Check for nvidia-smi
  if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)
    success "NVIDIA GPU detected: $GPU_INFO"
  else
    warning "nvidia-smi not found. GPU acceleration may not work."
  fi

  echo
}

# Check if .env file exists, create it if not
check_env_file() {
  section "Checking Environment Configuration"

  if [ ! -f .env ]; then
    warning ".env file not found. Creating from .env.example..."
    
    if [ -f backend/.env.example ]; then
      cp backend/.env.example .env
      success "Created .env file from backend/.env.example"
      echo -e "${YELLOW}Please edit the .env file with your SAP HANA Cloud credentials${RESET}"
    else
      error "backend/.env.example not found. Cannot create .env file."
    fi
  else
    success ".env file found"
  fi

  # Check for required environment variables
  source .env
  REQUIRED_VARS=("HANA_HOST" "HANA_USER" "HANA_PASSWORD")
  MISSING_VARS=()

  for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
      MISSING_VARS+=("$var")
    fi
  done

  if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    warning "The following required environment variables are missing or empty:"
    for var in "${MISSING_VARS[@]}"; do
      echo -e "${YELLOW}  - $var${RESET}"
    done
    echo -e "${YELLOW}Please update the .env file before deployment${RESET}"
  else
    success "All required environment variables are set"
  fi

  echo
}

# Deploy the backend with Docker Compose
deploy_backend() {
  section "Deploying Backend with Docker Compose"
  
  # Check if directories exist, create if needed
  mkdir -p backend/data backend/logs backend/config prometheus

  # Copy connection configuration if it doesn't exist
  if [ ! -f backend/config/connection.json ]; then
    cp backend/config/connection.json backend/config/connection.json 2>/dev/null || echo "{}" > backend/config/connection.json
    success "Created default connection.json"
  fi

  # Build and start containers
  echo "Building and starting containers..."
  docker compose -f docker-compose.backend.yml up -d --build

  # Check if deployment was successful
  if [ $? -eq 0 ]; then
    success "Backend deployment successful"
    
    # Get backend URL
    BACKEND_URL="http://localhost:8000"
    echo -e "${BOLD}Backend URL:${RESET} $BACKEND_URL"
    
    # Wait for the API to become available
    echo "Waiting for the API to become available..."
    for i in {1..30}; do
      if curl -s "$BACKEND_URL/health/ping" > /dev/null; then
        success "API is up and running"
        break
      fi
      if [ $i -eq 30 ]; then
        warning "API not responding. Check the logs with: docker compose -f docker-compose.backend.yml logs backend"
      fi
      sleep 1
    done
  else
    error "Backend deployment failed"
  fi

  echo
}

# Show frontend deployment instructions
show_frontend_instructions() {
  section "Frontend Deployment on Vercel"
  
  BACKEND_URL="http://localhost:8000"
  if [ -n "$EXTERNAL_BACKEND_URL" ]; then
    BACKEND_URL="$EXTERNAL_BACKEND_URL"
  fi
  
  echo -e "To deploy the frontend on Vercel, follow these steps:"
  echo
  echo -e "1. ${BOLD}Update environment variables:${RESET}"
  echo -e "   Edit frontend/.env.production and set BACKEND_URL=$BACKEND_URL"
  echo
  echo -e "2. ${BOLD}Deploy to Vercel:${RESET}"
  echo -e "   cd frontend"
  echo -e "   vercel --prod"
  echo
  echo -e "3. ${BOLD}Alternative: Manual deployment:${RESET}"
  echo -e "   - Go to https://vercel.com/new"
  echo -e "   - Import your GitHub repository"
  echo -e "   - Set the root directory to 'frontend'"
  echo -e "   - Add the environment variable BACKEND_URL=$BACKEND_URL"
  echo -e "   - Deploy"
  echo
  echo -e "${YELLOW}See VERCEL_DEPLOYMENT.md for detailed instructions${RESET}"
  echo
}

# Show monitoring instructions
show_monitoring_instructions() {
  section "Monitoring"
  
  echo -e "The following monitoring services are available:"
  echo
  echo -e "1. ${BOLD}Prometheus:${RESET}"
  echo -e "   URL: http://localhost:9090"
  echo
  echo -e "2. ${BOLD}Grafana:${RESET}"
  echo -e "   URL: http://localhost:3001"
  echo -e "   Default credentials: admin/admin"
  echo
  echo -e "${YELLOW}Note: For security, please change the default Grafana password after first login${RESET}"
  echo
}

# Main deployment process
main() {
  echo -e "${BOLD}SAP HANA Cloud LangChain Integration Deployment${RESET}"
  echo -e "Backend: FastAPI on NVIDIA GPU"
  echo -e "Frontend: React on Vercel"
  echo

  # Check if script is running with required permissions
  if [ "$EUID" -eq 0 ]; then
    warning "Running as root. This is not recommended unless required by your environment."
  fi

  # Parse command line arguments
  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --backend-only) BACKEND_ONLY=true ;;
      --external-url) EXTERNAL_BACKEND_URL="$2"; shift ;;
      --help) 
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  --backend-only         Deploy only the backend"
        echo "  --external-url URL     External URL for the backend (for frontend configuration)"
        echo "  --help                 Show this help message"
        exit 0
        ;;
      *) error "Unknown parameter: $1" ;;
    esac
    shift
  done

  check_requirements
  check_env_file
  deploy_backend
  
  if [ "$BACKEND_ONLY" != true ]; then
    show_frontend_instructions
  fi
  
  show_monitoring_instructions

  section "Deployment Complete"
  echo -e "Your SAP HANA Cloud LangChain Integration has been deployed successfully!"
  echo -e "${BOLD}Next steps:${RESET}"
  echo -e "1. Deploy your frontend on Vercel (see instructions above)"
  echo -e "2. Update the backend connection configuration with your frontend URL"
  echo -e "3. Test the complete solution"
  echo
  echo -e "For more information, see the following documentation:"
  echo -e "- ${BOLD}DEPLOYMENT.md${RESET}: Detailed deployment guide"
  echo -e "- ${BOLD}VERCEL_DEPLOYMENT.md${RESET}: Vercel frontend deployment guide"
  echo -e "- ${BOLD}EXTENSIBILITY.md${RESET}: Adding support for other platforms"
  echo
}

# Run the main function
main "$@"