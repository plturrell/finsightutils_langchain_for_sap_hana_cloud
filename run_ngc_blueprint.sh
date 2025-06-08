#!/bin/bash

# NGC Blueprint quick start script for SAP HANA Cloud LangChain Integration
# This script helps users quickly deploy the NGC Blueprint

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print section headers
print_header() {
  echo -e "\n${BLUE}=== $1 ===${NC}\n"
}

# Function to check if NVIDIA GPU is available
check_nvidia_gpu() {
  if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    return 0
  else
    echo -e "${YELLOW}No NVIDIA GPU detected. You can still run in test mode without GPU acceleration.${NC}"
    return 1
  fi
}

# Function to check Docker and Docker Compose
check_docker() {
  if command -v docker &> /dev/null; then
    echo -e "${GREEN}Docker is installed.${NC}"
  else
    echo -e "${RED}Docker is not installed. Please install Docker before continuing.${NC}"
    exit 1
  fi

  if docker compose version &> /dev/null; then
    echo -e "${GREEN}Docker Compose is available.${NC}"
  else
    echo -e "${YELLOW}Docker Compose V2 not detected. Checking for docker-compose...${NC}"
    if command -v docker-compose &> /dev/null; then
      echo -e "${GREEN}Docker Compose (standalone) is installed.${NC}"
      USE_STANDALONE_COMPOSE=1
    else
      echo -e "${RED}Docker Compose is not installed. Please install Docker Compose before continuing.${NC}"
      exit 1
    fi
  fi
}

# Function to prompt for SAP HANA Cloud credentials
prompt_credentials() {
  print_header "SAP HANA Cloud Configuration"
  
  echo -e "Do you have SAP HANA Cloud credentials? (y/n)"
  read -r has_credentials
  
  if [[ "$has_credentials" =~ ^[Yy]$ ]]; then
    echo -e "Enter SAP HANA Cloud host (e.g., hana-host.hanacloud.ondemand.com):"
    read -r HANA_HOST
    
    echo -e "Enter SAP HANA Cloud port [443]:"
    read -r HANA_PORT
    HANA_PORT=${HANA_PORT:-443}
    
    echo -e "Enter SAP HANA Cloud username:"
    read -r HANA_USER
    
    echo -e "Enter SAP HANA Cloud password:"
    read -rs HANA_PASSWORD
    echo ""
    
    echo -e "Enter table name for vector storage [EMBEDDINGS]:"
    read -r DEFAULT_TABLE_NAME
    DEFAULT_TABLE_NAME=${DEFAULT_TABLE_NAME:-EMBEDDINGS}
    
    # Export variables
    export HANA_HOST
    export HANA_PORT
    export HANA_USER
    export HANA_PASSWORD
    export DEFAULT_TABLE_NAME
    export TEST_MODE=false
    
    echo -e "${GREEN}SAP HANA Cloud credentials configured.${NC}"
  else
    echo -e "${YELLOW}Running in TEST MODE without real SAP HANA Cloud connection.${NC}"
    export TEST_MODE=true
  fi
}

# Function to set GPU configuration
configure_gpu() {
  print_header "GPU Configuration"
  
  if check_nvidia_gpu; then
    echo -e "Enable TensorRT optimization? (y/n) [y]:"
    read -r use_tensorrt
    if [[ "$use_tensorrt" =~ ^[Nn]$ ]]; then
      export USE_TENSORRT=false
      echo -e "${YELLOW}TensorRT optimization disabled.${NC}"
    else
      export USE_TENSORRT=true
      
      echo -e "Select TensorRT precision:"
      echo -e "1) FP32 (32-bit floating point, higher accuracy)"
      echo -e "2) FP16 (16-bit floating point, faster, recommended) [default]"
      echo -e "3) INT8 (8-bit integer, fastest, lower accuracy)"
      read -r precision_choice
      
      case $precision_choice in
        1) export TENSORRT_PRECISION=fp32 ;;
        3) export TENSORRT_PRECISION=int8 ;;
        *) export TENSORRT_PRECISION=fp16 ;;
      esac
      
      echo -e "${GREEN}TensorRT optimization enabled with ${TENSORRT_PRECISION} precision.${NC}"
    fi
    
    echo -e "Enable multi-GPU support (if multiple GPUs are available)? (y/n) [y]:"
    read -r use_multi_gpu
    if [[ "$use_multi_gpu" =~ ^[Nn]$ ]]; then
      export ENABLE_MULTI_GPU=false
      echo -e "${YELLOW}Multi-GPU support disabled.${NC}"
    else
      export ENABLE_MULTI_GPU=true
      echo -e "${GREEN}Multi-GPU support enabled.${NC}"
    fi
  else
    export GPU_ENABLED=false
    export USE_TENSORRT=false
    export ENABLE_MULTI_GPU=false
    echo -e "${YELLOW}Running without GPU acceleration.${NC}"
  fi
}

# Main function
main() {
  print_header "SAP HANA Cloud LangChain Integration - NGC Blueprint Launcher"
  
  # Check prerequisites
  check_docker
  
  # Configure settings
  prompt_credentials
  configure_gpu
  
  print_header "Starting the NGC Blueprint"
  
  # Start containers
  if [ "$USE_STANDALONE_COMPOSE" == "1" ]; then
    docker-compose -f ngc-blueprint.yml up -d
  else
    docker compose -f ngc-blueprint.yml up -d
  fi
  
  # Check if containers started successfully
  sleep 5
  if [ "$USE_STANDALONE_COMPOSE" == "1" ]; then
    CONTAINER_STATUS=$(docker-compose -f ngc-blueprint.yml ps | grep "Up")
  else
    CONTAINER_STATUS=$(docker compose -f ngc-blueprint.yml ps | grep "Up")
  fi
  
  if [ -n "$CONTAINER_STATUS" ]; then
    print_header "Success!"
    echo -e "${GREEN}The SAP HANA Cloud LangChain Integration is now running.${NC}"
    echo -e "\nAPI is available at:          ${BLUE}http://localhost:8000${NC}"
    echo -e "API Documentation:            ${BLUE}http://localhost:8000/docs${NC}"
    echo -e "Frontend (if enabled):        ${BLUE}http://localhost:3000${NC}"
    
    if [ "$TEST_MODE" == "true" ]; then
      echo -e "\n${YELLOW}Running in TEST MODE without real SAP HANA Cloud connection.${NC}"
    fi
    
    echo -e "\nTo stop the services:"
    if [ "$USE_STANDALONE_COMPOSE" == "1" ]; then
      echo -e "  docker-compose -f ngc-blueprint.yml down"
    else
      echo -e "  docker compose -f ngc-blueprint.yml down"
    fi
  else
    print_header "Error"
    echo -e "${RED}Failed to start containers. Please check the logs:${NC}"
    if [ "$USE_STANDALONE_COMPOSE" == "1" ]; then
      echo -e "  docker-compose -f ngc-blueprint.yml logs"
    else
      echo -e "  docker compose -f ngc-blueprint.yml logs"
    fi
  fi
}

# Run main function
main