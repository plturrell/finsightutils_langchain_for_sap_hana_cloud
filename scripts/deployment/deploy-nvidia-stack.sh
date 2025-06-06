#!/bin/bash

# Master Deployment Script for SAP HANA Cloud LangChain Integration with NVIDIA GPU
# This script orchestrates the entire deployment process:
# 1. GitHub repository synchronization
# 2. NVIDIA T4 GPU backend setup and deployment
# 3. Vercel frontend deployment with TensorRT optimization

# Exit on error
set -e

# ANSI color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Print script header
echo -e "${BLUE}===========================================================${NC}"
echo -e "${BLUE}    NVIDIA T4 GPU Enhanced Deployment for${NC}"
echo -e "${BLUE}    SAP HANA Cloud LangChain Integration${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Display usage information
function show_usage {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 [options]"
    echo
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  --skip-github       Skip GitHub synchronization step"
    echo -e "  --skip-backend      Skip NVIDIA T4 backend deployment"
    echo -e "  --skip-frontend     Skip Vercel frontend deployment"
    echo -e "  --backend-url URL   Specify backend URL (default: auto-detected)"
    echo -e "  --help              Show this help message"
    echo
}

# Parse command line arguments
SKIP_GITHUB=false
SKIP_BACKEND=false
SKIP_FRONTEND=false
BACKEND_URL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-github)
            SKIP_GITHUB=true
            shift
            ;;
        --skip-backend)
            SKIP_BACKEND=true
            shift
            ;;
        --skip-frontend)
            SKIP_FRONTEND=true
            shift
            ;;
        --backend-url)
            BACKEND_URL="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Check for required environment variables
function check_env_vars {
    local missing=false
    
    # Common requirements
    if [ -z "$GITHUB_TOKEN" ] && [ "$SKIP_GITHUB" = false ]; then
        echo -e "${RED}Error: GITHUB_TOKEN is not set${NC}"
        missing=true
    fi
    
    # Backend requirements
    if [ "$SKIP_BACKEND" = false ]; then
        if [ -z "$BREV_API_KEY" ]; then
            echo -e "${RED}Error: BREV_API_KEY is not set (required for backend deployment)${NC}"
            missing=true
        fi
    fi
    
    # Frontend requirements
    if [ "$SKIP_FRONTEND" = false ]; then
        if [ -z "$VERCEL_TOKEN" ]; then
            echo -e "${RED}Error: VERCEL_TOKEN is not set (required for frontend deployment)${NC}"
            missing=true
        fi
    fi
    
    if [ "$missing" = true ]; then
        echo
        echo -e "${YELLOW}Please set the required environment variables and try again:${NC}"
        echo -e "  ${CYAN}export GITHUB_TOKEN=your_github_token${NC}"
        echo -e "  ${CYAN}export BREV_API_KEY=your_brev_api_key${NC}"
        echo -e "  ${CYAN}export VERCEL_TOKEN=your_vercel_token${NC}"
        echo
        exit 1
    fi
}

# Check for required commands
function check_commands {
    local missing=false
    
    # Common requirements
    if ! command -v jq &> /dev/null; then
        echo -e "${RED}Error: jq is not installed${NC}"
        missing=true
    fi
    
    # GitHub requirements
    if [ "$SKIP_GITHUB" = false ]; then
        if ! command -v git &> /dev/null; then
            echo -e "${RED}Error: git is not installed${NC}"
            missing=true
        fi
    fi
    
    # Frontend requirements
    if [ "$SKIP_FRONTEND" = false ]; then
        if ! command -v npm &> /dev/null; then
            echo -e "${RED}Error: npm is not installed${NC}"
            missing=true
        fi
    fi
    
    if [ "$missing" = true ]; then
        echo
        echo -e "${YELLOW}Please install the required commands and try again.${NC}"
        echo
        exit 1
    fi
}

# Get user confirmation before proceeding
function get_confirmation {
    echo
    echo -e "${YELLOW}Deployment Plan:${NC}"
    echo -e "  GitHub Sync: $([ "$SKIP_GITHUB" = true ] && echo "${RED}SKIP${NC}" || echo "${GREEN}YES${NC}")"
    echo -e "  NVIDIA T4 Backend: $([ "$SKIP_BACKEND" = true ] && echo "${RED}SKIP${NC}" || echo "${GREEN}YES${NC}")"
    echo -e "  Vercel Frontend: $([ "$SKIP_FRONTEND" = true ] && echo "${RED}SKIP${NC}" || echo "${GREEN}YES${NC}")"
    
    if [ -n "$BACKEND_URL" ]; then
        echo -e "  Backend URL: ${CYAN}$BACKEND_URL${NC}"
    fi
    
    echo
    read -p "Do you want to proceed with deployment? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}Deployment cancelled by user${NC}"
        exit 0
    fi
    
    echo
}

# Step 1: GitHub repository synchronization
function github_sync {
    echo -e "${BLUE}===========================================================${NC}"
    echo -e "${BLUE}Step 1: GitHub Repository Synchronization${NC}"
    echo -e "${BLUE}===========================================================${NC}"
    
    # Run the GitHub repository setup script
    echo -e "${YELLOW}Running GitHub repository setup script...${NC}"
    bash scripts/development/setup-github-repo.sh --repo-url "https://${GITHUB_TOKEN}@github.com/USERNAME/enhanced-sap-hana-langchain.git"
    
    echo -e "${GREEN}GitHub synchronization complete!${NC}"
    echo
}

# Step 2: NVIDIA T4 GPU backend setup and deployment
function deploy_backend {
    echo -e "${BLUE}===========================================================${NC}"
    echo -e "${BLUE}Step 2: NVIDIA T4 GPU Backend Setup and Deployment${NC}"
    echo -e "${BLUE}===========================================================${NC}"
    
    # Check if deployment script exists
    if [ ! -f "scripts/deployment/deploy-to-nvidia-t4.sh" ]; then
        echo -e "${RED}Error: Backend deployment script not found${NC}"
        echo -e "Expected location: scripts/deployment/deploy-to-nvidia-t4.sh"
        exit 1
    fi
    
    # Run deployment script
    echo -e "${YELLOW}Deploying NVIDIA T4 GPU backend...${NC}"
    bash scripts/deployment/deploy-to-nvidia-t4.sh
    
    # Get the backend URL from the deployment file
    if [ -f "deployment_url.txt" ]; then
        BACKEND_URL=$(cat deployment_url.txt)
        echo -e "${GREEN}Backend deployed successfully at: ${CYAN}$BACKEND_URL${NC}"
    else
        echo -e "${RED}Error: Backend deployment URL not found${NC}"
        echo -e "Please specify the backend URL manually with --backend-url"
        exit 1
    fi
    
    echo -e "${GREEN}NVIDIA T4 GPU backend deployment complete!${NC}"
    echo
}

# Step 3: Vercel frontend deployment with TensorRT optimization
function deploy_frontend {
    echo -e "${BLUE}===========================================================${NC}"
    echo -e "${BLUE}Step 3: Vercel Frontend Deployment with TensorRT Optimization${NC}"
    echo -e "${BLUE}===========================================================${NC}"
    
    # Check if deployment script exists
    if [ ! -f "scripts/deployment/deploy-nvidia-vercel.sh" ]; then
        echo -e "${RED}Error: Frontend deployment script not found${NC}"
        echo -e "Expected location: scripts/deployment/deploy-nvidia-vercel.sh"
        exit 1
    fi
    
    # Check if backend URL is set
    if [ -z "$BACKEND_URL" ]; then
        echo -e "${RED}Error: Backend URL is not set${NC}"
        echo -e "Please specify the backend URL with --backend-url or deploy the backend first"
        exit 1
    fi
    
    # Run deployment script
    echo -e "${YELLOW}Deploying Vercel frontend with TensorRT optimization...${NC}"
    BACKEND_URL="$BACKEND_URL" bash scripts/deployment/deploy-nvidia-vercel.sh
    
    echo -e "${GREEN}Vercel frontend deployment complete!${NC}"
    echo
}

# Main deployment flow
function main {
    # Check requirements
    check_env_vars
    check_commands
    
    # Get user confirmation
    get_confirmation
    
    # Step 1: GitHub synchronization
    if [ "$SKIP_GITHUB" = false ]; then
        github_sync
    else
        echo -e "${YELLOW}Skipping GitHub synchronization as requested${NC}"
    fi
    
    # Step 2: NVIDIA T4 GPU backend deployment
    if [ "$SKIP_BACKEND" = false ]; then
        deploy_backend
    else
        echo -e "${YELLOW}Skipping NVIDIA T4 backend deployment as requested${NC}"
    fi
    
    # Step 3: Vercel frontend deployment
    if [ "$SKIP_FRONTEND" = false ]; then
        deploy_frontend
    else
        echo -e "${YELLOW}Skipping Vercel frontend deployment as requested${NC}"
    fi
    
    # Final success message
    echo -e "${BLUE}===========================================================${NC}"
    echo -e "${GREEN}Deployment Complete!${NC}"
    echo -e "${BLUE}===========================================================${NC}"
    
    # Get frontend URL
    local frontend_url=""
    if [ -f "deployment_url.txt" ]; then
        frontend_url=$(cat deployment_url.txt)
    fi
    
    echo -e "${YELLOW}Deployment Summary:${NC}"
    
    if [ -n "$BACKEND_URL" ]; then
        echo -e "  Backend URL: ${CYAN}$BACKEND_URL${NC}"
    fi
    
    if [ -n "$frontend_url" ]; then
        echo -e "  Frontend URL: ${CYAN}$frontend_url${NC}"
    fi
    
    echo
    echo -e "${YELLOW}Thank you for using the NVIDIA Enhanced SAP HANA Cloud LangChain Integration!${NC}"
    echo -e "${YELLOW}For more information, please visit: https://github.com/USERNAME/enhanced-sap-hana-langchain${NC}"
    echo
}

# Run the main function
main