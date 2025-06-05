#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}SAP HANA Cloud LangChain Integration - Full Deployment${NC}"
echo -e "${BLUE}=========================================================${NC}"

echo -e "${YELLOW}This script will perform a complete deployment:${NC}"
echo -e "1. GitHub Remote Sync"
echo -e "2. NVIDIA Backend FastAPI Setup"
echo -e "3. Vercel Frontend Deployment"
echo -e ""

read -p "Continue with the deployment? (y/n) " continue_decision
if [[ ! "$continue_decision" =~ ^[Yy]$ ]]; then
    echo -e "${RED}Exiting.${NC}"
    exit 1
fi

# Step 1: GitHub Remote Sync
echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Step 1: GitHub Remote Sync${NC}"
echo -e "${BLUE}=========================================================${NC}"

echo -e "${YELLOW}The code has already been synced to:${NC}"
echo -e "${CYAN}https://github.com/plturrell/langchain-integration-for-sap-hana-cloud${NC}"
echo -e "${YELLOW}Branch: ${CYAN}nvidia-vercel-deployment${NC}"
echo -e ""

read -p "Would you like to configure a new GitHub repository? (y/n) " setup_github
if [[ "$setup_github" =~ ^[Yy]$ ]]; then
    # Check if GitHub CLI is installed
    if command -v gh &> /dev/null; then
        echo -e "${GREEN}GitHub CLI detected. We'll use it for repository creation.${NC}"
        
        # Check if user is logged in to GitHub CLI
        if ! gh auth status &> /dev/null; then
            echo -e "${YELLOW}You need to log in to GitHub CLI${NC}"
            gh auth login
        fi
        
        # Ask for repository name
        read -p "Enter repository name [sap-hana-langchain-nvidia]: " REPO_NAME
        REPO_NAME=${REPO_NAME:-sap-hana-langchain-nvidia}
        
        # Ask if repository should be private
        read -p "Should the repository be private? (y/n) [n]: " PRIVATE_REPO
        PRIVATE_REPO=${PRIVATE_REPO:-n}
        
        echo -e "${BLUE}Creating GitHub repository...${NC}"
        if [[ "$PRIVATE_REPO" =~ ^[Yy]$ ]]; then
            gh repo create "$REPO_NAME" --private --source=. --remote=github
        else
            gh repo create "$REPO_NAME" --public --source=. --remote=github
        fi
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}GitHub repository created successfully!${NC}"
            echo -e "${YELLOW}Repository URL: ${CYAN}https://github.com/$(gh api user | jq -r .login)/${REPO_NAME}${NC}"
        else
            echo -e "${RED}Failed to create GitHub repository.${NC}"
        fi
    else
        echo -e "${YELLOW}GitHub CLI not detected. Manual instructions:${NC}"
        echo -e "1. Create a new repository on GitHub: https://github.com/new"
        echo -e "2. Push your code to the new repository:"
        echo -e "   git remote add github https://github.com/username/repo-name.git"
        echo -e "   git push -u github nvidia-vercel-deployment"
    fi
fi

# Step 2: NVIDIA Backend FastAPI Setup
echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Step 2: NVIDIA Backend FastAPI Setup${NC}"
echo -e "${BLUE}=========================================================${NC}"

echo -e "${YELLOW}This step will set up the NVIDIA-accelerated backend.${NC}"
read -p "Continue with NVIDIA backend setup? (y/n) " setup_backend
if [[ "$setup_backend" =~ ^[Yy]$ ]]; then
    ./setup_nvidia_backend.sh
else
    echo -e "${YELLOW}Skipping NVIDIA backend setup.${NC}"
fi

# Step 3: Vercel Frontend Deployment
echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Step 3: Vercel Frontend Deployment${NC}"
echo -e "${BLUE}=========================================================${NC}"

echo -e "${YELLOW}This step will prepare and deploy the frontend to Vercel.${NC}"
read -p "Continue with Vercel frontend setup? (y/n) " setup_frontend
if [[ "$setup_frontend" =~ ^[Yy]$ ]]; then
    ./setup_vercel_frontend.sh
else
    echo -e "${YELLOW}Skipping Vercel frontend setup.${NC}"
fi

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${BLUE}=========================================================${NC}"

echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Start your NVIDIA backend: ${CYAN}./start_nvidia_backend.sh${NC}"
echo -e "2. Access your backend API: ${CYAN}http://your-server:8000/docs${NC}"
echo -e "3. Access your Vercel frontend: ${CYAN}https://your-frontend.vercel.app${NC}"
echo -e ""
echo -e "${GREEN}Congratulations on your deployment!${NC}"
echo -e "${BLUE}=========================================================${NC}"