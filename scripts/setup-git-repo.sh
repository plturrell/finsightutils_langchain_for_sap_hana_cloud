#!/bin/bash
set -e

# Setup GitHub repository for the SAP HANA Cloud LangChain Integration
# This script will help you create a new GitHub repository and push the code

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Setup GitHub Repository for SAP HANA Cloud LangChain Integration${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo -e "${RED}Error: Git is not installed. Please install Git and try again.${NC}"
    exit 1
fi

# Check if GitHub CLI is installed
if command -v gh &> /dev/null; then
    USE_GH_CLI=true
    echo -e "${GREEN}GitHub CLI detected. We'll use it for repository creation.${NC}"
else
    USE_GH_CLI=false
    echo -e "${YELLOW}GitHub CLI not detected. You'll need to create the repository manually.${NC}"
fi

# Ask for GitHub username and repository name
read -p "Enter your GitHub username: " GITHUB_USERNAME
read -p "Enter repository name [langchain-integration-for-sap-hana-cloud]: " REPO_NAME
REPO_NAME=${REPO_NAME:-langchain-integration-for-sap-hana-cloud}

# Ask if repository should be private
read -p "Should the repository be private? (y/n) [n]: " PRIVATE_REPO
PRIVATE_REPO=${PRIVATE_REPO:-n}
if [[ $PRIVATE_REPO =~ ^[Yy]$ ]]; then
    PRIVATE="--private"
else
    PRIVATE="--public"
fi

# Initialize Git repository if not already initialized
if [ ! -d ".git" ]; then
    echo -e "${BLUE}Initializing Git repository...${NC}"
    git init
fi

# Create GitHub repository
if [ "$USE_GH_CLI" = true ]; then
    # Check if user is logged in to GitHub CLI
    if ! gh auth status &> /dev/null; then
        echo -e "${YELLOW}You need to log in to GitHub CLI${NC}"
        gh auth login
    fi
    
    echo -e "${BLUE}Creating GitHub repository...${NC}"
    if [ "$PRIVATE_REPO" = "y" ]; then
        gh repo create "$REPO_NAME" --private --source=. --remote=origin
    else
        gh repo create "$REPO_NAME" --public --source=. --remote=origin
    fi
else
    echo -e "${YELLOW}Please create a GitHub repository manually:${NC}"
    echo -e "1. Go to https://github.com/new"
    echo -e "2. Repository name: $REPO_NAME"
    echo -e "3. Set visibility (private or public)"
    echo -e "4. Click 'Create repository'"
    echo -e "5. Don't initialize with README, .gitignore, or license"
    
    read -p "Press Enter when you've created the repository..."
    
    echo -e "${BLUE}Setting up remote origin...${NC}"
    git remote add origin "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
fi

# Make sure all files are added
echo -e "${BLUE}Adding files to Git repository...${NC}"
git add .

# Commit changes
echo -e "${BLUE}Committing changes...${NC}"
git commit -m "Initial commit of SAP HANA Cloud LangChain Integration"

# Push to GitHub
echo -e "${BLUE}Pushing to GitHub...${NC}"
git push -u origin main || git push -u origin master

echo -e "${GREEN}Successfully set up GitHub repository for SAP HANA Cloud LangChain Integration!${NC}"
echo -e "${BLUE}Repository URL: https://github.com/$GITHUB_USERNAME/$REPO_NAME${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Deploy the backend API using Docker: docker-compose -f docker-compose.api.yml up -d"
echo -e "2. Deploy the frontend on Vercel as described in docs/deployment_guide.md"
echo -e "3. Configure the frontend to connect to your backend API"
echo -e ""
echo -e "${GREEN}Happy coding!${NC}"