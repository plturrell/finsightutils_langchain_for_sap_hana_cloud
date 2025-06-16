#!/bin/bash

# Prepare GitHub Commit Script
# This script organizes changes into logical commits for GitHub synchronization

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
echo -e "${BLUE}    Prepare GitHub Commit${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Check if .git directory exists
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: No Git repository found in the current directory${NC}"
    echo -e "${YELLOW}Run setup_github_sync.sh first to initialize the repository${NC}"
    exit 1
fi

# Organize files into categories
echo -e "${YELLOW}Organizing files into categories...${NC}"

# Docker-related changes
echo -e "${CYAN}Stage Docker-related changes...${NC}"
git add docker-compose*.yml
git add docker/ Dockerfile*
git add scripts/docker_*.sh
git add README_DOCKER.md

# Documentation changes
echo -e "${CYAN}Stage documentation changes...${NC}"
git add README.md
git add docs/
git add *.md
git add NVIDIA_*.md
git add NGC_*.md

# Deployment scripts
echo -e "${CYAN}Stage deployment scripts...${NC}"
git add scripts/deploy_*.sh
git add deploy_*.sh
git add scripts/setup_*.sh
git add scripts/sync_*.sh
git add scripts/verify_*.sh

# Testing and CI/CD
echo -e "${CYAN}Stage testing and CI/CD files...${NC}"
git add .github/
git add tests/
git add scripts/test_*.sh
git add scripts/test_*.py

# Backend improvements
echo -e "${CYAN}Stage backend improvements...${NC}"
git add api/
git add backend/

# Frontend improvements
echo -e "${CYAN}Stage frontend improvements...${NC}"
git add frontend/

# Infrastructure as code
echo -e "${CYAN}Stage infrastructure as code...${NC}"
git add terraform/
git add kubernetes/
git add scripts/*terraform*.sh
git add scripts/*k8s*.sh

# Benchmarks
echo -e "${CYAN}Stage benchmarks...${NC}"
git add benchmarks/
git add scripts/*_performance*.py

# Vercel integration
echo -e "${CYAN}Stage Vercel integration...${NC}"
git add vercel*.json
git add *VERCEL*.md

# Check if there are still unstaged changes
if [ -n "$(git status --porcelain | grep -v '^A')" ]; then
    echo -e "${YELLOW}There are still unstaged changes:${NC}"
    git status --porcelain | grep -v '^A'
    
    echo
    read -p "Do you want to stage all remaining changes? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add -A
        echo -e "${GREEN}All changes staged${NC}"
    else
        echo -e "${YELLOW}Remaining changes left unstaged${NC}"
    fi
fi

# Show status
echo
echo -e "${BLUE}===========================================================${NC}"
echo -e "${GREEN}Changes Staged for Commit${NC}"
echo -e "${BLUE}===========================================================${NC}"
echo
git status
echo
echo -e "${YELLOW}Suggested commit message:${NC}"
echo -e "${CYAN}Enhance SAP HANA Cloud LangChain Integration with NVIDIA GPU and Vercel support${NC}"
echo
echo -e "${YELLOW}To commit these changes:${NC}"
echo -e "${CYAN}git commit -m \"Enhance SAP HANA Cloud LangChain Integration with NVIDIA GPU and Vercel support\"${NC}"
echo
echo -e "${YELLOW}To push to GitHub:${NC}"
echo -e "${CYAN}git push github main${NC}"
echo
echo -e "${BLUE}===========================================================${NC}"