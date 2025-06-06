#!/bin/bash

# GitHub Repository Synchronization Setup Script
# This script configures your local repository to sync with a GitHub remote

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
echo -e "${BLUE}    GitHub Repository Synchronization Setup${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Function to show usage
function show_usage {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 [options]"
    echo
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  --repo-url URL     GitHub repository URL (required)"
    echo -e "  --branch NAME      Branch name (default: main)"
    echo -e "  --remote NAME      Remote name (default: github)"
    echo -e "  --token TOKEN      GitHub personal access token"
    echo -e "  --help             Show this help message"
    echo
    echo -e "${YELLOW}Example:${NC}"
    echo -e "  $0 --repo-url https://github.com/username/repo.git --branch main"
    echo
}

# Default values
REPO_URL=""
BRANCH="main"
REMOTE="github"
GITHUB_TOKEN=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --repo-url)
            REPO_URL="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --remote)
            REMOTE="$2"
            shift 2
            ;;
        --token)
            GITHUB_TOKEN="$2"
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

# Check for required parameters
if [ -z "$REPO_URL" ]; then
    echo -e "${RED}Error: Repository URL is required${NC}"
    show_usage
    exit 1
fi

# If GitHub token is provided, modify the URL to include it
if [ -n "$GITHUB_TOKEN" ]; then
    # Extract the URL without the protocol
    URL_WITHOUT_PROTOCOL=$(echo "$REPO_URL" | sed 's|https://||')
    
    # Create the URL with the token
    REPO_URL_WITH_TOKEN="https://$GITHUB_TOKEN@$URL_WITHOUT_PROTOCOL"
else
    REPO_URL_WITH_TOKEN="$REPO_URL"
    echo -e "${YELLOW}Warning: No GitHub token provided. You may encounter authentication issues.${NC}"
    echo -e "${YELLOW}Consider using the --token parameter or setting the GITHUB_TOKEN environment variable.${NC}"
fi

# Check if .git directory exists
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}No Git repository found. Initializing...${NC}"
    git init
    echo -e "${GREEN}Git repository initialized.${NC}"
fi

# Check if the remote already exists
if git remote | grep -q "$REMOTE"; then
    echo -e "${YELLOW}Remote '$REMOTE' already exists. Updating URL...${NC}"
    git remote set-url "$REMOTE" "$REPO_URL_WITH_TOKEN"
    echo -e "${GREEN}Remote URL updated.${NC}"
else
    echo -e "${YELLOW}Adding remote '$REMOTE'...${NC}"
    git remote add "$REMOTE" "$REPO_URL_WITH_TOKEN"
    echo -e "${GREEN}Remote added.${NC}"
fi

# Check if the branch exists
if git branch | grep -q "$BRANCH"; then
    echo -e "${YELLOW}Branch '$BRANCH' already exists.${NC}"
else
    echo -e "${YELLOW}Creating branch '$BRANCH'...${NC}"
    git checkout -b "$BRANCH"
    echo -e "${GREEN}Branch created.${NC}"
fi

# Set up tracking
echo -e "${YELLOW}Setting up tracking for branch '$BRANCH'...${NC}"
git branch --set-upstream-to="$REMOTE/$BRANCH" "$BRANCH" 2>/dev/null || true
echo -e "${GREEN}Tracking set up.${NC}"

# Configure Git user information if not already set
if [ -z "$(git config user.name)" ]; then
    echo -e "${YELLOW}Git user.name not set. Setting to 'NVIDIA Integration Bot'...${NC}"
    git config user.name "NVIDIA Integration Bot"
    echo -e "${GREEN}Git user.name set.${NC}"
fi

if [ -z "$(git config user.email)" ]; then
    echo -e "${YELLOW}Git user.email not set. Setting to 'nvidia-integration-bot@example.com'...${NC}"
    git config user.email "nvidia-integration-bot@example.com"
    echo -e "${GREEN}Git user.email set.${NC}"
fi

# Display final configuration
echo
echo -e "${BLUE}===========================================================${NC}"
echo -e "${GREEN}GitHub Repository Synchronization Setup Complete${NC}"
echo -e "${BLUE}===========================================================${NC}"
echo
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Repository URL: ${CYAN}$REPO_URL${NC}"
echo -e "  Remote Name: ${CYAN}$REMOTE${NC}"
echo -e "  Branch: ${CYAN}$BRANCH${NC}"
echo
echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Make your changes to the codebase"
echo -e "  2. Commit your changes with: ${CYAN}git add . && git commit -m \"Your message\"${NC}"
echo -e "  3. Push to GitHub with: ${CYAN}git push $REMOTE $BRANCH${NC}"
echo
echo -e "${YELLOW}Automatic Sync:${NC}"
echo -e "  To set up automatic synchronization, add this to your crontab:"
echo -e "  ${CYAN}*/30 * * * * cd $(pwd) && git pull $REMOTE $BRANCH && git push $REMOTE $BRANCH${NC}"
echo
echo -e "${BLUE}===========================================================${NC}"