#!/bin/bash

# GitHub Repository Synchronization Script
# This script synchronizes your local repository with a GitHub remote

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
echo -e "${BLUE}    GitHub Repository Synchronization${NC}"
echo -e "${BLUE}===========================================================${NC}"

# Function to show usage
function show_usage {
    echo -e "${YELLOW}Usage:${NC}"
    echo -e "  $0 [options]"
    echo
    echo -e "${YELLOW}Options:${NC}"
    echo -e "  --remote NAME      Remote name (default: github)"
    echo -e "  --branch NAME      Branch name (default: main)"
    echo -e "  --message MSG      Commit message (default: \"Auto-sync: \$(date)\")"
    echo -e "  --push-only        Only push changes, don't pull"
    echo -e "  --pull-only        Only pull changes, don't push"
    echo -e "  --force            Force push (use with caution)"
    echo -e "  --help             Show this help message"
    echo
    echo -e "${YELLOW}Example:${NC}"
    echo -e "  $0 --remote github --branch main --message \"Update documentation\""
    echo
}

# Default values
REMOTE="github"
BRANCH="main"
COMMIT_MESSAGE="Auto-sync: $(date)"
PUSH_ONLY=false
PULL_ONLY=false
FORCE_PUSH=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --remote)
            REMOTE="$2"
            shift 2
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        --message)
            COMMIT_MESSAGE="$2"
            shift 2
            ;;
        --push-only)
            PUSH_ONLY=true
            shift
            ;;
        --pull-only)
            PULL_ONLY=true
            shift
            ;;
        --force)
            FORCE_PUSH=true
            shift
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

# Check if both push-only and pull-only are specified
if [ "$PUSH_ONLY" = true ] && [ "$PULL_ONLY" = true ]; then
    echo -e "${RED}Error: Cannot specify both --push-only and --pull-only${NC}"
    show_usage
    exit 1
fi

# Check if .git directory exists
if [ ! -d ".git" ]; then
    echo -e "${RED}Error: No Git repository found in the current directory${NC}"
    echo -e "${YELLOW}Run setup_github_sync.sh first to initialize the repository${NC}"
    exit 1
fi

# Check if remote exists
if ! git remote | grep -q "$REMOTE"; then
    echo -e "${RED}Error: Remote '$REMOTE' does not exist${NC}"
    echo -e "${YELLOW}Run setup_github_sync.sh first to set up the remote${NC}"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "$BRANCH" ]; then
    echo -e "${YELLOW}Current branch is '$CURRENT_BRANCH', switching to '$BRANCH'...${NC}"
    git checkout "$BRANCH"
    echo -e "${GREEN}Switched to branch '$BRANCH'${NC}"
fi

# Pull changes if not in push-only mode
if [ "$PUSH_ONLY" = false ]; then
    echo -e "${YELLOW}Pulling changes from '$REMOTE/$BRANCH'...${NC}"
    git pull "$REMOTE" "$BRANCH" || {
        echo -e "${RED}Error: Failed to pull from '$REMOTE/$BRANCH'${NC}"
        echo -e "${YELLOW}There may be merge conflicts. Resolve them manually and try again.${NC}"
        exit 1
    }
    echo -e "${GREEN}Pull completed successfully${NC}"
fi

# Check for changes
if [ "$(git status --porcelain | wc -l)" -gt 0 ]; then
    echo -e "${YELLOW}Changes detected. Committing...${NC}"
    git add .
    git commit -m "$COMMIT_MESSAGE" || {
        echo -e "${YELLOW}Nothing to commit or commit failed${NC}"
    }
    echo -e "${GREEN}Commit completed${NC}"
else
    echo -e "${YELLOW}No changes to commit${NC}"
fi

# Push changes if not in pull-only mode
if [ "$PULL_ONLY" = false ]; then
    echo -e "${YELLOW}Pushing changes to '$REMOTE/$BRANCH'...${NC}"
    if [ "$FORCE_PUSH" = true ]; then
        echo -e "${RED}Warning: Force pushing to '$REMOTE/$BRANCH'${NC}"
        git push "$REMOTE" "$BRANCH" --force || {
            echo -e "${RED}Error: Failed to push to '$REMOTE/$BRANCH'${NC}"
            exit 1
        }
    else
        git push "$REMOTE" "$BRANCH" || {
            echo -e "${RED}Error: Failed to push to '$REMOTE/$BRANCH'${NC}"
            echo -e "${YELLOW}If there are upstream changes, pull first or use --force (with caution)${NC}"
            exit 1
        }
    fi
    echo -e "${GREEN}Push completed successfully${NC}"
fi

# Display final status
echo
echo -e "${BLUE}===========================================================${NC}"
echo -e "${GREEN}GitHub Repository Synchronization Complete${NC}"
echo -e "${BLUE}===========================================================${NC}"
echo
echo -e "${YELLOW}Summary:${NC}"
echo -e "  Remote: ${CYAN}$REMOTE${NC}"
echo -e "  Branch: ${CYAN}$BRANCH${NC}"
if [ "$PUSH_ONLY" = false ] && [ "$PULL_ONLY" = false ]; then
    echo -e "  Action: ${CYAN}Pull and Push${NC}"
elif [ "$PUSH_ONLY" = true ]; then
    echo -e "  Action: ${CYAN}Push Only${NC}"
else
    echo -e "  Action: ${CYAN}Pull Only${NC}"
fi
if [ "$FORCE_PUSH" = true ]; then
    echo -e "  Force Push: ${CYAN}Yes${NC}"
fi
echo
echo -e "${YELLOW}Current Status:${NC}"
git status
echo
echo -e "${BLUE}===========================================================${NC}"