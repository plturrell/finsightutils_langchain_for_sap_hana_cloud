#!/bin/bash
# Script to trigger the GitHub Actions workflow for Docker Build Cloud

set -e

# Constants
REPO_OWNER="finsightintelligence"
REPO_NAME="langchain-integration-for-sap-hana-cloud"
WORKFLOW_ID="docker-build-push.yml"

# Default tag
TAG=${1:-"cpu-secure"}

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI is not installed. Please install it first:"
    echo "https://cli.github.com/manual/installation"
    exit 1
fi

# Check if logged in to GitHub
if ! gh auth status &> /dev/null; then
    echo "You are not logged in to GitHub. Please run 'gh auth login' first."
    exit 1
fi

echo "Triggering GitHub Actions workflow: $WORKFLOW_ID"
echo "With tag: $TAG"

# Trigger the workflow
gh workflow run "$WORKFLOW_ID" -R "$REPO_OWNER/$REPO_NAME" -f tag="$TAG"

echo "Workflow triggered successfully!"
echo "Check status at: https://github.com/$REPO_OWNER/$REPO_NAME/actions"