#!/bin/bash

# Script to deploy to Vercel without requiring a token
# This script deploys the frontend to Vercel using their CLI without specifying a token
# and saves the deployment URL to a file for later use

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Deploying to Vercel...${NC}"

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}Vercel CLI not found. Installing...${NC}"
    npm install -g vercel
fi

# Deploy to Vercel
echo -e "${BLUE}Deploying to Vercel using logged-in account...${NC}"
DEPLOY_OUTPUT=$(vercel --prod --yes)
DEPLOYMENT_URL=$(echo "$DEPLOY_OUTPUT" | grep -o 'https://[^ ]*' | head -1)

# Check if deployment was successful
if [ -z "$DEPLOYMENT_URL" ]; then
    echo -e "${RED}Failed to extract deployment URL. Running vercel again in prod mode...${NC}"
    vercel --prod --yes
    DEPLOYMENT_URL=$(vercel --prod)
fi

# Save deployment URL to file for later use
echo "$DEPLOYMENT_URL" > deployment_url.txt
echo -e "${GREEN}Deployment URL saved to deployment_url.txt${NC}"

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${BLUE}Deployment URL:${NC} $DEPLOYMENT_URL"
echo -e "\n${BLUE}Key endpoints to test:${NC}"
echo -e "- Minimal Test API: ${YELLOW}$DEPLOYMENT_URL/minimal-test${NC}"
echo -e "- Debug Proxy: ${YELLOW}$DEPLOYMENT_URL/debug-proxy/proxy-health${NC}"
echo -e "- Enhanced Debug: ${YELLOW}$DEPLOYMENT_URL/enhanced-debug${NC}"
echo -e "\n${BLUE}To test the minimal API (recommended first test):${NC}"
echo -e "${YELLOW}./scripts/test_minimal_api.sh${NC}"