#!/bin/bash

# Script to deploy a minimal test to Vercel to diagnose function invocation issues
# This script focuses on deploying only the minimal_test.py file to isolate issues

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Minimal Test Deployment for Vercel Function Debugging${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}Vercel CLI not found. Installing...${NC}"
    npm install -g vercel
fi

# Create a temporary directory for the minimal test
TEMP_DIR=$(mktemp -d)
echo -e "${BLUE}Created temporary directory:${NC} $TEMP_DIR"

# Create minimal vercel.json
echo -e "${BLUE}Creating minimal vercel.json...${NC}"
cat > "$TEMP_DIR/vercel.json" << EOF
{
  "version": 2,
  "builds": [
    {
      "src": "api/minimal_test.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/minimal_test.py"
    }
  ],
  "env": {
    "ENVIRONMENT": "development",
    "LOG_LEVEL": "DEBUG"
  },
  "public": true
}
EOF

# Create api directory
mkdir -p "$TEMP_DIR/api"

# Create minimal requirements.txt
echo -e "${BLUE}Creating minimal requirements.txt...${NC}"
cat > "$TEMP_DIR/requirements.txt" << EOF
fastapi>=0.100.0,<1.0.0
uvicorn>=0.22.0,<1.0.0
EOF

# Copy the minimal test file
echo -e "${BLUE}Copying minimal_test.py...${NC}"
cp "$(pwd)/api/minimal_test.py" "$TEMP_DIR/api/"

# Display the files
echo -e "${BLUE}Files ready for deployment:${NC}"
ls -la "$TEMP_DIR"
ls -la "$TEMP_DIR/api"

# Change to the temporary directory
cd "$TEMP_DIR"

# Deploy to Vercel
echo -e "${BLUE}Deploying minimal test to Vercel...${NC}"
DEPLOY_OUTPUT=$(vercel --prod --yes)
DEPLOYMENT_URL=$(echo "$DEPLOY_OUTPUT" | grep -o 'https://[^ ]*' | head -1)

# Check if deployment was successful
if [ -z "$DEPLOYMENT_URL" ]; then
    echo -e "${RED}Failed to extract deployment URL. Running vercel again in prod mode...${NC}"
    vercel --prod --yes
    DEPLOYMENT_URL=$(vercel --prod)
fi

# Save deployment URL to a file in the original directory
echo "$DEPLOYMENT_URL" > "$(pwd)/minimal_deployment_url.txt"
cp "$(pwd)/minimal_deployment_url.txt" "$(pwd)/../deployment_url.txt"

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${BLUE}Minimal Test Deployment URL:${NC} $DEPLOYMENT_URL"

# Test the deployment
echo -e "${BLUE}Testing the deployment...${NC}"
echo -e "${YELLOW}Waiting 5 seconds for deployment to stabilize...${NC}"
sleep 5

# Function to test an endpoint
test_endpoint() {
    local url=$1
    local description=$2
    local bypass_header=$3
    
    echo -e "\n${BLUE}Testing $description:${NC} $url"
    
    if [ -n "$bypass_header" ]; then
        echo "Using protection bypass: $bypass_header"
        curl -s -H "x-vercel-protection-bypass: $bypass_header" "$url"
    else
        curl -s "$url"
    fi
    
    echo # Add a newline after the response
}

# Test the endpoints
BYPASS_TOKEN="jyuthfgjugjuiytioytkkilytkgjhkui"

echo -e "\n${GREEN}Testing endpoints:${NC}"
test_endpoint "$DEPLOYMENT_URL" "Root endpoint" "$BYPASS_TOKEN"
test_endpoint "$DEPLOYMENT_URL/environment" "Environment endpoint" "$BYPASS_TOKEN"
test_endpoint "$DEPLOYMENT_URL/sys-path" "Sys path endpoint" "$BYPASS_TOKEN"

# Return to original directory
cd - > /dev/null

echo -e "\n${GREEN}Minimal test deployment and testing complete!${NC}"
echo -e "${BLUE}If the minimal test endpoints work, the issue is likely with the dependencies or code in your main project.${NC}"
echo -e "${BLUE}If the minimal test endpoints fail, the issue is likely with Vercel's function execution environment.${NC}"
echo -e "\n${BLUE}To continue testing, run:${NC}"
echo -e "${YELLOW}./scripts/test_minimal_api.sh -u $DEPLOYMENT_URL${NC}"

# Clean up temporary directory
echo -e "${BLUE}Cleaning up temporary directory...${NC}"
rm -rf "$TEMP_DIR"
echo -e "${GREEN}Done!${NC}"