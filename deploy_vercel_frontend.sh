#!/bin/bash
# Deploy the frontend to Vercel

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Deploying SAP HANA Cloud LangChain Frontend to Vercel${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}Vercel CLI not found. Installing...${NC}"
    npm install -g vercel
fi

# Ask for backend URL if not provided
if [ "$#" -lt 1 ]; then
    echo -e "${BLUE}Enter your T4 GPU Backend URL (e.g., http://your-ip-address:8000):${NC}"
    read -p "> " BACKEND_URL
else
    BACKEND_URL=$1
fi

# Validate backend URL
if [ -z "$BACKEND_URL" ]; then
    echo -e "${RED}Error: Backend URL is required${NC}"
    exit 1
fi

echo -e "${BLUE}Using backend URL:${NC} $BACKEND_URL"

# Check if we can reach the backend
echo -e "${BLUE}Checking if backend is reachable...${NC}"
if curl -s --head --fail "$BACKEND_URL/health" > /dev/null || curl -s --head --fail "$BACKEND_URL/api/health" > /dev/null; then
    echo -e "${GREEN}Backend is reachable!${NC}"
else
    echo -e "${YELLOW}Warning: Backend does not seem to be reachable. Deployment will continue, but frontend may not work correctly.${NC}"
    read -p "Continue anyway? (y/n): " continue_prompt
    if [[ ! "$continue_prompt" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Navigate to the frontend directory
cd frontend

# Create .env file for the frontend
echo -e "${BLUE}Creating frontend configuration...${NC}"
cat > .env << EOF
REACT_APP_API_URL=$BACKEND_URL
REACT_APP_ENABLE_AUTH=false
EOF

# Create Vercel configuration
echo -e "${BLUE}Creating Vercel configuration...${NC}"
cat > vercel.json << EOF
{
  "version": 2,
  "builds": [
    {
      "src": "package.json",
      "use": "@vercel/static-build",
      "config": {
        "distDir": "build"
      }
    }
  ],
  "routes": [
    {
      "src": "/static/(.*)",
      "dest": "/static/$1"
    },
    {
      "src": "/favicon.ico",
      "dest": "/favicon.ico"
    },
    {
      "src": "/logo192.png",
      "dest": "/logo192.png"
    },
    {
      "src": "/logo512.png",
      "dest": "/logo512.png"
    },
    {
      "src": "/manifest.json",
      "dest": "/manifest.json"
    },
    {
      "src": "/robots.txt",
      "dest": "/robots.txt"
    },
    {
      "src": "/(.*)",
      "dest": "/index.html"
    }
  ],
  "env": {
    "REACT_APP_API_URL": "$BACKEND_URL"
  }
}
EOF

# Add build script to package.json if needed
if ! grep -q '"build":' package.json; then
    echo -e "${YELLOW}Adding build script to package.json...${NC}"
    sed -i.bak 's/"scripts": {/"scripts": {\n    "build": "react-scripts build",/g' package.json
    rm -f package.json.bak
fi

# Deploy to Vercel
echo -e "${BLUE}Deploying to Vercel...${NC}"
vercel --prod

echo -e "${GREEN}==========================================${NC}"
echo -e "${GREEN}Frontend deployment completed!${NC}"
echo -e "${GREEN}==========================================${NC}"
echo -e "${BLUE}Your frontend should now be accessible at the URL provided by Vercel.${NC}"
echo -e "${BLUE}It is configured to connect to your T4 GPU backend at:${NC} $BACKEND_URL"
echo -e "\n${YELLOW}NOTE:${NC} If your T4 GPU backend is not publicly accessible, you'll need to:"
echo -e "1. Make sure your T4 server is accessible from the internet"
echo -e "2. Configure any necessary firewall rules to allow access to port 8000"
echo -e "3. Consider setting up HTTPS for secure communication"