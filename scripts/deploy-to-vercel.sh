#!/bin/bash

# Deploy to Vercel script
# This script deploys the frontend to Vercel and connects it to the T4 GPU backend

# Exit on error
set -e

# Define variables
BACKEND_URL=${BACKEND_URL:-"https://jupyter0-513syzm60.brevlab.com"}
PROJECT_ROOT=$(pwd)
VERCEL_PROJECT_NAME=${VERCEL_PROJECT_NAME:-"sap-hana-langchain-t4"}
ENVIRONMENT=${ENVIRONMENT:-"production"}
JWT_SECRET=${JWT_SECRET:-"sap-hana-langchain-t4-integration-secret-key-2025"}

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Ensure we're in the project root
cd "$PROJECT_ROOT"

# Update frontend configuration with the correct backend URL
echo "Updating frontend configuration with backend URL: $BACKEND_URL"
sed -i.bak "s|const API_BASE_URL = '.*'|const API_BASE_URL = '$BACKEND_URL'|g" frontend/index.html

# Create vercel.json if it doesn't exist
if [ ! -f "vercel.json" ]; then
    echo "Creating vercel.json configuration..."
    cat > vercel.json << EOFINNER
{
  "version": 2,
  "builds": [
    {
      "src": "api/vercel_integration.py",
      "use": "@vercel/python"
    },
    {
      "src": "frontend/**",
      "use": "@vercel/static"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "api/vercel_integration.py"
    },
    {
      "src": "/(.*)",
      "dest": "frontend/\$1"
    }
  ],
  "env": {
    "T4_GPU_BACKEND_URL": "$BACKEND_URL",
    "ENVIRONMENT": "$ENVIRONMENT",
    "JWT_SECRET": "$JWT_SECRET"
  }
}
EOFINNER
fi

# Create requirements.txt for Vercel Python if it doesn't exist
if [ ! -f "requirements.txt" ]; then
    echo "Creating requirements.txt for Vercel deployment..."
    cat > requirements.txt << EOFINNER
fastapi==0.100.0
uvicorn==0.22.0
requests==2.31.0
pyjwt==2.8.0
pydantic==2.0.3
python-multipart==0.0.6
python-dotenv==1.0.0
EOFINNER
fi

# Deploy to Vercel
echo "Deploying to Vercel..."
if [ "$ENVIRONMENT" = "production" ]; then
    # Production deployment
    vercel --prod --confirm --token "$VERCEL_TOKEN" --name "$VERCEL_PROJECT_NAME"
else
    # Preview deployment
    vercel --confirm --token "$VERCEL_TOKEN" --name "$VERCEL_PROJECT_NAME"
fi

# Get the deployment URL
DEPLOYMENT_URL=$(vercel --token "$VERCEL_TOKEN" ls "$VERCEL_PROJECT_NAME" -j | jq -r '.deployments[0].url')
echo "Deployment successful\! Your application is available at: https://$DEPLOYMENT_URL"

# Save the deployment URL to a file for reference
echo "https://$DEPLOYMENT_URL" > deployment_url.txt
echo "Deployment URL saved to deployment_url.txt"

# Print connection information
echo ""
echo "======================================"
echo "Frontend-Backend Connection Information"
echo "======================================"
echo "Frontend URL: https://$DEPLOYMENT_URL"
echo "Backend URL: $BACKEND_URL"
echo ""
echo "To test the connection, visit:"
echo "https://$DEPLOYMENT_URL"
echo ""
echo "If you need to update the backend URL, run:"
echo "BACKEND_URL=<new-url> ./scripts/deploy_to_vercel.sh"
echo "======================================="