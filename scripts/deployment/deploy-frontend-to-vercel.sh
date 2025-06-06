#!/bin/bash
set -e

# Deploy frontend to Vercel
# This script deploys the frontend to Vercel

echo "Deploying frontend to Vercel"

# Check if VERCEL_TOKEN is set
if [ -z "$VERCEL_TOKEN" ]; then
    echo "Error: VERCEL_TOKEN is not set"
    exit 1
fi

# Check if VERCEL_ORG_ID is set
if [ -z "$VERCEL_ORG_ID" ]; then
    echo "Error: VERCEL_ORG_ID is not set"
    exit 1
fi

# Check if VERCEL_PROJECT_ID is set
if [ -z "$VERCEL_PROJECT_ID" ]; then
    echo "Error: VERCEL_PROJECT_ID is not set"
    exit 1
fi

# Install Vercel CLI
echo "Installing Vercel CLI"
npm install -g vercel

# Navigate to frontend directory
cd frontend

# Get the backend URL
BACKEND_URL=""
if [ -f "../deployment_url.txt" ]; then
    BACKEND_URL=$(cat ../deployment_url.txt)
    echo "Using backend URL from deployment: $BACKEND_URL"
else
    # Check if set in environment
    if [ -n "$BACKEND_URL" ]; then
        echo "Using backend URL from environment: $BACKEND_URL"
    else
        echo "Warning: No backend URL found. Please set BACKEND_URL environment variable."
        BACKEND_URL="http://localhost:8000"  # Default fallback
    fi
fi

# Update environment variables
echo "NEXT_PUBLIC_API_URL=$BACKEND_URL" > .env.production

# Deploy to Vercel
echo "Deploying to Vercel"
vercel --token "$VERCEL_TOKEN" --yes --prod

echo "Frontend deployment to Vercel completed"