#!/bin/bash

# Script to deploy to Vercel without requiring a token
# This script deploys the frontend to Vercel using their CLI without specifying a token

# Exit on error
set -e

echo "Deploying to Vercel..."

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "Vercel CLI not found. Installing..."
    npm install -g vercel
fi

# Deploy to Vercel
echo "Deploying to Vercel using logged-in account..."
vercel --prod --confirm

echo "Deployment complete!"
echo "To test the debug-proxy endpoint, visit: <your-vercel-domain>/debug-proxy/proxy-health"
echo "This will show if the backend connection is working properly."