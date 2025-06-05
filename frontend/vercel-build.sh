#!/bin/bash
set -e

# Simple build script for Vercel deployment
echo "Starting Vercel build for frontend..."

# Ensure CI=false to prevent build failing on warnings
export CI=false

# Set the API URL from environment variables
if [ -n "$BACKEND_URL" ]; then
  echo "Setting API_BASE_URL to $BACKEND_URL"
  echo "VITE_API_BASE_URL=$BACKEND_URL" > .env.production
else
  echo "BACKEND_URL not set, using default API URL"
  echo "VITE_API_BASE_URL=https://sap-hana-langchain-api.vercel.app" > .env.production
fi

# Make vector visualization files available
echo "Copying vector visualization files..."
mkdir -p public
cp vector-visualization.js public/
cp vector-visualization.css public/

# Run the React build script
echo "Running build..."
npm run build

echo "Frontend build completed successfully!"