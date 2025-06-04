#!/bin/bash

# Simple build script for Vercel deployment
echo "Starting Vercel build for frontend..."

# Ensure CI=false to prevent build failing on warnings
export CI=false

# Run the React build script
npm run build

echo "Frontend build completed."