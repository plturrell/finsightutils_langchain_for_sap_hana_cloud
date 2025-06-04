#!/bin/bash
# Script to update the backend URL in the Vercel configuration

# Exit on error
set -e

# Check if the new URL is provided
if [ -z "$1" ]; then
    echo "Error: Backend URL not provided"
    echo "Usage: $0 <backend_url>"
    echo "Example: $0 https://new-backend-url.example.com"
    exit 1
fi

# Get the new backend URL from the command line argument
NEW_BACKEND_URL=$1

# Path to the vercel.json file
VERCEL_JSON_PATH="vercel.json"

# Check if the file exists
if [ ! -f "$VERCEL_JSON_PATH" ]; then
    echo "Error: vercel.json file not found at $VERCEL_JSON_PATH"
    exit 1
fi

# Create backup of the original file
cp "$VERCEL_JSON_PATH" "${VERCEL_JSON_PATH}.bak"
echo "Created backup of original file: ${VERCEL_JSON_PATH}.bak"

# Update the T4_GPU_BACKEND_URL in the file
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|\"T4_GPU_BACKEND_URL\": \"[^\"]*\"|\"T4_GPU_BACKEND_URL\": \"$NEW_BACKEND_URL\"|g" "$VERCEL_JSON_PATH"
else
    # Linux and others
    sed -i "s|\"T4_GPU_BACKEND_URL\": \"[^\"]*\"|\"T4_GPU_BACKEND_URL\": \"$NEW_BACKEND_URL\"|g" "$VERCEL_JSON_PATH"
fi

echo "Updated T4_GPU_BACKEND_URL to: $NEW_BACKEND_URL"
echo "To deploy with the updated URL, run: ./scripts/deploy_to_vercel.sh"