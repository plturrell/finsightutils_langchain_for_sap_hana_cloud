#!/bin/bash
# Script to update the environment setting in vercel.json and redeploy

# Exit on error
set -e

# Default value
NEW_ENVIRONMENT="development"

# Parse command line arguments
if [ $# -ge 1 ]; then
    NEW_ENVIRONMENT=$1
fi

# Path to vercel.json
VERCEL_JSON_PATH="vercel.json"

# Check if the file exists
if [ ! -f "$VERCEL_JSON_PATH" ]; then
    echo "Error: vercel.json file not found at $VERCEL_JSON_PATH"
    exit 1
fi

# Create backup of the original file
cp "$VERCEL_JSON_PATH" "${VERCEL_JSON_PATH}.bak"
echo "Created backup of original file: ${VERCEL_JSON_PATH}.bak"

# Update the ENVIRONMENT setting in the file
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s|\"ENVIRONMENT\": \"[^\"]*\"|\"ENVIRONMENT\": \"$NEW_ENVIRONMENT\"|g" "$VERCEL_JSON_PATH"
else
    # Linux and others
    sed -i "s|\"ENVIRONMENT\": \"[^\"]*\"|\"ENVIRONMENT\": \"$NEW_ENVIRONMENT\"|g" "$VERCEL_JSON_PATH"
fi

echo "Updated ENVIRONMENT to: $NEW_ENVIRONMENT"

# Ask if user wants to deploy now
read -p "Do you want to deploy to Vercel now? (y/n): " deploy_now
if [[ "$deploy_now" == "y" ]]; then
    echo "Deploying to Vercel..."
    ./scripts/deploy_vercel_no_token.sh
else
    echo "To deploy manually, run: ./scripts/deploy_vercel_no_token.sh"
fi