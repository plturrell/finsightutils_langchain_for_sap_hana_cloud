#!/bin/bash

# Test script for the minimal test API
# This script is specifically designed to diagnose Vercel function invocation issues

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BYPASS_TOKEN="jyuthfgjugjuiytioytkkilytkgjhkui"
URL=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -u|--url)
      URL="$2"
      shift 2
      ;;
    -b|--bypass)
      BYPASS_TOKEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -u|--url <deployment-url> [-b|--bypass <bypass-token>]"
      exit 1
      ;;
  esac
done

# Check if URL is provided
if [ -z "$URL" ]; then
    if [ -f "deployment_url.txt" ]; then
        URL=$(cat deployment_url.txt)
        echo -e "${YELLOW}Using URL from deployment_url.txt: ${NC}$URL"
    else
        echo -e "${RED}Error: No URL provided and deployment_url.txt not found${NC}"
        echo "Usage: $0 -u|--url <deployment-url> [-b|--bypass <bypass-token>]"
        exit 1
    fi
fi

# Function to test an endpoint
test_endpoint() {
    local url=$1
    local description=$2
    local use_bypass=$3
    
    echo -e "\n${BLUE}Testing $description:${NC} $url"
    echo -e "${YELLOW}--------------------------------------------------------------------------${NC}"
    
    # Set up curl command with options
    curl_cmd="curl -s"
    if [ "$use_bypass" = true ]; then
        curl_cmd="$curl_cmd -H \"x-vercel-protection-bypass: $BYPASS_TOKEN\""
    fi
    curl_cmd="$curl_cmd \"$url\""
    
    # Execute the curl command
    echo -e "${YELLOW}Command:${NC} $curl_cmd"
    echo -e "${YELLOW}Response:${NC}"
    
    if [ "$use_bypass" = true ]; then
        curl -s -H "x-vercel-protection-bypass: $BYPASS_TOKEN" "$url" | jq . || echo "Failed to parse response as JSON"
    else
        curl -s "$url" | jq . || echo "Failed to parse response as JSON"
    fi
    
    echo -e "${YELLOW}--------------------------------------------------------------------------${NC}"
}

# Print header
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Testing Minimal API to diagnose Vercel function issues${NC}"
echo -e "${BLUE}=========================================================${NC}"

echo -e "${BLUE}Deployment URL:${NC} $URL"
echo -e "${BLUE}Using bypass token:${NC} $BYPASS_TOKEN"

# Test the main endpoint with and without bypass
test_endpoint "${URL}/minimal-test" "Minimal Test Root (No Bypass)" false
test_endpoint "${URL}/minimal-test" "Minimal Test Root (With Bypass)" true

# Test the environment endpoint with bypass
test_endpoint "${URL}/minimal-test/environment" "Environment Info (With Bypass)" true

# Test the sys-path endpoint with bypass
test_endpoint "${URL}/minimal-test/sys-path" "System Path Info (With Bypass)" true

echo -e "\n${GREEN}Testing complete!${NC}"
echo -e "${BLUE}If any of these endpoints fail, it indicates a basic configuration issue with Vercel functions.${NC}"
echo -e "${BLUE}Check the following:${NC}"
echo -e "1. Python version compatibility"
echo -e "2. Missing dependencies in requirements.txt"
echo -e "3. Invalid routes in vercel.json"
echo -e "4. Syntax errors in the minimal_test.py file"
echo -e "5. Vercel function size limits (code should be minimal)"