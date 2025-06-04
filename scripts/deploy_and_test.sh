#!/bin/bash
# Script to deploy to Vercel and test the API with enhanced debugging

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}SAP HANA Cloud LangChain T4 GPU Integration Deployment${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo -e "${YELLOW}Vercel CLI not found. Installing...${NC}"
    npm install -g vercel
fi

# Display current settings
echo -e "${BLUE}Current backend URL:${NC} $(grep "T4_GPU_BACKEND_URL" vercel.json | cut -d'"' -f4)"

# Ask if user wants to update the backend URL
read -p "Do you want to update the backend URL? (y/n): " update_url
if [[ "$update_url" == "y" ]]; then
    read -p "Enter new backend URL: " new_backend_url
    
    # Update the backend URL
    echo -e "${BLUE}Updating backend URL to:${NC} $new_backend_url"
    ./scripts/update_backend_url.sh "$new_backend_url"
fi

# Ask if user wants to update other settings
read -p "Do you want to update other environment settings? (y/n): " update_settings
if [[ "$update_settings" == "y" ]]; then
    # Timeout settings
    read -p "Default timeout (default: 30): " default_timeout
    if [[ ! -z "$default_timeout" ]]; then
        # Replace DEFAULT_TIMEOUT in vercel.json
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|\"DEFAULT_TIMEOUT\": \"[^\"]*\"|\"DEFAULT_TIMEOUT\": \"$default_timeout\"|g" vercel.json
        else
            # Linux and others
            sed -i "s|\"DEFAULT_TIMEOUT\": \"[^\"]*\"|\"DEFAULT_TIMEOUT\": \"$default_timeout\"|g" vercel.json
        fi
        echo -e "${GREEN}Updated DEFAULT_TIMEOUT to:${NC} $default_timeout"
    fi
    
    read -p "Health check timeout (default: 10): " health_timeout
    if [[ ! -z "$health_timeout" ]]; then
        # Replace HEALTH_CHECK_TIMEOUT in vercel.json
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|\"HEALTH_CHECK_TIMEOUT\": \"[^\"]*\"|\"HEALTH_CHECK_TIMEOUT\": \"$health_timeout\"|g" vercel.json
        else
            # Linux and others
            sed -i "s|\"HEALTH_CHECK_TIMEOUT\": \"[^\"]*\"|\"HEALTH_CHECK_TIMEOUT\": \"$health_timeout\"|g" vercel.json
        fi
        echo -e "${GREEN}Updated HEALTH_CHECK_TIMEOUT to:${NC} $health_timeout"
    fi
    
    read -p "Embedding timeout (default: 60): " embedding_timeout
    if [[ ! -z "$embedding_timeout" ]]; then
        # Replace EMBEDDING_TIMEOUT in vercel.json
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|\"EMBEDDING_TIMEOUT\": \"[^\"]*\"|\"EMBEDDING_TIMEOUT\": \"$embedding_timeout\"|g" vercel.json
        else
            # Linux and others
            sed -i "s|\"EMBEDDING_TIMEOUT\": \"[^\"]*\"|\"EMBEDDING_TIMEOUT\": \"$embedding_timeout\"|g" vercel.json
        fi
        echo -e "${GREEN}Updated EMBEDDING_TIMEOUT to:${NC} $embedding_timeout"
    fi
    
    # Log level settings
    read -p "Log level (DEBUG, INFO, WARNING, ERROR): " log_level
    if [[ ! -z "$log_level" ]]; then
        # Replace LOG_LEVEL in vercel.json
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|\"LOG_LEVEL\": \"[^\"]*\"|\"LOG_LEVEL\": \"$log_level\"|g" vercel.json
        else
            # Linux and others
            sed -i "s|\"LOG_LEVEL\": \"[^\"]*\"|\"LOG_LEVEL\": \"$log_level\"|g" vercel.json
        fi
        echo -e "${GREEN}Updated LOG_LEVEL to:${NC} $log_level"
    fi
    
    # Environment setting
    read -p "Environment (development, production): " environment
    if [[ ! -z "$environment" ]]; then
        # Replace ENVIRONMENT in vercel.json
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|\"ENVIRONMENT\": \"[^\"]*\"|\"ENVIRONMENT\": \"$environment\"|g" vercel.json
        else
            # Linux and others
            sed -i "s|\"ENVIRONMENT\": \"[^\"]*\"|\"ENVIRONMENT\": \"$environment\"|g" vercel.json
        fi
        echo -e "${GREEN}Updated ENVIRONMENT to:${NC} $environment"
    fi
fi

# Deploy to Vercel
echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Deploying to Vercel...${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Run the deployment script
vercel --prod --confirm

# Get the deployment URL
DEPLOYMENT_URL=$(vercel --prod)

echo -e "${GREEN}Deployment successful!${NC}"
echo -e "${BLUE}Deployment URL:${NC} $DEPLOYMENT_URL"

# Ask if user wants to run diagnostics
read -p "Do you want to run diagnostics tests? (y/n): " run_diagnostics
if [[ "$run_diagnostics" == "y" ]]; then
    echo -e "${BLUE}=========================================================${NC}"
    echo -e "${BLUE}Running diagnostics...${NC}"
    echo -e "${BLUE}=========================================================${NC}"
    
    # Make sure python script is executable
    chmod +x scripts/api_diagnostics.py
    
    # Run the diagnostics script
    ./scripts/api_diagnostics.py --url "$DEPLOYMENT_URL" --test-all --use-direct-proxy
fi

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}Deployment and testing complete!${NC}"
echo -e "${BLUE}=========================================================${NC}"
echo ""
echo -e "${BLUE}Testing URLs:${NC}"
echo -e "- Regular API: ${YELLOW}$DEPLOYMENT_URL/api${NC}"
echo -e "- Direct Proxy: ${YELLOW}$DEPLOYMENT_URL/debug-proxy${NC}"
echo -e "- Enhanced Debug: ${YELLOW}$DEPLOYMENT_URL/enhanced-debug${NC}"
echo ""
echo -e "${BLUE}Debug Endpoints:${NC}"
echo -e "- Connection Diagnostics: ${YELLOW}$DEPLOYMENT_URL/debug-proxy/connection-diagnostics${NC}"
echo -e "- Proxy Health: ${YELLOW}$DEPLOYMENT_URL/debug-proxy/proxy-health${NC}"
echo -e "- Proxy Info: ${YELLOW}$DEPLOYMENT_URL/debug-proxy/proxy-info${NC}"
echo ""
echo -e "${BLUE}To run diagnostics later:${NC}"
echo -e "${YELLOW}./scripts/api_diagnostics.py --url $DEPLOYMENT_URL --test-all --use-direct-proxy${NC}"
echo ""
echo -e "${BLUE}To check logs:${NC}"
echo -e "${YELLOW}vercel logs $DEPLOYMENT_URL${NC}"