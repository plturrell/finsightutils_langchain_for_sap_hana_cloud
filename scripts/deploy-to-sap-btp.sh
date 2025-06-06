#!/bin/bash
#
# SAP BTP Deployment Script
#
# This script handles deploying the backend API to SAP Business Technology Platform
# with GPU support.
#
# Prerequisites:
# - SAP Cloud Foundry CLI installed and configured
# - SAP BTP account with proper entitlements
# - Cloud Foundry environment configured

set -e

# Configuration
APP_NAME="langchain-hana-backend"
ENVIRONMENT="production"  # Options: dev, test, prod
INSTANCES=1
MEMORY="2G"
DISK="2G"
GPU_ENABLED=true

# Project root directory (parent of the script directory)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --app-name)
      APP_NAME="$2"
      shift
      shift
      ;;
    --env)
      ENVIRONMENT="$2"
      shift
      shift
      ;;
    --instances)
      INSTANCES="$2"
      shift
      shift
      ;;
    --memory)
      MEMORY="$2"
      shift
      shift
      ;;
    --disk)
      DISK="$2"
      shift
      shift
      ;;
    --no-gpu)
      GPU_ENABLED=false
      shift
      ;;
    --help)
      echo "SAP BTP Deployment Script"
      echo ""
      echo "Usage:"
      echo "  ./deploy_to_sap_btp.sh [options]"
      echo ""
      echo "Options:"
      echo "  --app-name <name>   Name of the application on SAP BTP (default: langchain-hana-backend)"
      echo "  --env <env>         Environment (dev, test, prod) (default: production)"
      echo "  --instances <num>   Number of instances (default: 1)"
      echo "  --memory <size>     Memory allocation (default: 2G)"
      echo "  --disk <size>       Disk allocation (default: 2G)"
      echo "  --no-gpu            Disable GPU support"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Check Cloud Foundry CLI is installed
if ! command -v cf &> /dev/null; then
    echo "Error: Cloud Foundry CLI is not installed or not in PATH"
    echo "Please install the Cloud Foundry CLI: https://docs.cloudfoundry.org/cf-cli/install-go-cli.html"
    exit 1
fi

# Check Cloud Foundry authentication
echo "Checking Cloud Foundry authentication..."
cf target &> /dev/null || { 
    echo "Error: Cloud Foundry authentication failed."
    echo "Please log in using 'cf login'"
    exit 1
}

# Prepare application for deployment
echo "Preparing application for SAP BTP deployment..."
cd "$PROJECT_ROOT"

# Ensure env file exists
ENV_FILE=".env.sap.$ENVIRONMENT"
if [ ! -f "$ENV_FILE" ]; then
    echo "Warning: Environment file $ENV_FILE not found. Using .env.template"
    cp "api/.env.template" "api/.env"
else
    echo "Using environment file: $ENV_FILE"
    cp "$ENV_FILE" "api/.env"
fi

# Create deployment directory
DEPLOY_DIR="$PROJECT_ROOT/deploy/sap"
mkdir -p "$DEPLOY_DIR"

# Copy necessary files
cp -r "$PROJECT_ROOT/api"/* "$DEPLOY_DIR/"
cp "$PROJECT_ROOT/api/.env" "$DEPLOY_DIR/"
cp "$PROJECT_ROOT/README.md" "$DEPLOY_DIR/"

# Create manifest.yml
cat > "$DEPLOY_DIR/manifest.yml" << EOL
---
applications:
- name: $APP_NAME
  memory: $MEMORY
  disk_quota: $DISK
  instances: $INSTANCES
  buildpacks:
  - python_buildpack
  command: python -m uvicorn app:app --host 0.0.0.0 --port \$PORT
  health-check-type: http
  health-check-http-endpoint: /api/health
  timeout: 180
  env:
    PLATFORM: sap_btp
    PLATFORM_SUPPORTS_GPU: $GPU_ENABLED
    ENVIRONMENT: $ENVIRONMENT
    ENABLE_CONTEXT_AWARE_ERRORS: true
    ENABLE_PRECISE_SIMILARITY: true
EOL

if [ "$GPU_ENABLED" = true ]; then
    # Add GPU-specific configuration
    cat >> "$DEPLOY_DIR/manifest.yml" << EOL
    USE_TENSORRT: true
    GPU_ENABLED: true
    GPU_DEVICE: auto
    EMBEDDING_MODEL: all-MiniLM-L6-v2
EOL
fi

# Create vars.yml for sensitive variables
cat > "$DEPLOY_DIR/vars.yml" << EOL
---
# Replace these with your actual values
hana_host: YOUR_HANA_HOST
hana_port: 443
hana_user: YOUR_HANA_USER
hana_password: YOUR_HANA_PASSWORD
EOL

echo "Created vars.yml. Please edit it to provide your actual SAP HANA Cloud credentials."

# Deploy to SAP BTP
echo "Deploying to SAP BTP..."
cd "$DEPLOY_DIR"

echo "WARNING: Before proceeding, make sure to update vars.yml with your actual credentials."
read -p "Continue with deployment? [y/N] " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled."
    exit 0
fi

# Deploy the application
cf push --vars-file vars.yml

# Get application URL
APP_URL=$(cf app $APP_NAME | grep routes | awk '{print $2}')

echo "SAP BTP deployment completed successfully!"
echo "Application URL: https://$APP_URL"
echo ""
echo "Next steps:"
echo "1. Update your frontend configuration to use this API URL"
echo "2. Test the API endpoint: curl https://$APP_URL/api/health"
echo "3. Monitor the application in the SAP BTP dashboard"
echo ""
echo "For more information, see the SAP BTP documentation in docs/flexible_deployment.md"