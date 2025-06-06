#!/bin/bash
#
# SAP BTP Frontend Deployment Script
#
# This script handles building and deploying the frontend to SAP BTP.
#
# Prerequisites:
# - SAP Cloud Foundry CLI installed and configured
# - Node.js and npm installed

set -e

# Configuration
APP_NAME="langchain-hana-frontend"
ENVIRONMENT="production"  # Options: dev, test, prod
INSTANCES=2
MEMORY="256M"
DISK="512M"

# Project root directory (parent of the script directory)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

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
    --help)
      echo "SAP BTP Frontend Deployment Script"
      echo ""
      echo "Usage:"
      echo "  ./deploy_frontend_to_sap_btp.sh [options]"
      echo ""
      echo "Options:"
      echo "  --app-name <name>   Name of the application on SAP BTP (default: langchain-hana-frontend)"
      echo "  --env <env>         Environment (dev, test, prod) (default: production)"
      echo "  --instances <num>   Number of instances (default: 2)"
      echo "  --memory <size>     Memory allocation (default: 256M)"
      echo "  --disk <size>       Disk allocation (default: 512M)"
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

# Prepare frontend for deployment
echo "Preparing frontend for SAP BTP deployment..."
cd "$FRONTEND_DIR"

# Ensure required files exist
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found in $FRONTEND_DIR"
    exit 1
fi

# Build the frontend
echo "Building frontend..."
npm install

# Use the appropriate build command based on environment
case "$ENVIRONMENT" in
    dev)
        npm run build:dev
        ;;
    test)
        npm run build:staging
        ;;
    prod)
        npm run build
        ;;
esac

# Create deployment directory
DEPLOY_DIR="$PROJECT_ROOT/deploy/sap_frontend"
mkdir -p "$DEPLOY_DIR"

# Copy build files
cp -r "$FRONTEND_DIR/build/"* "$DEPLOY_DIR/"

# Create Staticfile for staticfile_buildpack
echo "Creating Staticfile configuration..."
cat > "$DEPLOY_DIR/Staticfile" << EOL
root: .
directory: Disallow
pushstate: enabled
EOL

# Create manifest.yml
echo "Creating manifest.yml..."
cat > "$DEPLOY_DIR/manifest.yml" << EOL
---
applications:
- name: $APP_NAME
  memory: $MEMORY
  disk_quota: $DISK
  instances: $INSTANCES
  buildpacks:
  - staticfile_buildpack
  routes:
  - route: $APP_NAME.cfapps.\${vcap.application.cf_api/api.cf.}.hana.ondemand.com
  env:
    FORCE_HTTPS: true
EOL

# Deploy to SAP BTP
echo "Deploying frontend to SAP BTP..."
cd "$DEPLOY_DIR"
cf push

# Get application URL
APP_URL=$(cf app $APP_NAME | grep routes | awk '{print $2}')

echo "SAP BTP frontend deployment completed successfully!"
echo "Application URL: https://$APP_URL"
echo ""
echo "Next steps:"
echo "1. Visit the deployment URL to verify the frontend"
echo "2. Configure the frontend to connect to your backend API"
echo "3. Set up a custom domain in the SAP BTP dashboard if needed"
echo ""
echo "For more information, see the SAP BTP documentation in docs/flexible_deployment.md"