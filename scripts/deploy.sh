#!/bin/bash
set -e

# Main deployment script that handles configuration and deployment
# to different platform combinations.

# Default settings
BACKEND="together"
FRONTEND="vercel"
ENVIRONMENT="prod"
DRY_RUN=false
PUSH_TO_GITHUB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --backend)
      BACKEND="$2"
      shift 2
      ;;
    --frontend)
      FRONTEND="$2"
      shift 2
      ;;
    --env)
      ENVIRONMENT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --push)
      PUSH_TO_GITHUB=true
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --backend BACKEND    Backend platform (together, nvidia, sap, vercel)"
      echo "  --frontend FRONTEND  Frontend platform (vercel, sap)"
      echo "  --env ENV            Environment (dev, test, staging, prod)"
      echo "  --dry-run            Simulate deployment without making changes"
      echo "  --push               Push changes to GitHub"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate backend platform
case $BACKEND in
  together|nvidia|sap|vercel)
    echo "Using backend platform: $BACKEND"
    ;;
  *)
    echo "Error: Invalid backend platform: $BACKEND"
    echo "Valid options: together, nvidia, sap, vercel"
    exit 1
    ;;
esac

# Validate frontend platform
case $FRONTEND in
  vercel|sap)
    echo "Using frontend platform: $FRONTEND"
    ;;
  *)
    echo "Error: Invalid frontend platform: $FRONTEND"
    echo "Valid options: vercel, sap"
    exit 1
    ;;
esac

# Validate environment
case $ENVIRONMENT in
  dev|test|staging|prod)
    echo "Using environment: $ENVIRONMENT"
    ;;
  *)
    echo "Error: Invalid environment: $ENVIRONMENT"
    echo "Valid options: dev, test, staging, prod"
    exit 1
    ;;
esac

# Configuration files
BACKEND_ENV_FILE=".env.${BACKEND}.${ENVIRONMENT}"
FRONTEND_ENV_FILE=".env.frontend.${FRONTEND}.${ENVIRONMENT}"

# Check if environment files exist
if [ ! -f "$BACKEND_ENV_FILE" ]; then
    echo "Error: Backend environment file $BACKEND_ENV_FILE not found"
    exit 1
fi

if [ ! -f "$FRONTEND_ENV_FILE" ]; then
    echo "Error: Frontend environment file $FRONTEND_ENV_FILE not found"
    exit 1
fi

echo "Using backend config: $BACKEND_ENV_FILE"
echo "Using frontend config: $FRONTEND_ENV_FILE"

# Prepare deployment
if [ "$DRY_RUN" = true ]; then
    echo "Dry run mode - simulating deployment"
else
    echo "Preparing deployment"
    
    # Copy environment files
    cp "$BACKEND_ENV_FILE" api/.env
    cp "$FRONTEND_ENV_FILE" frontend/.env
    
    # Run deployment scripts
    if [ "$BACKEND" = "together" ]; then
        echo "Deploying backend to Together.ai"
        chmod +x scripts/deploy_to_together.sh
        scripts/deploy_to_together.sh
    elif [ "$BACKEND" = "nvidia" ]; then
        echo "Deploying backend to NVIDIA LaunchPad"
        chmod +x scripts/deploy_to_nvidia.sh
        scripts/deploy_to_nvidia.sh
    elif [ "$BACKEND" = "sap" ]; then
        echo "Deploying backend to SAP BTP"
        chmod +x scripts/deploy_to_sap_btp.sh
        scripts/deploy_to_sap_btp.sh
    elif [ "$BACKEND" = "vercel" ]; then
        echo "Deploying backend to Vercel"
        # The Vercel deployment will be handled by GitHub Actions
        echo "Vercel deployment will be handled by GitHub Actions"
    fi
    
    if [ "$FRONTEND" = "vercel" ]; then
        echo "Deploying frontend to Vercel"
        chmod +x scripts/deploy_frontend_to_vercel.sh
        scripts/deploy_frontend_to_vercel.sh
    elif [ "$FRONTEND" = "sap" ]; then
        echo "Deploying frontend to SAP BTP"
        chmod +x scripts/deploy_frontend_to_sap_btp.sh
        scripts/deploy_frontend_to_sap_btp.sh
    fi
fi

# Push changes to GitHub
if [ "$PUSH_TO_GITHUB" = true ]; then
    echo "Pushing changes to GitHub"
    
    # Add all files
    git add .
    
    # Commit changes
    git commit -m "Deploy to ${BACKEND}/${FRONTEND} (${ENVIRONMENT})"
    
    # Push to GitHub
    git push origin main
    
    echo "Changes pushed to GitHub"
fi

echo "Deployment complete!"