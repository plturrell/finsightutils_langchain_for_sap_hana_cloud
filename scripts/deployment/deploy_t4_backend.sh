#!/bin/bash
# Deploy the backend to NVIDIA T4 GPU server

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Deploying SAP HANA Cloud LangChain Backend to T4 GPU${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Docker Compose is not installed. Please install Docker Compose first.${NC}"
    exit 1
fi

# Check if NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}NVIDIA drivers not detected. Please install NVIDIA drivers and NVIDIA Container Toolkit.${NC}"
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${RED}NVIDIA Container Toolkit not detected. Please install it.${NC}"
    echo -e "${YELLOW}See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
    exit 1
fi

# Check GPU type
GPU_TYPE=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -i "T4" || echo "")
if [ -z "$GPU_TYPE" ]; then
    echo -e "${YELLOW}Warning: NVIDIA T4 GPU not detected. This script is optimized for T4 GPUs.${NC}"
    read -p "Continue anyway? (y/n): " continue_prompt
    if [[ ! "$continue_prompt" =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}NVIDIA T4 GPU detected: $GPU_TYPE${NC}"
fi

# Create .env file for the backend
echo -e "${BLUE}Creating backend configuration...${NC}"
cat > api/.env << EOF
# GPU Configuration
GPU_ENABLED=true
USE_TENSORRT=true
TENSORRT_PRECISION=fp16
TENSORRT_CACHE_DIR=/app/tensorrt_cache

# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG
ENABLE_CORS=true
APP_NAME=langchain-hana-integration

# Performance
DEFAULT_TIMEOUT=30
HEALTH_CHECK_TIMEOUT=10
EMBEDDING_TIMEOUT=60
SEARCH_TIMEOUT=45
CONNECTION_TEST_TIMEOUT=5

# Authentication
JWT_SECRET=sap-hana-langchain-t4-integration-secret-key-2025
REQUIRE_AUTH=false

# Error Handling
ENABLE_ERROR_CONTEXT=true
ENABLE_DETAILED_LOGGING=true
ENABLE_MEMORY_TRACKING=true
MAX_RETRY_COUNT=3
RETRY_DELAY_MS=1000
ENABLE_SSL_VERIFICATION=true

# CORS Configuration - Allow Vercel frontend
CORS_ORIGINS=*
EOF

# Create Docker Compose configuration for T4 GPU
echo -e "${BLUE}Creating Docker Compose configuration...${NC}"
cat > docker-compose.t4.yml << EOF
version: '3'

services:
  api:
    build:
      context: ./api
      dockerfile: Dockerfile.ngc
    ports:
      - "8000:8000"
    env_file:
      - ./api/.env
    volumes:
      - ./api:/app
      - tensorrt_cache:/app/tensorrt_cache
    restart: unless-stopped
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  tensorrt_cache:
EOF

# Make start script executable
chmod +x api/start.sh

# Build and start the backend service
echo -e "${BLUE}Building and starting the backend service...${NC}"
docker-compose -f docker-compose.t4.yml up -d --build

# Wait for the service to start
echo -e "${BLUE}Waiting for the service to start...${NC}"
sleep 10

# Check if the service is running
if docker-compose -f docker-compose.t4.yml ps | grep -q "Up"; then
    echo -e "${GREEN}Backend service is running!${NC}"
    
    # Get server IP for frontend configuration
    SERVER_IP=$(hostname -I | awk '{print $1}')
    
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${GREEN}Backend deployment completed successfully!${NC}"
    echo -e "${GREEN}==========================================${NC}"
    echo -e "${BLUE}Backend URL:${NC} http://$SERVER_IP:8000"
    echo -e "${BLUE}API Documentation:${NC} http://$SERVER_IP:8000/docs"
    echo -e "${BLUE}GPU Info:${NC} http://$SERVER_IP:8000/benchmark/gpu_info"
    
    echo -e "\n${BLUE}Next Steps for Vercel Frontend Deployment:${NC}"
    echo -e "1. Update the API URL in the frontend:"
    echo -e "   ${YELLOW}cd frontend${NC}"
    echo -e "   ${YELLOW}echo \"REACT_APP_API_URL=http://$SERVER_IP:8000\" > .env${NC}"
    echo -e "2. Deploy the frontend to Vercel:"
    echo -e "   ${YELLOW}vercel --prod${NC}"
    echo -e "3. Alternatively, set environment variables in the Vercel dashboard:"
    echo -e "   Key: ${YELLOW}REACT_APP_API_URL${NC}"
    echo -e "   Value: ${YELLOW}http://$SERVER_IP:8000${NC}"
    
    echo -e "\n${BLUE}To view backend logs:${NC}"
    echo -e "docker-compose -f docker-compose.t4.yml logs -f"
    
    echo -e "\n${BLUE}To stop the backend:${NC}"
    echo -e "docker-compose -f docker-compose.t4.yml down"
    
    # Create frontend deployment instructions
    echo -e "${BLUE}Creating Vercel frontend deployment instructions...${NC}"
    cat > VERCEL_FRONTEND_DEPLOY.md << EOF
# Deploying the Frontend to Vercel

Follow these steps to deploy the frontend to Vercel:

## Prerequisites
- Vercel CLI installed: \`npm install -g vercel\`
- Vercel account and logged in via CLI: \`vercel login\`

## Steps

1. Update the frontend environment to point to your T4 backend:

\`\`\`bash
cd frontend
echo "REACT_APP_API_URL=http://$SERVER_IP:8000" > .env
\`\`\`

2. Deploy to Vercel:

\`\`\`bash
vercel --prod
\`\`\`

3. After deployment, your frontend will be available at the URL provided by Vercel.

## Alternative: Configure via Vercel Dashboard

1. Go to your project settings in the Vercel dashboard
2. Navigate to the "Environment Variables" section
3. Add the following variable:
   - Name: \`REACT_APP_API_URL\`
   - Value: \`http://$SERVER_IP:8000\`
4. Redeploy your project

## Important Notes

- Ensure your T4 server is accessible from the internet if your Vercel frontend needs to connect to it
- For production, consider setting up HTTPS for your backend API
- Update CORS settings in your backend if needed
EOF
    
    echo -e "${GREEN}Vercel frontend deployment instructions created in VERCEL_FRONTEND_DEPLOY.md${NC}"
else
    echo -e "${RED}Backend service failed to start. Check the logs:${NC}"
    docker-compose -f docker-compose.t4.yml logs
    exit 1
fi