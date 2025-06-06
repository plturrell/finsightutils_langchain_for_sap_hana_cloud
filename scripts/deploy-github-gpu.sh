#!/bin/bash
# GPU-accelerated SAP HANA Cloud LangChain Integration GitHub Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Configuration
REPO_URL=${1:-"https://github.com/yourusername/langchain-integration-for-sap-hana-cloud.git"}
BRANCH=${2:-"main"}
DEPLOYMENT_DIR="/tmp/sap-hana-langchain-deploy"
ENV_FILE=".env.deployment"

echo -e "${BLUE}=== SAP HANA Cloud LangChain GPU Deployment ===${NC}"
echo -e "${BLUE}Repository: ${REPO_URL}${NC}"
echo -e "${BLUE}Branch: ${BRANCH}${NC}"

# Step 1: Clone or update the repository
if [ -d "$DEPLOYMENT_DIR" ]; then
    echo -e "${YELLOW}Updating existing repository...${NC}"
    cd "$DEPLOYMENT_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo -e "${YELLOW}Cloning repository...${NC}"
    git clone --branch "$BRANCH" "$REPO_URL" "$DEPLOYMENT_DIR"
    cd "$DEPLOYMENT_DIR"
fi

# Step 2: Check for GPU availability
echo -e "${YELLOW}Checking GPU availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    GPU_AVAILABLE=true
    echo -e "${GREEN}GPU detected! Using GPU-accelerated configuration.${NC}"
else
    GPU_AVAILABLE=false
    echo -e "${YELLOW}No GPU detected. Using CPU configuration.${NC}"
fi

# Step 3: Set up environment variables
echo -e "${YELLOW}Setting up environment variables...${NC}"
if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Using existing environment file: $ENV_FILE${NC}"
else
    echo -e "${YELLOW}Creating new environment file: $ENV_FILE${NC}"
    cat > "$ENV_FILE" << EOF
# SAP HANA Cloud Connection
HANA_HOST=your-hana-host
HANA_PORT=443
HANA_USER=your-hana-user
HANA_PASSWORD=your-hana-password
DEFAULT_TABLE_NAME=EMBEDDINGS

# GPU Settings
GPU_ENABLED=$GPU_AVAILABLE
USE_TENSORRT=$GPU_AVAILABLE
TENSORRT_PRECISION=fp16
BATCH_SIZE=32
MAX_BATCH_SIZE=128
ENABLE_MULTI_GPU=false
GPU_MEMORY_FRACTION=0.9

# Blue-Green Deployment
BLUE_VERSION=1.0.0
GREEN_VERSION=1.0.0
EOF
    echo -e "${YELLOW}Please edit $DEPLOYMENT_DIR/$ENV_FILE with your SAP HANA Cloud credentials${NC}"
    echo -e "${YELLOW}Then re-run this script.${NC}"
    exit 1
fi

# Step 4: Run deployment
echo -e "${YELLOW}Starting deployment...${NC}"
if [ "$GPU_AVAILABLE" = true ]; then
    # Check Docker Compose configuration
    if [ -f "config/docker/docker-compose.blue-green.yml" ]; then
        echo -e "${YELLOW}Using blue-green deployment for high availability...${NC}"
        docker-compose -f config/docker/docker-compose.blue-green.yml --env-file "$ENV_FILE" up -d
    else
        echo -e "${YELLOW}Using standard GPU-accelerated deployment...${NC}"
        docker-compose -f config/docker/docker-compose.nvidia.yml --env-file "$ENV_FILE" up -d
    fi
else
    echo -e "${YELLOW}Using CPU-only deployment...${NC}"
    docker-compose -f docker-compose.yml --env-file "$ENV_FILE" up -d
fi

# Step 5: Verify deployment
echo -e "${YELLOW}Verifying deployment...${NC}"
if [ "$GPU_AVAILABLE" = true ]; then
    if [ -f "config/docker/docker-compose.blue-green.yml" ]; then
        # Check blue-green deployment
        sleep 10
        HEALTH_BLUE=$(curl -s http://localhost:8000/health/status | grep -o '"status":"healthy"' || echo "")
        HEALTH_GREEN=$(curl -s http://localhost:8001/health/status | grep -o '"status":"healthy"' || echo "")
        
        if [ -n "$HEALTH_BLUE" ]; then
            echo -e "${GREEN}Blue deployment is healthy!${NC}"
        else
            echo -e "${RED}Blue deployment health check failed.${NC}"
        fi
        
        if [ -n "$HEALTH_GREEN" ]; then
            echo -e "${GREEN}Green deployment is healthy!${NC}"
        else
            echo -e "${RED}Green deployment health check failed.${NC}"
        fi
        
        # Check Traefik
        echo -e "${YELLOW}Checking Traefik router configuration...${NC}"
        curl -s http://localhost:8080/api/http/routers | grep -o '"service":"api-[^"]*"'
    else
        # Check standard deployment
        sleep 5
        HEALTH=$(curl -s http://localhost:8000/health/ping)
        if [ "$HEALTH" == "pong" ]; then
            echo -e "${GREEN}Deployment successful!${NC}"
        else
            echo -e "${RED}Deployment health check failed.${NC}"
        fi
    fi
else
    # Check CPU deployment
    sleep 5
    HEALTH=$(curl -s http://localhost:8000/health/ping)
    if [ "$HEALTH" == "pong" ]; then
        echo -e "${GREEN}Deployment successful!${NC}"
    else
        echo -e "${RED}Deployment health check failed.${NC}"
    fi
fi

# Step 6: Show access URLs
echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${BLUE}Access your application:${NC}"
echo -e "  API: http://localhost:8000"
echo -e "  Frontend: http://localhost:3000"
if [ -f "config/docker/docker-compose.blue-green.yml" ]; then
    echo -e "  Traefik Dashboard: http://localhost:8080"
    echo -e "  Prometheus: http://localhost:9090"
    echo -e "  Grafana: http://localhost:3001 (admin/admin)"
fi

echo -e "${YELLOW}For troubleshooting, check container logs:${NC}"
echo -e "  docker-compose logs -f api"

exit 0