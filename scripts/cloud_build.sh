#!/bin/bash
# Script to build SAP HANA Cloud LangChain Nvidia Blueprint using Docker Build Cloud

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
PUSH=false
TAG="latest"
ENV_FILE=".env.test"
USE_EXISTING_BUILDER=false

# Help text
function show_help {
  echo -e "${BLUE}SAP HANA Cloud LangChain Nvidia Blueprint - Docker Build Cloud Script${NC}"
  echo ""
  echo "Usage: $0 [options]"
  echo ""
  echo "Options:"
  echo "  -h, --help              Show this help message"
  echo "  -p, --push              Push images to registry after build"
  echo "  -t, --tag TAG           Tag to use for images (default: latest)"
  echo "  -e, --env-file FILE     Environment file to use (default: .env.test)"
  echo "  -b, --use-builder       Use existing cloud builder instead of creating a new one"
  echo ""
  echo "Example:"
  echo "  $0 --push --tag v1.0.2 --env-file .env.prod"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -h|--help)
      show_help
      exit 0
      ;;
    -p|--push)
      PUSH=true
      shift
      ;;
    -t|--tag)
      TAG="$2"
      shift
      shift
      ;;
    -e|--env-file)
      ENV_FILE="$2"
      shift
      shift
      ;;
    -b|--use-builder)
      USE_EXISTING_BUILDER=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

echo -e "${BLUE}=== Docker Build Cloud - SAP HANA Cloud LangChain Nvidia Blueprint ===${NC}"
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Push images: ${PUSH}"
echo -e "  Tag: ${TAG}"
echo -e "  Environment file: ${ENV_FILE}"
echo -e "  Use existing builder: ${USE_EXISTING_BUILDER}"
echo ""

# Check for environment file
if [ ! -f "$ENV_FILE" ]; then
  echo -e "${YELLOW}Warning: Environment file $ENV_FILE does not exist. Creating an example file...${NC}"
  cat > "$ENV_FILE" << EOL
# SAP HANA Cloud connection
HANA_HOST=your-hana-host.hana.prod-region.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your-username
HANA_PASSWORD=your-password

# NVIDIA NGC credentials
NGC_API_KEY=your-ngc-api-key
EOL
  echo -e "${YELLOW}Created $ENV_FILE - please edit with your credentials${NC}"
  exit 1
fi

# Load environment variables
echo -e "${GREEN}Loading environment variables from $ENV_FILE...${NC}"
set -a
source "$ENV_FILE"
set +a

# Setup cloud builder
BUILDER_NAME="hana-langchain-cloud-builder"

if [ "$USE_EXISTING_BUILDER" = false ]; then
  echo -e "${GREEN}Creating new cloud builder: $BUILDER_NAME...${NC}"
  docker buildx create --name "$BUILDER_NAME" --driver cloud --bootstrap || {
    echo -e "${YELLOW}Failed to create cloud builder. Using existing one...${NC}"
    USE_EXISTING_BUILDER=true
  }
fi

if [ "$USE_EXISTING_BUILDER" = true ]; then
  echo -e "${GREEN}Using existing cloud builder...${NC}"
  docker buildx use "$BUILDER_NAME" || {
    echo -e "${YELLOW}Failed to use existing builder. Creating a default cloud builder...${NC}"
    docker buildx create --name "$BUILDER_NAME" --driver cloud --bootstrap
    docker buildx use "$BUILDER_NAME"
  }
else
  docker buildx use "$BUILDER_NAME"
fi

# Build API image
echo -e "${GREEN}Building API image with cloud builder...${NC}"
PUSH_FLAG=""
if [ "$PUSH" = true ]; then
  PUSH_FLAG="--push"
fi

docker buildx build $PUSH_FLAG \
  --file docker/Dockerfile.nvidia \
  --tag "langchain-hana-nvidia:$TAG" \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/pytorch:23.12-py3 \
  --cache-from type=registry,ref=langchain-hana-nvidia:cache \
  --cache-to type=registry,ref=langchain-hana-nvidia:cache,mode=max \
  .

# Build Frontend image (if exists)
if [ -d "frontend" ]; then
  echo -e "${GREEN}Building Frontend image with cloud builder...${NC}"
  docker buildx build $PUSH_FLAG \
    --file frontend/Dockerfile \
    --tag "langchain-hana-frontend:$TAG" \
    --cache-from type=registry,ref=langchain-hana-frontend:cache \
    --cache-to type=registry,ref=langchain-hana-frontend:cache,mode=max \
    ./frontend
fi

echo -e "${GREEN}Done!${NC}"
echo -e "${BLUE}To run the containers:${NC}"
echo -e "docker-compose -f docker/docker-compose.nvidia.yml --env-file $ENV_FILE up -d"
