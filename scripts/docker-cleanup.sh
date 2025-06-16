#!/bin/bash
# Script to clean up Docker environment before testing

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}     Docker Environment Cleanup Script   ${NC}"
echo -e "${BLUE}=========================================${NC}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo -e "${RED}Docker is not running. Please start Docker first.${NC}"
  exit 1
fi

# Stop all running containers
echo -e "${YELLOW}Stopping all running containers...${NC}"
RUNNING_CONTAINERS=$(docker ps -q)
if [ -n "$RUNNING_CONTAINERS" ]; then
  docker stop $RUNNING_CONTAINERS
  echo -e "${GREEN}✅ All containers stopped${NC}"
else
  echo -e "${GREEN}✅ No running containers to stop${NC}"
fi

# Remove all containers
echo -e "${YELLOW}Removing all containers...${NC}"
ALL_CONTAINERS=$(docker ps -a -q)
if [ -n "$ALL_CONTAINERS" ]; then
  docker rm $ALL_CONTAINERS
  echo -e "${GREEN}✅ All containers removed${NC}"
else
  echo -e "${GREEN}✅ No containers to remove${NC}"
fi

# Remove dangling images
echo -e "${YELLOW}Removing dangling images...${NC}"
DANGLING_IMAGES=$(docker images -f "dangling=true" -q)
if [ -n "$DANGLING_IMAGES" ]; then
  docker rmi $DANGLING_IMAGES
  echo -e "${GREEN}✅ Dangling images removed${NC}"
else
  echo -e "${GREEN}✅ No dangling images to remove${NC}"
fi

# Remove all finsightintelligence images
echo -e "${YELLOW}Removing finsightintelligence images...${NC}"
REPO_IMAGES=$(docker images "finsightintelligence/*" -q)
if [ -n "$REPO_IMAGES" ]; then
  docker rmi $REPO_IMAGES
  echo -e "${GREEN}✅ Repository images removed${NC}"
else
  echo -e "${GREEN}✅ No repository images to remove${NC}"
fi

# Prune volumes
echo -e "${YELLOW}Pruning Docker volumes...${NC}"
docker volume prune -f
echo -e "${GREEN}✅ Volumes pruned${NC}"

# Prune networks
echo -e "${YELLOW}Pruning Docker networks...${NC}"
docker network prune -f
echo -e "${GREEN}✅ Networks pruned${NC}"

# Prune build cache
echo -e "${YELLOW}Pruning build cache...${NC}"
docker builder prune -f
echo -e "${GREEN}✅ Build cache pruned${NC}"

# Clean up Docker Buildx cache
echo -e "${YELLOW}Cleaning Docker Buildx cache...${NC}"
docker buildx prune -f
echo -e "${GREEN}✅ Buildx cache cleaned${NC}"

# System prune as final step
echo -e "${YELLOW}Performing system-wide prune...${NC}"
docker system prune -f
echo -e "${GREEN}✅ System pruned${NC}"

echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}Docker environment successfully cleaned!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo -e "You can now test the Docker build implementation with:"
echo -e "${YELLOW}./docker-build.sh --type cpu-secure${NC}"

exit 0