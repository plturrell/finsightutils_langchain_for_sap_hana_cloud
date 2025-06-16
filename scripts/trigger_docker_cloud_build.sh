#!/bin/bash
# Script to trigger a build on Docker Hub's automated build system for
# the finsightintelligence/langchainsaphana repository

set -e

# Log function
log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check for required tools
if ! command -v curl &> /dev/null; then
  log "ERROR: curl is required but not installed"
  exit 1
fi

# Prompt for credentials if not provided
if [ -z "$DOCKER_USERNAME" ]; then
  read -p "Docker Hub username: " DOCKER_USERNAME
fi

if [ -z "$DOCKER_PASSWORD" ]; then
  read -s -p "Docker Hub password/token: " DOCKER_PASSWORD
  echo
fi

# Variables
ORGANIZATION="finsightintelligence"
REPOSITORY="langchainsaphana"

# Get Docker Hub authentication token
log "Authenticating with Docker Hub..."
TOKEN=$(curl -s -H "Content-Type: application/json" \
  -X POST \
  -d "{\"username\": \"${DOCKER_USERNAME}\", \"password\": \"${DOCKER_PASSWORD}\"}" \
  https://hub.docker.com/v2/users/login/ | jq -r .token)

if [ -z "$TOKEN" ] || [ "$TOKEN" == "null" ]; then
  log "ERROR: Failed to authenticate with Docker Hub"
  exit 1
fi

# Trigger build for CPU image
log "Triggering build for CPU image..."
curl -s -X POST \
  -H "Authorization: JWT ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "Branch", "source_name": "main"}' \
  "https://hub.docker.com/v2/repositories/${ORGANIZATION}/${REPOSITORY}/autobuild/trigger-build/"

# Trigger build for GPU image
log "Triggering build for GPU image..."
curl -s -X POST \
  -H "Authorization: JWT ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{"source_type": "Branch", "source_name": "main"}' \
  "https://hub.docker.com/v2/repositories/${ORGANIZATION}/${REPOSITORY}/autobuild/trigger-build/"

log "Build requests sent to Docker Hub"
log "Check the status at: https://hub.docker.com/repository/docker/${ORGANIZATION}/${REPOSITORY}/builds"
