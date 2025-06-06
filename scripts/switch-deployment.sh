#!/bin/bash
# Blue-Green Deployment Switcher
# Usage: ./switch-deployment.sh [blue|green]

set -e

# Configuration
COMPOSE_FILE="../config/docker/docker-compose.blue-green.yml"
ENV_FILE="../.env.blue-green"
HEALTH_CONTAINER="sap-hana-langchain-healthcheck"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Validate input
if [ "$1" != "blue" ] && [ "$1" != "green" ]; then
    echo -e "${RED}Error: Must specify 'blue' or 'green' as the target deployment${NC}"
    echo "Usage: $0 [blue|green]"
    exit 1
fi

TARGET_COLOR="$1"

echo -e "${BLUE}Switching active deployment to ${TARGET_COLOR}...${NC}"

# Get current active deployment
CURRENT_ACTIVE=$(curl -s http://localhost/api/deployment/status | grep -o '"color":"[^"]*"' | cut -d'"' -f4)

if [ "$CURRENT_ACTIVE" == "$TARGET_COLOR" ]; then
    echo -e "${YELLOW}Deployment ${TARGET_COLOR} is already active. No switch needed.${NC}"
    exit 0
fi

# Check health of target deployment
if [ "$TARGET_COLOR" == "blue" ]; then
    HEALTH_URL="http://localhost:8000/health/status"
else
    HEALTH_URL="http://localhost:8001/health/status"
fi

echo -e "${BLUE}Checking health of ${TARGET_COLOR} deployment...${NC}"
HEALTH_STATUS=$(curl -s "$HEALTH_URL")
HEALTHY=$(echo "$HEALTH_STATUS" | grep -o '"status":"healthy"' || echo "")

if [ -z "$HEALTHY" ]; then
    echo -e "${RED}Error: ${TARGET_COLOR} deployment is not healthy!${NC}"
    echo "$HEALTH_STATUS"
    echo -e "${YELLOW}Do you want to force the switch anyway? (y/N)${NC}"
    read -r FORCE
    if [ "$FORCE" != "y" ] && [ "$FORCE" != "Y" ]; then
        echo "Aborting switch."
        exit 1
    fi
    echo -e "${YELLOW}Forcing switch despite unhealthy status...${NC}"
fi

# Perform the switch using the healthcheck container
echo -e "${BLUE}Executing deployment switch to ${TARGET_COLOR}...${NC}"
docker exec "$HEALTH_CONTAINER" python -c "from healthcheck import switch_traffic; print(switch_traffic('$TARGET_COLOR'))"

# Verify the switch
sleep 5
VERIFY_ACTIVE=$(curl -s http://localhost/api/deployment/status | grep -o '"color":"[^"]*"' | cut -d'"' -f4)

if [ "$VERIFY_ACTIVE" == "$TARGET_COLOR" ]; then
    echo -e "${GREEN}Successfully switched to ${TARGET_COLOR} deployment!${NC}"
    
    # Print version information
    VERSION=$(curl -s http://localhost/api/deployment/status | grep -o '"version":"[^"]*"' | cut -d'"' -f4)
    echo -e "${GREEN}Active deployment version: ${VERSION}${NC}"
else
    echo -e "${RED}Failed to switch to ${TARGET_COLOR} deployment. Currently active: ${VERIFY_ACTIVE}${NC}"
    exit 1
fi

# Display next steps for updating the other environment
OTHER_COLOR=$([ "$TARGET_COLOR" = "blue" ] && echo "green" || echo "blue")
echo -e "${BLUE}Next steps:${NC}"
echo "1. Update the $OTHER_COLOR deployment to the new version"
echo "   Edit .env.blue-green to update ${OTHER_COLOR^^}_VERSION"
echo "2. Apply the update to the $OTHER_COLOR deployment:"
echo "   docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d api-$OTHER_COLOR"

exit 0