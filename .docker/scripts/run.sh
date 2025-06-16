#!/bin/bash
# Standardized Docker run script for LangChain SAP HANA Integration
# Based on the standardized template

set -e  # Exit immediately if a command fails

# Default values
PROFILE="full"
ENV_FILE=""
DETACH=false
PRUNE=false
SECURE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display help message
show_help() {
    echo -e "${BLUE}Docker Run Script for LangChain SAP HANA Integration${NC}"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -p, --profile PROFILE    Service profile to run (default: full)"
    echo "                           Available profiles: full, api, arrow-flight, frontend, monitoring, dev"
    echo "  -e, --env-file FILE      Environment file to use"
    echo "  -d, --detach             Run in detached mode"
    echo "  --prune                  Prune unused containers, networks, and volumes before starting"
    echo "  --secure                 Use secure configuration with enhanced security settings"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run all services in interactive mode"
    echo "  $0 -p api                 # Run only API service"
    echo "  $0 -p dev -e .env.dev     # Run development profile with specific env file"
    echo "  $0 -d                     # Run all services in detached mode"
    echo "  $0 --secure               # Run with enhanced security settings"
    echo ""
    exit 0
}

# Log message with timestamp
log() {
    local level=$1
    local message=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    case $level in
        "INFO")
            echo -e "${GREEN}[INFO]${NC} ${timestamp} - ${message}"
            ;;
        "WARN")
            echo -e "${YELLOW}[WARN]${NC} ${timestamp} - ${message}"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} ${timestamp} - ${message}"
            ;;
        *)
            echo -e "${BLUE}[${level}]${NC} ${timestamp} - ${message}"
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--profile)
            PROFILE="$2"
            shift 2
            ;;
        -e|--env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        --prune)
            PRUNE=true
            shift
            ;;
        --secure)
            SECURE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            show_help
            ;;
    esac
done

# Project root directory
PROJECT_ROOT=$(dirname $(dirname $(dirname $0)))
cd $PROJECT_ROOT

# Check if Docker Compose is available
if ! command -v docker compose &> /dev/null; then
    log "ERROR" "Docker Compose is not installed or not in PATH"
    exit 1
fi

# Available profiles
AVAILABLE_PROFILES=("full" "api" "arrow-flight" "frontend" "monitoring" "dev")

# Check if the specified profile is valid
if [[ ! " ${AVAILABLE_PROFILES[*]} " =~ " ${PROFILE} " ]]; then
    log "ERROR" "Invalid profile: $PROFILE"
    log "INFO" "Available profiles: ${AVAILABLE_PROFILES[*]}"
    exit 1
fi

# Determine compose files to use
COMPOSE_FILES="-f .docker/compose/docker-compose.yml"

# Add dev override if dev profile
if [[ "$PROFILE" == "dev" ]]; then
    COMPOSE_FILES="$COMPOSE_FILES -f .docker/compose/overrides/docker-compose.dev.yml"
fi

# Add secure override if secure flag is set
if [[ "$SECURE" == "true" ]]; then
    COMPOSE_FILES="$COMPOSE_FILES -f .docker/compose/overrides/docker-compose.secure.yml"
    log "INFO" "Using secure configuration with enhanced security settings"
fi

# Add env file if specified
if [[ -n "$ENV_FILE" ]]; then
    if [[ -f "$ENV_FILE" ]]; then
        COMPOSE_FILES="$COMPOSE_FILES --env-file $ENV_FILE"
    else
        log "ERROR" "Environment file not found: $ENV_FILE"
        exit 1
    fi
fi

# Prune docker resources if requested
if [[ "$PRUNE" == "true" ]]; then
    log "INFO" "Pruning unused Docker resources..."
    docker system prune -f
fi

# Stop any running containers with the same name
log "INFO" "Stopping any existing containers..."
docker compose $COMPOSE_FILES --profile $PROFILE down 2>/dev/null || true

# Build the command
CMD="docker compose $COMPOSE_FILES --profile $PROFILE"
if [[ "$DETACH" == "true" ]]; then
    CMD="$CMD up -d"
    log "INFO" "Starting services in detached mode with profile: $PROFILE"
else
    CMD="$CMD up"
    log "INFO" "Starting services with profile: $PROFILE"
fi

# Run docker compose
log "INFO" "Running command: $CMD"
eval $CMD

# Show info if detached
if [[ "$DETACH" == "true" ]]; then
    # Wait a moment for containers to start
    sleep 3
    
    # Show running containers
    log "INFO" "Running containers:"
    docker compose $COMPOSE_FILES --profile $PROFILE ps
    
    # Show access info
    case "$PROFILE" in
        "full"|"frontend")
            log "INFO" "Frontend is available at: http://localhost:${FRONTEND_PORT:-3000}"
            log "INFO" "API is available at: http://localhost:${API_PORT:-8000}"
            log "INFO" "Arrow Flight service is available at: http://localhost:${ARROW_API_PORT:-8001}"
            ;;
        "api")
            log "INFO" "API is available at: http://localhost:${API_PORT:-8000}"
            log "INFO" "API Documentation: http://localhost:${API_PORT:-8000}/docs"
            ;;
        "arrow-flight")
            log "INFO" "Arrow Flight API is available at: http://localhost:${ARROW_API_PORT:-8001}"
            log "INFO" "Arrow Flight service is available at: http://localhost:${ARROW_FLIGHT_PORT:-8816}"
            ;;
        "monitoring")
            log "INFO" "Prometheus is available at: http://localhost:${PROMETHEUS_PORT:-9090}"
            log "INFO" "Grafana is available at: http://localhost:${GRAFANA_PORT:-3000}"
            ;;
        "dev")
            log "INFO" "API is available at: http://localhost:${API_PORT:-8000}"
            log "INFO" "Frontend is available at: http://localhost:${FRONTEND_PORT:-3000}"
            log "INFO" "Jupyter Notebook is available at: http://localhost:${JUPYTER_PORT:-8888}"
            ;;
    esac
    
    log "INFO" "To stop the services, run: docker compose $COMPOSE_FILES --profile $PROFILE down"
fi

exit 0