#!/bin/bash
# Script to deploy the SAP HANA Cloud LangChain Integration to a Jupyter VM

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
VM_USER="ubuntu"
VM_PORT="22"
LOCAL_PROJECT_PATH="$(pwd)"
REMOTE_PROJECT_PATH="/home/ubuntu/langchain-integration-for-sap-hana-cloud"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--host)
      VM_HOST="$2"
      shift 2
      ;;
    -u|--user)
      VM_USER="$2"
      shift 2
      ;;
    -p|--port)
      VM_PORT="$2"
      shift 2
      ;;
    -i|--identity)
      IDENTITY_FILE="$2"
      shift 2
      ;;
    --path)
      REMOTE_PROJECT_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 -h|--host <hostname> [-u|--user <username>] [-p|--port <port>] [-i|--identity <identity_file>] [--path <remote_path>]"
      exit 1
      ;;
  esac
done

# Check if host is provided
if [ -z "$VM_HOST" ]; then
    echo -e "${RED}Error: VM hostname not provided${NC}"
    echo "Usage: $0 -h|--host <hostname> [-u|--user <username>] [-p|--port <port>] [-i|--identity <identity_file>] [--path <remote_path>]"
    exit 1
fi

# Build SSH command with appropriate options
SSH_CMD="ssh"
SCP_CMD="scp"

if [ -n "$IDENTITY_FILE" ]; then
    SSH_CMD="$SSH_CMD -i $IDENTITY_FILE"
    SCP_CMD="$SCP_CMD -i $IDENTITY_FILE"
fi

SSH_CMD="$SSH_CMD -p $VM_PORT $VM_USER@$VM_HOST"
SCP_CMD="$SCP_CMD -P $VM_PORT"

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Deploying to Jupyter VM at $VM_HOST${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if we can connect to the VM
echo -e "${BLUE}Testing connection to VM...${NC}"
if ! $SSH_CMD "echo Connection successful"; then
    echo -e "${RED}Failed to connect to VM. Check your credentials and network.${NC}"
    exit 1
fi

# Create a temporary directory for the deployment package
TEMP_DIR=$(mktemp -d)
echo -e "${BLUE}Creating deployment package in ${TEMP_DIR}...${NC}"

# Copy essential files to the temporary directory
mkdir -p $TEMP_DIR/api
mkdir -p $TEMP_DIR/frontend

# Copy API files
cp -r $LOCAL_PROJECT_PATH/api/*.py $TEMP_DIR/api/
cp -r $LOCAL_PROJECT_PATH/api/requirements*.txt $TEMP_DIR/api/
cp -r $LOCAL_PROJECT_PATH/api/Dockerfile* $TEMP_DIR/api/
cp -r $LOCAL_PROJECT_PATH/api/docker-compose*.yml $TEMP_DIR/api/
cp -r $LOCAL_PROJECT_PATH/api/start.sh $TEMP_DIR/api/
cp -r $LOCAL_PROJECT_PATH/api/.env $TEMP_DIR/api/

# Copy langchain_hana package
cp -r $LOCAL_PROJECT_PATH/langchain_hana $TEMP_DIR/

# Copy frontend files
cp -r $LOCAL_PROJECT_PATH/frontend/Dockerfile $TEMP_DIR/frontend/
cp -r $LOCAL_PROJECT_PATH/frontend/package*.json $TEMP_DIR/frontend/
cp -r $LOCAL_PROJECT_PATH/frontend/public $TEMP_DIR/frontend/
cp -r $LOCAL_PROJECT_PATH/frontend/src $TEMP_DIR/frontend/
cp -r $LOCAL_PROJECT_PATH/frontend/nginx.conf $TEMP_DIR/frontend/
cp -r $LOCAL_PROJECT_PATH/frontend/entrypoint.sh $TEMP_DIR/frontend/

# Copy deployment scripts
cp $LOCAL_PROJECT_PATH/deploy_to_vm.sh $TEMP_DIR/

# Create deployment archive
DEPLOYMENT_ARCHIVE="langchain_hana_deployment.tar.gz"
echo -e "${BLUE}Creating deployment archive...${NC}"
tar -czf $DEPLOYMENT_ARCHIVE -C $TEMP_DIR .

# Copy the deployment archive to the VM
echo -e "${BLUE}Copying deployment archive to VM...${NC}"
$SCP_CMD $DEPLOYMENT_ARCHIVE $VM_USER@$VM_HOST:/tmp/

# Extract and deploy on the VM
echo -e "${BLUE}Extracting and deploying on VM...${NC}"
$SSH_CMD "mkdir -p $REMOTE_PROJECT_PATH && tar -xzf /tmp/$DEPLOYMENT_ARCHIVE -C $REMOTE_PROJECT_PATH && cd $REMOTE_PROJECT_PATH && chmod +x deploy_to_vm.sh && ./deploy_to_vm.sh"

# Clean up
echo -e "${BLUE}Cleaning up temporary files...${NC}"
rm -rf $TEMP_DIR
rm $DEPLOYMENT_ARCHIVE

echo -e "${GREEN}Deployment completed!${NC}"
echo -e "${BLUE}You can access the API at:${NC} http://$VM_HOST:8000"
echo -e "${BLUE}You can access the frontend at:${NC} http://$VM_HOST:3000"
echo -e "${BLUE}API documentation is available at:${NC} http://$VM_HOST:8000/docs"
echo -e "${BLUE}To check logs on the VM, run:${NC} docker-compose -f $REMOTE_PROJECT_PATH/api/docker-compose.yml logs -f"