#!/bin/bash
# Post-deployment script to configure SAP HANA Cloud connection
# Run this script after the API is running to configure it for real SAP HANA Cloud connection

# Print colored messages
function echo_info() {
  echo -e "\033[1;34m[INFO]\033[0m $1"
}

function echo_success() {
  echo -e "\033[1;32m[SUCCESS]\033[0m $1"
}

function echo_error() {
  echo -e "\033[1;31m[ERROR]\033[0m $1"
}

function echo_warning() {
  echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# Set up logging
LOG_DIR="logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/hana_config_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo_info "==================== SAP HANA CLOUD CONFIGURATION ===================="
echo_info "This script will configure the connection to SAP HANA Cloud"
echo_info "=================================================================="

# Check if API is running
echo_info "Checking if API is running..."
if curl -s http://localhost:8000/health/ping > /dev/null; then
  echo_success "API is running!"
else
  echo_warning "API doesn't seem to be running. Make sure to start it first with ./brev_deploy.sh"
  echo_info "Continuing with configuration anyway..."
fi

# Prompt for HANA connection details
read -p "Enter SAP HANA Cloud host (e.g., hana-instance.hanacloud.ondemand.com): " HANA_HOST
read -p "Enter SAP HANA Cloud port [443]: " HANA_PORT
HANA_PORT=${HANA_PORT:-443}
read -p "Enter SAP HANA Cloud username: " HANA_USER
read -sp "Enter SAP HANA Cloud password: " HANA_PASSWORD
echo ""
read -p "Enter default table name for vector store [EMBEDDINGS]: " DEFAULT_TABLE
DEFAULT_TABLE=${DEFAULT_TABLE:-EMBEDDINGS}

# Create .env file
echo_info "Creating .env file with provided credentials..."
cat > .env << EOF
HANA_HOST=$HANA_HOST
HANA_PORT=$HANA_PORT
HANA_USER=$HANA_USER
HANA_PASSWORD=$HANA_PASSWORD
DEFAULT_TABLE_NAME=$DEFAULT_TABLE
TEST_MODE=false
ENABLE_CORS=true
LOG_LEVEL=INFO
EOF

echo_success "Created .env file with SAP HANA Cloud credentials"

# Create connection.json file (alternative configuration)
echo_info "Creating connection.json file (alternative configuration)..."
mkdir -p config
cat > config/connection.json << EOF
{
  "host": "$HANA_HOST",
  "port": $HANA_PORT,
  "user": "$HANA_USER",
  "password": "$HANA_PASSWORD",
  "encrypt": true,
  "sslValidateCertificate": true,
  "connectTimeout": 30000
}
EOF

echo_success "Created config/connection.json file"

# Test the connection
echo_info "Testing connection to SAP HANA Cloud..."
python -c "
import sys
try:
    from hdbcli import dbapi
    print('Connecting to SAP HANA Cloud...')
    conn = dbapi.connect(
        address='$HANA_HOST',
        port=$HANA_PORT,
        user='$HANA_USER',
        password='$HANA_PASSWORD',
        encrypt=True,
        sslValidateCertificate=True
    )
    print('Connection successful!')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM DUMMY')
    result = cursor.fetchone()
    print(f'Dummy query result: {result}')
    print('Testing completed successfully')
    conn.close()
    sys.exit(0)
except Exception as e:
    print(f'Error: {str(e)}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
  echo_success "Connection to SAP HANA Cloud successful!"
else
  echo_error "Failed to connect to SAP HANA Cloud. Please check credentials and try again."
  echo_warning "The configuration files have been created but the connection test failed."
fi

# Provide instructions to restart the API
echo_info ""
echo_info "Configuration completed. To use these settings:"
echo_info ""
echo_info "1. Stop the current API instance (if running):"
echo_info "   pkill -f \"uvicorn app:app\""
echo_info ""
echo_info "2. Restart the API with the new configuration:"
echo_info "   cd $(pwd)"
echo_info "   nohup ./brev_deploy.sh > logs/api_$(date +%Y%m%d_%H%M%S).log 2>&1 &"
echo_info ""
echo_info "3. Verify the API is using the real SAP HANA connection:"
echo_info "   curl http://localhost:8000/health/database"
echo_info ""
echo_info "Configuration logs saved to: $LOG_FILE"