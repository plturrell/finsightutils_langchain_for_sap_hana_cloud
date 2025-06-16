#!/bin/bash
# Script to securely set up SAP HANA Cloud credentials for langchain integration

set -e

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

echo_info "SAP HANA Cloud Credentials Setup"
echo_info "================================"
echo_info "This script will help you set up your SAP HANA Cloud credentials"
echo_info "The credentials will be stored in a secure environment file"
echo_info ""

# Check if .env file already exists
ENV_FILE=".env"

if [ -f "$ENV_FILE" ]; then
  echo_warning "An existing .env file was found."
  read -p "Do you want to override it? (y/n): " override
  if [[ $override != "y" && $override != "Y" ]]; then
    echo_info "Keeping existing configuration."
    exit 0
  fi
fi

# Copy template if it exists
if [ -f "config/.env.template" ]; then
  cp config/.env.template $ENV_FILE
  echo_info "Created .env file from template."
else
  touch $ENV_FILE
  echo_info "Created empty .env file."
fi

# Prompt for SAP HANA credentials
echo_info "Please enter your SAP HANA Cloud credentials:"
read -p "SAP HANA Host (e.g., your-hana-host.hanacloud.ondemand.com): " hana_host
read -p "SAP HANA Port [443]: " hana_port
hana_port=${hana_port:-443}
read -p "SAP HANA User [SYSTEM]: " hana_user
hana_user=${hana_user:-SYSTEM}
read -sp "SAP HANA Password: " hana_password
echo ""
read -p "Default Table Name [EMBEDDINGS]: " table_name
table_name=${table_name:-EMBEDDINGS}

# Update the .env file
if grep -q "HANA_HOST" $ENV_FILE; then
  sed -i.bak "s/^HANA_HOST=.*/HANA_HOST=$hana_host/" $ENV_FILE
else
  echo "HANA_HOST=$hana_host" >> $ENV_FILE
fi

if grep -q "HANA_PORT" $ENV_FILE; then
  sed -i.bak "s/^HANA_PORT=.*/HANA_PORT=$hana_port/" $ENV_FILE
else
  echo "HANA_PORT=$hana_port" >> $ENV_FILE
fi

if grep -q "HANA_USER" $ENV_FILE; then
  sed -i.bak "s/^HANA_USER=.*/HANA_USER=$hana_user/" $ENV_FILE
else
  echo "HANA_USER=$hana_user" >> $ENV_FILE
fi

if grep -q "HANA_PASSWORD" $ENV_FILE; then
  sed -i.bak "s/^HANA_PASSWORD=.*/HANA_PASSWORD=$hana_password/" $ENV_FILE
else
  echo "HANA_PASSWORD=$hana_password" >> $ENV_FILE
fi

if grep -q "DEFAULT_TABLE_NAME" $ENV_FILE; then
  sed -i.bak "s/^DEFAULT_TABLE_NAME=.*/DEFAULT_TABLE_NAME=$table_name/" $ENV_FILE
else
  echo "DEFAULT_TABLE_NAME=$table_name" >> $ENV_FILE
fi

# Set TEST_MODE to false
if grep -q "TEST_MODE" $ENV_FILE; then
  sed -i.bak "s/^TEST_MODE=.*/TEST_MODE=false/" $ENV_FILE
else
  echo "TEST_MODE=false" >> $ENV_FILE
fi

# Remove backup file if it exists
if [ -f "$ENV_FILE.bak" ]; then
  rm "$ENV_FILE.bak"
fi

# Set proper permissions
chmod 600 $ENV_FILE

echo_success "SAP HANA Cloud credentials have been set up successfully!"
echo_info "Credentials are stored in $ENV_FILE with restricted permissions."
echo_info ""
echo_info "To use these credentials with docker-compose, run:"
echo_info "docker-compose --env-file $ENV_FILE up -d"
echo_info ""
echo_info "Or to load them into your current environment, run:"
echo_info "set -a; source $ENV_FILE; set +a"