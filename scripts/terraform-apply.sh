#!/bin/bash
# Script to initialize and apply Terraform configurations

set -e

# Default values
ENVIRONMENT="staging"
ACTION="plan"
AUTO_APPROVE=false

# Function to display usage information
usage() {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  -e, --environment <env>  Specify environment: staging or production (default: staging)"
  echo "  -a, --action <action>    Specify action: plan, apply, destroy (default: plan)"
  echo "  -y, --yes                Auto-approve apply or destroy actions"
  echo "  -h, --help               Display this help message"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--environment)
      ENVIRONMENT="$2"
      if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
        echo "Error: Environment must be 'staging' or 'production'"
        exit 1
      fi
      shift 2
      ;;
    -a|--action)
      ACTION="$2"
      if [[ ! "$ACTION" =~ ^(plan|apply|destroy)$ ]]; then
        echo "Error: Action must be 'plan', 'apply', or 'destroy'"
        exit 1
      fi
      shift 2
      ;;
    -y|--yes)
      AUTO_APPROVE=true
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Error: Unknown option $1"
      usage
      ;;
  esac
done

# Set working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TF_DIR="${PROJECT_ROOT}/terraform/environments/${ENVIRONMENT}"

# Check if Terraform directory exists
if [ ! -d "$TF_DIR" ]; then
  echo "Error: Terraform environment directory not found: $TF_DIR"
  exit 1
fi

# Navigate to Terraform directory
cd "$TF_DIR"
echo "Working in directory: $(pwd)"

# Initialize Terraform
echo "Initializing Terraform..."
terraform init

# Run format check
echo "Running format check..."
terraform fmt -check -recursive

# Run validation
echo "Validating Terraform configuration..."
terraform validate

# Perform requested action
case "$ACTION" in
  plan)
    echo "Planning Terraform changes for $ENVIRONMENT environment..."
    terraform plan -out=tfplan
    ;;
  apply)
    echo "Applying Terraform changes to $ENVIRONMENT environment..."
    if [ "$AUTO_APPROVE" = true ]; then
      terraform apply -auto-approve
    else
      terraform plan -out=tfplan
      echo ""
      read -p "Do you want to apply these changes? (y/n): " confirm
      if [[ "$confirm" =~ ^[Yy]$ ]]; then
        terraform apply tfplan
      else
        echo "Apply cancelled."
        exit 0
      fi
    fi
    ;;
  destroy)
    echo "CAUTION: Destroying $ENVIRONMENT environment infrastructure..."
    if [ "$AUTO_APPROVE" = true ]; then
      terraform destroy -auto-approve
    else
      terraform plan -destroy -out=tfplan
      echo ""
      echo "WARNING: This will destroy all resources in the $ENVIRONMENT environment!"
      read -p "Are you absolutely sure you want to destroy? (type 'yes' to confirm): " confirm
      if [ "$confirm" = "yes" ]; then
        terraform apply tfplan
      else
        echo "Destroy cancelled."
        exit 0
      fi
    fi
    ;;
esac

echo "Terraform operation completed successfully!"