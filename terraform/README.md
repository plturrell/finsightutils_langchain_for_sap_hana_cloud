# Terraform Infrastructure as Code

This directory contains the Terraform configuration for deploying the LangChain HANA Cloud Integration application to Kubernetes.

## Directory Structure

```
terraform/
├── modules/                  # Reusable Terraform modules
│   ├── kubernetes/           # Kubernetes resources (deployment, service, etc.)
│   └── monitoring/           # Prometheus and Grafana stack
├── environments/             # Environment-specific configurations
│   ├── staging/              # Staging environment
│   └── production/           # Production environment
├── main.tf                   # Main Terraform configuration
├── variables.tf              # Common variables
└── README.md                 # This file
```

## Prerequisites

- Terraform v1.0.0 or later
- kubectl configured with access to your Kubernetes cluster
- Helm v3.0.0 or later (for monitoring stack)

## Getting Started

### Manual Deployment

1. Navigate to the environment directory:
   ```bash
   cd terraform/environments/staging   # or production
   ```

2. Initialize Terraform:
   ```bash
   terraform init
   ```

3. Plan the changes:
   ```bash
   terraform plan -out=tfplan
   ```

4. Apply the changes:
   ```bash
   terraform apply tfplan
   ```

### Using the Helper Script

We provide a helper script for easier deployment:

```bash
# Plan changes for staging (default)
./scripts/terraform_apply.sh

# Apply changes to staging with confirmation
./scripts/terraform_apply.sh -a apply

# Apply changes to production with auto-approval
./scripts/terraform_apply.sh -e production -a apply -y

# Show help
./scripts/terraform_apply.sh -h
```

## Configuration

### Environment Variables

You can use environment variables to configure the deployment:

- `TF_VAR_kube_config_path`: Path to the Kubernetes config file
- `TF_VAR_kube_context`: Kubernetes context to use
- `TF_VAR_image_repository`: Docker image repository for the API
- `TF_VAR_image_tag`: Docker image tag for the API

### Sensitive Variables

For sensitive information, use environment variables or a `.tfvars` file (do not commit this file). We've provided a template (`credentials.tfvars.example`) that you should copy and fill with your actual values:

```bash
# Copy the example file
cp credentials.tfvars.example credentials.tfvars

# Edit the file with your actual credentials
vi credentials.tfvars
```

The file should contain your SAP HANA Cloud and DataSphere credentials:

```hcl
# credentials.tfvars
hana_credentials = {
  host     = "your-hana-hostname.hanacloud.ondemand.com"
  port     = "443"
  user     = "your-username"
  password = "your-password"
}

datasphere_credentials = {
  client_id     = "your-client-id"
  client_secret = "your-client-secret"
  auth_url      = "https://your-tenant.authentication.eu10.hana.ondemand.com/oauth/authorize"
  token_url     = "https://your-tenant.authentication.eu10.hana.ondemand.com/oauth/token"
  api_url       = "https://your-tenant.eu10.hcs.cloud.sap/api/v1"
}

grafana_admin_password = "your-secure-password"
```

Then apply with:
```bash
terraform apply -var-file=credentials.tfvars
```

IMPORTANT: Always add `*.tfvars` files to your `.gitignore` to prevent accidental commit of credentials.

### Managing and Testing Credentials

Before applying Terraform configurations, you can verify your SAP HANA Cloud and DataSphere credentials using the provided scripts:

#### Generate .env file from Terraform variables

Convert your Terraform credentials to environment variables:

```bash
# Create .env file from credentials.tfvars
./scripts/generate_env_from_tfvars.py

# Specify custom paths
./scripts/generate_env_from_tfvars.py --input /path/to/credentials.tfvars --output /path/to/.env

# Force overwrite existing .env file
./scripts/generate_env_from_tfvars.py --force
```

#### Test Credentials Locally

Test connections from your local machine:

```bash
# Test using local Python installation
export HANA_HOST=your-hana-hostname.hanacloud.ondemand.com
export HANA_PORT=443
export HANA_USER=your-username
export HANA_PASSWORD=your-password
export DATASPHERE_CLIENT_ID=your-client-id
export DATASPHERE_CLIENT_SECRET=your-client-secret
export DATASPHERE_AUTH_URL=https://your-tenant.authentication.eu10.hana.ondemand.com/oauth/authorize
export DATASPHERE_TOKEN_URL=https://your-tenant.authentication.eu10.hana.ondemand.com/oauth/token
export DATASPHERE_API_URL=https://your-tenant.eu10.hcs.cloud.sap/api/v1

# Run the test script
./scripts/verify_credentials.sh

# Alternatively, use Docker to avoid installing dependencies
./scripts/test_connections_docker.sh

# Test only HANA connection
./scripts/verify_credentials.sh --hana-only

# Test only DataSphere connection
./scripts/verify_credentials.sh --datasphere-only

# Use environment file with Docker
./scripts/test_connections_docker.sh --env-file .env
```

#### Test Credentials in Kubernetes

Test connections from within your Kubernetes cluster:

```bash
# Generate .env file if you haven't already
./scripts/generate_env_from_tfvars.py

# Test connections using a temporary pod in the default namespace
./scripts/k8s_test_connections.sh --env-file .env

# Test only HANA connection in a specific namespace
./scripts/k8s_test_connections.sh --namespace langchain-hana-staging --hana-only --env-file .env

# Keep resources after testing (for debugging)
./scripts/k8s_test_connections.sh --keep-resources --env-file .env
```

These tests will verify that your credentials are valid and can successfully connect to the services from both your local environment and the Kubernetes cluster.

## Remote State

The current configuration uses local state. For production use, it's recommended to configure remote state storage using one of:

1. **Terraform Cloud**:
   ```hcl
   terraform {
     cloud {
       organization = "your-org"
       workspaces {
         name = "langchain-hana-production"
       }
     }
   }
   ```

2. **S3 Backend**:
   ```hcl
   terraform {
     backend "s3" {
       bucket = "terraform-state-langchain-hana"
       key    = "terraform.tfstate"
       region = "us-east-1"
     }
   }
   ```

## Secrets Management

The current configuration uses Kubernetes Secrets with placeholder values. In a production environment, these should be populated using a secure method such as:

1. Using a CI/CD pipeline to inject secrets from a secure storage
2. Using External Secrets Operator to sync secrets from external sources
3. Using HashiCorp Vault for secret management

## CI/CD Integration

This Terraform configuration is integrated with our GitHub Actions workflow. See `.github/workflows/terraform.yml` for details.

The workflow:
1. Runs `terraform plan` on pull requests
2. Posts plan results as PR comments
3. Applies changes automatically when merging to staging/main branches

## Monitoring

The monitoring module deploys:

- **Prometheus**: For metrics collection
- **Grafana**: For visualization and dashboards

Prometheus is configured to scrape metrics from the API service, and Grafana includes a pre-configured dashboard for monitoring the application.

## Cleanup

To destroy the infrastructure:

```bash
# Using the helper script
./scripts/terraform_apply.sh -a destroy -e staging

# Or manually
cd terraform/environments/staging
terraform destroy
```

CAUTION: This will remove all resources managed by Terraform.