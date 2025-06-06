# CI/CD Guide for langchain-integration-for-sap-hana-cloud

This document provides an overview of the Continuous Integration and Continuous Deployment (CI/CD) setup for this project with a focus on our streamlined deployment architecture.

## Overview

The CI/CD pipeline automates the following processes:

1. **Continuous Integration (CI)**:
   - Code linting and static analysis with ruff
   - Type checking with mypy
   - Security scanning with bandit and safety
   - Unit tests with pytest and coverage reporting
   - Docker image building for NVIDIA GPU backend

2. **Continuous Deployment (CD)**:
   - Backend deployment to Kubernetes with NVIDIA GPU support
   - Frontend deployment to Vercel
   - Environment-specific configuration management
   - Secure secrets handling and injection
   - Deployment verification with automated tests

## Streamlined Deployment Architecture

Our deployment architecture has been streamlined to focus on two main components:

1. **Backend**: FastAPI application running on Kubernetes with NVIDIA GPU support
2. **Frontend**: React application deployed on Vercel

This simplification provides:
- Clearer separation of concerns
- More reliable deployments
- Better resource utilization
- Simplified maintenance

## Local Development Setup

### Prerequisites

- Python 3.10+ installed
- Git installed
- Access to the GitHub repository
- Docker and Docker Compose installed for local testing

### Setting Up Local Development Environment

Run the setup script to install pre-commit hooks and development dependencies:

```bash
./scripts/setup_local_dev.sh
```

For local testing with Docker Compose:

```bash
# Run the backend with NVIDIA GPU support
docker-compose -f docker-compose.backend.yml up

# In a separate terminal, start the frontend development server
cd frontend
npm install
npm start
```

## CI/CD Pipeline

### Pipeline Triggers

The CI/CD pipeline is triggered by:
- **Push to `main` branch**: Deploys to staging environment
- **Push of version tags (e.g., `v1.0.0`)**: Deploys to production environment
- **Pull requests to `main` branch**: Runs tests and builds without deploying
- **Manual trigger via GitHub Actions**: Can specify the target environment

### CI Jobs

1. **Lint and Type Check**:
   - Runs code linting with ruff
   - Performs type checking with mypy

2. **Security Scan**:
   - Runs security analysis with bandit
   - Checks dependencies for vulnerabilities with safety

3. **Test**:
   - Runs unit tests with pytest
   - Generates code coverage reports
   - Uploads coverage to Codecov

4. **Build Backend**:
   - Builds Docker image for backend with NVIDIA GPU support
   - Pushes to GitHub Container Registry with appropriate tags

### CD Jobs

1. **Deploy Frontend**:
   - Builds and deploys the frontend application to Vercel
   - Configures environment-specific settings (API endpoints, etc.)

2. **Deploy Backend**:
   - Deploys the backend to Kubernetes with NVIDIA GPU support
   - Applies environment-specific configurations
   - Sets up autoscaling for production environment
   - Runs verification tests to ensure deployment success

3. **Notify**:
   - Sends notifications via Slack about deployment status

## Deployment Environments

The deployment workflow supports two environments:

### Staging Environment

- Purpose: Testing and validation before production release
- Triggered by: Pushes to the main branch or manual workflow dispatches
- Configuration:
  - Fewer replicas (2)
  - Less resource allocation
  - Basic smoke testing
  - Preview Vercel deployment

### Production Environment

- Purpose: Live, user-facing deployment
- Triggered by: Release tags (v*) or manual workflow dispatches with production selection
- Configuration:
  - Higher replica count (3+)
  - Auto-scaling enabled via HPA
  - More resource allocation
  - Comprehensive verification testing
  - Production Vercel deployment

## Infrastructure as Code with Terraform

The project now uses Terraform to manage all infrastructure components. This approach provides several benefits:

1. **Declarative Configuration**: Infrastructure defined as code
2. **Version Control**: Infrastructure changes tracked alongside application code
3. **Consistency**: Ensures consistent environments across deployments
4. **Automation**: Integrated with CI/CD for automated infrastructure updates

### Terraform Workflow

The Terraform workflow is integrated into the CI/CD pipeline:

1. **Pull Requests**:
   - Terraform plan is generated and added as a PR comment
   - Shows planned infrastructure changes for review

2. **Merges to Branches**:
   - `staging` branch: Applies changes to staging environment
   - `main` branch: Applies changes to production environment

### Terraform Components

The Terraform configuration manages the following infrastructure:

1. **Kubernetes Resources**:
   - Namespaces for environment isolation
   - Deployments with GPU support
   - Services for API access
   - ConfigMaps and Secrets for configuration
   - HPA for auto-scaling

2. **Monitoring Stack**:
   - Prometheus for metrics collection
   - Grafana for visualization
   - Pre-configured dashboards

For more detailed information, see [Infrastructure as Code](./infrastructure_as_code.md) documentation.

## Kubernetes Configuration

The backend is deployed to Kubernetes using Terraform with the following components:

1. **Namespace**: Isolated environment for each deployment stage
2. **ConfigMap**: Environment-specific, non-sensitive configuration
3. **Secrets**: Sensitive information like database credentials and API keys
4. **Deployment**: Pod specifications including GPU requirements
5. **Service**: Load balancer for external access
6. **HPA (Production only)**: Auto-scaling based on CPU and memory usage

## Secrets Management

The pipeline requires several secrets to be configured in your GitHub repository:

### Kubernetes and Database Secrets
- `KUBE_CONFIG`: Kubernetes configuration for cluster access
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`: Database credentials
- `API_KEY`: API authentication key

### Vercel Deployment Secrets
- `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID_FRONTEND`: Vercel deployment credentials
- `STAGING_BACKEND_URL`, `PRODUCTION_BACKEND_URL`: Backend API endpoints for frontend configuration

### Terraform Secrets
- `TF_API_TOKEN`: HashiCorp Terraform Cloud API token for remote state management
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`: AWS credentials for S3 backend (if using S3 for state storage)
- `TF_ENCRYPTION_KEY`: Encryption key for Terraform state (if using encryption)

### Notification Secrets
- `SLACK_WEBHOOK_URL`: Webhook for deployment notifications

## Versioning

This project follows semantic versioning (SEMVER):
- **MAJOR**: Incompatible API changes
- **MINOR**: Added functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

To create a new release:

```bash
# Create a new tag locally
git tag -a v1.0.0 -m "Release v1.0.0"

# Push the tag to trigger deployment
git push origin v1.0.0
```

## Verification Process

### Staging Verification

Performs basic smoke tests to ensure core functionality:
- Health checks
- API endpoint availability
- Basic embedding generation

### Production Verification

Performs comprehensive verification:
- Functional testing
- Load testing with concurrent requests
- Performance benchmarking
- Error handling verification
- Results are saved as artifacts for analysis

## High Availability and Scaling

The production deployment includes:

- Multiple replicas (3+) for high availability
- Horizontal Pod Autoscaler (HPA) for automatic scaling based on CPU/memory usage
- Resource requests and limits for optimal node placement
- Node selector for GPU requirements
- Health checks for automatic recovery

## Troubleshooting

### Common Issues

1. **CI Failures**:
   - Check the GitHub Actions logs for details
   - Run linting and tests locally to catch issues before pushing

2. **CD Failures**:
   - Verify secrets and environment variables are correctly set in GitHub
   - Check Kubernetes cluster access and configuration
   - Verify GPU resources are available in the cluster
   - Check the deployment verification logs for specific errors

3. **Local Development Issues**:
   - Run `docker-compose -f docker-compose.backend.yml config` to validate configuration
   - Check NVIDIA GPU availability with `nvidia-smi`
   - Verify environment variables are set correctly

### Getting Help

If you encounter issues with the CI/CD pipeline:
1. Check the GitHub Actions logs
2. Review this documentation
3. Check the deployment verification results
4. Create an issue in the GitHub repository