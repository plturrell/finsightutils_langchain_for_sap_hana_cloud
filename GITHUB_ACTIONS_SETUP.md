# GitHub Actions Setup for Docker Security Pipeline

This guide explains how to set up the required secrets and configurations for the automated Docker security pipeline.

## Required GitHub Secrets

The following secrets need to be configured in your GitHub repository to enable the automated security pipeline:

### 1. DOCKERHUB_USERNAME
- **Value**: Your Docker Hub username (e.g., `finsightintelligence`)
- **Description**: Used for logging into Docker Hub to push images

### 2. DOCKERHUB_TOKEN
- **Value**: A Docker Hub Personal Access Token (PAT)
- **Description**: Used for authentication with Docker Hub
- **How to create**:
  1. Log in to [Docker Hub](https://hub.docker.com)
  2. Go to Account Settings → Security
  3. Click "New Access Token"
  4. Give it a name (e.g., "GitHub Actions")
  5. Set appropriate permissions (minimum: "Read, Write, Delete")
  6. Copy the generated token

### 3. DOCKER_BUILD_CLOUD_TOKEN (Optional)
- **Value**: Docker Build Cloud API token
- **Description**: Used to notify Docker Build Cloud of new pushes
- **How to create**:
  1. Log in to [Docker Hub](https://hub.docker.com)
  2. Navigate to your organization settings
  3. Go to Build Cloud → API Tokens
  4. Generate a new token for automated builds

## Setting Up GitHub Secrets

1. Go to your GitHub repository
2. Click on "Settings" tab
3. Select "Secrets and variables" → "Actions" from the sidebar
4. Click "New repository secret"
5. Enter the name and value for each required secret
6. Click "Add secret"

## Manual Workflow Trigger

To manually trigger the workflow:

1. Go to your GitHub repository
2. Click on "Actions" tab
3. Select "Automated Docker Security Pipeline" workflow
4. Click "Run workflow" button
5. Optionally enter a custom version tag (e.g., "1.0.0")
6. Click "Run workflow" to start the build process

## Monitoring Workflow Results

After the workflow runs, you can:

1. View the workflow status and logs in the Actions tab
2. Check the security-reports directory for generated vulnerability reports
3. See security scanning results in the GitHub Security tab
4. Verify the images have been pushed to Docker Hub

## Troubleshooting

If the workflow fails:

1. Check that all required secrets are set correctly
2. Ensure your Docker Hub user has push access to the organization repositories
3. Verify that your repository has appropriate GitHub Actions permissions
4. Review the detailed workflow logs for specific error messages
