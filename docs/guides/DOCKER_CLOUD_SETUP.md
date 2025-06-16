# Docker Build Cloud Setup Guide for SAP HANA Langchain Integration

This guide explains how to set up automated builds on Docker Hub for the `finsightintelligence/langchainsaphana` repository.

## Prerequisites

1. A Docker Hub account with access to the `finsightintelligence` organization
2. The repository code pushed to GitHub
3. Administrator access to both GitHub repository and Docker Hub organization

## Step 1: Connect GitHub to Docker Hub

1. Log in to [Docker Hub](https://hub.docker.com)
2. Navigate to the `finsightintelligence` organization
3. Click on "Create repository" or go to an existing `langchainsaphana` repository
4. In the repository settings, click on "Builds" tab
5. Click "Link to GitHub" (or equivalent option to set up automated builds)
6. Authenticate with GitHub and authorize Docker Hub

## Step 2: Configure Automated Builds

1. Select the GitHub repository containing this code
2. Set the main branch to build from (typically `main` or `master`)
3. Under "Build Rules," set up two build configurations:
   - CPU Build:
     - Source Type: `Branch`
     - Source: `main` (or your primary branch)
     - Docker Tag: `cpu-latest`
     - Dockerfile Location: `Dockerfile.cpu`
     - Build Context: `/` (root of repository)
     - Add build arguments: `FORCE_CPU=1`
   
   - GPU Build:
     - Source Type: `Branch`
     - Source: `main` (or your primary branch)
     - Docker Tag: `gpu-latest`
     - Dockerfile Location: `Dockerfile`
     - Build Context: `/` (root of repository)
     - Add build arguments: `INSTALL_GPU=true`

4. Enable "Autobuild" option
5. Optionally, configure build notifications

## Step 3: Using the Docker Build Cloud Configuration File

Docker Build Cloud can also use the included `docker-build.yml` configuration:

1. In the repository settings on Docker Hub, go to the "Builds" tab
2. Look for an option to use configuration file
3. Specify path to configuration file: `docker-build.yml`
4. Save the settings

## Step 4: Trigger Manual Build

1. On the Docker Hub repository page, go to "Builds" tab
2. Click "Trigger Build" to manually start the build process for testing
3. Monitor the build logs for any issues

## Step 5: Pull and Test Images

After the builds are complete:

```bash
# Pull and run CPU image
docker pull finsightintelligence/langchainsaphana:cpu-latest
docker run -p 8000:8000 finsightintelligence/langchainsaphana:cpu-latest

# Pull and run GPU image (with GPU access)
docker pull finsightintelligence/langchainsaphana:gpu-latest
docker run --gpus all -p 8000:8000 finsightintelligence/langchainsaphana:gpu-latest
```

## Notes on Image Tags

The automated builds will create the following tags:
- `finsightintelligence/langchainsaphana:cpu-latest` - CPU-optimized image
- `finsightintelligence/langchainsaphana:gpu-latest` - GPU-enabled image
- Additional tags with branch name or git tag if configured
- Timestamp-based tags for version tracking

## Troubleshooting

If builds fail on Docker Cloud:
1. Check the build logs for specific errors
2. Verify Dockerfile syntax and paths
3. Ensure all required files are present in the repository
4. Check that build arguments are correctly defined
5. Confirm the GitHub repository permissions are properly set
