# Docker Build System

This document explains the fully automated Docker build system for the SAP HANA LangChain integration project.

## Components

The system consists of:

1. **Universal Docker Build Script**: `docker-build.sh`
2. **GitHub Actions Workflows**:
   - `docker-automated-build.yml`: Automated builds triggered by code changes, schedule, or manual dispatch
   - `docker-image-validation.yml`: Validates Docker images
3. **Docker Compose Configuration**: `docker-compose.integrated.yml` for integrated testing

## 1. Universal Docker Build Script

The `docker-build.sh` script provides a flexible way to build Docker images in both local and cloud environments.

### Features

- Works in both local and cloud environments
- Supports multiple image types (CPU, GPU, minimal, etc.)
- Optional image pushing to Docker Hub
- Built-in image testing

### Usage

```bash
# Build CPU image and push to Docker Hub
./docker-build.sh --type cpu-secure --push

# Build GPU image using Docker Build Cloud
./docker-build.sh --type gpu-secure --cloud

# Build default image and test it
./docker-build.sh --test

# See all options
./docker-build.sh --help
```

## 2. GitHub Actions Workflows

### Automated Build Workflow

The `docker-automated-build.yml` workflow automates the Docker build process.

#### Triggers

- Code changes in relevant files (Dockerfile, API code, etc.)
- Weekly scheduled builds (Sunday at midnight)
- Manual dispatch with build type selection

#### Features

- Builds multiple image types in parallel
- Verifies images are available on Docker Hub
- Creates status comments on pull requests

### Image Validation Workflow

The `docker-image-validation.yml` workflow validates Docker images.

#### Triggers

- Weekly scheduled validation (Monday at midnight)
- Manual dispatch with image tag selection

#### Validation Steps

1. Pulls the specified Docker image
2. Runs the container
3. Tests health endpoints
4. Checks GPU info (for GPU images)
5. Reports any failures

## 3. Docker Compose Configuration

The `docker-compose.integrated.yml` file provides an integrated testing environment.

### Features

- Runs the API and frontend together
- Optional monitoring with Prometheus and Grafana
- Configurable through environment variables

### Usage

```bash
# Start the basic stack
docker-compose -f docker-compose.integrated.yml up -d

# Start with monitoring
docker-compose -f docker-compose.integrated.yml --profile monitoring up -d

# Use custom images
REPOSITORY=myrepo API_TAG=mytag docker-compose -f docker-compose.integrated.yml up -d
```

## CI/CD Integration

The Docker build system integrates with CI/CD pipelines through GitHub Actions.

1. **On code changes**:
   - Builds and pushes new images
   - Validates the images

2. **On schedule**:
   - Rebuilds images weekly
   - Validates existing images

3. **On demand**:
   - Manually trigger builds or validation

## Requirements

- Docker with Buildx extension
- GitHub repository with secrets set up:
  - `DOCKERHUB_USERNAME`: Docker Hub username
  - `DOCKERHUB_TOKEN`: Docker Hub access token

## Troubleshooting

If you encounter issues:

1. **Build failures**:
   - Check Dockerfile for errors
   - Verify Docker Hub credentials
   - Ensure sufficient resources

2. **Validation failures**:
   - Check container logs
   - Verify API endpoints
   - Check network connectivity

3. **Cloud build issues**:
   - Verify Docker Build Cloud is enabled
   - Check Docker Buildx installation
   - Ensure proper authentication