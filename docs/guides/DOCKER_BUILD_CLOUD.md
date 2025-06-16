# Docker Build Cloud Integration Guide

This document explains how to use Docker Build Cloud with GitHub Actions for building the SAP HANA Cloud LangChain Nvidia Blueprint images.

## Overview

Docker Build Cloud accelerates build times by:
- Caching layers across builds and branches
- Offloading build workloads to Docker's cloud infrastructure
- Supporting parallel multi-architecture builds

## Prerequisites

1. A GitHub repository with your Dockerfile(s)
2. Docker Hub account
3. GitHub repository secrets configured (see below)

## Required Secrets

Add these secrets to your GitHub repository (Settings > Secrets and variables > Actions):

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: A Docker Hub access token (not your password)
- `NGC_USERNAME`: Your NVIDIA NGC username (typically `$oauthtoken`)
- `NGC_API_KEY`: Your NVIDIA NGC API key

## How It Works

1. The GitHub Actions workflow triggers on pushes to main/master, pull requests, or manual dispatches
2. The workflow uses Docker's cloud infrastructure instead of GitHub runners
3. Builds are cached in Docker's cloud for faster subsequent builds
4. On main/master branch pushes, images are automatically pushed to Docker Hub

## Local Development

For local development with Docker Build Cloud:

```bash
# Create a cloud builder
docker buildx create --name cloud-builder --driver cloud

# Use the cloud builder
docker buildx use cloud-builder

# Build your image with the cloud builder
docker buildx build --file docker/Dockerfile.nvidia --tag your-username/langchain-hana-nvidia:latest .
```

## Troubleshooting

If you encounter "unexpected EOF" or similar network errors during local builds, using Docker Build Cloud can help by:

1. Offloading network-intensive operations to Docker's cloud
2. Providing more reliable builds with better caching
3. Avoiding local environment issues

## Further Resources

- [Docker Build Cloud Documentation](https://docs.docker.com/build/cloud/)
- [GitHub Actions with Docker Build Cloud](https://docs.docker.com/build/cloud/github-actions/)
