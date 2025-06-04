# Deployment Guide

This guide covers how to deploy the SAP HANA Cloud LangChain integration to different environments.

## Table of Contents

1. [NVIDIA LaunchPad Deployment](#nvidia-launchpad-deployment)
2. [Vercel Deployment](#vercel-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Local Development Setup](#local-development-setup)

## NVIDIA LaunchPad Deployment

NVIDIA LaunchPad provides a GPU-accelerated environment for running the SAP HANA Cloud LangChain integration.

### Prerequisites

- NVIDIA LaunchPad account with NGC API key
- Docker installed
- NGC CLI installed

### Building the Container

1. **Build the API container**:

```bash
cd api
docker build -t nvcr.io/your-org/langchain-hana-gpu:latest -f Dockerfile.ngc .
```

2. **Build the frontend container**:

```bash
cd frontend
docker build -t nvcr.io/your-org/langchain-hana-frontend:latest .
```

### Pushing to NGC Registry

1. **Log in to the NGC Registry**:

```bash
# Configure NGC CLI with your API key
ngc config set

# Log in to Docker with NGC credentials
docker login nvcr.io
```

2. **Push the containers**:

```bash
docker push nvcr.io/your-org/langchain-hana-gpu:latest
docker push nvcr.io/your-org/langchain-hana-frontend:latest
```

### Deploying on LaunchPad

1. **Modify the NVIDIA LaunchPad configuration**:

Edit `nvidia-launchable.yaml` to update your organization and container image names.

2. **Publish to LaunchPad**:

```bash
ngc registry resource publish --file nvidia-launchable.yaml
```

3. **Run on LaunchPad**:

Navigate to your LaunchPad dashboard, select the published resource, and launch it.

## Vercel Deployment

Vercel provides a simple way to deploy the frontend and serverless API.

### Prerequisites

- Vercel account
- Vercel CLI installed
- Git repository set up

### Deployment Steps

1. **Install Vercel CLI**:

```bash
npm install -g vercel
```

2. **Login to Vercel**:

```bash
vercel login
```

3. **Deploy the project**:

```bash
# From the project root
vercel
```

4. **Set up environment variables**:

You'll need to configure the following environment variables in the Vercel dashboard:
- `HANA_HOST`: Your SAP HANA Cloud host
- `HANA_PORT`: Your SAP HANA Cloud port (usually 443)
- `HANA_USER`: Your SAP HANA Cloud username
- `HANA_PASSWORD`: Your SAP HANA Cloud password
- `ENABLE_ERROR_CONTEXT`: Set to "true" to enable context-aware error messages
- `CACHE_VECTOR_REDUCTION`: Set to "true" to enable vector reduction caching
- `ENABLE_ADVANCED_CLUSTERING`: Set to "true" to enable advanced clustering algorithms

5. **Set up production deployment**:

```bash
vercel --prod
```

### Vercel Project Configuration

The `vercel.json` file contains the necessary configuration for Vercel deployment:
- Routes configuration for API and frontend
- Build settings
- Environment variables

## Docker Deployment

You can also deploy the application using Docker Compose.

### Prerequisites

- Docker and Docker Compose installed
- Access to a GPU for optimal performance (optional)

### Deployment Steps

1. **Set up environment variables**:

Create a `.env` file with your SAP HANA Cloud credentials:

```bash
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password
GPU_ENABLED=true
USE_TENSORRT=true
ENABLE_ERROR_CONTEXT=true
CACHE_VECTOR_REDUCTION=true
ENABLE_ADVANCED_CLUSTERING=true
```

2. **Start the containers**:

```bash
# Standard deployment
docker-compose up -d

# GPU-enabled deployment
docker-compose -f docker-compose.gpu.yml up -d
```

3. **Access the application**:

- Frontend: http://localhost:3000
- API: http://localhost:8000

## Local Development Setup

For local development, you can run the API and frontend separately.

### API Setup

1. **Set up a Python virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:

```bash
cd api
pip install -r requirements.txt
```

3. **Run the API server**:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Install dependencies**:

```bash
cd frontend
npm install
```

2. **Run the development server**:

```bash
npm start
```

3. **Access the frontend**:

Open http://localhost:3000 in your browser.

## Error Handling Configuration

The new error handling system can be customized through environment variables:

- `ENABLE_ERROR_CONTEXT`: Set to "true" to enable context-aware error messages
- `ERROR_DETAIL_LEVEL`: Set to "minimal", "standard", or "verbose" to control error detail level
- `INCLUDE_SUGGESTIONS`: Set to "true" to include suggested actions in error messages

## Vector Visualization Configuration

The vector visualization can be customized through environment variables:

- `CACHE_VECTOR_REDUCTION`: Set to "true" to enable caching of reduced vectors
- `MAX_VECTOR_CACHE_SIZE`: Maximum size of the vector cache in MB
- `ENABLE_ADVANCED_CLUSTERING`: Set to "true" to enable advanced clustering algorithms
- `DEFAULT_CLUSTERING_ALGORITHM`: Set to "kmeans", "dbscan", or "hdbscan"
- `DEFAULT_DIMENSIONALITY_REDUCTION`: Set to "tsne", "umap", or "pca"