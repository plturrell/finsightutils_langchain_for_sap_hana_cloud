# Deployment Guide

This guide covers how to deploy the SAP HANA Cloud LangChain integration to different environments, with a focus on the new split architecture that separates the backend FastAPI and frontend components.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Docker Deployment](#docker-deployment)
   - [Backend API Deployment](#backend-api-deployment)
   - [Frontend Deployment](#frontend-deployment)
   - [Full Stack Deployment](#full-stack-deployment)
3. [Vercel Deployment](#vercel-deployment)
   - [Frontend on Vercel](#frontend-on-vercel)
   - [Backend API Deployment](#backend-api-on-vercel)
4. [NVIDIA LaunchPad Deployment](#nvidia-launchpad-deployment)
5. [Local Development Setup](#local-development-setup)

## Architecture Overview

The SAP HANA Cloud LangChain integration now uses a split architecture:

1. **Backend FastAPI Service**: Handles connections to SAP HANA Cloud, embedding generation, vector operations, and GPU acceleration
2. **Frontend Service**: Provides the user interface, visualization, and user interaction

This split architecture allows for more flexible deployment options, such as:
- Frontend on Vercel with backend on Docker
- Both services on Docker
- Frontend on static hosting with backend on Kubernetes

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

The new split architecture is optimized for deployment on Vercel. The frontend can be deployed directly to Vercel, while the backend API can be deployed either to Vercel Serverless Functions (with limitations) or to another platform like Docker or Kubernetes.

### Frontend on Vercel

Deploying the frontend to Vercel is the simplest approach:

#### Prerequisites

- Vercel account
- GitHub repository with your code
- Backend API deployed and accessible

#### Deployment Steps

1. **Push your code to GitHub**:

Make sure your project is in a GitHub repository.

2. **Create a new project on Vercel**:

- Go to https://vercel.com/new
- Import your GitHub repository
- Configure the project:
  - **Framework Preset**: Other
  - **Root Directory**: `frontend`
  - **Build Command**: `./vercel-build.sh`
  - **Output Directory**: `build`

3. **Set up environment variables**:

Add the following environment variables in the Vercel dashboard:
- `BACKEND_URL`: URL of your deployed backend API
- `VITE_APP_VERSION`: Application version (optional)
- `VITE_ENABLE_ANALYTICS`: Set to "true" to enable analytics (optional)

4. **Deploy the project**:

Click "Deploy" to start the deployment process. Vercel will build and deploy your frontend.

5. **Access your deployed frontend**:

Once deployed, Vercel will provide a URL to access your frontend.

### Backend API on Vercel

While the backend API is better suited for Docker or Kubernetes deployment due to GPU acceleration requirements, you can deploy a simplified version to Vercel Serverless Functions:

#### Prerequisites

- Vercel account
- GitHub repository with your code
- SAP HANA Cloud credentials

#### Deployment Steps

1. **Modify `api/vercel.json`**:

Make sure your Vercel configuration is set up correctly:

```json
{
  "version": 2,
  "functions": {
    "api/core/main.py": {
      "memory": 1024,
      "maxDuration": 10
    }
  },
  "routes": [
    { "src": "/api/(.*)", "dest": "/api/core/main.py" }
  ]
}
```

2. **Set up environment variables**:

Add the following environment variables in the Vercel dashboard:
- `HANA_HOST`: Your SAP HANA Cloud host
- `HANA_PORT`: Your SAP HANA Cloud port (usually 443)
- `HANA_USER`: Your SAP HANA Cloud username
- `HANA_PASSWORD`: Your SAP HANA Cloud password
- `JWT_SECRET`: Secret for JWT authentication
- `ENABLE_CORS`: Set to "true" to enable CORS support
- `CORS_ORIGINS`: Comma-separated list of allowed origins

3. **Deploy the API**:

```bash
# From the api directory
vercel
```

4. **Deploy to production**:

```bash
vercel --prod
```

### Limitations of Backend API on Vercel

The backend API on Vercel has some limitations compared to Docker deployment:

- No GPU acceleration support
- Limited execution time (10 seconds maximum)
- Limited memory (1GB maximum)
- No persistent storage

For production use with large embedding operations or GPU acceleration, we recommend deploying the backend API using Docker or Kubernetes.

### Vercel Project Configuration

The `vercel.json` files in both the frontend and api directories contain the necessary configuration for Vercel deployment:
- Routes configuration
- Build settings
- Environment variables
- Serverless function configuration

## Docker Deployment

With the split architecture, you can deploy the backend API and frontend separately or together using Docker Compose.

### Prerequisites

- Docker and Docker Compose installed
- Access to a GPU for optimal performance (optional)
- SAP HANA Cloud credentials

### Backend API Deployment

The backend API can be deployed using the dedicated `docker-compose.api.yml` file:

1. **Set up environment variables**:

Create a `.env` file with your SAP HANA Cloud credentials:

```bash
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password
JWT_SECRET=your-secure-jwt-secret
ENABLE_CORS=true
CORS_ORIGINS=http://localhost:3000,https://your-frontend.vercel.app
```

2. **Start the API container**:

```bash
# Standard CPU deployment
docker-compose -f docker-compose.api.yml up -d

# GPU-enabled deployment
docker-compose -f docker-compose.api.yml -f docker-compose.gpu.yml up -d
```

3. **Access the API**:

The API will be available at http://localhost:8000 with OpenAPI documentation at http://localhost:8000/docs

### Frontend Deployment

The frontend can be deployed using the dedicated `docker-compose.frontend.yml` file:

1. **Set up environment variables**:

Add frontend environment variables to your `.env` file:

```bash
BACKEND_URL=http://localhost:8000  # or your deployed API URL
```

2. **Start the frontend container**:

```bash
docker-compose -f docker-compose.frontend.yml up -d
```

3. **Access the frontend**:

The frontend will be available at http://localhost:3000

### Full Stack Deployment

To deploy both the backend API and frontend together:

```bash
# Deploy both services (CPU)
docker-compose -f docker-compose.api.yml -f docker-compose.frontend.yml up -d

# Deploy with GPU support for the backend
docker-compose -f docker-compose.api.yml -f docker-compose.gpu.yml -f docker-compose.frontend.yml up -d
```

### Docker Compose Files

The project includes several Docker Compose files for different deployment scenarios:

- `docker-compose.api.yml`: Backend API service only
- `docker-compose.frontend.yml`: Frontend service only
- `docker-compose.gpu.yml`: GPU acceleration extension for the API
- `docker-compose.nvidia.yml`: NVIDIA-specific deployment for specialized environments

To use these files together, you can specify multiple `-f` flags with docker-compose.

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
uvicorn api.core.main:app --reload --host 0.0.0.0 --port 8000
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