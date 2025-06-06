# Deployment Guide

This document outlines the simplified deployment process for the SAP HANA Cloud LangChain Integration. The deployment has been streamlined to focus on two primary components:

1. **Backend**: FastAPI application running on NVIDIA GPU via Docker Compose
2. **Frontend**: React application deployed on Vercel

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and [Docker Compose](https://docs.docker.com/compose/install/) for backend deployment
- [Vercel CLI](https://vercel.com/docs/cli) (optional for local frontend deployment)
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) for GPU support
- Access to an SAP HANA Cloud instance

## Backend Deployment (NVIDIA GPU with Docker Compose)

The backend runs a FastAPI application with GPU acceleration via NVIDIA Docker.

### 1. Configure Environment Variables

Create a `.env` file in the project root based on the example:

```bash
cp backend/.env.example .env
```

Edit the `.env` file to include your SAP HANA Cloud connection details:

```
HANA_HOST=your-hana-cloud-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password
FRONTEND_URL=https://your-frontend-url.vercel.app
```

### 2. Deploy with Docker Compose

Build and start the backend services:

```bash
docker compose -f docker-compose.backend.yml up -d
```

This will start:
- The FastAPI backend with NVIDIA GPU acceleration
- Prometheus for metrics collection
- Grafana for metrics visualization

### 3. Verify Backend Deployment

Check if the services are running:

```bash
docker compose -f docker-compose.backend.yml ps
```

Verify the API is working by accessing the health endpoint:

```bash
curl http://localhost:8000/health/ping
```

The Grafana dashboard is available at http://localhost:3001 (default credentials: admin/admin)

## Frontend Deployment (Vercel)

The frontend is a React application deployed on Vercel.

### 1. Configure Vercel Project

If you haven't already, install the Vercel CLI and log in:

```bash
npm i -g vercel
vercel login
```

Link your project to Vercel:

```bash
cd frontend
vercel link
```

### 2. Configure Environment Variables

Set the backend URL in Vercel:

```bash
vercel env add BACKEND_URL
```

Enter your backend URL when prompted (e.g., `http://your-backend-server-ip:8000`).

### 3. Deploy to Vercel

Deploy your frontend:

```bash
vercel --prod
```

Alternatively, connect your GitHub repository to Vercel for automatic deployments.

## Connection Setup

The frontend and backend communicate via HTTP/HTTPS. Make sure:

1. CORS is properly configured in the backend to allow requests from your frontend URL
2. The frontend has the correct backend URL configured
3. If using authentication, ensure both frontend and backend have matching JWT secrets

The connection configuration can be found in:
- Backend: `backend/config/connection.json`
- Frontend: `.env.production` (for Vercel deployment)

## Monitoring

The backend includes Prometheus and Grafana for monitoring:

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

Predefined dashboards include:
- GPU Performance
- API Request Metrics
- Database Connections

## Troubleshooting

### Backend Issues

1. **Docker container not starting:**
   Check GPU availability with `nvidia-smi` and ensure the NVIDIA Container Toolkit is installed.

2. **Connection to SAP HANA Cloud failing:**
   Verify your connection parameters and ensure network connectivity to the SAP HANA Cloud instance.

3. **GPU not being utilized:**
   Check the logs with `docker compose -f docker-compose.backend.yml logs backend` and verify GPU_ENABLED=true.

### Frontend Issues

1. **API connection failing:**
   Check that the BACKEND_URL environment variable is correctly set in Vercel.

2. **Vercel build failing:**
   Review the build logs in the Vercel dashboard for specific errors.

3. **CORS errors:**
   Ensure the frontend URL is added to the CORS_ORIGINS list in the backend configuration.

## Extending for SAP BTP or Other Platforms

The deployment configuration has been designed to be extensible. To add support for additional platforms:

1. Create a new Docker Compose file for the target platform
2. Update the backend configuration to support the platform-specific requirements
3. Configure the frontend to connect to the new backend endpoint

See the `EXTENSIBILITY.md` document for more details on adding deployment options for other platforms.