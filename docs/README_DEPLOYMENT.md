# SAP HANA Cloud LangChain Integration with NVIDIA GPU and Vercel

This project integrates LangChain with SAP HANA Cloud, leveraging NVIDIA GPUs for backend processing and Vercel for frontend hosting. The deployment architecture has been streamlined to focus on these core components for simplicity and maintainability.

## Deployment Architecture

Our streamlined deployment architecture consists of two primary components:

1. **Backend**: FastAPI application with NVIDIA GPU acceleration
   - Containerized with Docker and deployable via Docker Compose for development
   - Deployed to Kubernetes for production with autoscaling
   - Optimized for NVIDIA GPUs with TensorRT acceleration

2. **Frontend**: React application 
   - Deployed on Vercel for global availability and easy updates
   - Configured to connect to the appropriate backend environment

```
┌───────────────────┐     HTTP/HTTPS     ┌────────────────────┐     JDBC     ┌────────────────┐
│                   │                     │                    │              │                │
│  React Frontend   │ <----------------> │  FastAPI Backend   │ <---------> │  SAP HANA Cloud │
│  (Vercel)         │                     │  (NVIDIA GPU)      │              │                │
│                   │                     │                    │              │                │
└───────────────────┘                     └────────────────────┘              └────────────────┘
                                                   ^
                                                   │
                                         ┌─────────┴──────────┐
                                         │                    │
                                         │  Prometheus/Grafana│
                                         │  (Monitoring)      │
                                         │                    │
                                         └────────────────────┘
```

## Quick Start

### Local Development

For local development:

```bash
# Clone the repository
git clone https://github.com/sap/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud

# Set up environment variables for backend
cp .env.example .env
# Edit .env with your SAP HANA Cloud credentials and settings

# Start the backend with NVIDIA GPU support
docker-compose -f docker-compose.backend.yml up -d

# Start the frontend
cd frontend
npm install
npm start
```

### Production Deployment

Production deployments are managed through our CI/CD pipeline:

1. **For Staging Deployment**:
   ```bash
   # Push changes to main branch
   git push origin main
   ```

2. **For Production Deployment**:
   ```bash
   # Create and push a version tag
   git tag -a v1.0.0 -m "Release v1.0.0"
   git push origin v1.0.0
   ```

3. **Manual Deployment Trigger**:
   - Go to GitHub Actions tab
   - Select "CI/CD Pipeline" workflow
   - Click "Run workflow"
   - Choose your target environment

## Key Features

- **GPU Acceleration**: Optimized for NVIDIA GPUs with TensorRT support
- **Kubernetes Deployment**: Scalable and highly available backend
- **Vercel Integration**: Global CDN for frontend with simple deployment
- **CI/CD Pipeline**: Automated testing, building, and deployment
- **High Availability**: Multiple replicas and auto-scaling
- **Security**: Proper secrets management and CORS configuration
- **Monitoring**: Built-in health checks and Prometheus integration

## Deployment Options

### Backend Deployment

The backend can be deployed in several ways:

1. **Local Development**:
   ```bash
   docker-compose -f docker-compose.backend.yml up -d
   ```

2. **Kubernetes (Automated via CI/CD)**:
   The CI/CD pipeline automatically deploys to Kubernetes when pushing to main branch or version tags.

3. **Manual Kubernetes Deployment**:
   ```bash
   kubectl apply -f kubernetes/production/namespace.yaml
   kubectl apply -f kubernetes/production/secrets.yaml
   kubectl apply -f kubernetes/production/configmap.yaml
   kubectl apply -f kubernetes/production/deployment.yaml
   kubectl apply -f kubernetes/production/service.yaml
   kubectl apply -f kubernetes/production/hpa.yaml
   ```

### Frontend Deployment

The frontend is deployed to Vercel:

1. **Local Development**:
   ```bash
   cd frontend
   npm install
   npm start
   ```

2. **Vercel (Automated via CI/CD)**:
   The CI/CD pipeline automatically deploys to Vercel when pushing to main branch or version tags.

3. **Manual Vercel Deployment**:
   ```bash
   cd frontend
   vercel --prod
   ```

## Environment Configuration

### Backend Environment Variables

Essential environment variables for the backend:

```
# Core Configuration
GPU_ENABLED=true
USE_TENSORRT=true
BATCH_SIZE=64
LOG_LEVEL=INFO

# Database Configuration
DB_HOST=your-hana-host.hanacloud.ondemand.com
DB_PORT=443
DB_USER=DBADMIN
DB_PASSWORD=your-password

# Security
API_KEY=your-api-key
CORS_ORIGINS=https://your-frontend-domain.vercel.app
```

### Frontend Environment Variables

Essential environment variables for the frontend:

```
BACKEND_URL=https://your-backend-url
AUTH_ENABLED=true
ENVIRONMENT=production
```

## Verification and Testing

After deployment, verify the application:

1. **Smoke Testing**:
   ```bash
   python scripts/smoke_test.py --api-url https://your-backend-url
   ```

2. **Production Verification**:
   ```bash
   python scripts/verify_deployment.py --api-url https://your-backend-url
   ```

## Monitoring

Monitor your deployment using:

1. **Health Endpoints**:
   - `GET /health`: Basic health information
   - `GET /health/ready`: Readiness status
   - `GET /metrics`: Prometheus metrics

2. **Kubernetes Commands**:
   ```bash
   # Check pod status
   kubectl get pods -n langchain-hana-production
   
   # View logs
   kubectl logs -n langchain-hana-production deployment/langchain-hana-api
   
   # Describe deployment
   kubectl describe deployment -n langchain-hana-production langchain-hana-api
   ```

3. **Prometheus and Grafana**:
   - Connect Prometheus to the `/metrics` endpoint
   - Import the dashboards from `prometheus/dashboards/`

## Troubleshooting

Common issues and solutions:

1. **Backend Connection Issues**:
   - Check if the backend is running: `kubectl get pods -n langchain-hana-production`
   - Verify network connectivity and firewall rules
   - Check backend logs for errors

2. **GPU Acceleration Issues**:
   - Verify GPU availability: `nvidia-smi`
   - Check container GPU configuration
   - Examine pod status for GPU-related errors

3. **Frontend Deployment Issues**:
   - Check Vercel deployment logs
   - Verify environment variables are set correctly
   - Test backend connectivity from frontend

For more detailed information, see [docs/troubleshooting.md](docs/troubleshooting.md).

## Additional Documentation

- [DEPLOYMENT_SUMMARY.md](DEPLOYMENT_SUMMARY.md): Summary of all deployment improvements
- [docs/cicd_guide.md](docs/cicd_guide.md): Detailed CI/CD pipeline documentation
- [docs/gpu_acceleration.md](docs/gpu_acceleration.md): NVIDIA GPU optimization details
- [docs/kubernetes_guide.md](docs/kubernetes_guide.md): Kubernetes deployment guide
- [docs/monitoring_guide.md](docs/monitoring_guide.md): Monitoring and alerting setup