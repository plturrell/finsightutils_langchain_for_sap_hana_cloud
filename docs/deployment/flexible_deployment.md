# Flexible Multi-Platform Deployment Guide

This guide outlines how to deploy the SAP HANA Cloud LangChain Integration across multiple platforms with a flexible architecture that allows mixing and matching backend and frontend deployments.

## Architecture Overview

The system is designed with a decoupled architecture that enables:

1. **Backend Deployment Options**:
   - NVIDIA LaunchPad (GPU-optimized)
   - Together.ai (managed AI platform)
   - SAP BTP with GPU support
   - Vercel (serverless, without GPU acceleration)

2. **Frontend Deployment Options**:
   - Vercel (static site hosting)
   - SAP BTP (standard hosting)

3. **Configuration Management**:
   - Environment-specific configuration files
   - Runtime environment detection
   - Feature toggling based on platform capabilities

```
┌───────────────────┐    API Requests    ┌──────────────────┐
│    FRONTEND       │◄─────────────────►│     BACKEND      │
│                   │                    │                  │
│  - Vercel         │                    │ - NVIDIA NGC     │
│  - SAP BTP        │                    │ - Together.ai    │
│                   │                    │ - SAP BTP        │
└───────────────────┘                    │ - Vercel         │
                                         └──────────────────┘
                                                 │
                                                 ▼
                                         ┌──────────────────┐
                                         │   SAP HANA       │
                                         │   CLOUD          │
                                         └──────────────────┘
```

## Backend Deployment Options

### 1. NVIDIA LaunchPad/NGC Deployment

**Best for**: Production environments requiring maximum performance with GPU acceleration.

**Key Features**:
- TensorRT optimization
- Multi-GPU scaling
- Dynamic batch sizing
- Full error handling capabilities

**Setup Instructions**: See [NVIDIA Deployment Guide](./nvidia_deployment.md)

### 2. Together.ai Deployment

**Best for**: Quick deployment with managed AI infrastructure.

**Key Features**:
- Managed GPU infrastructure
- Simplified deployment
- API key-based authentication
- Pay-as-you-go pricing

**Setup Instructions**: See [Together.ai Deployment Guide](#together-ai-section)

### 3. SAP BTP with GPU Deployment

**Best for**: Enterprise environments already using SAP BTP.

**Key Features**:
- Integration with SAP ecosystem
- Enterprise-grade security
- SAP identity management
- Direct connection to SAP HANA Cloud

**Setup Instructions**: See [SAP BTP Deployment Guide](#sap-btp-section)

### 4. Vercel Deployment (Serverless)

**Best for**: Development, testing, or lower-traffic production without GPU requirements.

**Key Features**:
- Serverless architecture
- Quick deployment
- Free tier available
- CI/CD integration

**Setup Instructions**: See [Vercel Deployment Guide](./vercel_deployment.md)

## Frontend Deployment Options

### 1. Vercel (Static Site)

**Best for**: Quick deployment, development and testing.

**Key Features**:
- Global CDN
- Automatic HTTPS
- Preview deployments
- GitHub integration

**Setup Instructions**: See [Vercel Frontend Guide](#vercel-frontend-section)

### 2. SAP BTP (Standard Hosting)

**Best for**: Enterprise environments already using SAP BTP.

**Key Features**:
- Integration with SAP ecosystem
- Enterprise-grade security
- SAP identity management

**Setup Instructions**: See [SAP BTP Frontend Guide](#sap-btp-frontend-section)

## Configuration Management

### Environment Configuration Files

The system uses environment-specific configuration files:

- `.env.nvidia` - NVIDIA LaunchPad configuration
- `.env.together` - Together.ai configuration
- `.env.sap` - SAP BTP configuration
- `.env.vercel` - Vercel configuration

### Configuration Switching Script

The `deploy.sh` script facilitates environment switching:

```bash
./deploy.sh --backend nvidia --frontend vercel
./deploy.sh --backend together --frontend sap
./deploy.sh --backend sap --frontend sap
```

### Feature Toggle System

Features are automatically enabled/disabled based on platform capabilities:

```javascript
// Feature toggle based on environment
const useGPU = process.env.PLATFORM_SUPPORTS_GPU === 'true';
const useTensorRT = useGPU && process.env.USE_TENSORRT === 'true';
const useMultiGPU = useGPU && process.env.GPU_COUNT > 1;
```

## Deployment Combinations

### Recommended Combinations

1. **Full Enterprise Deployment**:
   - Backend: SAP BTP with GPU
   - Frontend: SAP BTP
   - Best for: SAP-centric enterprise environments

2. **Maximum Performance Deployment**:
   - Backend: NVIDIA LaunchPad
   - Frontend: Vercel
   - Best for: Performance-critical applications

3. **Managed AI Deployment**:
   - Backend: Together.ai
   - Frontend: Vercel
   - Best for: Simplified deployment with managed infrastructure

4. **Development/Testing Deployment**:
   - Backend: Vercel
   - Frontend: Vercel
   - Best for: Development, testing, and demos

### Mix and Match Considerations

When mixing deployment platforms:

1. **CORS Configuration**:
   - Ensure CORS headers are properly configured
   - Update frontend API URL configuration

2. **Authentication**:
   - Implement consistent authentication across platforms
   - Consider using JWT or API keys

3. **Networking**:
   - Ensure backend is accessible from frontend
   - Consider IP allowlisting for security

4. **Environment Variables**:
   - Maintain consistent naming across platforms
   - Document platform-specific limitations

## API Compatibility

The API maintains consistent endpoints across all platforms:

```
GET /api/health
GET /api/feature/vector-similarity
GET /api/feature/error-handling
POST /api/search
```

Backend-specific endpoints are conditionally enabled:

```
GET /api/gpu/info             # Only on GPU-enabled backends
POST /api/benchmark/tensorrt  # Only on NVIDIA backend
```

## Deployment Matrix

| Backend Platform | GPU Support | TensorRT | Error Handling | Vector Similarity |
|------------------|-------------|----------|----------------|-------------------|
| NVIDIA LaunchPad | ✅          | ✅       | ✅             | ✅                |
| Together.ai      | ✅          | ❌       | ✅             | ✅                |
| SAP BTP w/GPU    | ✅          | ✅       | ✅             | ✅                |
| Vercel           | ❌          | ❌       | ✅             | ✅                |

## Implementation Details

<a id="together-ai-section"></a>
### Together.ai Backend Deployment

1. **Register and Setup**:
   ```bash
   # Login to Together.ai
   together login

   # Deploy the backend
   together deploy \
     --name langchain-hana-backend \
     --source ./api \
     --requirements api/requirements.txt \
     --env-file .env.together
   ```

2. **Environment Variables**:
   ```
   TOGETHER_API_KEY=your_api_key
   HANA_HOST=your_hana_host
   HANA_PORT=443
   HANA_USER=your_username
   HANA_PASSWORD=your_password
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

3. **API URL**:
   ```
   https://api.together.xyz/v1/langchain-hana-backend
   ```

<a id="sap-btp-section"></a>
### SAP BTP Backend Deployment

1. **Prerequisites**:
   - SAP BTP account
   - Cloud Foundry CLI
   - SAP Cloud SDK

2. **Deployment**:
   ```bash
   # Login to Cloud Foundry
   cf login -a https://api.cf.{region}.hana.ondemand.com

   # Deploy the backend
   cf push langchain-hana-backend \
     -f manifest.yml \
     --vars-file vars.yml
   ```

3. **manifest.yml**:
   ```yaml
   applications:
   - name: langchain-hana-backend
     path: api
     memory: 2G
     disk_quota: 2G
     instances: 1
     buildpacks:
     - python_buildpack
     env:
       PLATFORM: sap_btp
       PLATFORM_SUPPORTS_GPU: true
       ENABLE_CONTEXT_AWARE_ERRORS: true
       ENABLE_PRECISE_SIMILARITY: true
   ```

<a id="vercel-frontend-section"></a>
### Vercel Frontend Deployment

1. **Deploy with Vercel CLI**:
   ```bash
   # Install Vercel CLI
   npm install -g vercel

   # Deploy frontend only
   cd frontend
   vercel --prod
   ```

2. **Environment Variables**:
   ```
   REACT_APP_API_URL=https://your-backend-url.com/api
   REACT_APP_ENVIRONMENT=production
   ```

<a id="sap-btp-frontend-section"></a>
### SAP BTP Frontend Deployment

1. **Build Frontend**:
   ```bash
   cd frontend
   npm install
   npm run build
   ```

2. **Deploy to SAP BTP**:
   ```bash
   cf push langchain-hana-frontend \
     -f frontend/manifest.yml \
     --vars-file frontend/vars.yml
   ```

3. **frontend/manifest.yml**:
   ```yaml
   applications:
   - name: langchain-hana-frontend
     path: build
     memory: 256M
     disk_quota: 512M
     instances: 2
     buildpacks:
     - staticfile_buildpack
     env:
       REACT_APP_API_URL: https://langchain-hana-backend.cfapps.{region}.hana.ondemand.com/api
   ```

## Environment Switching

### Using the Deployment Script

The repository includes a deployment script for environment switching:

```bash
# Switch to NVIDIA backend and Vercel frontend
./deploy.sh --backend nvidia --frontend vercel

# Switch to Together.ai backend and SAP BTP frontend
./deploy.sh --backend together --frontend sap
```

### Manual Configuration

For manual environment switching:

1. **Backend Configuration**:
   ```bash
   # Copy appropriate environment file
   cp .env.nvidia .env

   # Deploy to selected platform
   ./scripts/deploy_to_nvidia.sh
   ```

2. **Frontend Configuration**:
   ```bash
   # Update API URL in frontend configuration
   cd frontend
   echo "REACT_APP_API_URL=https://your-backend-url.com/api" > .env.production

   # Build and deploy
   npm run build
   ./scripts/deploy_to_vercel.sh
   ```

## Troubleshooting

### Common Issues

1. **CORS Errors**:
   - Verify CORS configuration in backend
   - Check frontend API URL configuration
   - Use browser developer tools to inspect request headers

2. **Authentication Failures**:
   - Verify API keys and credentials
   - Check environment variable configuration
   - Confirm network connectivity between services

3. **Missing GPU Features**:
   - Confirm GPU availability on selected platform
   - Verify TensorRT installation
   - Check environment variable configuration

4. **Deployment Failures**:
   - Review platform-specific deployment logs
   - Verify resource allocation
   - Check for size limits and quotas

## Further Resources

- [NVIDIA LaunchPad Documentation](https://docs.nvidia.com/launchpad/)
- [Together.ai API Documentation](https://docs.together.ai/)
- [SAP BTP Documentation](https://help.sap.com/docs/btp)
- [Vercel Documentation](https://vercel.com/docs)
- [LangChain Documentation](https://python.langchain.com/docs/get_started)
- [SAP HANA Cloud Documentation](https://help.sap.com/docs/HANA_CLOUD)