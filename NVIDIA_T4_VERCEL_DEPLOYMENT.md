# SAP HANA Cloud LangChain Integration: T4 GPU Backend with Vercel Frontend

This guide explains how to deploy the SAP HANA Cloud LangChain integration with the backend running on an NVIDIA T4 GPU and the frontend on Vercel.

## Architecture

```
┌───────────────┐              ┌───────────────┐              ┌───────────────┐
│   Vercel      │    REST      │    NVIDIA     │    HANA      │  SAP HANA     │
│   Frontend    │◄───────────►│    T4 GPU     │◄───────────►│  Cloud        │
└───────────────┘    API       └───────────────┘    SQL       └───────────────┘
```

## Deployment Steps

### 1. Deploy Backend to NVIDIA T4 GPU Server

#### Prerequisites
- Access to an NVIDIA T4 GPU server (like NVIDIA LaunchPad or a VM with T4 GPU)
- Docker and Docker Compose installed
- NGC CLI installed and configured (optional, for NGC repository deployment)

#### Option A: Deploy with Docker Compose (Recommended for local deployments)

1. SSH into your T4 GPU server
2. Clone the repository:
   ```bash
   git clone https://github.com/plturrell/langchain-integration-for-sap-hana-cloud.git
   cd langchain-integration-for-sap-hana-cloud
   ```

3. Create a `.env` file in the `api` directory:
   ```bash
   cat > api/.env << EOF
   # GPU Configuration
   GPU_ENABLED=true
   USE_TENSORRT=true
   TENSORRT_PRECISION=fp16
   TENSORRT_CACHE_DIR=/app/tensorrt_cache

   # Environment
   ENVIRONMENT=development
   LOG_LEVEL=DEBUG
   ENABLE_CORS=true
   APP_NAME=langchain-hana-integration

   # Performance
   DEFAULT_TIMEOUT=30
   HEALTH_CHECK_TIMEOUT=10
   EMBEDDING_TIMEOUT=60
   SEARCH_TIMEOUT=45
   CONNECTION_TEST_TIMEOUT=5

   # Authentication
   JWT_SECRET=sap-hana-langchain-t4-integration-secret-key-2025
   REQUIRE_AUTH=false

   # Error Handling
   ENABLE_ERROR_CONTEXT=true
   ENABLE_DETAILED_LOGGING=true
   ENABLE_MEMORY_TRACKING=true
   MAX_RETRY_COUNT=3
   RETRY_DELAY_MS=1000
   ENABLE_SSL_VERIFICATION=true

   # CORS Configuration - Allow Vercel frontend
   CORS_ORIGINS=*
   EOF
   ```

4. Create a Docker Compose file for GPU deployment:
   ```bash
   cat > docker-compose.gpu.yml << EOF
   version: '3'

   services:
     api:
       build:
         context: ./api
         dockerfile: Dockerfile.ngc
       ports:
         - "8000:8000"
       env_file:
         - ./api/.env
       volumes:
         - ./api:/app
         - tensorrt_cache:/app/tensorrt_cache
       restart: unless-stopped
       runtime: nvidia
       environment:
         - NVIDIA_VISIBLE_DEVICES=all
       deploy:
         resources:
           reservations:
             devices:
               - driver: nvidia
                 count: 1
                 capabilities: [gpu]

   volumes:
     tensorrt_cache:
   EOF
   ```

5. Build and start the backend service:
   ```bash
   docker-compose -f docker-compose.gpu.yml up -d
   ```

6. Test that the API is working:
   ```bash
   curl http://localhost:8000/benchmark/gpu_info
   ```

#### Option B: Deploy with NGC (for NVIDIA LaunchPad)

1. Make the deployment script executable and run it:
   ```bash
   chmod +x scripts/deploy_to_nvidia_t4.sh
   ./scripts/deploy_to_nvidia_t4.sh --skip-frontend
   ```

2. Launch on NVIDIA LaunchPad:
   ```bash
   ngc launchpod launch --config nvidia-launchable-t4.yaml
   ```

### 2. Deploy Frontend to Vercel

1. Update the backend URL in the frontend:
   ```bash
   cat > frontend/.env << EOF
   REACT_APP_API_URL=https://your-t4-server-address:8000
   REACT_APP_ENABLE_AUTH=false
   EOF
   ```

2. Deploy to Vercel:
   ```bash
   cd frontend
   vercel --prod
   ```

3. Alternative: Configure environment variables directly in Vercel dashboard:
   - Go to your project settings in Vercel
   - Add REACT_APP_API_URL environment variable with your backend URL
   - Redeploy your project

### 3. Configure CORS and Security

1. Ensure your T4 backend accepts requests from your Vercel frontend:
   - Update the CORS settings in your backend `.env` file:
     ```
     CORS_ORIGINS=https://your-vercel-frontend-url.vercel.app
     ```

2. If your T4 server is behind a firewall:
   - Open port 8000 for inbound connections
   - Consider setting up a reverse proxy with SSL termination (nginx/caddy)

### 4. Testing the Deployment

1. Test frontend-to-backend connectivity:
   - Visit your Vercel frontend URL
   - Try the embedding generation feature
   - Check browser developer console for any CORS or connection errors

2. Test backend GPU acceleration:
   - Visit `https://your-t4-server-address:8000/benchmark/gpu_info`
   - Run the TensorRT benchmark: `https://your-t4-server-address:8000/benchmark/tensorrt`

## Troubleshooting

### Backend Issues

1. Check if GPU is detected:
   ```bash
   docker exec -it <container_id> nvidia-smi
   ```

2. Check container logs:
   ```bash
   docker logs <container_id>
   ```

3. Verify CORS settings if frontend can't connect:
   ```bash
   curl -I -H "Origin: https://your-vercel-frontend-url.vercel.app" https://your-t4-server-address:8000
   ```

### Frontend Issues

1. Check Vercel build logs for errors
2. Verify environment variables are correctly set
3. Test API connectivity from the browser console:
   ```javascript
   fetch('https://your-t4-server-address:8000/benchmark/gpu_info')
     .then(response => response.json())
     .then(data => console.log(data))
     .catch(error => console.error('Error:', error));
   ```

## Performance Optimization

1. Optimize TensorRT for T4:
   - Use FP16 precision for optimal T4 performance
   - Adjust batch sizes based on available memory
   - Consider model quantization for larger models

2. Backend Scaling:
   - Use dynamic batch sizing based on load
   - Monitor GPU memory utilization
   - Consider auto-scaling if multiple T4 GPUs are available

## Maintenance

1. Update backend:
   ```bash
   git pull
   docker-compose -f docker-compose.gpu.yml build
   docker-compose -f docker-compose.gpu.yml up -d
   ```

2. Update frontend:
   ```bash
   vercel --prod
   ```

## Security Considerations

1. Set up JWT authentication for production
2. Use HTTPS for all communications
3. Implement rate limiting on your API
4. Consider using a private network for backend-database communication