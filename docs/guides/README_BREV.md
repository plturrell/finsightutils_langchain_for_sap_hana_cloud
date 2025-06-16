# Deploying SAP HANA Cloud LangChain Integration on NVIDIA Brev

This guide explains how to deploy the SAP HANA Cloud LangChain Integration with GPU acceleration on NVIDIA Brev.

## Prerequisites

- An NVIDIA Brev environment with GPU support
- Docker and Docker Compose installed
- Git access to this repository

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/plturrell/finsightutils_langchain_for_sap_hana_cloud.git
   cd finsightutils_langchain_for_sap_hana_cloud
   ```

2. Set up SAP HANA Cloud credentials (optional):
   ```bash
   ./setup_hana_credentials.sh
   ```
   
   This script will prompt you for your SAP HANA Cloud connection details and store them securely.

3. Deploy the application:
   ```bash
   ./brev_deploy.sh
   ```

4. Access the services:
   - API: http://localhost:8000/docs
   - Frontend: http://localhost:3000

## Using Without SAP HANA Cloud Connection

If you don't have SAP HANA Cloud credentials, the application will run in TEST_MODE by default, using in-memory storage instead of connecting to a real SAP HANA instance.

## Architecture

The deployment consists of two main services:

1. **API Service**: Python FastAPI application with GPU-accelerated embedding generation using TensorRT optimization.
   - Runs on port 8000
   - Uses NVIDIA GPU for acceleration
   - Connects to SAP HANA Cloud (when not in TEST_MODE)

2. **Frontend Service**: Web interface for interacting with the API.
   - Runs on port 3000
   - Includes vector visualization tools
   - Provides a user-friendly interface for embedding operations

## Common Operations

### View logs
```bash
docker-compose logs -f
```

### Stop services
```bash
docker-compose down
```

### Restart services
```bash
docker-compose restart
```

### Update deployment
```bash
git pull
./brev_deploy.sh
```

## Persistence

The deployment uses Docker volumes for persistence:
- `trt-engines`: Stores TensorRT engine caches for faster startup
- `api-data`: Stores application data

## Troubleshooting

If you encounter issues:

1. Check GPU availability:
   ```bash
   nvidia-smi
   ```

2. Verify Docker GPU support:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

3. Check service logs:
   ```bash
   docker-compose logs api
   ```

4. Ensure environment variables are set correctly:
   ```bash
   docker-compose config
   ```

## Performance Tuning

For optimal performance:

1. Adjust batch size based on your GPU memory:
   ```
   BATCH_SIZE=64
   MAX_BATCH_SIZE=256
   ```

2. Set GPU memory fraction:
   ```
   GPU_MEMORY_FRACTION=0.8
   ```

3. Choose appropriate TensorRT precision:
   ```
   TENSORRT_PRECISION=fp16  # For better performance
   TENSORRT_PRECISION=fp32  # For better accuracy
   ```