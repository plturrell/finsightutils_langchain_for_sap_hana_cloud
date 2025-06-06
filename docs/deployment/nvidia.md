# NVIDIA Deployment Guide for SAP HANA Cloud LangChain Integration

This guide provides instructions for deploying the SAP HANA Cloud LangChain integration on NVIDIA GPU infrastructure, both locally and on NVIDIA Launchables.

## Prerequisites

- Docker installed
- NVIDIA Container Toolkit installed (for local deployment with GPU support)
- NGC account with API key (for NVIDIA Launchables deployment)

## Local Deployment with NVIDIA GPUs

### Option 1: Using Docker Compose

1. Build and run using Docker Compose:

```bash
docker-compose -f docker-compose.nvidia.yml up -d --build
```

2. Verify the service is running:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/gpu-info
```

### Option 2: Using the Build Script

1. Run the build script:

```bash
chmod +x build_nvidia_local.sh
./build_nvidia_local.sh
```

2. Follow the prompts to build and optionally run the container.

## Deployment to NVIDIA Launchables

### Option 1: Manual Process

1. Build and tag the Docker image:

```bash
docker build -t langchain-nvidia:latest -f Dockerfile.nvidia .
```

2. Tag the image for NGC:

```bash
docker tag langchain-nvidia:latest nvcr.io/your-org/your-collection/langchain-nvidia:latest
```

3. Login to NGC Registry:

```bash
docker login nvcr.io
```

4. Push the image:

```bash
docker push nvcr.io/your-org/your-collection/langchain-nvidia:latest
```

5. Create a NVIDIA Launchable configuration file (`launchable-config.yaml`):

```yaml
# NVIDIA Launchable Configuration
name: langchain-sap-hana
version: 1.0.0
description: SAP HANA Cloud LangChain Integration with GPU Acceleration

resources:
  gpu:
    count: 1
    type: T4  # Specify GPU type (T4, A100, etc.)
  memory: 8Gi
  cpu:
    count: 4

container:
  image: nvcr.io/your-org/your-collection/langchain-nvidia:latest
  ports:
    - containerPort: 8000
      name: http
  env:
    - name: GPU_ENABLED
      value: "true"
    - name: USE_TENSORRT
      value: "true"
    - name: TENSORRT_PRECISION
      value: "fp16"
    - name: LOG_LEVEL
      value: "INFO"

health:
  path: /health
  initialDelaySeconds: 30
  periodSeconds: 30
  timeoutSeconds: 10
  failureThreshold: 3
```

6. Deploy using the NVIDIA Launchable CLI or web interface.

### Option 2: Using the Automated Build Script

1. Run the build_launchable script:

```bash
chmod +x build_launchable.sh
./build_launchable.sh
```

2. Follow the prompts to build, tag, and push the image to NGC.

3. Use the generated `launchable-config.yaml` file to deploy to NVIDIA Launchables.

## Verifying the Deployment

After deployment, you can verify the service is working correctly by accessing:

- Health check: `http://<service-url>/health`
- GPU information: `http://<service-url>/gpu-info`
- Tensor test: `http://<service-url>/tensor-test`
- Generate embeddings: `http://<service-url>/embeddings` (POST request)

## Testing Embeddings

You can test the embedding functionality with a simple curl command:

```bash
curl -X POST "http://<service-url>/embeddings" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["This is a test sentence for embeddings"]}'
```

## Troubleshooting

### Container Health Checks Failing

If the container health checks are failing:

1. Check container logs:
```bash
docker logs <container_id>
```

2. Verify the application is running inside the container:
```bash
docker exec <container_id> curl -f http://localhost:8000/health
```

3. Ensure the NVIDIA runtime is being used correctly:
```bash
docker exec <container_id> nvidia-smi
```

### GPU Not Being Detected

If the GPU is not being detected:

1. Verify NVIDIA driver installation:
```bash
nvidia-smi
```

2. Check that NVIDIA Container Toolkit is installed and configured:
```bash
docker info | grep -i nvidia
```

3. Verify that the container can access the GPU:
```bash
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Additional Resources

- [NVIDIA Container Toolkit Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/overview.html)
- [NVIDIA NGC Registry Documentation](https://docs.nvidia.com/ngc/ngc-private-registry-user-guide/index.html)
- [NVIDIA Launchables Documentation](https://docs.nvidia.com/launchables/)