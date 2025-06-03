# VM Setup Guide for SAP HANA Cloud LangChain Integration

This guide provides detailed instructions for setting up the SAP HANA Cloud LangChain Integration with GPU Acceleration in a Virtual Machine environment. This is the recommended deployment method for NVIDIA Launchable users.

## Prerequisites

- VM with NVIDIA GPU (Tesla, Quadro, RTX, A100, or H100 series)
- CUDA 11.8+ and cuDNN installed
- Docker and NVIDIA Container Toolkit installed
- Internet access for container pulls
- SAP HANA Cloud instance credentials

## Step 1: Install NVIDIA Drivers and CUDA

If not already installed, set up NVIDIA drivers and CUDA:

```bash
# Install NVIDIA drivers (Ubuntu example)
sudo apt update
sudo apt install -y nvidia-driver-525

# Verify driver installation
nvidia-smi

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

## Step 2: Install Docker and NVIDIA Container Toolkit

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Step 3: Authenticate with NGC

```bash
# Install NGC CLI
wget -O ngccli.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip
unzip -o ngccli.zip
chmod u+x ngc-cli/ngc
echo 'export PATH="$PATH:$HOME/ngc-cli"' >> ~/.bashrc
source ~/.bashrc

# Configure NGC CLI with API key
# Generate your API key at https://ngc.nvidia.com/setup/api-key
ngc config set

# Verify NGC authentication
ngc registry image list
```

## Step 4: Pull the Container Image

```bash
# Log in to Docker with NGC credentials
docker login nvcr.io

# Pull the container image
docker pull nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest
```

## Step 5: Configure SAP HANA Cloud Connection

To connect to your SAP HANA Cloud instance, you need to provide the following credentials:

1. **Host**: Your SAP HANA Cloud hostname (e.g., `myhana.hanacloud.ondemand.com`)
2. **Port**: Your SAP HANA Cloud port (typically `443` for cloud instances)
3. **User**: Your SAP HANA Cloud username
4. **Password**: Your SAP HANA Cloud password

Create a `.env` file to store these credentials:

```bash
# Create .env file
cat > .env << EOL
HANA_HOST=your-hana-host.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your_username
HANA_PASSWORD=your_password
EOL
```

## Step 6: Run the Container

```bash
# Run the container with GPU support and environment variables
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  -e GPU_ENABLED=true \
  -e USE_TENSORRT=true \
  -e TENSORRT_PRECISION=fp16 \
  nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest
```

## Step 7: Verify Installation

Access the API and check that it's running properly:

```bash
# Check the health endpoint
curl http://localhost:8000/health

# Check GPU information
curl http://localhost:8000/benchmark/gpu_info
```

## Step 8: Run Benchmarks

```bash
# Run TensorRT benchmark to verify GPU acceleration
curl -X POST "http://localhost:8000/benchmark/tensorrt" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "all-MiniLM-L6-v2",
    "precision": "fp16",
    "batch_sizes": [1, 8, 32, 64],
    "input_length": 128,
    "iterations": 10
  }'
```

## Troubleshooting

### GPU Not Detected

If the GPU is not detected:

```bash
# Check NVIDIA driver and CUDA installation
nvidia-smi
nvcc --version

# Verify Docker can access GPUs
docker run --gpus all --rm nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Enable container GPU access
sudo systemctl restart nvidia-container-toolkit.service
sudo systemctl restart docker
```

### NGC Authentication Issues

If you encounter issues with NGC authentication:

```bash
# Regenerate your NGC API key at https://ngc.nvidia.com/setup/api-key
# Then reconfigure NGC CLI
ngc config set

# Verify authentication
ngc auth test
```

### Database Connection Issues

If you experience issues connecting to SAP HANA Cloud:

```bash
# Test connection to SAP HANA Cloud
nc -zv your-hana-host.hanacloud.ondemand.com 443

# Verify environment variables are passed correctly
docker inspect --format='{{range .Config.Env}}{{println .}}{{end}}' <container_id>
```

## Advanced Configuration

### Multi-GPU Setup

For multi-GPU environments:

```bash
# Specify which GPUs to use
docker run --gpus '"device=0,1"' -p 8000:8000 \
  --env-file .env \
  -e GPU_ENABLED=true \
  -e USE_TENSORRT=true \
  nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest
```

### Persistent TensorRT Cache

To persist the TensorRT engine cache between container restarts:

```bash
# Create a volume for the cache
docker volume create tensorrt-cache

# Mount the volume when running the container
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  -e GPU_ENABLED=true \
  -e USE_TENSORRT=true \
  -e TENSORRT_CACHE_DIR=/cache \
  -v tensorrt-cache:/cache \
  nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest
```

### Running in Production Mode

For production deployments:

```bash
# Run in production mode
docker run --gpus all -p 8000:8000 \
  --env-file .env \
  -e GPU_ENABLED=true \
  -e USE_TENSORRT=true \
  -e ENVIRONMENT=production \
  -e LOG_LEVEL=WARNING \
  --restart always \
  nvcr.io/plturrell/sap-enhanced/langchain-hana-gpu:latest
```

## Resources

- [NVIDIA NGC Documentation](https://docs.nvidia.com/ngc/)
- [NVIDIA TensorRT Documentation](https://developer.nvidia.com/tensorrt)
- [SAP HANA Cloud Documentation](https://help.sap.com/docs/hana-cloud)
- [GitHub Repository](https://github.com/plturrell/langchain-integration-for-sap-hana-cloud)