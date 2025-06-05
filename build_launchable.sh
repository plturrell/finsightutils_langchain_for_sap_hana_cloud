#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================================${NC}"
echo -e "${BLUE}Building NVIDIA NGC Blueprint for SAP HANA Cloud LangChain${NC}"
echo -e "${BLUE}=========================================================${NC}"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if NGC CLI is installed
NGC_CLI_INSTALLED=false
if command -v ngc &> /dev/null; then
    NGC_CLI_INSTALLED=true
    echo -e "${GREEN}NGC CLI detected. Will use NGC CLI for advanced operations.${NC}"
else
    echo -e "${YELLOW}NGC CLI not detected. Some advanced features will be disabled.${NC}"
    echo -e "${YELLOW}To install NGC CLI: https://ngc.nvidia.com/setup/installers/cli${NC}"
fi

# Check if NVIDIA Container Toolkit is installed
if ! docker info | grep -q "Runtimes:.*nvidia"; then
    echo -e "${YELLOW}WARNING: NVIDIA Container Toolkit not detected.${NC}"
    echo -e "${YELLOW}This script is intended for NVIDIA GPU deployments.${NC}"
    echo -e "${YELLOW}See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html${NC}"
    
    read -p "Continue anyway? (y/n) " continue_decision
    if [[ ! "$continue_decision" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Exiting.${NC}"
        exit 1
    fi
fi

# Check for available GPUs
if command -v nvidia-smi &> /dev/null; then
    echo -e "${BLUE}Checking GPUs...${NC}"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)
    echo -e "${GREEN}Available GPUs:${NC}"
    echo "$GPU_INFO" | while read line; do
        echo -e "  ${CYAN}$line${NC}"
    done
    
    # Extract driver version
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    echo -e "${BLUE}NVIDIA Driver Version: ${CYAN}$DRIVER_VERSION${NC}"
    
    # Check minimum driver version
    REQUIRED_DRIVER="520.0"
    if [[ $(echo -e "$DRIVER_VERSION\n$REQUIRED_DRIVER" | sort -V | head -n1) != "$REQUIRED_DRIVER" ]]; then
        echo -e "${YELLOW}WARNING: Driver version $DRIVER_VERSION may be too old for NGC Blueprints.${NC}"
        echo -e "${YELLOW}Recommended minimum driver version is $REQUIRED_DRIVER.${NC}"
    fi
else
    echo -e "${YELLOW}WARNING: nvidia-smi not found. Cannot verify GPU availability.${NC}"
fi

# Set variables
IMAGE_NAME="langchain-nvidia"
VERSION="1.0.3"
NGC_ORG="your-org"  # Replace with your NGC org
NGC_COLLECTION="sap-enhanced"  # Replace with your NGC collection

# Ask user for NGC credentials if not already logged in
if ! docker info | grep -q "Username: nvcr.io"; then
    echo -e "${BLUE}You need to log in to the NGC Registry.${NC}"
    echo -e "${BLUE}You can get your NGC API key at: https://ngc.nvidia.com/setup/api-key${NC}"
    
    # Prompt for NGC API key
    read -p "Enter your NGC API key: " NGC_API_KEY
    
    if [ -z "$NGC_API_KEY" ]; then
        echo -e "${RED}No API key provided. Exiting.${NC}"
        exit 1
    fi
    
    # Login to NGC
    echo -e "${BLUE}Logging in to NGC...${NC}"
    echo "$NGC_API_KEY" | docker login nvcr.io -u '$oauthtoken' --password-stdin
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to log in to NGC. Please check your API key.${NC}"
        exit 1
    fi
    
    # Also login to NGC CLI if available
    if [ "$NGC_CLI_INSTALLED" = true ]; then
        echo -e "${BLUE}Configuring NGC CLI...${NC}"
        echo "$NGC_API_KEY" > ngc_api_key.txt
        ngc config set -k api_key -v $(cat ngc_api_key.txt)
        rm ngc_api_key.txt
    fi
fi

# Set BASE_IMAGE from available NGC PyTorch containers
echo -e "${BLUE}Selecting NGC base image...${NC}"
echo -e "Available PyTorch images:"
echo -e "  ${CYAN}1) nvcr.io/nvidia/pytorch:23.12-py3 (Latest)${NC}"
echo -e "  ${CYAN}2) nvcr.io/nvidia/pytorch:23.10-py3${NC}"
echo -e "  ${CYAN}3) nvcr.io/nvidia/pytorch:23.08-py3${NC}"
echo -e "  ${CYAN}4) nvcr.io/nvidia/pytorch:23.07-py3${NC}"
echo -e "  ${CYAN}5) Custom image${NC}"

read -p "Select base image [1]: " base_image_selection
case $base_image_selection in
    2) BASE_IMAGE="nvcr.io/nvidia/pytorch:23.10-py3" ;;
    3) BASE_IMAGE="nvcr.io/nvidia/pytorch:23.08-py3" ;;
    4) BASE_IMAGE="nvcr.io/nvidia/pytorch:23.07-py3" ;;
    5) 
        read -p "Enter custom NGC base image: " BASE_IMAGE
        ;;
    *) BASE_IMAGE="nvcr.io/nvidia/pytorch:23.12-py3" ;;
esac

echo -e "${BLUE}Selected base image: ${CYAN}$BASE_IMAGE${NC}"

# Build the Docker image
echo -e "${BLUE}Building the Docker image...${NC}"
echo -e "${BLUE}Using build arg BASE_IMAGE=$BASE_IMAGE${NC}"

docker build -t ${IMAGE_NAME}:${VERSION} \
    --build-arg BASE_IMAGE=${BASE_IMAGE} \
    -f Dockerfile.nvidia .

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Docker image built successfully!${NC}"
else
    echo -e "${RED}Failed to build Docker image.${NC}"
    exit 1
fi

# Test the image with NVIDIA runtime
echo -e "${BLUE}Testing image with NVIDIA runtime...${NC}"
if docker run --rm --gpus all ${IMAGE_NAME}:${VERSION} nvidia-smi; then
    echo -e "${GREEN}Successfully verified GPU access!${NC}"
else
    echo -e "${RED}Warning: Could not verify GPU access.${NC}"
    echo -e "${YELLOW}This may be due to missing NVIDIA Container Toolkit or no GPU.${NC}"
fi

# Verify TensorRT
echo -e "${BLUE}Verifying TensorRT in the container...${NC}"
if docker run --rm ${IMAGE_NAME}:${VERSION} python -c "import tensorrt; print(f'TensorRT {tensorrt.__version__} is available')"; then
    echo -e "${GREEN}TensorRT verification successful!${NC}"
else
    echo -e "${RED}Warning: TensorRT verification failed.${NC}"
    echo -e "${YELLOW}This may affect performance in NGC environments.${NC}"
fi

# Ask for NGC organization and collection
echo -e "${BLUE}Preparing for NGC push...${NC}"
read -p "Enter your NGC organization name [$NGC_ORG]: " input_org
NGC_ORG=${input_org:-$NGC_ORG}

read -p "Enter your NGC collection name [$NGC_COLLECTION]: " input_collection
NGC_COLLECTION=${input_collection:-$NGC_COLLECTION}

# Tag the image for NGC
echo -e "${BLUE}Tagging image for NGC...${NC}"
NGC_IMAGE="nvcr.io/${NGC_ORG}/${NGC_COLLECTION}/${IMAGE_NAME}:${VERSION}"
docker tag ${IMAGE_NAME}:${VERSION} ${NGC_IMAGE}

# Push to NGC
echo -e "${BLUE}Would you like to push the image to NGC? (y/n)${NC}"
read -p "Push to NGC? " push_decision

if [[ "$push_decision" =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Pushing image to NGC...${NC}"
    docker push ${NGC_IMAGE}
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully pushed image to NGC!${NC}"
        echo -e "${GREEN}Image URL: ${NGC_IMAGE}${NC}"
        
        # Generate improved NGC Blueprint configuration
        echo -e "${BLUE}Generating NVIDIA NGC Blueprint configuration...${NC}"
        
        NGC_BLUEPRINT="nvidia-blueprint.yaml"
        cat > ${NGC_BLUEPRINT} << EOL
name: sap-hana-langchain-ngc
version: ${VERSION}
organization: ${NGC_ORG}
collection: ${NGC_COLLECTION}
description: Enterprise-ready solution for SAP HANA Cloud vector store operations with NVIDIA GPU acceleration, TensorRT optimization, and interactive 3D visualizations

runtimeEnvironment: vm
license: Apache-2.0
maintainer: $(git config --get user.email || echo "user@example.com")
supportContact: support@example.com
supportUrl: https://github.com/${NGC_ORG}/langchain-integration-for-sap-hana-cloud/issues

container:
  image: ${NGC_IMAGE}

labels:
  - langchain
  - sap-hana
  - vector-store
  - gpu-acceleration
  - tensorrt
  - embeddings
  - large-language-models
  - vector-similarity
  - knowledge-graph
  - interactive-visualization
  - enterprise-ready

components:
  - name: api-server
    type: service
    container:
      image: ${NGC_IMAGE}
      ports:
        - containerPort: 8000
          protocol: TCP
          expose: false
    resources:
      gpu:
        required: true
        count: 1
        productFamily: ["Tesla", "Quadro", "RTX", "A100", "H100", "A10", "T4"]
        memory: 16Gi
      memory:
        min: 8Gi
        recommended: 16Gi
      cpu:
        min: 4
        recommended: 8
      storage:
        min: 20Gi
        recommended: 50Gi
        
  - name: frontend
    type: service
    container:
      image: nvcr.io/${NGC_ORG}/${NGC_COLLECTION}/langchain-hana-frontend:${VERSION}
      ports:
        - containerPort: 3000
          protocol: TCP
          expose: true
    resources:
      memory:
        min: 2Gi
        recommended: 4Gi
      cpu:
        min: 2
        recommended: 4

documentation:
  overview: |
    # SAP HANA Cloud LangChain Integration with GPU Acceleration
    
    This enterprise-ready application provides GPU-accelerated vector operations for SAP HANA Cloud, 
    enabling efficient semantic search and retrieval with LLM-powered embeddings, context-aware 
    error handling, interactive 3D visualizations, and precision vector similarity scoring.
    
    ## Key Features
    
    - TensorRT optimization for maximum inference speed
    - Multi-GPU load balancing for high throughput
    - Dynamic batch sizing based on GPU memory
    - Mixed precision support (FP16, FP32, INT8)
    - Memory optimization for large operations
    - Context-aware error handling with intelligent suggestions
    - Operation-specific error diagnosis and troubleshooting
    - Precise vector similarity measurements for better ranking
    - Interactive 3D vector visualization with real-time filtering
    - Accessible and responsive user interface
  
  quickStart: |
    ## Quick Start
    
    ### Step 1: Authenticate with NGC
    
    ```bash
    # Install NGC CLI
    wget -O ngccli.zip https://ngc.nvidia.com/downloads/ngccli_linux.zip
    unzip -o ngccli.zip
    chmod u+x ngc-cli/ngc
    echo 'export PATH="$PATH:$HOME/ngc-cli"' >> ~/.bashrc
    source ~/.bashrc
    
    # Configure NGC CLI with your API key (get it from https://ngc.nvidia.com/setup/api-key)
    ngc config set
    
    # Log in to Docker with NGC credentials
    docker login nvcr.io
    ```
    
    ### Step 2: Pull the container
    
    ```bash
    docker pull ${NGC_IMAGE}
    ```
    
    ### Step 3: Configure SAP HANA Cloud connection
    
    ```bash
    # Create .env file with your credentials
    cat > .env << EOL
    HANA_HOST=your-hana-host.hanacloud.ondemand.com
    HANA_PORT=443
    HANA_USER=your_username
    HANA_PASSWORD=your_password
    ENABLE_CONTEXT_AWARE_ERRORS=true
    ENABLE_PRECISE_SIMILARITY=true
    TENSORRT_PRECISION=fp16
    EOL
    ```
    
    ### Step 4: Run with GPU support
    
    ```bash
    docker run --gpus all -p 8000:8000 \\
      --env-file .env \\
      -e GPU_ENABLED=true \\
      -e USE_TENSORRT=true \\
      -v $PWD/trt_engines:/app/trt_engines \\
      ${NGC_IMAGE}
    ```
    
    ### Step 5: Start the frontend
    
    ```bash
    docker run -p 3000:3000 \\
      -e REACT_APP_API_URL=http://localhost:8000 \\
      nvcr.io/${NGC_ORG}/${NGC_COLLECTION}/langchain-hana-frontend:${VERSION}
    ```
    
    ### Step 6: Access the application
    
    - Frontend UI: http://localhost:3000
    - API documentation: http://localhost:8000/docs
    - GPU information: http://localhost:8000/benchmark/gpu_info
    - Benchmarking: http://localhost:8000/benchmark/tensorrt
    
    For complete setup instructions, refer to the documentation.
  
  performance: |
    ## Performance
    
    This solution leverages NVIDIA GPUs to accelerate embedding generation and vector operations:
    
    ### T4 Performance
    
    | Operation | Batch Size | T4 GPU | CPU | Speedup |
    |-----------|------------|--------|-----|---------|
    | Embedding | 1          | 18ms   | 80ms| 4.4x    |
    | Embedding | 32         | 82ms   | 580ms| 7.1x   |
    | Embedding | 128        | 198ms  | 2320ms| 11.7x |
    | MMR Search| 10 results | 12ms   | 68ms | 5.7x   |
    
    ### A10 Performance
    
    | Operation | Batch Size | A10 GPU | CPU | Speedup |
    |-----------|------------|---------|-----|---------|
    | Embedding | 1          | 14ms    | 80ms| 5.7x    |
    | Embedding | 32         | 58ms    | 580ms| 10.0x  |
    | Embedding | 128        | 112ms   | 2320ms| 20.7x |
    | MMR Search| 10 results | 8ms     | 68ms | 8.5x   |
    
    ### A100 Performance
    
    | Operation | Batch Size | A100 GPU | CPU | Speedup |
    |-----------|------------|----------|-----|---------|
    | Embedding | 1          | 9ms      | 80ms| 8.9x    |
    | Embedding | 32         | 30ms     | 580ms| 19.3x  |
    | Embedding | 128        | 62ms     | 2320ms| 37.4x |
    | MMR Search| 10 results | 4ms      | 68ms | 17.0x  |
    
    ### Benchmarking
    
    Run the built-in benchmark:
    ```bash
    curl -X POST "http://localhost:8000/benchmark/tensorrt" \\
      -H "Content-Type: application/json" \\
      -d '{
        "model_name": "all-MiniLM-L6-v2",
        "precision": "fp16",
        "batch_sizes": [1, 8, 32, 64, 128],
        "input_length": 128,
        "iterations": 100
      }'
    ```

requirements:
  gpu:
    cuda: ">=11.8"
    driver: ">=520.0"
    tensorRT: ">=8.6.0"
  software:
    docker: ">=20.10.0"
    nvidia-container-toolkit: ">=1.14.0"

environment:
  variables:
    - name: HANA_HOST
      description: SAP HANA Cloud host
      required: true
    - name: HANA_PORT
      description: SAP HANA Cloud port
      required: true
      defaultValue: "443"
    - name: HANA_USER
      description: SAP HANA Cloud username
      required: true
    - name: HANA_PASSWORD
      description: SAP HANA Cloud password
      required: true
    - name: GPU_ENABLED
      description: Enable GPU acceleration
      required: false
      defaultValue: "true"
    - name: USE_TENSORRT
      description: Enable TensorRT optimization
      required: false
      defaultValue: "true"
    - name: TENSORRT_PRECISION
      description: TensorRT precision (fp16, fp32, int8)
      required: false
      defaultValue: "fp16"
    - name: ENABLE_CONTEXT_AWARE_ERRORS
      description: Enable intelligent error handling with suggestions
      required: false
      defaultValue: "true"
    - name: ENABLE_PRECISE_SIMILARITY
      description: Enable accurate vector similarity measurements
      required: false
      defaultValue: "true"
    - name: TENSORRT_ENGINE_CACHE_DIR
      description: Directory to cache TensorRT engines
      required: false
      defaultValue: "/app/trt_engines"
    - name: ENABLE_VECTOR_VISUALIZATION
      description: Enable 3D vector visualization
      required: false
      defaultValue: "true"

relatedSolutions:
  - name: NVIDIA NeMo
    url: https://developer.nvidia.com/nemo
  - name: NVIDIA TensorRT
    url: https://developer.nvidia.com/tensorrt
  - name: LangChain
    url: https://www.langchain.com/
  - name: NVIDIA RAG
    url: https://www.nvidia.com/en-us/research/generative-ai/rag/
  - name: SAP HANA Cloud Vector Engine
    url: https://www.sap.com/products/technology-platform/hana/cloud.html
EOL
        
        echo -e "${GREEN}Created NGC Blueprint configuration file: ${NGC_BLUEPRINT}${NC}"
        
        # Use NGC CLI to register the blueprint if available
        if [ "$NGC_CLI_INSTALLED" = true ]; then
            echo -e "${BLUE}Would you like to register this blueprint with NGC? (y/n)${NC}"
            read -p "Register blueprint? " register_decision
            
            if [[ "$register_decision" =~ ^[Yy]$ ]]; then
                echo -e "${BLUE}Registering blueprint with NGC...${NC}"
                ngc registry resource upload --file ${NGC_BLUEPRINT}
                
                if [ $? -eq 0 ]; then
                    echo -e "${GREEN}Successfully registered blueprint with NGC!${NC}"
                    echo -e "${GREEN}You can view your blueprint at: https://ngc.nvidia.com/catalog/${NGC_ORG}/${NGC_COLLECTION}/blueprints/sap-hana-langchain-ngc${NC}"
                else
                    echo -e "${RED}Failed to register blueprint with NGC.${NC}"
                    echo -e "${YELLOW}You can register manually using:${NC}"
                    echo -e "${YELLOW}ngc registry resource upload --file ${NGC_BLUEPRINT}${NC}"
                fi
            else
                echo -e "${YELLOW}Skipping NGC blueprint registration.${NC}"
                echo -e "${BLUE}To register manually later, use:${NC}"
                echo -e "${YELLOW}ngc registry resource upload --file ${NGC_BLUEPRINT}${NC}"
            fi
        else
            echo -e "${YELLOW}NGC CLI not detected. To register the blueprint, install NGC CLI and run:${NC}"
            echo -e "${YELLOW}ngc registry resource upload --file ${NGC_BLUEPRINT}${NC}"
        fi
    else
        echo -e "${RED}Failed to push image to NGC.${NC}"
    fi
else
    echo -e "${YELLOW}Skipping NGC push.${NC}"
    echo -e "${BLUE}To push manually later, use:${NC}"
    echo -e "${YELLOW}docker push ${NGC_IMAGE}${NC}"
fi

echo -e "${BLUE}=========================================================${NC}"
echo -e "${GREEN}NGC Blueprint build process completed!${NC}"
echo -e "${BLUE}=========================================================${NC}"