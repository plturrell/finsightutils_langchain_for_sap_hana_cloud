#!/bin/bash
#
# NVIDIA T4 GPU Deployment Script
#
# This script handles building and deploying the application to NVIDIA LaunchPad
# specifically optimized for T4 GPUs.
#
# Prerequisites:
# - NGC CLI installed and configured (https://ngc.nvidia.com/setup)
# - Docker installed and configured for NGC authentication
# - Proper NGC organization and team access

set -e

# Configuration
# Project root directory (parent of the scripts/deployment directory)
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"

ORG_NAME="plturrell"
COLLECTION_NAME="sap-enhanced"
BACKEND_IMAGE_NAME="langchain-hana-t4"
FRONTEND_IMAGE_NAME="langchain-hana-frontend"
VERSION=$(cat "$PROJECT_ROOT/VERSION" 2>/dev/null || echo "1.2.0")
GPU_TYPE="T4"

# Parse command line arguments
SKIP_BUILD=false
SKIP_PUSH=false
SKIP_FRONTEND=false
TAG="latest"
ENV_FILE="config/environment/.env.nvidia.t4.prod"

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --skip-push)
      SKIP_PUSH=true
      shift
      ;;
    --skip-frontend)
      SKIP_FRONTEND=true
      shift
      ;;
    --tag)
      TAG="$2"
      shift
      shift
      ;;
    --env-file)
      ENV_FILE="$2"
      shift
      shift
      ;;
    --help)
      echo "NVIDIA T4 GPU Deployment Script"
      echo ""
      echo "Usage:"
      echo "  ./deploy_to_nvidia_t4.sh [options]"
      echo ""
      echo "Options:"
      echo "  --skip-build     Skip the build step"
      echo "  --skip-push      Skip the push step"
      echo "  --skip-frontend  Skip frontend build and push"
      echo "  --tag <tag>      Specify tag (default: latest)"
      echo "  --env-file <file> Specify environment file (default: .env.nvidia.t4.prod)"
      echo "  --help           Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $key"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Ensure environment directory exists
mkdir -p "$PROJECT_ROOT/config/environment"

# Check if environment file exists
if [ ! -f "$PROJECT_ROOT/$ENV_FILE" ]; then
    echo "Warning: Environment file $ENV_FILE not found, creating a default one"
    
    # Create default environment file
    cat > "$PROJECT_ROOT/$ENV_FILE" << 'EOF'
# NVIDIA T4 GPU Environment Configuration
GPU_ENABLED=true
GPU_TYPE=T4
USE_TENSORRT=true
TENSORRT_PRECISION=fp16
GPU_BATCH_SIZE=24
GPU_MEMORY_THRESHOLD=15.0
PLATFORM=nvidia
PLATFORM_SUPPORTS_GPU=true
EOF
    
    echo "Created default environment file at $PROJECT_ROOT/$ENV_FILE"
fi

# Check NGC CLI is installed
if ! command -v ngc &> /dev/null; then
    echo "Error: NGC CLI is not installed or not in PATH"
    echo "Please install NGC CLI from: https://ngc.nvidia.com/setup/installers/cli"
    exit 1
fi

# Verify NGC authentication
echo "Verifying NGC authentication..."
ngc config get > /dev/null || { echo "Error: NGC authentication failed. Please run 'ngc config set' to configure your API key."; exit 1; }

# Create a T4-specific Dockerfile if it doesn't exist
T4_DOCKERFILE="$PROJECT_ROOT/api/Dockerfile.t4"
if [ ! -f "$T4_DOCKERFILE" ]; then
    echo "Creating T4-specific Dockerfile..."
    cat > "$T4_DOCKERFILE" << 'EOF'
FROM nvcr.io/nvidia/pytorch:22.12-py3

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorRT for optimized inference
RUN pip install --no-cache-dir nvidia-tensorrt

# Install additional dependencies
RUN pip install --no-cache-dir tritonclient[http] pynvml

# Copy application code
COPY . .

# Copy the environment file
COPY ../config/environment/.env.nvidia.t4.prod .env

# TensorRT optimization script
COPY optimize_models.py .
RUN python optimize_models.py --gpu_type T4 --precision fp16

# Set up the entrypoint
ENTRYPOINT ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
    
    # Create the model optimization script
    cat > "$PROJECT_ROOT/api/optimize_models.py" << 'EOF'
import os
import argparse
import torch
import tensorrt as trt
from transformers import AutoTokenizer, AutoModel

def optimize_model(model_name, gpu_type, precision, output_dir):
    """Optimize a Hugging Face model for TensorRT inference on specific GPU."""
    print(f"Optimizing {model_name} for {gpu_type} with {precision} precision...")
    
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    
    # Configure builder
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    
    # Set precision
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Using FP16 precision")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("Using INT8 precision")
    else:
        print("Using FP32 precision")
    
    # Set workspace size based on GPU type
    if gpu_type == "T4":
        config.max_workspace_size = 1 << 30  # 1GB for T4
    else:
        config.max_workspace_size = 4 << 30  # 4GB for other GPUs
    
    # Create ONNX parser
    parser = trt.OnnxParser(network, logger)
    
    # Export model to ONNX
    dummy_input = torch.ones(1, 128, dtype=torch.long).cuda()
    dummy_attention = torch.ones(1, 128, dtype=torch.long).cuda()
    
    # Define dynamic batch size
    dynamic_axes = {
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'output': {0: 'batch_size'},
    }
    
    # Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input, dummy_attention),
        f"{output_dir}/{model_name.replace('/', '_')}.onnx",
        input_names=['input_ids', 'attention_mask'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        opset_version=12
    )
    
    # Parse ONNX model
    with open(f"{output_dir}/{model_name.replace('/', '_')}.onnx", 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX model")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return
    
    # Build engine
    engine = builder.build_engine(network, config)
    if engine is None:
        print("Failed to build TensorRT engine")
        return
    
    # Save engine
    with open(f"{output_dir}/{model_name.replace('/', '_')}.engine", 'wb') as f:
        f.write(engine.serialize())
    
    print(f"Successfully optimized {model_name} for {gpu_type}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize models for TensorRT inference")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--gpu_type", type=str, default="T4")
    parser.add_argument("--precision", type=str, default="fp16")
    parser.add_argument("--output_dir", type=str, default="/app/trt_engines")
    
    args = parser.parse_args()
    optimize_model(args.model_name, args.gpu_type, args.precision, args.output_dir)
EOF
    
    echo "T4-specific Dockerfile and optimization script created"
fi

# Build backend image
if [ "$SKIP_BUILD" != true ]; then
    echo "Building backend Docker image optimized for T4 GPU..."
    cd "$PROJECT_ROOT/api"
    
    # Create environment directory if it doesn't exist
    mkdir -p "$PROJECT_ROOT/config/environment"
    
    # Copy environment file to api directory
    cp "$PROJECT_ROOT/$ENV_FILE" .env.t4
    
    # Build the image with T4-specific optimizations
    docker build -t "$BACKEND_IMAGE_NAME:$TAG" -f Dockerfile.t4 --build-arg GPU_TYPE=T4 .
    
    # Tag with NGC registry name
    docker tag "$BACKEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$TAG"
    docker tag "$BACKEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$VERSION"
    
    echo "T4-optimized backend image built successfully."
    
    # Build frontend image (if not skipped)
    if [ "$SKIP_FRONTEND" != true ]; then
        echo "Building frontend Docker image..."
        cd "$PROJECT_ROOT/frontend"
        
        # Check if Dockerfile exists
        if [ ! -f "Dockerfile" ]; then
            echo "Error: Dockerfile not found in $PROJECT_ROOT/frontend"
            exit 1
        fi
        
        # Build the frontend image
        docker build -t "$FRONTEND_IMAGE_NAME:$TAG" .
        
        # Tag with NGC registry name
        docker tag "$FRONTEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$FRONTEND_IMAGE_NAME:$TAG"
        docker tag "$FRONTEND_IMAGE_NAME:$TAG" "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$FRONTEND_IMAGE_NAME:$VERSION"
        
        echo "Frontend image built successfully."
    fi
fi

# Push images to NGC
if [ "$SKIP_PUSH" != true ]; then
    echo "Pushing T4-optimized backend image to NGC..."
    docker push "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$TAG"
    docker push "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$VERSION"
    
    echo "T4-optimized backend image pushed successfully."
    
    # Push frontend image (if not skipped)
    if [ "$SKIP_FRONTEND" != true ]; then
        echo "Pushing frontend image to NGC..."
        docker push "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$FRONTEND_IMAGE_NAME:$TAG"
        docker push "nvcr.io/$ORG_NAME/$COLLECTION_NAME/$FRONTEND_IMAGE_NAME:$VERSION"
        
        echo "Frontend image pushed successfully."
    fi
fi

# Create T4-specific LaunchPad configuration
mkdir -p "$PROJECT_ROOT/config/nvidia"
NVIDIA_T4_CONFIG="$PROJECT_ROOT/config/nvidia/nvidia-launchable-t4.yaml"
cat > "$NVIDIA_T4_CONFIG" << EOF
name: langchain-hana-t4
version: $VERSION
description: SAP HANA Cloud LangChain Integration optimized for T4 GPU
runtime:
  name: docker
  image: nvcr.io/$ORG_NAME/$COLLECTION_NAME/$BACKEND_IMAGE_NAME:$TAG
resources:
  gpus: 1
  gpu_type: T4
  cpu: 4
  memory: 16Gi
  disk: 20Gi
env:
  - name: GPU_TYPE
    value: "T4"
  - name: GPU_ENABLED
    value: "true"
  - name: PLATFORM
    value: "nvidia"
  - name: PLATFORM_SUPPORTS_GPU
    value: "true"
  - name: USE_TENSORRT
    value: "true"
  - name: TENSORRT_PRECISION
    value: "fp16"
  - name: GPU_BATCH_SIZE
    value: "24"
  - name: GPU_MEMORY_THRESHOLD
    value: "15.0"
ports:
  - name: http
    containerPort: 8000
    servicePort: 80
healthCheck:
  httpGet:
    path: /health/ping
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
EOF

# Create a symlink to the configuration file for backward compatibility
ln -sf "$NVIDIA_T4_CONFIG" "$PROJECT_ROOT/nvidia-launchable-t4.yaml"

echo "NVIDIA T4 LaunchPad configuration created successfully at $NVIDIA_T4_CONFIG"
echo "Symlink created at $PROJECT_ROOT/nvidia-launchable-t4.yaml for backward compatibility"

echo "T4-optimized NVIDIA LaunchPad deployment prepared successfully!"
echo ""
echo "Next steps:"
echo "1. Launch the application on NVIDIA LaunchPad with the T4 configuration:"
echo "   ngc launchpod launch --config $NVIDIA_T4_CONFIG"
echo "   or use the symlink for backward compatibility:"
echo "   ngc launchpod launch --config $PROJECT_ROOT/nvidia-launchable-t4.yaml"
echo "2. Connect to the application using the provided URL"
echo ""
echo "Note: This deployment is specifically optimized for T4 GPUs with:"
echo "- TensorRT FP16 precision for optimal T4 performance"
echo "- Reduced batch size for better memory utilization"
echo "- GPU memory optimizations specific to T4's 16GB VRAM"
echo "- Pre-built model optimizations for sentence embedding"