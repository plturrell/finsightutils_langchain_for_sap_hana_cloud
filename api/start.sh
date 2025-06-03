#!/bin/bash
# Startup script for the SAP HANA Cloud LangChain Integration API with GPU acceleration

set -e

# Check if .env file exists (for local development)
if [ ! -f .env ] && [ "$KUBERNETES_SERVICE_HOST" == "" ] && [ "$DOCKER_CONTAINER" == "" ]; then
    echo "Warning: .env file not found. Using environment variables."
    echo "For local development, create .env file from .env.example: cp .env.example .env"
fi

# Check if NVIDIA GPUs are available
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU acceleration may not be available."
    export GPU_ENABLED=false
    export USE_TENSORRT=false
else
    # Check if GPUs are accessible
    if ! nvidia-smi &> /dev/null; then
        echo "WARNING: NVIDIA GPUs detected but not accessible. GPU acceleration may not be available."
        export GPU_ENABLED=false
        export USE_TENSORRT=false
    else
        echo "NVIDIA GPUs detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        
        # Check if TensorRT is available
        python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')" &> /dev/null || \
        { 
            echo "WARNING: TensorRT not found. TensorRT optimization will be disabled."
            export USE_TENSORRT=false
        }
        
        if [ "$USE_TENSORRT" = "true" ]; then
            echo "TensorRT optimization enabled with precision: $TENSORRT_PRECISION"
        fi
    fi
fi

# Check for database connection
if [ -z "$HANA_HOST" ] || [ -z "$HANA_PORT" ] || [ -z "$HANA_USER" ] || [ -z "$HANA_PASSWORD" ]; then
    echo "WARNING: SAP HANA Cloud credentials not fully provided. The API will start but might not be able to connect to the database."
    echo "Please set HANA_HOST, HANA_PORT, HANA_USER, and HANA_PASSWORD environment variables."
fi

# Determine run mode
if [ "$1" == "prod" ] || [ "$ENVIRONMENT" == "production" ]; then
    echo "Starting SAP HANA Cloud LangChain Integration API in PRODUCTION mode..."
    RELOAD_FLAG=""
else
    echo "Starting SAP HANA Cloud LangChain Integration API in DEVELOPMENT mode..."
    RELOAD_FLAG="--reload"
fi

echo "API will be available at http://0.0.0.0:8000"
echo "Documentation available at http://0.0.0.0:8000/docs"
echo "GPU Info endpoint: http://0.0.0.0:8000/benchmark/gpu_info"

if [ "$GPU_ENABLED" = "true" ]; then
    echo "GPU acceleration is ENABLED"
else
    echo "GPU acceleration is DISABLED"
fi

# Start the API
LOG_LEVEL=${LOG_LEVEL:-"info"}
exec uvicorn app:app --host 0.0.0.0 --port 8000 --log-level ${LOG_LEVEL,,} $RELOAD_FLAG