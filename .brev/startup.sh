#!/bin/bash
# Brev LaunchPad startup script

set -e

# Display banner
echo "======================================================================"
echo "  NVIDIA Brev LaunchPad - SAP HANA Cloud LangChain Integration"
echo "======================================================================"

# Check NVIDIA GPU availability
echo "Checking GPU availability..."
nvidia-smi
if [ $? -ne 0 ]; then
  echo "Warning: NVIDIA GPU not detected or nvidia-smi not available."
  echo "This application requires GPU acceleration to function properly."
fi

# Set up environment
echo "Setting up environment..."

# Create TensorRT cache directory
mkdir -p /app/trt_engines
mkdir -p /app/data
mkdir -p /app/logs

# Install dependencies if needed
if [ ! -f "/.dependencies_installed" ]; then
  echo "Installing dependencies..."
  pip install --no-cache-dir -r requirements.txt
  pip install --no-cache-dir nvidia-tensorrt
  touch "/.dependencies_installed"
fi

# Verify Python environment and GPU availability
echo "Verifying Python environment..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

# Start services
echo "Starting services..."

# Start API service
echo "Starting API service..."
python -m uvicorn api.core.main:app --host 0.0.0.0 --port 8000 --workers 4 &
API_PID=$!

# Wait for API to be ready
echo "Waiting for API to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:8000/health/ping > /dev/null; then
    echo "API is ready!"
    break
  fi
  echo "Waiting for API to start... ($i/30)"
  sleep 2
done

# Start frontend service if it exists
if [ -d "./frontend" ]; then
  echo "Starting frontend service..."
  cd frontend
  if [ -f "package.json" ]; then
    npm install
    npm start &
    FRONTEND_PID=$!
  fi
  cd ..
fi

echo "======================================================================"
echo "  Services started successfully!"
echo "======================================================================"
echo "API URL: http://localhost:8000"
echo "Frontend URL: http://localhost:3000 (if available)"
echo "API Documentation: http://localhost:8000/docs"
echo "======================================================================"

# Keep the script running to maintain the services
wait $API_PID