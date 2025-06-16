#!/bin/bash
# Setup script for CPU-only mode

# Update and install dependencies
apt-get update && apt-get install -y curl build-essential

# Create required directories
mkdir -p /app/docs/pr_notes

# Install Python dependencies
pip install --no-cache-dir numpy scipy pandas scikit-learn torch
pip install --no-cache-dir -r /app/requirements.txt
pip install --no-cache-dir langchain langchain_core langchain_hana transformers sentence-transformers

# Create CPU-compatible dummy modules with proper fallbacks
mkdir -p /usr/local/lib/python3.10/site-packages/api/embeddings

# Copy multi_gpu module
cp /app/api/multi_gpu.py /usr/local/lib/python3.10/site-packages/multi_gpu.py

# Create gpu_utils.py
cat > /usr/local/lib/python3.10/site-packages/gpu_utils.py << 'EOL'
import logging
logger = logging.getLogger(__name__)

def get_gpu_info():
    return {"gpu_count": 0, "gpu_names": [], "cpu_only": True}

def get_gpu_utilization():
    return [{"id": 0, "name": "CPU", "utilization": 0, "memory": 0}]
EOL

# Create tensorrt_utils.py
cat > /usr/local/lib/python3.10/site-packages/tensorrt_utils.py << 'EOL'
import logging
logger = logging.getLogger(__name__)

class TensorRTEngine:
    def __init__(self, *args, **kwargs):
        logger.warning("TensorRT not available in CPU mode. Using dummy implementation.")

    def run(self, *args, **kwargs):
        logger.warning("TensorRT execution simulated in CPU mode.")
        return None

def optimize_model(*args, **kwargs):
    logger.warning("TensorRT optimization simulated in CPU mode.")
    return None
EOL

# Debug: Print Python path
python -c "import sys; print(sys.path)"

# Start the API server
PYTHONPATH=/app:/usr/local/lib/python3.10/site-packages uvicorn api.core.main:app --host 0.0.0.0 --port 8000 --reload
