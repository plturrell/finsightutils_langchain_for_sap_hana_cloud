#\!/bin/bash
#
# Brev LaunchPad startup script for SAP HANA Cloud LangChain Integration
# This script initializes the environment and prepares the application for deployment

set -e  # Exit on error

# Print environment information
echo "Starting Brev LaunchPad initialization..."
echo "=============================================="
echo "Environment: $(uname -a)"
date
echo "=============================================="

# Create required directories
mkdir -p /app/trt_engines /app/models /app/calibration_cache /app/data /app/logs

# Setup monitoring directories
mkdir -p /etc/dcgm-exporter /etc/prometheus

# Verify GPU is available
if \! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: NVIDIA GPU drivers not found. This application requires GPU acceleration."
    echo "Please ensure NVIDIA drivers are properly installed."
    exit 1
fi

echo "NVIDIA GPU information:"
nvidia-smi

# Check for required environment variables
required_vars=("HANA_HOST" "HANA_USER" "HANA_PASSWORD" "JWT_SECRET")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [[ -z "${\!var}" ]]; then
        missing_vars+=("$var")
    fi
done

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: Missing required environment variables: ${missing_vars[*]}"
    echo "Please set these variables before continuing."
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir -r requirements-monitoring.txt

# Install NVIDIA libraries if not already installed
if \! python -c "import tensorrt" &> /dev/null; then
    echo "Installing NVIDIA TensorRT..."
    pip install --no-cache-dir nvidia-tensorrt
fi

if \! python -c "import triton" &> /dev/null; then
    echo "Installing NVIDIA Triton client..."
    pip install --no-cache-dir tritonclient[all]
fi

if \! python -c "import dali" &> /dev/null; then
    echo "Installing NVIDIA DALI..."
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110
fi

# Setup DCGM Exporter configuration
echo "Configuring DCGM Exporter..."
cat > /etc/dcgm-exporter/default-counters.csv << DCGMEOF
# Format,,
# DCGM FI Field ID, Prometheus metric type, help message

# GPU utilization
DCGM_FI_DEV_GPU_UTIL, gauge, GPU utilization (in %)

# Memory utilization
DCGM_FI_DEV_FB_USED, gauge, GPU framebuffer memory used (in MiB)
DCGM_FI_DEV_FB_FREE, gauge, GPU framebuffer memory free (in MiB)
DCGM_FI_DEV_FB_TOTAL, gauge, GPU framebuffer memory total (in MiB)

# SM clocks
DCGM_FI_DEV_SM_CLOCK, gauge, SM clock frequency (in MHz)

# Memory clocks
DCGM_FI_DEV_MEM_CLOCK, gauge, Memory clock frequency (in MHz)

# Power usage
DCGM_FI_DEV_POWER_USAGE, gauge, Power usage (in W)

# Temperature
DCGM_FI_DEV_GPU_TEMP, gauge, GPU temperature (in C)

# PCIe throughput
DCGM_FI_DEV_PCIE_TX_THROUGHPUT, gauge, PCIe transmit throughput (in KB/s)
DCGM_FI_DEV_PCIE_RX_THROUGHPUT, gauge, PCIe receive throughput (in KB/s)
DCGMEOF

# Setup Prometheus configuration
echo "Configuring Prometheus..."
cat > /etc/prometheus/prometheus.yml << PROMEOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s
scrape_configs:
  - job_name: 'dcgm'
    static_configs:
      - targets: ['dcgm-exporter:9400']
  - job_name: 'api'
    static_configs:
      - targets: ['api:8000']
  - job_name: 'triton'
    static_configs:
      - targets: ['triton-server:8002']
PROMEOF

# Optimize models with TensorRT if enabled
if [[ "${USE_TENSORRT:-true}" == "true" ]]; then
    echo "Optimizing models with TensorRT..."
    python -m scripts.optimize_models \
        --model-name="${MODEL_NAME:-all-MiniLM-L6-v2}" \
        --precision="${TENSORRT_PRECISION:-fp16}" \
        --batch-sizes="${BATCH_SIZES:-1,2,4,8,16,32,64,128}" \
        --calibration-cache="/app/calibration_cache" \
        --export-format="tensorrt,onnx" \
        --output-dir="/app/models" \
        --cache-dir="/app/trt_engines" \
        --triton-model-repository="/models"
fi

# Create metadata file
echo "Creating metadata file..."
cat > /app/metadata.json << METAEOF
{
  "name": "finsightutils-langchain-for-sap-hana-cloud",
  "version": "1.0.0",
  "description": "SAP HANA Cloud LangChain Integration with NVIDIA GPU acceleration",
  "startup_time": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
  "environment": "brev-launchpad"
}
METAEOF

echo "Initialization complete. Starting services..."
