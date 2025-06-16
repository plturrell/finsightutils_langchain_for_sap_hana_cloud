#!/usr/bin/env python
"""
Integration test for FastAPI endpoints to verify proper embedding initialization
in both CPU and GPU environments.
"""

import os
import sys
import json
import logging
import requests
from pathlib import Path
from contextlib import contextmanager
import subprocess
import time
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_integration_test")

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# API constants
API_HOST = "localhost"
API_PORT = 8000
API_URL = f"http://{API_HOST}:{API_PORT}"

# Test Flow data
TEST_FLOW = {
    "id": "test-flow-id",
    "name": "Test Embedding Flow",
    "description": "Flow to test embedding initialization",
    "nodes": [
        {
            "id": "embedding-node",
            "type": "embedding",
            "data": {
                "type": "embedding",
                "params": {
                    "model_name": "all-MiniLM-L6-v2",
                    "useHanaInternal": True
                }
            },
            "position": {"x": 0, "y": 0}
        }
    ],
    "edges": []
}

@contextmanager
def start_api_server(env_vars=None):
    """Start the API server in a subprocess and yield, then terminate it."""
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    # Start the server
    cmd = [sys.executable, "-m", "uvicorn", "api.main:app", "--host", API_HOST, "--port", str(API_PORT)]
    logger.info(f"Starting API server: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start (up to 10 seconds)
    for _ in range(10):
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200:
                logger.info("API server is ready")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        logger.error("Failed to start API server")
        process.terminate()
        stdout, stderr = process.communicate()
        logger.error(f"Server stdout: {stdout}")
        logger.error(f"Server stderr: {stderr}")
        raise RuntimeError("API server failed to start")
    
    try:
        yield process
    finally:
        logger.info("Terminating API server")
        process.send_signal(signal.SIGINT)
        process.wait(timeout=5)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)
        if process.poll() is None:
            process.kill()

def test_run_flow():
    """Test the run flow endpoint."""
    url = f"{API_URL}/api/v1/developer/flow/run"
    response = requests.post(url, json={"flow": TEST_FLOW})
    
    assert response.status_code == 200, f"API request failed with status {response.status_code}: {response.text}"
    data = response.json()
    
    assert data["success"], f"Flow execution failed: {data.get('error', 'Unknown error')}"
    logger.info("Flow execution succeeded")
    
    # Check the generated code to verify embedding initialization
    code = data.get("generated_code", "")
    assert "embeddings" in code, "Generated code does not contain embeddings initialization"
    logger.info("Generated code contains embedding initialization")
    
    return data

def test_with_cpu_mode():
    """Run tests in CPU-only mode."""
    logger.info("TESTING WITH CPU-ONLY MODE")
    with start_api_server(env_vars={"FORCE_CPU": "1"}):
        data = test_run_flow()
        # Could do additional verification of the response here
        logger.info("CPU-only mode test completed successfully")
        return data

def test_with_native_environment():
    """Run tests in the native environment (may use GPU if available)."""
    logger.info("TESTING WITH NATIVE ENVIRONMENT")
    with start_api_server():
        data = test_run_flow()
        # Could do additional verification of the response here
        logger.info("Native environment test completed successfully")
        return data

def test_with_gpu_simulation():
    """Run tests with simulated GPU environment."""
    logger.info("TESTING WITH SIMULATED GPU ENVIRONMENT")
    # Set environment variables to simulate GPU environment
    env_vars = {
        "SIMULATE_GPU": "1",
        "CUDA_VISIBLE_DEVICES": "0"
    }
    with start_api_server(env_vars=env_vars):
        data = test_run_flow()
        # Could do additional verification of the response here
        logger.info("Simulated GPU environment test completed successfully")
        return data

if __name__ == "__main__":
    logger.info("STARTING API INTEGRATION TESTS")
    
    try:
        # Test with CPU-only mode
        cpu_data = test_with_cpu_mode()
        
        # Test with native environment
        native_data = test_with_native_environment()
        
        # Test with simulated GPU environment
        # Uncomment if you have the ability to simulate GPU
        # gpu_data = test_with_gpu_simulation()
        
        logger.info("ALL TESTS COMPLETED SUCCESSFULLY")
    except Exception as e:
        logger.exception(f"Test failed: {str(e)}")
        sys.exit(1)
