"""
Simplified deployment script for Brev Container with Python and CUDA preinstalled.
This script sets up and runs the minimal API with test mode enabled.
"""

import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("brev_deploy")

def check_gpu():
    """Check if GPU is available and print information."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        
        logger.info(f"CUDA available: {cuda_available}")
        logger.info(f"GPU count: {gpu_count}")
        
        if cuda_available and gpu_count > 0:
            for i in range(gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        return cuda_available
    except ImportError:
        logger.warning("PyTorch not available, skipping GPU check")
        return False
    except Exception as e:
        logger.error(f"Error checking GPU: {e}")
        return False

def install_dependencies():
    """Install the required Python dependencies."""
    try:
        logger.info("Installing basic requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "--no-cache-dir", 
            "fastapi", "uvicorn", "pydantic"
        ], check=True)
        
        logger.info("Installing additional requirements...")
        api_requirements = os.path.join("api", "requirements.txt")
        if os.path.exists(api_requirements):
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--no-cache-dir", 
                "-r", api_requirements
            ], check=True)
        
        # Try to install GPU-related packages if needed
        has_gpu = check_gpu()
        if has_gpu:
            logger.info("Installing GPU-related packages...")
            subprocess.run([
                sys.executable, "-m", "pip", "install", "--no-cache-dir",
                "torch", "sentence-transformers"
            ], check=True)
        
        return True
    except Exception as e:
        logger.error(f"Error installing dependencies: {e}")
        return False

def run_api(test_mode=True):
    """Run the FastAPI application."""
    try:
        # Set environment variables
        os.environ["TEST_MODE"] = "true" if test_mode else "false"
        os.environ["ENABLE_CORS"] = "true"
        os.environ["LOG_LEVEL"] = "INFO"
        
        # Determine which API to run
        if os.path.exists(os.path.join("api", "ultra_minimal.py")):
            logger.info("Running ultra minimal API...")
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "api.ultra_minimal:app", "--host", "0.0.0.0", "--port", "8000"
            ]
        else:
            logger.info("Running standard API...")
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "api.app:app", "--host", "0.0.0.0", "--port", "8000"
            ]
        
        # Run the API
        process = subprocess.Popen(cmd)
        
        # Wait a bit for the API to start
        time.sleep(5)
        
        # Check if the API is running
        try:
            health_check = subprocess.run(
                ["curl", "-s", "http://localhost:8000/health/ping"],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"API health check: {health_check.stdout.strip()}")
            logger.info("API is running successfully")
        except subprocess.CalledProcessError:
            logger.warning("API health check failed, but process is still running")
        
        # Keep the process running
        process.wait()
    except Exception as e:
        logger.error(f"Error running API: {e}")

def main():
    """Main entry point for the deployment script."""
    logger.info("Starting Brev deployment script")
    
    # Check if we're in the correct directory
    if not os.path.exists("api"):
        # Try to find the correct directory
        if os.path.exists("langchain-integration-for-sap-hana-cloud"):
            os.chdir("langchain-integration-for-sap-hana-cloud")
        else:
            logger.error("Cannot find the API directory")
            return False
    
    # Install dependencies
    if not install_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Run the API
    run_api(test_mode=True)
    
    return True

if __name__ == "__main__":
    main()