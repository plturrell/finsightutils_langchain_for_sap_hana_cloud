"""
Minimal test endpoint for diagnosing API functionality with minimal dependencies.
"""

from fastapi import FastAPI
import os
import sys
import json
import time

# Create FastAPI app
app = FastAPI(
    title="SAP HANA Cloud LangChain API - Test Mode",
    description="Minimal API for testing deployment",
    version="1.0.0",
)

@app.get("/")
async def root():
    """Root endpoint providing basic information"""
    return {
        "message": "SAP HANA Cloud LangChain API - Test Mode",
        "status": "active",
        "timestamp": time.time()
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "API is healthy",
        "timestamp": time.time()
    }

@app.get("/environment")
async def environment():
    """Get basic environment information"""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    return {
        "python_version": python_version,
        "platform": sys.platform,
        "environment_variables": {k: v for k, v in os.environ.items() 
                                if "password" not in k.lower() 
                                and "secret" not in k.lower()
                                and not k.startswith("AWS_")},
        "timestamp": time.time()
    }

@app.get("/sys-path")
async def sys_path():
    """Get Python sys.path information"""
    return {
        "sys_path": sys.path,
        "timestamp": time.time()
    }

@app.get("/gpu/info")
async def gpu_info():
    """Get GPU information if available"""
    gpu_available = False
    cuda_version = None
    device_count = 0
    devices = []
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            
            for i in range(device_count):
                device_info = {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_allocated": f"{torch.cuda.memory_allocated(i) / 1024**2:.2f} MB",
                    "memory_reserved": f"{torch.cuda.memory_reserved(i) / 1024**2:.2f} MB",
                }
                devices.append(device_info)
    except ImportError:
        pass
    
    return {
        "gpu_available": gpu_available,
        "cuda_version": cuda_version,
        "device_count": device_count,
        "devices": devices
    }

@app.get("/api/feature/vector-similarity")
async def vector_similarity_info():
    """Information about the vector similarity feature."""
    return {
        "feature": "Vector Similarity",
        "version": "1.0.0",
        "description": "Vector similarity measurement for embeddings",
        "status": "enabled in test mode"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)