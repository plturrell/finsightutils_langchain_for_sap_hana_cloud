"""
Minimal test endpoint for diagnosing Vercel function invocation issues.
This file uses minimal dependencies to isolate issues with function deployment.
"""

from fastapi import FastAPI
import os
import sys
import json
import time

# Create FastAPI app
app = FastAPI(
    title="Minimal Test API",
    description="Minimal API for testing Vercel function invocation",
    version="1.0.0",
)

@app.get("/")
async def root():
    """Root endpoint providing basic information"""
    return {
        "message": "Minimal Test API",
        "status": "active",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)