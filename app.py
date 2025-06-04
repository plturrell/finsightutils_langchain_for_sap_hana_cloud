"""
Simple FastAPI application to test NVIDIA GPU deployment
"""

import os
import sys
from fastapi import FastAPI, HTTPException
import torch
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any

# Create FastAPI app
app = FastAPI(
    title="Simple NVIDIA GPU Test",
    description="A simple API to test GPU availability and basic functionality",
    version="1.0.0",
)

# Define model for embeddings request
class EmbeddingRequest(BaseModel):
    texts: List[str]

# Define model for response
class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    gpu_used: bool
    dimensions: int

@app.get("/")
async def root():
    """Root endpoint providing basic information"""
    gpu_info = get_gpu_info()
    return {
        "message": "Simple NVIDIA GPU Test API is running",
        "status": "active",
        "gpu_available": torch.cuda.is_available(),
        "gpu_info": gpu_info,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/gpu-info")
async def gpu_info():
    """Get GPU information"""
    return get_gpu_info()

def get_gpu_info() -> Dict[str, Any]:
    """Get detailed GPU information"""
    gpu_info = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if hasattr(torch.version, "cuda") else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "devices": []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            device_info = {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated": f"{torch.cuda.memory_allocated(i) / 1024**2:.2f} MB",
                "memory_reserved": f"{torch.cuda.memory_reserved(i) / 1024**2:.2f} MB",
            }
            gpu_info["devices"].append(device_info)
    
    return gpu_info

@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """Generate embeddings for the given texts"""
    # Check if texts are provided
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")
    
    # Check GPU availability
    use_gpu = torch.cuda.is_available()
    device = "cuda" if use_gpu else "cpu"
    
    try:
        # Create simple random embeddings (for testing)
        embeddings = []
        dim = 384  # Standard dimension for testing
        
        for text in request.texts:
            # Create a deterministic but text-dependent embedding
            # This is just for testing - not a real embedding model
            seed = sum(ord(c) for c in text)
            torch.manual_seed(seed)
            
            if use_gpu:
                embedding = torch.rand(dim, device=device).cpu().numpy().tolist()
            else:
                embedding = torch.rand(dim).numpy().tolist()
            
            embeddings.append(embedding)
        
        return {
            "embeddings": embeddings,
            "model": "test-model",
            "gpu_used": use_gpu,
            "dimensions": dim
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)