"""
Enhanced FastAPI application for NVIDIA Launchable deployment
Includes health checks and GPU capability testing
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sys
import torch
import platform
import psutil
import json
import time
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="SAP HANA Cloud LangChain Integration",
    description="API for SAP HANA Cloud vector store operations with NVIDIA GPU acceleration",
    version="1.0.2",
)

# Add CORS middleware if enabled
if os.environ.get("ENABLE_CORS", "false").lower() == "true":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Define model for embeddings request
class EmbeddingRequest(BaseModel):
    texts: List[str]
    model: Optional[str] = "test-embedding-model"

# Define model for response
class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    gpu_used: bool
    dimensions: int
    time_taken: float

# Basic health check routes
@app.get("/")
async def root():
    """Root endpoint providing basic information"""
    return {
        "message": "SAP HANA Cloud LangChain Integration API",
        "status": "UP",
        "version": "1.0.2",
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "UP"}

@app.get("/healthz")
async def healthz():
    """Kubernetes-style health check endpoint"""
    return {"status": "UP"}

@app.get("/live")
async def live():
    """Liveness probe endpoint"""
    return {"status": "UP"}

@app.get("/ready")
async def ready():
    """Readiness probe endpoint"""
    return {"status": "UP"}

@app.get("/api/health")
async def api_health():
    """API health check endpoint"""
    return {"status": "UP"}

@app.get("/system-info")
async def system_info():
    """Get detailed system information"""
    info = {
        "hostname": platform.node(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "memory_total": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "memory_available": f"{psutil.virtual_memory().available / (1024**3):.2f} GB",
        "gpu_info": get_gpu_info()
    }
    return info

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
                "max_memory_cached": f"{torch.cuda.max_memory_reserved(i) / 1024**2:.2f} MB",
            }
            gpu_info["devices"].append(device_info)
        
        # Get additional TensorRT information if available
        if os.environ.get("USE_TENSORRT", "false").lower() == "true":
            try:
                import tensorrt as trt
                gpu_info["tensorrt_version"] = trt.__version__
                gpu_info["tensorrt_precision"] = os.environ.get("TENSORRT_PRECISION", "fp32")
            except ImportError:
                gpu_info["tensorrt_available"] = False
    
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
        # Record start time
        start_time = time.time()
        
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
        
        # Record end time
        end_time = time.time()
        time_taken = end_time - start_time
        
        return {
            "embeddings": embeddings,
            "model": request.model,
            "gpu_used": use_gpu,
            "dimensions": dim,
            "time_taken": time_taken
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

@app.get("/tensor-test")
async def tensor_test():
    """Run a simple tensor operation to test GPU performance"""
    try:
        # Record start time
        start_time = time.time()
        
        # Create tensors
        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda" if use_gpu else "cpu")
        
        # Create large tensors for multiplication
        size = 2000
        a = torch.rand(size, size, device=device)
        b = torch.rand(size, size, device=device)
        
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        
        # Ensure operation is complete (especially important for GPU)
        if use_gpu:
            torch.cuda.synchronize()
        
        # Record end time
        end_time = time.time()
        time_taken = end_time - start_time
        
        # Get result shape and a small sample
        result_shape = c.shape
        result_sample = c[0:3, 0:3].cpu().numpy().tolist() if use_gpu else c[0:3, 0:3].numpy().tolist()
        
        return {
            "operation": "matrix_multiplication",
            "matrix_size": f"{size}x{size}",
            "device": "GPU" if use_gpu else "CPU",
            "time_taken": time_taken,
            "result_shape": result_shape,
            "result_sample": result_sample
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in tensor test: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)