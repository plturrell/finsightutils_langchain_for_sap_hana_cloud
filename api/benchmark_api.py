"""API endpoints for benchmarking tools."""

import logging
import time
from typing import Dict, List, Optional, Union, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field

from benchmark import Benchmark
import gpu_utils
from tensorrt_utils import TENSORRT_AVAILABLE, tensorrt_optimizer
from app import get_embeddings
from embeddings_tensorrt import TensorRTEmbeddings, TensorRTHybridEmbeddings

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/benchmark", tags=["benchmark"])

# Global benchmark instance
benchmark = Benchmark()

# Current benchmark status
benchmark_status = {
    "is_running": False,
    "current_benchmark": None,
    "progress": 0.0,
    "start_time": None,
    "results": {},
}


class EmbeddingBenchmarkRequest(BaseModel):
    """Request for embedding benchmark."""
    texts: List[str] = Field(
        default=["This is a sample text for benchmarking embedding performance."],
        description="List of texts to embed for benchmark.",
    )
    count: int = Field(
        default=100,
        description="Number of times to run the embedding operation.",
        ge=1,
        le=10000,
    )
    batch_size: Optional[int] = Field(
        default=None,
        description="Batch size for embedding. If not provided, uses default from configuration.",
    )


class VectorSearchBenchmarkRequest(BaseModel):
    """Request for vector search benchmark."""
    query: str = Field(
        default="Sample query text",
        description="Query text to use for benchmark.",
    )
    k: int = Field(
        default=10,
        description="Number of results to return.",
        ge=1,
        le=1000,
    )
    iterations: int = Field(
        default=100,
        description="Number of search iterations to perform.",
        ge=1,
        le=10000,
    )


class TensorRTBenchmarkRequest(BaseModel):
    """Request for TensorRT benchmark."""
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model to benchmark",
    )
    precision: str = Field(
        default="fp16",
        description="Precision to use for TensorRT",
    )
    batch_sizes: List[int] = Field(
        default=[1, 8, 32, 64, 128],
        description="Batch sizes to benchmark",
    )
    input_length: int = Field(
        default=128,
        description="Input text length to use for benchmark",
    )
    iterations: int = Field(
        default=100,
        description="Number of iterations to run for each benchmark",
    )


@router.post("/embedding")
async def benchmark_embedding(request: EmbeddingBenchmarkRequest):
    """Benchmark embedding performance."""
    if not gpu_utils.is_gpu_available():
        raise HTTPException(
            status_code=400,
            detail="GPU not available for benchmarking.",
        )
    
    if benchmark_status["is_running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark already running: {benchmark_status['current_benchmark']}",
        )
    
    try:
        benchmark_status["is_running"] = True
        benchmark_status["current_benchmark"] = "embedding"
        benchmark_status["start_time"] = time.time()
        
        result = benchmark.benchmark_embedding(
            texts=request.texts,
            count=request.count,
            batch_size=request.batch_size,
        )
        
        benchmark_status["results"] = result
        benchmark_status["is_running"] = False
        
        return result
    except Exception as e:
        benchmark_status["is_running"] = False
        logger.exception("Error running embedding benchmark")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}",
        )


@router.post("/search")
async def benchmark_vector_search(request: VectorSearchBenchmarkRequest):
    """Benchmark vector search performance."""
    if benchmark_status["is_running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark already running: {benchmark_status['current_benchmark']}",
        )
    
    try:
        benchmark_status["is_running"] = True
        benchmark_status["current_benchmark"] = "vector_search"
        benchmark_status["start_time"] = time.time()
        
        result = benchmark.benchmark_vector_search(
            query=request.query,
            k=request.k,
            iterations=request.iterations,
        )
        
        benchmark_status["results"] = result
        benchmark_status["is_running"] = False
        
        return result
    except Exception as e:
        benchmark_status["is_running"] = False
        logger.exception("Error running vector search benchmark")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}",
        )


@router.post("/tensorrt")
async def benchmark_tensorrt(request: TensorRTBenchmarkRequest):
    """Benchmark TensorRT vs PyTorch embedding performance."""
    if not TENSORRT_AVAILABLE:
        raise HTTPException(
            status_code=400,
            detail="TensorRT not available. Please install TensorRT and torch-tensorrt.",
        )
    
    if not gpu_utils.is_gpu_available():
        raise HTTPException(
            status_code=400,
            detail="GPU not available for benchmarking.",
        )
    
    if benchmark_status["is_running"]:
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark already running: {benchmark_status['current_benchmark']}",
        )
    
    try:
        benchmark_status["is_running"] = True
        benchmark_status["current_benchmark"] = "tensorrt"
        benchmark_status["start_time"] = time.time()
        
        # Create benchmark text of specified length
        import random
        import string
        random_text = ''.join(random.choices(string.ascii_letters + ' ', k=request.input_length))
        
        results = {
            "model": request.model_name,
            "precision": request.precision,
            "input_length": request.input_length,
            "iterations": request.iterations,
            "gpu_info": gpu_utils.get_gpu_info(),
            "batch_results": [],
        }
        
        # Run benchmarks for each batch size
        for batch_size in request.batch_sizes:
            # Create standard PyTorch embeddings
            from sentence_transformers import SentenceTransformer
            pytorch_model = SentenceTransformer(request.model_name, device="cuda")
            
            # Warmup
            _ = pytorch_model.encode([random_text])
            
            # Benchmark PyTorch
            batch = [random_text] * batch_size
            start_time = time.time()
            for _ in range(request.iterations):
                _ = pytorch_model.encode(batch, convert_to_numpy=True)
            pytorch_time = (time.time() - start_time) / request.iterations
            
            # Create TensorRT optimized embeddings
            tensorrt_embeddings = TensorRTEmbeddings(
                model_name=request.model_name,
                device="cuda",
                batch_size=batch_size,
                use_tensorrt=True,
                precision=request.precision,
            )
            
            # Warmup
            _ = tensorrt_embeddings.embed_query(random_text)
            
            # Benchmark TensorRT
            start_time = time.time()
            for _ in range(request.iterations):
                _ = tensorrt_embeddings.embed_documents(batch)
            tensorrt_time = (time.time() - start_time) / request.iterations
            
            # Record results
            batch_result = {
                "batch_size": batch_size,
                "pytorch_time_ms": pytorch_time * 1000,
                "tensorrt_time_ms": tensorrt_time * 1000,
                "speedup_factor": pytorch_time / tensorrt_time if tensorrt_time > 0 else 0,
                "pytorch_throughput": batch_size / pytorch_time,
                "tensorrt_throughput": batch_size / tensorrt_time,
            }
            results["batch_results"].append(batch_result)
        
        benchmark_status["results"] = results
        benchmark_status["is_running"] = False
        
        return results
    except Exception as e:
        benchmark_status["is_running"] = False
        logger.exception("Error running TensorRT benchmark")
        raise HTTPException(
            status_code=500,
            detail=f"Benchmark failed: {str(e)}",
        )


@router.get("/status")
async def get_benchmark_status():
    """Get current benchmark status."""
    status = dict(benchmark_status)
    
    if status["is_running"] and status["start_time"]:
        status["elapsed_time"] = time.time() - status["start_time"]
    
    return status


@router.get("/gpu_info")
async def get_gpu_info():
    """Get GPU information."""
    return gpu_utils.get_gpu_info()


@router.post("/compare_embeddings")
async def compare_embedding_models(embeddings_service=Depends(get_embeddings)):
    """Compare different embedding models."""
    if isinstance(embeddings_service, TensorRTHybridEmbeddings):
        # Run benchmark comparison between internal and TensorRT embeddings
        return embeddings_service.benchmark_comparison()
    else:
        raise HTTPException(
            status_code=400,
            detail="TensorRT hybrid embeddings not configured. Set USE_TENSORRT=true and USE_INTERNAL_EMBEDDINGS=true.",
        )