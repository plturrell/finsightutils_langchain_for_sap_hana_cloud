"""
TensorRT utilities for optimizing embedding operations.
"""
import os
import logging
from typing import List, Optional, Dict, Any, Union
import time
import torch
from pathlib import Path
import numpy as np

# Check if TensorRT is available
try:
    import tensorrt as trt
    import torch_tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

from gpu_utils import get_available_gpu_memory, is_gpu_available

logger = logging.getLogger(__name__)

class TensorRTOptimizer:
    """
    Handles TensorRT optimization for embedding models.
    """
    def __init__(
        self,
        cache_dir: str = "/tmp/tensorrt_engines",
        precision: str = "fp16",
        enable_caching: bool = True,
    ):
        """
        Initialize TensorRT optimizer.
        
        Args:
            cache_dir: Directory to cache compiled TensorRT engines
            precision: Precision to use for TensorRT ('fp32', 'fp16', or 'int8')
            enable_caching: Whether to cache and reuse compiled engines
        """
        self.cache_dir = cache_dir
        self.precision = precision
        self.enable_caching = enable_caching
        self.engines = {}
        
        if not TENSORRT_AVAILABLE:
            logger.warning("TensorRT not available. Using PyTorch directly.")
            return
            
        if not is_gpu_available():
            logger.warning("GPU not available. TensorRT optimization disabled.")
            return
            
        # Create cache directory if it doesn't exist
        if self.enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
            
        logger.info(f"TensorRT optimizer initialized with precision {precision}")
        
    def _get_engine_path(self, model_name: str) -> str:
        """Get path for cached TensorRT engine."""
        sanitized_name = model_name.replace('/', '_').replace('\\', '_')
        return os.path.join(self.cache_dir, f"{sanitized_name}_{self.precision}.engine")
        
    def optimize_model(
        self, 
        model: torch.nn.Module,
        model_name: str,
        input_shape: List[int] = [1, 512],
        max_batch_size: int = 128,
        dynamic_shapes: bool = True,
    ) -> Optional[torch.nn.Module]:
        """
        Optimize PyTorch model with TensorRT.
        
        Args:
            model: PyTorch model to optimize
            model_name: Name of the model (used for caching)
            input_shape: Input shape for compilation
            max_batch_size: Maximum batch size
            dynamic_shapes: Whether to use dynamic shapes
            
        Returns:
            Optimized TensorRT model or original model if optimization fails
        """
        if not TENSORRT_AVAILABLE or not is_gpu_available():
            return model
            
        engine_path = self._get_engine_path(model_name)
        
        # Check if engine already exists in cache
        if self.enable_caching and os.path.exists(engine_path):
            try:
                logger.info(f"Loading cached TensorRT engine for {model_name}")
                return self._load_engine(engine_path, model)
            except Exception as e:
                logger.warning(f"Failed to load cached engine: {e}. Recompiling...")
                
        # Configure precision
        enabled_precisions = {torch.float32}
        if self.precision == "fp16":
            enabled_precisions.add(torch.float16)
            
        # Create dynamic shapes if needed
        if dynamic_shapes:
            input_shapes = [
                (1, input_shape[1]),             # min shape
                (max_batch_size//2, input_shape[1]),  # opt shape
                (max_batch_size, input_shape[1])      # max shape
            ]
            dynamic_batch = True
        else:
            input_shapes = [(input_shape[0], input_shape[1])]
            dynamic_batch = False
            
        logger.info(f"Optimizing model {model_name} with TensorRT (precision: {self.precision})")
        start_time = time.time()
        
        try:
            # Move model to GPU for compilation
            model = model.cuda().eval()
            
            # Compile with Torch-TensorRT
            optimized_model = torch_tensorrt.compile(
                model,
                inputs=[
                    torch_tensorrt.Input(
                        input_shapes,
                        dtype=torch.float32,
                        dynamic_batch=dynamic_batch,
                    )
                ],
                enabled_precisions=enabled_precisions,
                workspace_size=1 << 30,  # 1GB workspace
                require_full_compilation=False,
            )
            
            logger.info(f"TensorRT optimization completed in {time.time() - start_time:.2f}s")
            
            # Cache the engine if enabled
            if self.enable_caching:
                self._save_engine(optimized_model, engine_path)
                
            return optimized_model
            
        except Exception as e:
            logger.error(f"TensorRT optimization failed: {e}")
            logger.info("Falling back to original PyTorch model")
            return model
            
    def _save_engine(self, optimized_model: torch.nn.Module, path: str) -> None:
        """Save TensorRT engine to disk."""
        try:
            torch.save(optimized_model.state_dict(), path)
            logger.info(f"Saved TensorRT engine to {path}")
        except Exception as e:
            logger.warning(f"Failed to save TensorRT engine: {e}")
            
    def _load_engine(self, path: str, original_model: torch.nn.Module) -> torch.nn.Module:
        """Load TensorRT engine from disk."""
        if path in self.engines:
            return self.engines[path]
            
        # Load the engine
        state_dict = torch.load(path)
        original_model.load_state_dict(state_dict)
        
        # Cache for future use
        self.engines[path] = original_model
        return original_model
        
    def get_optimal_precision(self) -> str:
        """Determine optimal precision based on GPU capabilities."""
        if not TENSORRT_AVAILABLE or not is_gpu_available():
            return "fp32"
            
        # Check if Tensor Cores are available (Volta, Turing, Ampere or newer)
        cuda_capability = torch.cuda.get_device_capability(0)
        major, minor = cuda_capability
        
        if major >= 7:  # Volta or newer architecture
            return "fp16"  # Use FP16 for best performance
        else:
            return "fp32"  # Use FP32 for older GPUs
            
    def benchmark_inference(
        self, 
        model: torch.nn.Module,
        input_shape: List[int] = [1, 512],
        iterations: int = 100,
        warmup: int = 10,
    ) -> Dict[str, Any]:
        """
        Benchmark inference performance.
        
        Args:
            model: Model to benchmark
            input_shape: Input shape for benchmark
            iterations: Number of iterations
            warmup: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        if not is_gpu_available():
            return {"error": "GPU not available"}
            
        # Move model to GPU and set to eval mode
        model = model.cuda().eval()
        
        # Create random input tensor
        input_tensor = torch.randn(*input_shape, device="cuda")
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_tensor)
                
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(iterations):
                start_time = time.time()
                _ = model(input_tensor)
                torch.cuda.synchronize()  # Wait for GPU operations to complete
                latencies.append((time.time() - start_time) * 1000)  # ms
                
        # Calculate statistics
        latencies = np.array(latencies)
        result = {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "iterations": iterations,
            "input_shape": input_shape,
            "precision": self.precision,
            "throughput_samples_per_second": float(1000 / np.mean(latencies)),
        }
        
        return result


# Global optimizer instance
tensorrt_optimizer = TensorRTOptimizer(
    precision=os.environ.get("TENSORRT_PRECISION", "fp16"),
    enable_caching=os.environ.get("TENSORRT_CACHE", "1") == "1",
)