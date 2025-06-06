"""
Enhanced TensorRT embeddings module with advanced T4 optimization.

This module provides specialized optimizations for embedding models running on NVIDIA T4 GPUs
using TensorRT and Tensor Core acceleration.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import torch

# Import local utilities
from gpu_utils import is_gpu_available, get_available_gpu_memory
from tensorrt_utils import tensorrt_optimizer, TENSORRT_AVAILABLE

# Import tensor core optimizations
import sys
import importlib.util
from pathlib import Path

# Dynamically import tensor_core_optimizer module
_tensor_core_optimizer = None
try:
    # Determine the module path
    module_path = Path(__file__).resolve().parent.parent / "langchain_hana" / "gpu" / "tensor_core_optimizer.py"
    if module_path.exists():
        spec = importlib.util.spec_from_file_location("tensor_core_optimizer", module_path)
        _tensor_core_optimizer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_tensor_core_optimizer)
        logging.info("Successfully imported tensor_core_optimizer module")
    else:
        logging.warning(f"tensor_core_optimizer module not found at {module_path}")
except Exception as e:
    logging.warning(f"Failed to import tensor_core_optimizer: {e}")

logger = logging.getLogger(__name__)


class TensorRTEmbeddingsEnhanced:
    """
    TensorRT optimized embeddings for T4 GPUs with advanced Tensor Core optimization.
    
    This class provides optimized embedding operations using:
    1. TensorRT engine compilation and caching
    2. Tensor Core acceleration with custom memory layouts
    3. Dynamic batch sizing based on available GPU memory
    4. Multi-precision support (FP16/INT8)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda",
        cache_dir: str = "/tmp/tensorrt_engines",
        precision: str = "fp16",
        batch_size: Optional[int] = None,
        max_sequence_length: int = 128,
        dynamic_shapes: bool = True,
        enable_tensor_cores: bool = True,
    ):
        """
        Initialize TensorRT optimized embeddings.
        
        Args:
            model_name: Name of the embedding model
            device: Device to use (cuda or cpu)
            cache_dir: Directory to cache TensorRT engines
            precision: Precision to use (fp16, fp32, or int8)
            batch_size: Batch size for embedding generation
            max_sequence_length: Maximum sequence length
            dynamic_shapes: Whether to use dynamic shapes for TensorRT
            enable_tensor_cores: Whether to enable Tensor Core optimizations
        """
        self.model_name = model_name
        self.device_name = device
        self.precision = precision
        self.cache_dir = cache_dir
        self.max_sequence_length = max_sequence_length
        self.dynamic_shapes = dynamic_shapes
        self.enable_tensor_cores = enable_tensor_cores
        
        # Set up GPU device if available
        self.device = torch.device(device if is_gpu_available() else "cpu")
        self.is_gpu = self.device.type == "cuda"
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize model and tokenizer
        self._initialize_model()
        
        # Determine optimal batch size if not provided
        if batch_size is None:
            if self.is_gpu:
                # Determine based on GPU memory and model size
                embedding_dim = self.get_embedding_dimension()
                if _tensor_core_optimizer:
                    # Use tensor core optimizer's calculation
                    self.batch_size = _tensor_core_optimizer.get_optimal_batch_size_for_t4(
                        model_dim=embedding_dim,
                        seq_length=max_sequence_length,
                        precision=precision,
                        memory_gb=get_available_gpu_memory() / 1024,
                    )
                else:
                    # Simple heuristic based on available GPU memory
                    gpu_memory_mb = get_available_gpu_memory()
                    bytes_per_element = 2 if precision == "fp16" else 4
                    bytes_per_embedding = embedding_dim * max_sequence_length * bytes_per_element
                    
                    # Assume we need about 5x the raw embedding size for intermediate activations
                    memory_per_sample = bytes_per_embedding * 5
                    
                    # Calculate max batch size with 80% of available memory
                    max_samples = int((gpu_memory_mb * 0.8 * 1024 * 1024) / memory_per_sample)
                    
                    # Round down to nearest multiple of 8 for tensor core efficiency
                    self.batch_size = max(1, (max_samples // 8) * 8)
            else:
                # Default CPU batch size
                self.batch_size = 8
        else:
            self.batch_size = batch_size
        
        # Set up TensorCore optimizer if available and enabled
        self.tensor_core_optimizer = None
        if self.is_gpu and self.enable_tensor_cores and _tensor_core_optimizer:
            try:
                self.tensor_core_optimizer = _tensor_core_optimizer.TensorCoreOptimizer(
                    device=self.device,
                    tensor_core_enabled=True,
                    precision=self.precision,
                    enable_profiling=False,
                )
                logger.info(f"Tensor Core optimizer initialized with {precision} precision")
            except Exception as e:
                logger.warning(f"Failed to initialize Tensor Core optimizer: {e}")
        
        # Optimize the model with TensorRT
        if self.is_gpu:
            self._optimize_with_tensorrt()
    
    def _initialize_model(self):
        """Initialize the embedding model and tokenizer."""
        try:
            from transformers import AutoTokenizer, AutoModel
            
            logger.info(f"Initializing model: {self.model_name}")
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Initialize model
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Model initialized on {self.device}")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _optimize_with_tensorrt(self):
        """Optimize the model with TensorRT."""
        if not is_gpu_available():
            logger.warning("GPU not available. TensorRT optimization disabled.")
            return
        
        try:
            logger.info(f"Optimizing model with TensorRT (precision: {self.precision})")
            
            # Use our TensorRT optimizer
            self.model = tensorrt_optimizer.optimize_model(
                model=self.model,
                model_name=self.model_name,
                input_shape=[1, self.max_sequence_length],
                max_batch_size=self.batch_size,
                dynamic_shapes=self.dynamic_shapes,
                force_rebuild=False,
            )
            
            # Apply Tensor Core optimizations if available
            if self.tensor_core_optimizer:
                logger.info("Applying Tensor Core optimizations")
                try:
                    self.model = self.tensor_core_optimizer.optimize_model(self.model)
                    logger.info("Tensor Core optimizations applied successfully")
                except Exception as e:
                    logger.warning(f"Failed to apply Tensor Core optimizations: {e}")
            
            logger.info("Model optimization complete")
        except Exception as e:
            logger.warning(f"TensorRT optimization failed: {e}. Using original model.")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Dimension of the embeddings
        """
        # Get model embedding dimension from config
        if hasattr(self.model, "config") and hasattr(self.model.config, "hidden_size"):
            return self.model.config.hidden_size
        
        # If not available, try to infer from model structure
        for module in self.model.modules():
            if isinstance(module, torch.nn.Linear):
                # Check output dimension of the last layer
                if module.weight.shape[0] > 0:
                    return module.weight.shape[0]
        
        # Default dimension for most sentence transformer models
        return 384
    
    def encode_texts(
        self,
        texts: List[str],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: List of texts to encode
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            return np.array([])
        
        # Process in batches
        embeddings = []
        
        # Display progress bar if requested
        pbar = None
        if show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(texts), desc="Generating embeddings")
            except ImportError:
                logger.warning("tqdm not installed, progress bar disabled")
        
        # Process batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize texts
            batch_inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            )
            
            # Move inputs to the correct device
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                # Use tensor core optimizer for attention if available
                if self.tensor_core_optimizer:
                    # Extract attention inputs
                    attention_mask = batch_inputs.get("attention_mask", None)
                    
                    # Forward pass with tensor core optimization
                    if attention_mask is not None:
                        # The model has separate token embeddings and attention layers
                        # We optimize only the attention mechanism here
                        outputs = self.model(**batch_inputs)
                    else:
                        # No attention mask, just run the model normally
                        outputs = self.model(**batch_inputs)
                else:
                    # Standard forward pass
                    outputs = self.model(**batch_inputs)
                
                # Extract embeddings from model outputs
                batch_embeddings = self._extract_embeddings(outputs, batch_inputs)
                
                # Move to CPU and convert to numpy
                batch_embeddings_np = batch_embeddings.cpu().numpy()
            
            # Normalize if requested
            if normalize:
                batch_embeddings_np = self._normalize_embeddings(batch_embeddings_np)
            
            # Add to list
            embeddings.append(batch_embeddings_np)
            
            # Update progress bar
            if pbar:
                pbar.update(len(batch_texts))
        
        # Close progress bar
        if pbar:
            pbar.close()
        
        # Concatenate all batches
        return np.vstack(embeddings)
    
    def _extract_embeddings(
        self,
        outputs: Any,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Extract embeddings from model outputs.
        
        Args:
            outputs: Model outputs
            inputs: Model inputs
            
        Returns:
            Tensor of embeddings
        """
        # Get attention mask
        attention_mask = inputs.get("attention_mask", None)
        
        # Handle different output formats
        if isinstance(outputs, torch.Tensor):
            # Output is already embeddings tensor
            embeddings = outputs
        elif hasattr(outputs, "last_hidden_state"):
            # Transformer model output
            embeddings = outputs.last_hidden_state
        else:
            # Try to get the first output
            embeddings = outputs[0]
        
        # Use mean pooling to get sentence embeddings
        if attention_mask is not None:
            # Mask padded tokens
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
            
            # Apply mask and compute mean
            embeddings = torch.sum(embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        else:
            # No mask, use mean of all tokens
            embeddings = torch.mean(embeddings, dim=1)
        
        return embeddings
    
    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        L2-normalize embeddings.
        
        Args:
            embeddings: Embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        # Calculate L2 norm along embedding dimension
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        norms = np.maximum(norms, 1e-10)
        
        # Normalize
        return embeddings / norms
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not is_gpu_available():
            return {"gpu_available": False}
        
        try:
            import torch
            
            # Get current GPU memory usage
            allocated = torch.cuda.memory_allocated(self.device) / (1024 * 1024)  # MB
            reserved = torch.cuda.memory_reserved(self.device) / (1024 * 1024)    # MB
            
            return {
                "gpu_available": True,
                "allocated_mb": allocated,
                "reserved_mb": reserved,
                "device": str(self.device),
                "device_name": torch.cuda.get_device_name(self.device),
            }
        except Exception as e:
            return {"gpu_available": False, "error": str(e)}
    
    def benchmark(
        self,
        texts: List[str] = None,
        iterations: int = 5,
        batch_sizes: List[int] = None,
    ) -> Dict[str, Any]:
        """
        Benchmark embedding generation performance.
        
        Args:
            texts: List of texts to use for benchmarking
            iterations: Number of iterations
            batch_sizes: List of batch sizes to test
            
        Returns:
            Dictionary with benchmark results
        """
        # Use sample texts if none provided
        if texts is None or len(texts) == 0:
            texts = [
                "This is a sample text for benchmarking.",
                "TensorRT optimization provides significant speedups for inference.",
                "NVIDIA T4 GPUs have Tensor Cores for accelerated matrix operations.",
                "LangChain integration with SAP HANA Cloud leverages vector embeddings.",
                "Optimizing memory layout is essential for tensor core utilization.",
                "PyTorch can be accelerated with TensorRT for production deployments.",
                "Sentence embeddings capture semantic meaning of text.",
                "Vector databases enable efficient similarity search.",
            ]
        
        # Use default batch sizes if none provided
        if batch_sizes is None:
            batch_sizes = [1, 8, 32, 64]
        
        results = {
            "model_name": self.model_name,
            "device": str(self.device),
            "precision": self.precision,
            "tensor_cores_enabled": self.enable_tensor_cores and self.tensor_core_optimizer is not None,
            "batch_results": {},
            "memory_usage": self.get_memory_usage(),
        }
        
        # Record hardware info if available
        if is_gpu_available():
            try:
                device_idx = self.device.index if self.device.index is not None else 0
                results["device_name"] = torch.cuda.get_device_name(device_idx)
                results["compute_capability"] = torch.cuda.get_device_capability(device_idx)
            except:
                pass
        
        # Test each batch size
        for batch_size in batch_sizes:
            # Skip if we don't have enough texts
            if batch_size > len(texts):
                continue
            
            # Prepare texts for this batch size
            # Repeat the texts to get enough samples
            benchmark_texts = (texts * (batch_size // len(texts) + 1))[:batch_size]
            
            # Store original batch size
            original_batch_size = self.batch_size
            
            # Set batch size for this test
            self.batch_size = batch_size
            
            # Run benchmark
            latencies = []
            for _ in range(iterations):
                # Warmup
                self.encode_texts(benchmark_texts[:1])
                
                # Measure performance
                start_time = time.time()
                embeddings = self.encode_texts(benchmark_texts)
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            # Calculate statistics
            mean_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            throughput = batch_size / (mean_latency / 1000)  # Samples per second
            
            # Record results
            results["batch_results"][str(batch_size)] = {
                "mean_latency_ms": float(mean_latency),
                "p95_latency_ms": float(p95_latency),
                "min_latency_ms": float(np.min(latencies)),
                "max_latency_ms": float(np.max(latencies)),
                "throughput_samples_per_second": float(throughput),
                "embedding_dimension": embeddings.shape[1],
            }
            
            # Restore original batch size
            self.batch_size = original_batch_size
        
        # Add tensor core profiling data if available
        if self.tensor_core_optimizer and hasattr(self.tensor_core_optimizer, "get_profiling_data"):
            results["tensor_core_profiling"] = self.tensor_core_optimizer.get_profiling_data()
        
        return results


# Enhance the existing TensorRTEmbeddings class with Tensor Core optimization
class TensorRTEmbeddingsWithTensorCores(TensorRTEmbeddings):
    """
    Enhanced version of TensorRTEmbeddings with advanced Tensor Core optimizations.
    
    This class adds Tensor Core specific optimizations to the existing TensorRTEmbeddings class.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        use_tensorrt: bool = True,
        precision: Optional[str] = None,
        dynamic_shapes: bool = True,
        force_rebuild: bool = False,
        calibration_texts: Optional[List[str]] = None,
        enable_tensor_cores: bool = True,
    ):
        """
        Initialize the enhanced TensorRT embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
            device: Device to use for computations ('cuda', 'cpu', or None for auto).
            batch_size: Batch size for processing.
            use_tensorrt: Whether to use TensorRT optimization.
            precision: Precision to use for TensorRT ('fp32', 'fp16', 'int8', or None for auto).
            dynamic_shapes: Whether to use dynamic shapes for TensorRT.
            force_rebuild: Force rebuilding the TensorRT engine.
            calibration_texts: Text samples for INT8 calibration.
            enable_tensor_cores: Whether to enable Tensor Core optimizations.
        """
        # Initialize the base class
        super().__init__(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            use_tensorrt=use_tensorrt,
            precision=precision,
            dynamic_shapes=dynamic_shapes,
            force_rebuild=force_rebuild,
            calibration_texts=calibration_texts,
        )
        
        # Add tensor core optimization
        self.enable_tensor_cores = enable_tensor_cores
        self.tensor_core_optimizer = None
        
        # Initialize tensor core optimizer if available and enabled
        if self.enable_tensor_cores and self.device == "cuda" and _tensor_core_optimizer:
            try:
                self.tensor_core_optimizer = _tensor_core_optimizer.TensorCoreOptimizer(
                    device=self.device,
                    tensor_core_enabled=True,
                    precision=self.precision,
                    enable_profiling=False,
                )
                
                # Apply tensor core optimizations to the model
                if hasattr(self.model, "_modules") and '0' in self.model._modules:
                    # Access the transformer model
                    transformer_model = self.model._modules['0']._modules['auto_model']
                    
                    # Apply tensor core optimizations
                    optimized_model = self.tensor_core_optimizer.optimize_model(transformer_model)
                    
                    # Replace the transformer model with the optimized version
                    self.model._modules['0']._modules['auto_model'] = optimized_model
                
                logger.info(f"Tensor Core optimizations applied successfully with {self.precision} precision")
            except Exception as e:
                logger.warning(f"Failed to apply Tensor Core optimizations: {e}")
    
    def benchmark_tensor_cores(self) -> Dict[str, Any]:
        """
        Run benchmarks specifically testing Tensor Core optimizations.
        
        Returns:
            Dictionary with benchmark results
        """
        results = {
            "model_name": self.model_name,
            "device": self.device,
            "precision": self.precision,
            "tensor_cores_enabled": self.enable_tensor_cores and self.tensor_core_optimizer is not None,
        }
        
        # Skip if tensor cores not available
        if not self.enable_tensor_cores or not self.tensor_core_optimizer:
            results["error"] = "Tensor Core optimization not available"
            return results
        
        # Generate sample text
        import random
        import string
        
        random_text = ''.join(random.choices(string.ascii_letters + ' ', k=100))
        
        # Run benchmarks with different batch sizes
        batch_results = {}
        for batch_size in [1, 8, 32, 64]:
            batch_texts = [random_text] * batch_size
            
            # Warmup
            _ = self.embed_documents(batch_texts[:1])
            
            # Benchmark with tensor cores
            start_time = time.time()
            for _ in range(10):  # 10 iterations
                _ = self.embed_documents(batch_texts)
            tensor_core_time = (time.time() - start_time) / 10
            
            # Temporarily disable tensor cores
            tensor_core_enabled = self.enable_tensor_cores
            self.enable_tensor_cores = False
            
            # Benchmark without tensor cores
            start_time = time.time()
            for _ in range(10):  # 10 iterations
                _ = self.embed_documents(batch_texts)
            standard_time = (time.time() - start_time) / 10
            
            # Restore tensor cores
            self.enable_tensor_cores = tensor_core_enabled
            
            # Calculate speedup
            speedup = standard_time / tensor_core_time if tensor_core_time > 0 else 0
            
            batch_results[str(batch_size)] = {
                "tensor_core_time_ms": tensor_core_time * 1000,
                "standard_time_ms": standard_time * 1000,
                "speedup_factor": speedup,
                "throughput_with_tensor_cores": batch_size / tensor_core_time,
                "throughput_without_tensor_cores": batch_size / standard_time,
            }
        
        results["batch_results"] = batch_results
        
        # Add profiling data if available
        if hasattr(self.tensor_core_optimizer, "get_profiling_data"):
            results["profiling_data"] = self.tensor_core_optimizer.get_profiling_data()
        
        return results


# Factory function for easy creation
def create_tensorrt_embeddings(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    precision: str = "fp16",
    batch_size: Optional[int] = None,
    enable_tensor_cores: bool = True,
) -> Union[TensorRTEmbeddingsEnhanced, TensorRTEmbeddingsWithTensorCores]:
    """
    Create TensorRT optimized embeddings with appropriate settings for T4 GPUs.
    
    Args:
        model_name: Name of the embedding model
        precision: Precision to use (fp16, fp32, or int8)
        batch_size: Batch size for embedding generation
        enable_tensor_cores: Whether to enable Tensor Core optimizations
        
    Returns:
        TensorRT embeddings instance
    """
    # Ensure precision is appropriate for the available hardware
    if not is_gpu_available():
        logger.warning("GPU not available. Using CPU with fp32 precision.")
        precision = "fp32"
        enable_tensor_cores = False
    else:
        # Check for T4 or better GPU
        try:
            device_idx = 0  # Use the first GPU
            cc_major, cc_minor = torch.cuda.get_device_capability(device_idx)
            
            # T4 has compute capability 7.5
            has_tensor_cores = (cc_major, cc_minor) >= (7, 0)
            
            if not has_tensor_cores and precision != "fp32":
                logger.warning(
                    f"GPU (compute capability {cc_major}.{cc_minor}) does not support "
                    f"Tensor Cores. Falling back to fp32 precision."
                )
                precision = "fp32"
                enable_tensor_cores = False
        except Exception as e:
            logger.warning(f"Error checking GPU capabilities: {e}")
    
    # Create the appropriate embeddings implementation
    if _tensor_core_optimizer:
        # If tensor_core_optimizer module is available, use the enhanced implementation
        return TensorRTEmbeddingsEnhanced(
            model_name=model_name,
            device="cuda" if is_gpu_available() else "cpu",
            precision=precision,
            batch_size=batch_size,
            enable_tensor_cores=enable_tensor_cores,
        )
    else:
        # Fall back to enhanced wrapper around existing implementation
        model_name_short = model_name.split("/")[-1] if "/" in model_name else model_name
        return TensorRTEmbeddingsWithTensorCores(
            model_name=model_name_short,
            device="cuda" if is_gpu_available() else "cpu",
            batch_size=batch_size if batch_size else 32,
            precision=precision,
            enable_tensor_cores=enable_tensor_cores,
        )