"""GPU-accelerated embeddings with TensorRT optimization."""

import logging
import os
from typing import Dict, List, Optional, Union, Any

import numpy as np
from langchain_core.embeddings import Embeddings

import gpu_utils
from tensorrt_utils import tensorrt_optimizer, TENSORRT_AVAILABLE

logger = logging.getLogger(__name__)


class TensorRTEmbeddings(Embeddings):
    """
    TensorRT-optimized embeddings using sentence-transformers.
    
    This class provides GPU acceleration with TensorRT optimization
    for maximum performance on NVIDIA GPUs.
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
    ):
        """
        Initialize the TensorRT-optimized embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
            device: Device to use for computations ('cuda', 'cpu', or None for auto).
            batch_size: Batch size for processing.
            use_tensorrt: Whether to use TensorRT optimization.
            precision: Precision to use for TensorRT ('fp32', 'fp16', 'int8', or None for auto).
            dynamic_shapes: Whether to use dynamic shapes for TensorRT.
            force_rebuild: Force rebuilding the TensorRT engine.
            calibration_texts: Text samples for INT8 calibration.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_tensorrt = use_tensorrt and TENSORRT_AVAILABLE
        self.precision = precision  # Will be set based on optimizer if None
        self.dynamic_shapes = dynamic_shapes
        self.force_rebuild = force_rebuild
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if gpu_utils.is_torch_available() else "cpu"
        else:
            self.device = device
        
        # Initialize the model
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading sentence-transformers model {model_name} on {self.device}")
            self.model = SentenceTransformer(model_name, device=self.device)
            
            # Apply TensorRT optimization if requested and available
            if self.use_tensorrt and self.device == "cuda":
                logger.info("Optimizing embedding model with TensorRT")
                
                # Get the transformer model component for optimization
                transformer_model = self.model._modules['0']._modules['auto_model']
                
                # Use provided precision or get from optimizer
                if self.precision is None:
                    self.precision = tensorrt_optimizer.precision
                    logger.info(f"Using auto-detected precision: {self.precision}")
                
                # Check if calibration data should be used for INT8
                needs_calibration = self.precision == "int8" and calibration_texts is None
                if needs_calibration:
                    # If no calibration texts provided but INT8 requested, use defaults
                    from tensorrt_utils import INT8CalibrationDataset
                    calibration_texts = INT8CalibrationDataset.get_default_calibration_texts()
                    logger.info(f"Using default calibration texts for INT8 quantization")
                
                # Optimize with TensorRT
                optimized_model = tensorrt_optimizer.optimize_model(
                    model=transformer_model,
                    model_name=model_name,
                    input_shape=[1, 512],  # Typical input shape for embedding models
                    max_batch_size=batch_size * 2,  # Allow for some growth
                    dynamic_shapes=dynamic_shapes,
                    calibration_data=calibration_texts,  # May be None if not INT8
                    force_rebuild=force_rebuild,
                )
                
                # Replace the transformer component with the optimized version
                self.model._modules['0']._modules['auto_model'] = optimized_model
                
                logger.info(f"TensorRT optimization complete with {self.precision} precision")
            
            if self.device == "cuda":
                if self.use_tensorrt:
                    logger.info(f"Using GPU acceleration with TensorRT optimization ({self.precision})")
                else:
                    logger.info("Using GPU acceleration (TensorRT not available or disabled)")
            else:
                logger.info("Using CPU for embeddings (TensorRT not applicable)")
                
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it with pip.")
            raise ImportError(
                "sentence-transformers not installed. "
                "Please install it with pip: pip install sentence-transformers"
            )
    
    def _process_batch(self, texts: List[str]) -> np.ndarray:
        """
        Process a batch of texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            NumPy array of embeddings.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents.
        
        Args:
            texts: List of documents to embed.
            
        Returns:
            List of embeddings.
        """
        if not texts:
            return []
        
        embeddings = self._process_batch(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: Query to embed.
            
        Returns:
            Embedding of the query.
        """
        embedding = self._process_batch([text])[0]
        return embedding.tolist()
        
    def benchmark(self, text_length: int = 100, iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark embedding performance.
        
        Args:
            text_length: Length of random text to use for benchmarking.
            iterations: Number of iterations for the benchmark.
            
        Returns:
            Dictionary with benchmark results.
        """
        import random
        import string
        import time
        
        # Generate random text
        random_text = ''.join(random.choices(string.ascii_letters + ' ', k=text_length))
        
        # Warmup
        _ = self.embed_query(random_text)
        
        # Benchmark single query
        start_time = time.time()
        for _ in range(iterations):
            _ = self.embed_query(random_text)
        single_query_time = (time.time() - start_time) / iterations
        
        # Benchmark batch of queries
        batch_size = min(32, self.batch_size)
        batch = [random_text] * batch_size
        
        start_time = time.time()
        for _ in range(iterations // batch_size + 1):
            _ = self.embed_documents(batch)
        batch_time = (time.time() - start_time) / (iterations // batch_size + 1)
        
        return {
            "model": self.model_name,
            "device": self.device,
            "tensorrt_enabled": self.use_tensorrt,
            "precision": self.precision if self.use_tensorrt else "native",
            "single_query_time_ms": single_query_time * 1000,
            "batch_query_time_ms": batch_time * 1000,
            "batch_size": batch_size,
            "throughput_tokens_per_second": (batch_size * text_length) / batch_time,
        }


class TensorRTHybridEmbeddings(Embeddings):
    """
    Hybrid embeddings that can switch between SAP HANA internal embeddings
    and TensorRT-optimized embeddings.
    
    This class provides maximum GPU acceleration with TensorRT optimization
    while still supporting SAP HANA's internal embedding capabilities.
    """
    
    def __init__(
        self,
        internal_embedding_model_id: str = "SAP_NEB.20240715",
        external_model_name: str = "all-MiniLM-L6-v2",
        use_internal: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
        use_tensorrt: bool = True,
        precision: Optional[str] = None,
        force_rebuild: bool = False,
        calibration_texts: Optional[List[str]] = None,
    ):
        """
        Initialize the hybrid embeddings with TensorRT optimization.
        
        Args:
            internal_embedding_model_id: ID of the SAP HANA internal embedding model.
            external_model_name: Name of the external sentence-transformers model.
            use_internal: Whether to use internal embeddings by default.
            device: Device to use for external embeddings.
            batch_size: Batch size for processing external embeddings.
            use_tensorrt: Whether to use TensorRT optimization.
            precision: Precision to use for TensorRT ('fp32', 'fp16', 'int8', or None for auto).
            force_rebuild: Force rebuilding the TensorRT engine.
            calibration_texts: Text samples for INT8 calibration.
        """
        self.use_internal = use_internal
        self.internal_embedding_model_id = internal_embedding_model_id
        
        # Initialize internal embeddings
        from langchain_hana import HanaInternalEmbeddings
        self.internal_embeddings = HanaInternalEmbeddings(
            internal_embedding_model_id=internal_embedding_model_id
        )
        
        # Initialize external TensorRT-optimized embeddings
        self.external_embeddings = TensorRTEmbeddings(
            model_name=external_model_name,
            device=device,
            batch_size=batch_size,
            use_tensorrt=use_tensorrt,
            precision=precision,
            force_rebuild=force_rebuild,
            calibration_texts=calibration_texts,
        )
        
        # Get the actual precision used (may be auto-detected)
        self.precision = self.external_embeddings.precision
        
        logger.info(
            f"Hybrid embeddings initialized: "
            f"Using {'internal' if use_internal else 'external TensorRT-optimized'} embeddings by default. "
            f"External precision: {self.precision}"
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents.
        
        Args:
            texts: List of documents to embed.
            
        Returns:
            List of embeddings.
        """
        if self.use_internal:
            return self.internal_embeddings.embed_documents(texts)
        else:
            return self.external_embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: Query to embed.
            
        Returns:
            Embedding of the query.
        """
        if self.use_internal:
            return self.internal_embeddings.embed_query(text)
        else:
            return self.external_embeddings.embed_query(text)
    
    def get_model_id(self) -> str:
        """
        Get the model ID.
        
        Returns:
            Model ID.
        """
        if self.use_internal:
            return self.internal_embedding_model_id
        else:
            return f"external:{self.external_embeddings.model_name}"
    
    def set_mode(self, use_internal: bool) -> None:
        """
        Set the embedding mode.
        
        Args:
            use_internal: Whether to use internal embeddings.
        """
        self.use_internal = use_internal
        logger.info(f"Embedding mode set to: {'internal' if use_internal else 'external TensorRT-optimized'}")
        
    def benchmark_comparison(self) -> Dict[str, Any]:
        """
        Run benchmark comparison between internal and external embeddings.
        
        Returns:
            Dictionary with benchmark results.
        """
        results = {
            "test_timestamp": time.time(),
            "device_info": {
                "device": self.external_embeddings.device,
                "precision": self.precision,
            }
        }
        
        # Get GPU info if available
        if gpu_utils.is_torch_available():
            try:
                results["device_info"]["gpu_name"] = torch.cuda.get_device_name(0)
                results["device_info"]["cuda_version"] = torch.version.cuda
                results["device_info"]["compute_capability"] = f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}"
                results["device_info"]["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            except Exception as e:
                results["device_info"]["gpu_info_error"] = str(e)
        
        # Temporarily switch to external embeddings
        original_mode = self.use_internal
        self.use_internal = False
        
        # Benchmark external embeddings with TensorRT
        results["external_tensorrt"] = {
            "model": self.external_embeddings.model_name,
            "precision": self.precision,
            "benchmark": self.external_embeddings.benchmark(),
        }
        
        # Switch to internal embeddings if available
        try:
            self.use_internal = True
            
            # For internal embeddings, we need a different benchmark approach
            import time
            import random
            import string
            
            # Generate random text
            text_length = 100
            iterations = 20  # Fewer iterations for internal as it may be slower
            random_text = ''.join(random.choices(string.ascii_letters + ' ', k=text_length))
            
            # Warmup
            _ = self.embed_query(random_text)
            
            # Benchmark single query
            start_time = time.time()
            for _ in range(iterations):
                _ = self.embed_query(random_text)
            single_query_time = (time.time() - start_time) / iterations
            
            # Benchmark with different batch sizes
            batch_results = {}
            for batch_size in [1, 8, 16, 32]:
                try:
                    batch = [random_text] * batch_size
                    
                    # Warmup
                    _ = self.embed_documents(batch)
                    
                    # Benchmark
                    start_time = time.time()
                    for _ in range(max(1, iterations // batch_size)):
                        _ = self.embed_documents(batch)
                    batch_time = (time.time() - start_time) / max(1, iterations // batch_size)
                    
                    batch_results[str(batch_size)] = {
                        "batch_query_time_ms": batch_time * 1000,
                        "throughput_samples_per_second": batch_size / batch_time,
                        "throughput_tokens_per_second": (batch_size * text_length) / batch_time,
                    }
                except Exception as e:
                    batch_results[str(batch_size)] = {"error": str(e)}
            
            results["internal_hana"] = {
                "model": self.internal_embedding_model_id,
                "single_query_time_ms": single_query_time * 1000,
                "batch_sizes": batch_results,
            }
        except Exception as e:
            logger.warning(f"Could not benchmark internal embeddings: {e}")
            results["internal_hana"] = {"error": str(e)}
        
        # Restore original mode
        self.use_internal = original_mode
        
        # Add comparison metrics
        if "internal_hana" in results and "error" not in results["internal_hana"] and "16" in results["internal_hana"].get("batch_sizes", {}):
            try:
                # Get throughput for batch size 16 for fair comparison
                ext_results = results["external_tensorrt"]["benchmark"]
                if isinstance(ext_results, dict) and "batch_sizes" in ext_results and "16" in ext_results["batch_sizes"]:
                    external_throughput = ext_results["batch_sizes"]["16"]["throughput_tokens_per_second"]
                    internal_throughput = results["internal_hana"]["batch_sizes"]["16"]["throughput_tokens_per_second"]
                    results["speedup_factor"] = external_throughput / internal_throughput
                    
                    # Add summary
                    results["summary"] = {
                        "internal_model": self.internal_embedding_model_id,
                        "external_model": self.external_embeddings.model_name,
                        "external_precision": self.precision,
                        "internal_throughput": internal_throughput,
                        "external_throughput": external_throughput,
                        "speedup_factor": external_throughput / internal_throughput,
                    }
            except (KeyError, ZeroDivisionError, TypeError) as e:
                results["speedup_error"] = str(e)
        
        return results
        
    def run_precision_comparison(self) -> Dict[str, Any]:
        """
        Run a comparison of different precision modes for the external embeddings.
        
        This method tests FP32, FP16, and INT8 precision for the external TensorRT model
        and compares throughput performance.
        
        Returns:
            Dictionary with benchmark results for different precision modes.
        """
        # Save original state
        original_mode = self.use_internal
        original_precision = self.precision
        
        # Create results dictionary
        results = {
            "model": self.external_embeddings.model_name,
            "device_info": {
                "name": torch.cuda.get_device_name(0) if gpu_utils.is_torch_available() else "CPU",
                "compute_capability": f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}" 
                                    if gpu_utils.is_torch_available() else "N/A",
            },
            "precision_modes": {}
        }
        
        # Temporarily switch to external embeddings
        self.use_internal = False
        
        # Get access to the underlying tensorrt_optimizer
        from tensorrt_utils import tensorrt_optimizer
        
        # Test each precision mode
        for precision in ["fp32", "fp16", "int8"]:
            try:
                # Skip INT8 if not supported
                if precision == "int8" and not gpu_utils.is_torch_available():
                    results["precision_modes"]["int8"] = {"error": "INT8 requires GPU acceleration"}
                    continue
                    
                if precision == "int8" and torch.cuda.get_device_capability()[0] < 7:
                    results["precision_modes"]["int8"] = {"error": "INT8 requires Volta (SM70) or newer GPU"}
                    continue
                
                logger.info(f"Testing {precision} precision")
                
                # Create a new embeddings model with this precision
                temp_embeddings = TensorRTEmbeddings(
                    model_name=self.external_embeddings.model_name,
                    device=self.external_embeddings.device,
                    batch_size=self.external_embeddings.batch_size,
                    use_tensorrt=True,
                    precision=precision,
                    force_rebuild=True,  # Force rebuild to ensure fair comparison
                )
                
                # Run benchmark
                benchmark_result = temp_embeddings.benchmark()
                results["precision_modes"][precision] = benchmark_result
                
                # Clean up
                del temp_embeddings
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error benchmarking {precision} precision: {e}")
                results["precision_modes"][precision] = {"error": str(e)}
        
        # Restore original state
        self.use_internal = original_mode
        
        # Calculate speedup factors
        if "fp32" in results["precision_modes"] and "throughput_tokens_per_second" in results["precision_modes"]["fp32"]:
            fp32_baseline = results["precision_modes"]["fp32"]["throughput_tokens_per_second"]
            
            # FP16 speedup
            if "fp16" in results["precision_modes"] and "throughput_tokens_per_second" in results["precision_modes"]["fp16"]:
                fp16_throughput = results["precision_modes"]["fp16"]["throughput_tokens_per_second"]
                results["fp16_vs_fp32_speedup"] = fp16_throughput / fp32_baseline
            
            # INT8 speedup
            if "int8" in results["precision_modes"] and "throughput_tokens_per_second" in results["precision_modes"]["int8"]:
                int8_throughput = results["precision_modes"]["int8"]["throughput_tokens_per_second"]
                results["int8_vs_fp32_speedup"] = int8_throughput / fp32_baseline
                
                # INT8 vs FP16
                if "fp16" in results["precision_modes"] and "throughput_tokens_per_second" in results["precision_modes"]["fp16"]:
                    fp16_throughput = results["precision_modes"]["fp16"]["throughput_tokens_per_second"]
                    results["int8_vs_fp16_speedup"] = int8_throughput / fp16_throughput
        
        # Add recommended precision based on results
        if "int8_vs_fp32_speedup" in results and results["int8_vs_fp32_speedup"] > 1.1:
            results["recommended_precision"] = "int8"
        elif "fp16_vs_fp32_speedup" in results and results["fp16_vs_fp32_speedup"] > 1.1:
            results["recommended_precision"] = "fp16"
        else:
            results["recommended_precision"] = "fp32"
            
        return results