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
        precision: str = "fp16",
        dynamic_shapes: bool = True,
    ):
        """
        Initialize the TensorRT-optimized embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
            device: Device to use for computations ('cuda', 'cpu', or None for auto).
            batch_size: Batch size for processing.
            use_tensorrt: Whether to use TensorRT optimization.
            precision: Precision to use for TensorRT ('fp32', 'fp16').
            dynamic_shapes: Whether to use dynamic shapes for TensorRT.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_tensorrt = use_tensorrt and TENSORRT_AVAILABLE
        self.precision = precision
        self.dynamic_shapes = dynamic_shapes
        
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
                
                # Optimize with TensorRT
                optimized_model = tensorrt_optimizer.optimize_model(
                    model=transformer_model,
                    model_name=model_name,
                    input_shape=[1, 512],  # Typical input shape for embedding models
                    max_batch_size=batch_size * 2,  # Allow for some growth
                    dynamic_shapes=dynamic_shapes,
                )
                
                # Replace the transformer component with the optimized version
                self.model._modules['0']._modules['auto_model'] = optimized_model
                
                logger.info("TensorRT optimization complete")
            
            if self.device == "cuda":
                if self.use_tensorrt:
                    logger.info("Using GPU acceleration with TensorRT optimization")
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
        precision: str = "fp16",
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
            precision: Precision to use for TensorRT ('fp32', 'fp16').
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
        )
        
        logger.info(
            f"Hybrid embeddings initialized: "
            f"Using {'internal' if use_internal else 'external TensorRT-optimized'} embeddings by default"
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
        results = {}
        
        # Temporarily switch to external embeddings
        original_mode = self.use_internal
        self.use_internal = False
        
        # Benchmark external embeddings
        results["external_tensorrt"] = self.external_embeddings.benchmark()
        
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
            
            # Benchmark batch
            batch_size = 16
            batch = [random_text] * batch_size
            
            start_time = time.time()
            for _ in range(iterations // batch_size + 1):
                _ = self.embed_documents(batch)
            batch_time = (time.time() - start_time) / (iterations // batch_size + 1)
            
            results["internal_hana"] = {
                "model": self.internal_embedding_model_id,
                "single_query_time_ms": single_query_time * 1000,
                "batch_query_time_ms": batch_time * 1000,
                "batch_size": batch_size,
                "throughput_tokens_per_second": (batch_size * text_length) / batch_time,
            }
        except Exception as e:
            logger.warning(f"Could not benchmark internal embeddings: {e}")
            results["internal_hana"] = {"error": str(e)}
        
        # Restore original mode
        self.use_internal = original_mode
        
        # Add comparison metrics
        if "internal_hana" in results and "error" not in results["internal_hana"]:
            try:
                external_throughput = results["external_tensorrt"]["throughput_tokens_per_second"]
                internal_throughput = results["internal_hana"]["throughput_tokens_per_second"]
                results["speedup_factor"] = external_throughput / internal_throughput
            except (KeyError, ZeroDivisionError):
                results["speedup_factor"] = "N/A"
        
        return results