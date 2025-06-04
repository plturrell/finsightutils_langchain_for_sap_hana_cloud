"""
TensorRT-optimized embedding generation for SAP HANA Cloud LangChain integration.

This module provides a TensorRT-optimized embedding class that accelerates embedding
generation on NVIDIA GPUs for improved performance when working with large document
collections or real-time embedding generation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any

import numpy as np
from langchain_core.embeddings import Embeddings

# Conditional imports based on GPU availability
try:
    import torch
    import tensorrt as trt
    from tensorrt.logger import Logger as TRTLogger
    import pycuda.driver as cuda
    import pycuda.autoinit
    HAS_GPU_DEPENDENCIES = True
except ImportError:
    HAS_GPU_DEPENDENCIES = False

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about an available GPU."""
    index: int
    name: str
    memory_total: int
    memory_free: int
    compute_capability: str

    @property
    def memory_usage_percent(self) -> float:
        """Get the percentage of GPU memory currently in use."""
        return 100.0 * (1.0 - (self.memory_free / self.memory_total))
    
    def __str__(self) -> str:
        """String representation of GPU info."""
        return (
            f"GPU {self.index}: {self.name} "
            f"(Compute: {self.compute_capability}, "
            f"Memory: {self.memory_free}/{self.memory_total} MB, "
            f"Usage: {self.memory_usage_percent:.1f}%)"
        )


def get_available_gpus() -> List[GPUInfo]:
    """
    Get information about all available NVIDIA GPUs on the system.
    
    Returns:
        List[GPUInfo]: List of GPUInfo objects for each available GPU
        
    Note:
        Returns an empty list if no GPUs are available or if the required
        dependencies (torch, tensorrt, pycuda) are not installed
    """
    if not HAS_GPU_DEPENDENCIES:
        logger.warning(
            "GPU dependencies not found. To enable GPU acceleration, install: "
            "torch, tensorrt, pycuda"
        )
        return []
    
    try:
        gpu_count = torch.cuda.device_count()
        if gpu_count == 0:
            logger.warning("No NVIDIA GPUs detected by PyTorch")
            return []
        
        result = []
        for i in range(gpu_count):
            try:
                # Get properties using torch
                props = torch.cuda.get_device_properties(i)
                # Get memory info using pycuda
                with torch.cuda.device(i):
                    free, total = torch.cuda.mem_get_info()
                    free_mb = free // (1024 * 1024)
                    total_mb = total // (1024 * 1024)
                
                # Add GPU to list
                result.append(GPUInfo(
                    index=i,
                    name=props.name,
                    memory_total=total_mb,
                    memory_free=free_mb,
                    compute_capability=f"{props.major}.{props.minor}"
                ))
            except Exception as e:
                logger.warning(f"Error getting properties for GPU {i}: {str(e)}")
        
        return result
    except Exception as e:
        logger.warning(f"Error detecting GPUs: {str(e)}")
        return []


class TensorRTEmbeddings(Embeddings):
    """
    TensorRT-optimized embeddings class for accelerated embedding generation on NVIDIA GPUs.
    
    This class provides GPU-accelerated embedding generation using NVIDIA TensorRT,
    which significantly improves performance for both batch processing and
    real-time embedding generation. The class automatically handles model optimization,
    GPU memory management, and fallback to CPU if needed.
    
    Key features:
    - Automatic TensorRT engine generation from Hugging Face models
    - Dynamic batch size adjustment based on available GPU memory
    - Multi-GPU support with automatic load balancing
    - Transparent CPU fallback when GPU is unavailable
    - Runtime performance optimization with mixed precision support (FP32, FP16, INT8)
    - INT8 quantization with calibration for maximum throughput
    
    Args:
        model_name: Name of the Hugging Face model to use for embeddings
                  (e.g., "sentence-transformers/all-MiniLM-L6-v2")
        cache_dir: Directory to cache optimized TensorRT engines
        force_engine_rebuild: If True, forces rebuilding the TensorRT engine even if
                             a cached version exists
        max_batch_size: Maximum batch size for embedding generation
        gpu_memory_threshold: Percentage of GPU memory that must be free to use GPU
        precision: Precision to use for TensorRT optimization ("fp32", "fp16", or "int8")
        calibration_cache_dir: Directory to cache INT8 calibration data
        calibration_data: Custom text samples for INT8 calibration
    
    Example:
        ```python
        # Create TensorRT-optimized embeddings with INT8 precision for maximum throughput
        embeddings = TensorRTEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_dir="./trt_engines",
            precision="int8"  # Use INT8 quantization for maximum throughput
        )
        
        # Generate embeddings for documents
        documents = ["This is a sample document", "Another example text"]
        doc_embeddings = embeddings.embed_documents(documents)
        
        # Generate embedding for a query
        query_embedding = embeddings.embed_query("Sample query text")
        ```
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        force_engine_rebuild: bool = False,
        max_batch_size: int = 32,
        gpu_memory_threshold: float = 20.0,  # Percentage of GPU memory that must be free
        precision: Optional[str] = None,  # "fp32", "fp16", "int8", or None for auto-detection
        calibration_cache_dir: Optional[str] = None,
        calibration_data: Optional[List[str]] = None,
    ) -> None:
        """Initialize the TensorRT Embeddings class."""
        self.model_name = model_name
        self.cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "hana_trt_engines")
        self.calibration_cache_dir = calibration_cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "hana_trt_calibration")
        self.force_engine_rebuild = force_engine_rebuild
        self.max_batch_size = max_batch_size
        self.gpu_memory_threshold = gpu_memory_threshold
        self.precision = precision  # Will be auto-detected if None
        self.calibration_data = calibration_data
        
        # Dynamically set during initialization
        self.model_dim: int = 0
        self.use_gpu: bool = False
        self.current_gpu: Optional[GPUInfo] = None
        self.engine: Any = None
        self.context: Any = None
        self.is_int8: bool = False
        
        # Initialize model
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the TensorRT engine or fall back to CPU."""
        # Check if GPU dependencies are available
        if not HAS_GPU_DEPENDENCIES:
            logger.warning(
                "TensorRT dependencies not found. Using CPU fallback mode. "
                "To enable GPU acceleration, install: torch, tensorrt, pycuda"
            )
            self._initialize_cpu_fallback()
            return
        
        # Get available GPUs
        gpus = get_available_gpus()
        if not gpus:
            logger.warning("No suitable GPUs found. Using CPU fallback mode.")
            self._initialize_cpu_fallback()
            return
        
        # Select the GPU with the most available memory
        selected_gpu = max(gpus, key=lambda g: g.memory_free)
        
        # Check if the selected GPU has enough free memory
        if selected_gpu.memory_usage_percent > (100 - self.gpu_memory_threshold):
            logger.warning(
                f"Selected GPU ({selected_gpu.name}) has insufficient free memory "
                f"({selected_gpu.memory_free} MB). Using CPU fallback mode."
            )
            self._initialize_cpu_fallback()
            return
        
        # Auto-detect optimal precision if not specified
        if self.precision is None:
            self.precision = self._get_optimal_precision(selected_gpu)
            logger.info(f"Auto-detected optimal precision: {self.precision}")
        
        # Set INT8 flag for later use
        self.is_int8 = self.precision == "int8"
        
        # Try to initialize TensorRT engine
        try:
            self._initialize_tensorrt(selected_gpu)
            self.use_gpu = True
            self.current_gpu = selected_gpu
            logger.info(f"Successfully initialized TensorRT on {selected_gpu} with {self.precision} precision")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorRT: {str(e)}. Using CPU fallback mode.")
            self._initialize_cpu_fallback()
    
    def _get_optimal_precision(self, gpu: GPUInfo) -> str:
        """Determine optimal precision based on GPU capabilities."""
        major, minor = map(int, gpu.compute_capability.split('.'))
        
        # Check for INT8 support (Turing+ or Volta+ with additional capabilities)
        if major >= 7 and minor >= 5:  # Turing or newer
            return "int8"  # Use INT8 for Turing or newer for best throughput
        elif major >= 7:  # Volta or newer architecture
            return "fp16"  # Use FP16 for Volta for best performance
        else:
            return "fp32"  # Use FP32 for older GPUs
    
    def _initialize_tensorrt(self, gpu: GPUInfo) -> None:
        """Initialize the TensorRT engine on the specified GPU."""
        # Set CUDA device
        cuda.Device(gpu.index).make_context()
        
        # Create TensorRT logger
        trt_logger = TRTLogger()
        
        # Determine engine path
        os.makedirs(self.cache_dir, exist_ok=True)
        engine_filename = f"{self.model_name.replace('/', '_')}_{self.precision}.engine"
        engine_path = os.path.join(self.cache_dir, engine_filename)
        
        # For INT8, also ensure calibration cache directory exists
        if self.is_int8:
            os.makedirs(self.calibration_cache_dir, exist_ok=True)
        
        # Check if engine exists and needs to be rebuilt
        if not os.path.exists(engine_path) or self.force_engine_rebuild:
            logger.info(f"Building TensorRT engine for {self.model_name} with {self.precision} precision")
            self._build_engine(engine_path, trt_logger)
        
        # Load the TensorRT engine
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        
        # Create runtime and engine
        runtime = trt.Runtime(trt_logger)
        self.engine = runtime.deserialize_cuda_engine(engine_data)
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get model output dimension from engine
        output_shape = self.engine.get_binding_shape(1)  # Assuming binding 1 is output
        self.model_dim = output_shape[1]  # [batch_size, dim]
        
        # Clean up CUDA context
        cuda.Context.pop()
    
    def _build_engine(self, engine_path: str, trt_logger: TRTLogger) -> None:
        """Build a TensorRT engine from the model and save it to disk."""
        # Load the model using transformers
        from transformers import AutoModel, AutoTokenizer
        
        # Load base model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        
        # Get embedding dimension
        self.model_dim = model.config.hidden_size
        
        # Create a TensorRT builder and network
        builder = trt.Builder(trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        
        # Create ONNX parser
        parser = trt.OnnxParser(network, trt_logger)
        
        # Export model to ONNX (temporary file)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
            # Create dynamic input for the model
            dummy_input = {
                "input_ids": torch.ones((1, 128), dtype=torch.long),
                "attention_mask": torch.ones((1, 128), dtype=torch.long),
            }
            
            # Export model to ONNX
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                tmp.name,
                input_names=list(dummy_input.keys()),
                output_names=["embeddings"],
                dynamic_axes={
                    "input_ids": {0: "batch_size"},
                    "attention_mask": {0: "batch_size"},
                    "embeddings": {0: "batch_size"},
                },
                opset_version=12,
            )
            
            # Parse ONNX model
            with open(tmp.name, "rb") as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                    raise RuntimeError("Failed to parse ONNX model")
        
        # Configure builder
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1 GB
        
        # Set up INT8 calibration if needed
        if self.is_int8:
            if not builder.platform_has_fast_int8:
                logger.warning("INT8 not supported on this platform. Falling back to FP16.")
                self.precision = "fp16"
            else:
                # Set up INT8 calibration
                config.set_flag(trt.BuilderFlag.INT8)
                
                # Create calibration cache path
                calibration_cache = os.path.join(
                    self.calibration_cache_dir, 
                    f"{self.model_name.replace('/', '_')}_calibration.cache"
                )
                
                # Create or use calibration data
                if self.calibration_data is None:
                    # Default calibration data if none provided
                    self.calibration_data = [
                        "The quick brown fox jumps over the lazy dog.",
                        "TensorRT provides high-performance inference for deep learning models.",
                        "Natural language processing enables computers to understand human language.",
                        "The speed of light in vacuum is approximately 299,792,458 meters per second.",
                        "Neural networks consist of layers of interconnected nodes.",
                        "The Internet of Things connects everyday devices to the internet.",
                        "Cloud computing provides on-demand access to computing resources.",
                        "Cybersecurity is essential for protecting digital systems from attacks.",
                        "Blockchain technology enables secure and transparent transactions.",
                        "Virtual reality creates immersive digital environments.",
                    ]
                
                # Set up calibrator
                calibrator = self._create_int8_calibrator(
                    tokenizer=tokenizer,
                    calibration_data=self.calibration_data,
                    calibration_cache=calibration_cache,
                    batch_size=min(8, self.max_batch_size),  # Small batch size for calibration
                )
                
                # Set the calibrator
                config.int8_calibrator = calibrator
        
        # Set precision flags
        if self.precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            # Also enable FP16 for INT8 as fallback precision
            if self.is_int8 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
        
        # Set max batch size and dynamic shapes
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input_ids",
            min=(1, 128),
            opt=(self.max_batch_size // 2, 128),
            max=(self.max_batch_size, 128),
        )
        profile.set_shape(
            "attention_mask",
            min=(1, 128),
            opt=(self.max_batch_size // 2, 128),
            max=(self.max_batch_size, 128),
        )
        config.add_optimization_profile(profile)
        
        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Failed to build TensorRT engine")
        
        # Save engine to file
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        logger.info(f"TensorRT engine built and saved to {engine_path}")
    
    def _create_int8_calibrator(self, tokenizer, calibration_data: List[str], calibration_cache: str, batch_size: int):
        """Create an INT8 calibrator for TensorRT."""
        
        class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
            """INT8 Calibrator implementing the IInt8EntropyCalibrator2 interface."""
            
            def __init__(self, tokenizer, texts, cache_file, batch_size=8):
                """Initialize the calibrator with tokenizer and texts."""
                super().__init__()
                self.tokenizer = tokenizer
                self.texts = texts
                self.cache_file = cache_file
                self.batch_size = batch_size
                self.current_batch = 0
                self.max_batch = len(texts) // batch_size
                
                # Tokenize all texts upfront for efficiency
                self.encodings = tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                
                # Allocate GPU memory
                self.input_ids_buffer = cuda.mem_alloc(batch_size * 128 * np.dtype(np.int32).itemsize)
                self.attention_mask_buffer = cuda.mem_alloc(batch_size * 128 * np.dtype(np.int32).itemsize)
            
            def get_batch_size(self):
                """Return the batch size."""
                return self.batch_size
            
            def get_batch(self, names):
                """Get the next batch of calibration data."""
                if self.current_batch >= self.max_batch:
                    return None
                
                # Get batch of input data
                start_idx = self.current_batch * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.texts))
                actual_batch_size = end_idx - start_idx
                
                # Extract batch from tokenized data
                input_ids = self.encodings['input_ids'][start_idx:end_idx].int().contiguous()
                attention_mask = self.encodings['attention_mask'][start_idx:end_idx].int().contiguous()
                
                # Copy to GPU
                cuda.memcpy_htod(self.input_ids_buffer, input_ids.numpy())
                cuda.memcpy_htod(self.attention_mask_buffer, attention_mask.numpy())
                
                self.current_batch += 1
                return [int(self.input_ids_buffer), int(self.attention_mask_buffer)]
            
            def read_calibration_cache(self):
                """Read calibration cache from file if it exists."""
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "rb") as f:
                        logger.info(f"Reading INT8 calibration cache from {self.cache_file}")
                        return f.read()
                return None
            
            def write_calibration_cache(self, cache):
                """Write calibration cache to file."""
                with open(self.cache_file, "wb") as f:
                    logger.info(f"Writing INT8 calibration cache to {self.cache_file}")
                    f.write(cache)
        
        # Create and return the calibrator
        return Int8EntropyCalibrator(tokenizer, calibration_data, calibration_cache, batch_size)
    
    def _initialize_cpu_fallback(self) -> None:
        """Initialize CPU fallback using standard Hugging Face models."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Initializing CPU fallback using {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.model_dim = self.model.get_sentence_embedding_dimension()
            self.use_gpu = False
            self.current_gpu = None
        except ImportError:
            logger.error(
                "Cannot initialize CPU fallback: sentence-transformers not installed. "
                "Please install with: pip install sentence-transformers"
            )
            raise ImportError(
                "The sentence-transformers package is required for CPU fallback mode. "
                "Please install with: pip install sentence-transformers"
            )
    
    def _get_embedding_gpu(self, text: str) -> List[float]:
        """Generate embedding using TensorRT on GPU."""
        # Import here to avoid dependency issues if GPU not available
        from transformers import AutoTokenizer
        
        # Set CUDA device
        if self.current_gpu:
            cuda.Device(self.current_gpu.index).make_context()
        
        try:
            # Tokenize the input text
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            encodings = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            # Prepare input bindings
            input_ids = encodings["input_ids"].contiguous().cuda().data_ptr()
            attention_mask = encodings["attention_mask"].contiguous().cuda().data_ptr()
            
            # Prepare output buffer
            output = torch.zeros(1, self.model_dim, device="cuda")
            output_ptr = output.data_ptr()
            
            # Set input shapes for dynamic batch size
            batch_size = 1
            self.context.set_binding_shape(0, (batch_size, 128))  # input_ids
            self.context.set_binding_shape(1, (batch_size, 128))  # attention_mask
            
            # Run inference
            bindings = [int(input_ids), int(attention_mask), int(output_ptr)]
            self.context.execute_v2(bindings)
            
            # Get result and normalize
            result = output.cpu().numpy()[0]
            normalized = result / np.linalg.norm(result)
            
            return normalized.tolist()
        finally:
            # Clean up CUDA context
            if self.current_gpu:
                cuda.Context.pop()
    
    def _get_embedding_cpu(self, text: str) -> List[float]:
        """Generate embedding using CPU fallback."""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    
    def _get_embeddings_gpu_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using TensorRT on GPU."""
        from transformers import AutoTokenizer
        
        # Set CUDA device
        if self.current_gpu:
            cuda.Device(self.current_gpu.index).make_context()
        
        try:
            # Tokenize all texts
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            encodings = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            
            # Prepare input bindings
            input_ids = encodings["input_ids"].contiguous().cuda()
            attention_mask = encodings["attention_mask"].contiguous().cuda()
            
            # Prepare output buffer
            batch_size = len(texts)
            output = torch.zeros(batch_size, self.model_dim, device="cuda")
            
            # Set input shapes for dynamic batch size
            self.context.set_binding_shape(0, (batch_size, 128))  # input_ids
            self.context.set_binding_shape(1, (batch_size, 128))  # attention_mask
            
            # Run inference
            bindings = [
                int(input_ids.data_ptr()),
                int(attention_mask.data_ptr()),
                int(output.data_ptr()),
            ]
            self.context.execute_v2(bindings)
            
            # Get results and normalize
            results = output.cpu().numpy()
            
            # Normalize each embedding
            normalized = np.zeros_like(results)
            for i, embedding in enumerate(results):
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    normalized[i] = embedding / norm
                else:
                    normalized[i] = embedding
            
            return normalized.tolist()
        finally:
            # Clean up CUDA context
            if self.current_gpu:
                cuda.Context.pop()
    
    def _get_embeddings_cpu_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts using CPU fallback."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings, one for each input text
        """
        if not texts:
            return []
        
        # Import the batch processor here to avoid circular imports
        from langchain_hana.gpu.batch_processor import EmbeddingBatchProcessor
        
        # Use dynamic batch processor if GPU is available
        if self.use_gpu:
            try:
                # Create embedding function
                def batch_embedding_fn(batch_texts: List[str]) -> List[List[float]]:
                    return self._get_embeddings_gpu_batch(batch_texts)
                
                # Create batch processor with appropriate settings
                processor = EmbeddingBatchProcessor(
                    embedding_fn=batch_embedding_fn,
                    model_name=self.model_name,
                    embedding_dim=self.model_dim,
                    device_id=0 if self.current_gpu is None else self.current_gpu.index,
                    initial_batch_size=self.max_batch_size,
                    min_batch_size=1,
                    max_batch_size=self.max_batch_size,
                    safety_factor=0.8,
                    oom_recovery_factor=0.5,
                    dtype="float16" if self.precision == "fp16" else "float32" if self.precision == "fp32" else "int8",
                    enable_caching=True
                )
                
                # Process documents with dynamic batching
                embeddings, stats = processor.embed_documents(texts)
                
                # Log statistics
                logger.info(
                    f"Generated {stats.total_items} embeddings in {stats.total_time:.2f}s "
                    f"({stats.items_per_second:.2f} items/s) using dynamic batching. "
                    f"Batch size: {stats.initial_batch_size} → {stats.final_batch_size} "
                    f"(min: {stats.min_batch_size}, max: {stats.max_batch_size})"
                )
                
                # Log any OOM events
                if stats.oom_events > 0:
                    logger.warning(
                        f"Encountered {stats.oom_events} OOM errors during embedding generation. "
                        f"Batch size was adjusted {stats.batch_size_adjustments} times."
                    )
                
                return embeddings
            except ImportError as e:
                logger.warning(f"Dynamic batch processor unavailable: {str(e)}. Using static batching.")
            except Exception as e:
                logger.warning(f"GPU embedding generation with dynamic batching failed: {str(e)}. Falling back to CPU.")
                return self._embed_documents_static_batching(texts)
        
        # Fall back to static batching if GPU is not available or dynamic batching failed
        return self._embed_documents_static_batching(texts)
    
    def _embed_documents_static_batching(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using static batching (legacy method).
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embeddings, one for each input text
        """
        # Process in batches to avoid memory issues
        batch_size = min(self.max_batch_size, len(texts))
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            if self.use_gpu:
                try:
                    batch_embeddings = self._get_embeddings_gpu_batch(batch_texts)
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.warning(f"GPU embedding generation failed: {str(e)}. Falling back to CPU.")
                    batch_embeddings = self._get_embeddings_cpu_batch(batch_texts)
                    embeddings.extend(batch_embeddings)
            else:
                batch_embeddings = self._get_embeddings_cpu_batch(batch_texts)
                embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query text.
        
        Args:
            text: The query text to generate an embedding for
            
        Returns:
            The embedding for the text
        """
        # Check if we should use dynamic batch processor
        try:
            from langchain_hana.gpu.batch_processor import EmbeddingBatchProcessor
            use_dynamic = True
        except ImportError:
            use_dynamic = False
        
        # For single queries, we can use either method - dynamic is preferable for consistency
        if self.use_gpu and use_dynamic:
            try:
                # Create embedding function that handles a batch of one
                def batch_embedding_fn(batch_texts: List[str]) -> List[List[float]]:
                    if len(batch_texts) == 1:
                        # Use single query embedding function for better performance
                        return [self._get_embedding_gpu(batch_texts[0])]
                    else:
                        return self._get_embeddings_gpu_batch(batch_texts)
                
                # Create batch processor with appropriate settings - high priority for query
                processor = EmbeddingBatchProcessor(
                    embedding_fn=batch_embedding_fn,
                    model_name=self.model_name,
                    embedding_dim=self.model_dim,
                    device_id=0 if self.current_gpu is None else self.current_gpu.index,
                    initial_batch_size=1,  # Always use batch size 1 for queries
                    min_batch_size=1,
                    max_batch_size=1,
                    safety_factor=0.9,  # Higher safety factor for queries
                    dtype="float16" if self.precision == "fp16" else "float32" if self.precision == "fp32" else "int8",
                    enable_caching=True  # Enable caching for repeated queries
                )
                
                # Process the query
                return processor.embed_query(text)
            except Exception as e:
                logger.warning(f"GPU embedding generation with dynamic batching failed: {str(e)}. Falling back to standard method.")
        
        # Fall back to standard method
        if self.use_gpu:
            try:
                return self._get_embedding_gpu(text)
            except Exception as e:
                logger.warning(f"GPU embedding generation failed: {str(e)}. Falling back to CPU.")
                return self._get_embedding_cpu(text)
        else:
            return self._get_embedding_cpu(text)
            
    def benchmark(self, batch_sizes: List[int] = None, iterations: int = 100, warmup: int = 10) -> Dict[str, Any]:
        """
        Benchmark embedding performance.
        
        Args:
            batch_sizes: List of batch sizes to benchmark (defaults to [1, 8, 16, 32, 64, 128])
            iterations: Number of iterations for benchmarking
            warmup: Number of warmup iterations
            
        Returns:
            Dictionary with benchmark results
        """
        if batch_sizes is None:
            batch_sizes = [1, 8, 16, 32, 64, 128]
            # For INT8, limit batch sizes based on available memory
            if self.is_int8:
                batch_sizes = [size for size in batch_sizes if size <= self.max_batch_size]
        
        # Get GPU info if available
        gpu_info = {}
        if self.use_gpu and self.current_gpu:
            gpu_info["name"] = self.current_gpu.name
            gpu_info["compute_capability"] = self.current_gpu.compute_capability
            gpu_info["memory_total_mb"] = self.current_gpu.memory_total
            gpu_info["memory_free_mb"] = self.current_gpu.memory_free
        
        # Prepare benchmark results dictionary
        results = {
            "model": self.model_name,
            "device": "GPU" if self.use_gpu else "CPU",
            "precision": self.precision if self.use_gpu else "fp32",
            "batch_sizes": {},
            "gpu_info": gpu_info,
        }
        
        # Random sample text for benchmarking
        import random
        import string
        
        random_text = ''.join(random.choices(string.ascii_letters + ' .', k=100))
        
        # Benchmark single query latency
        if self.use_gpu:
            try:
                # Warmup
                for _ in range(warmup):
                    _ = self.embed_query(random_text)
                
                # Benchmark
                import time
                latencies = []
                for _ in range(iterations):
                    start_time = time.time()
                    _ = self.embed_query(random_text)
                    if self.use_gpu:
                        torch.cuda.synchronize()  # Wait for GPU operations to complete
                    latencies.append((time.time() - start_time) * 1000)  # ms
                
                # Add single query results
                results["single_query"] = {
                    "mean_latency_ms": float(np.mean(latencies)),
                    "median_latency_ms": float(np.median(latencies)),
                    "min_latency_ms": float(np.min(latencies)),
                    "max_latency_ms": float(np.max(latencies)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)),
                    "p99_latency_ms": float(np.percentile(latencies, 99)),
                    "throughput_queries_per_second": float(1000 / np.mean(latencies)),
                }
            except Exception as e:
                results["single_query_error"] = str(e)
        
        # Benchmark batched queries
        for batch_size in batch_sizes:
            try:
                # Create batch of random texts
                batch = [random_text] * batch_size
                
                # Warmup
                for _ in range(max(1, warmup // batch_size)):
                    _ = self.embed_documents(batch)
                
                # Benchmark
                import time
                latencies = []
                for _ in range(max(1, iterations // batch_size)):
                    start_time = time.time()
                    _ = self.embed_documents(batch)
                    if self.use_gpu:
                        torch.cuda.synchronize()  # Wait for GPU operations to complete
                    latencies.append((time.time() - start_time) * 1000)  # ms
                
                # Calculate statistics
                mean_latency = float(np.mean(latencies))
                
                # Add batch results
                results["batch_sizes"][str(batch_size)] = {
                    "mean_latency_ms": mean_latency,
                    "median_latency_ms": float(np.median(latencies)),
                    "min_latency_ms": float(np.min(latencies)),
                    "max_latency_ms": float(np.max(latencies)),
                    "p95_latency_ms": float(np.percentile(latencies, 95)),
                    "p99_latency_ms": float(np.percentile(latencies, 99)),
                    "throughput_samples_per_second": float(batch_size * 1000 / mean_latency),
                    "throughput_tokens_per_second": float(batch_size * 100 * 1000 / mean_latency),  # Assuming ~100 tokens per text
                }
            except Exception as e:
                results["batch_sizes"][str(batch_size)] = {"error": str(e)}
        
        return results
    
    def benchmark_precision_comparison(self) -> Dict[str, Any]:
        """
        Run a comparison of different precision modes (FP32, FP16, INT8).
        
        This method creates temporary instances with different precision modes
        and benchmarks them for comparison.
        
        Returns:
            Dictionary with benchmark results for different precision modes
        """
        # Check if GPU is available
        if not self.use_gpu or not HAS_GPU_DEPENDENCIES:
            return {"error": "GPU not available for precision comparison"}
        
        # Current GPU info
        gpu_info = {}
        if self.current_gpu:
            gpu_info["name"] = self.current_gpu.name
            gpu_info["compute_capability"] = self.current_gpu.compute_capability
        
        # Prepare results dictionary
        results = {
            "model": self.model_name,
            "gpu_info": gpu_info,
            "precision_modes": {},
        }
        
        # Get all available precision modes based on GPU capabilities
        available_modes = ["fp32"]
        
        # Check for FP16 support
        if self.current_gpu and float(self.current_gpu.compute_capability) >= 7.0:
            available_modes.append("fp16")
        
        # Check for INT8 support
        if self.current_gpu and float(self.current_gpu.compute_capability) >= 7.5:
            available_modes.append("int8")
        
        # Test each available precision mode
        for precision in available_modes:
            try:
                logger.info(f"Benchmarking {precision} precision mode...")
                
                # Create a temporary embeddings instance with this precision
                temp_embeddings = TensorRTEmbeddings(
                    model_name=self.model_name,
                    cache_dir=self.cache_dir,
                    precision=precision,
                    max_batch_size=self.max_batch_size,
                    calibration_cache_dir=self.calibration_cache_dir,
                    calibration_data=self.calibration_data,
                )
                
                # Run benchmark with common batch sizes
                batch_sizes = [1, 8, 16, 32]
                benchmark_result = temp_embeddings.benchmark(batch_sizes=batch_sizes, iterations=50)
                
                # Add to results
                results["precision_modes"][precision] = benchmark_result
                
                # Clean up temporary embeddings
                del temp_embeddings
                if HAS_GPU_DEPENDENCIES:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error benchmarking {precision} precision: {str(e)}")
                results["precision_modes"][precision] = {"error": str(e)}
        
        # Calculate speedup factors if we have fp32 as baseline
        if "fp32" in results["precision_modes"] and "batch_sizes" in results["precision_modes"]["fp32"]:
            try:
                # Use batch size 16 for comparison
                fp32_throughput = results["precision_modes"]["fp32"]["batch_sizes"]["16"]["throughput_tokens_per_second"]
                
                # FP16 speedup
                if "fp16" in results["precision_modes"] and "batch_sizes" in results["precision_modes"]["fp16"]:
                    fp16_throughput = results["precision_modes"]["fp16"]["batch_sizes"]["16"]["throughput_tokens_per_second"]
                    results["fp16_vs_fp32_speedup"] = fp16_throughput / fp32_throughput
                
                # INT8 speedup
                if "int8" in results["precision_modes"] and "batch_sizes" in results["precision_modes"]["int8"]:
                    int8_throughput = results["precision_modes"]["int8"]["batch_sizes"]["16"]["throughput_tokens_per_second"]
                    results["int8_vs_fp32_speedup"] = int8_throughput / fp32_throughput
                    
                    # INT8 vs FP16
                    if "fp16" in results["precision_modes"] and "batch_sizes" in results["precision_modes"]["fp16"]:
                        fp16_throughput = results["precision_modes"]["fp16"]["batch_sizes"]["16"]["throughput_tokens_per_second"]
                        results["int8_vs_fp16_speedup"] = int8_throughput / fp16_throughput
            except (KeyError, ZeroDivisionError, TypeError) as e:
                results["speedup_calculation_error"] = str(e)
        
        # Add recommended precision based on results
        if "int8_vs_fp32_speedup" in results and results["int8_vs_fp32_speedup"] > 1.1:
            results["recommended_precision"] = "int8"
        elif "fp16_vs_fp32_speedup" in results and results["fp16_vs_fp32_speedup"] > 1.1:
            results["recommended_precision"] = "fp16"
        else:
            results["recommended_precision"] = "fp32"
        
        return results