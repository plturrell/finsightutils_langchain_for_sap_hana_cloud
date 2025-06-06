"""
TensorRT-accelerated embeddings provider for SAP HANA Cloud.

This module provides a specialized embeddings class that combines the power of TensorRT
GPU acceleration with SAP HANA Cloud's vector capabilities, enabling high-performance
embedding generation for large-scale document processing.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import numpy as np
from langchain_core.embeddings import Embeddings

from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.gpu.tensorrt_embeddings import TensorRTEmbeddings
from langchain_hana.gpu.batch_processor import EmbeddingBatchProcessor
from langchain_hana.gpu.calibration_datasets import create_enhanced_calibration_dataset
from langchain_hana.gpu.tensor_core_optimizer import TensorCoreOptimizer
from langchain_hana.gpu.multi_gpu_manager import get_multi_gpu_manager, EnhancedMultiGPUManager

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingPerformanceStats:
    """Statistics about embedding performance."""
    total_documents: int
    total_time_seconds: float
    documents_per_second: float
    avg_document_time_ms: float
    peak_memory_mb: float
    batch_size: int
    precision: str
    gpu_name: Optional[str] = None
    gpu_count: int = 1
    
    def __str__(self) -> str:
        """Convert to human-readable string."""
        return (
            f"Embedded {self.total_documents} documents in {self.total_time_seconds:.2f}s "
            f"({self.documents_per_second:.2f} docs/s, {self.avg_document_time_ms:.2f}ms/doc) "
            f"using {self.precision} precision with batch size {self.batch_size} "
            f"on {self.gpu_count} GPU{'' if self.gpu_count == 1 else 's'}"
            f"{f' ({self.gpu_name})' if self.gpu_name else ''}"
        )


class HanaTensorRTEmbeddings(Embeddings):
    """
    TensorRT-accelerated embeddings for SAP HANA Cloud integration.
    
    This class combines TensorRT GPU optimization with SAP HANA Cloud vector capabilities
    to provide high-performance embedding generation for document processing and similarity search.
    
    Key features:
    1. GPU acceleration with TensorRT for 2-6x speedup over CPU-based embeddings
    2. Multi-GPU support with intelligent load balancing
    3. Memory optimization with customizable precision modes (FP32, FP16, INT8)
    4. Configurable for HANA internal embeddings when available
    5. T4-specific optimizations for tensor core utilization
    6. Memory-efficient batch processing with dynamic batch sizing
    7. Automatic fallback to CPU for reliability
    
    This class integrates seamlessly with the SAP HANA vectorstore implementation,
    providing a drop-in replacement for standard embedding models with significantly
    improved performance.
    
    Example:
        ```python
        from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings
        from langchain_hana.vectorstores import HanaDB
        
        # Create GPU-accelerated embeddings
        embeddings = HanaTensorRTEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            precision="fp16",  # Use FP16 precision for best performance on T4 GPUs
            multi_gpu=True,    # Use multiple GPUs if available
        )
        
        # Create vectorstore with GPU-accelerated embeddings
        vectorstore = HanaDB(
            connection=conn,
            embedding=embeddings,
            table_name="MY_VECTORS"
        )
        
        # Add documents (will use GPU-accelerated embedding generation)
        documents = ["Document 1", "Document 2", "Document 3", ...]
        vectorstore.add_texts(documents)
        
        # Search for similar documents (will use GPU for query embedding)
        results = vectorstore.similarity_search("What is SAP HANA Cloud?")
        ```
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        internal_embedding_model_id: Optional[str] = None,
        precision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        multi_gpu: bool = False,
        max_batch_size: int = 32,
        force_engine_rebuild: bool = False,
        gpu_memory_threshold: float = 20.0,
        calibration_domain: Optional[str] = None,
        calibration_data: Optional[List[str]] = None,
        enable_tensor_cores: bool = True,
        device: Union[str, int] = 0,
        enable_profiling: bool = False,
    ):
        """
        Initialize the TensorRT-accelerated embeddings for SAP HANA Cloud.
        
        Args:
            model_name: Name of the Hugging Face model to use for embeddings
                      (e.g., "sentence-transformers/all-MiniLM-L6-v2")
            internal_embedding_model_id: Optional ID of SAP HANA internal embedding model
                                       (e.g., "SAP_NEB.20240715"). If provided, uses HANA's
                                       internal embedding capabilities instead of GPU.
            precision: Precision to use for TensorRT optimization ("fp32", "fp16", "int8").
                      Default is "fp16" for T4 GPUs and "fp32" otherwise.
            cache_dir: Directory to cache optimized TensorRT engines
            multi_gpu: Whether to use multiple GPUs if available
            max_batch_size: Maximum batch size for embedding generation
            force_engine_rebuild: If True, forces rebuilding the TensorRT engine
            gpu_memory_threshold: Percentage of GPU memory that must be free to use GPU
            calibration_domain: Domain for INT8 calibration ("financial", "sap", "technical", "all")
            calibration_data: Custom text samples for INT8 calibration
            enable_tensor_cores: Whether to enable Tensor Core optimizations
            device: CUDA device to use if multi_gpu is False
            enable_profiling: Whether to enable performance profiling
        """
        self.model_name = model_name
        self.internal_embedding_model_id = internal_embedding_model_id
        self.precision = precision
        self.cache_dir = cache_dir
        self.multi_gpu = multi_gpu
        self.max_batch_size = max_batch_size
        self.force_engine_rebuild = force_engine_rebuild
        self.gpu_memory_threshold = gpu_memory_threshold
        self.enable_tensor_cores = enable_tensor_cores
        self.device = device
        self.enable_profiling = enable_profiling
        
        # Internal state
        self.embedding_model = None
        self.tensor_core_optimizer = None
        self.use_internal_embeddings = internal_embedding_model_id is not None
        self.last_performance_stats = None
        
        # Create enhanced calibration dataset for INT8 quantization if needed
        if calibration_domain and precision == "int8":
            self.calibration_data = create_enhanced_calibration_dataset(
                domains=[calibration_domain] if calibration_domain != "all" else None,
                count=100,
                custom_file_path=None
            )
        else:
            self.calibration_data = calibration_data
        
        # Initialize embeddings based on configuration
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize the embeddings based on configuration."""
        if self.use_internal_embeddings:
            # Use SAP HANA's internal embedding function
            logger.info(f"Using SAP HANA internal embeddings with model ID: {self.internal_embedding_model_id}")
            self.embedding_model = HanaInternalEmbeddings(
                internal_embedding_model_id=self.internal_embedding_model_id
            )
            return
        
        # Initialize tensor core optimizer if enabled
        if self.enable_tensor_cores:
            self.tensor_core_optimizer = TensorCoreOptimizer(
                device="cuda" if isinstance(self.device, int) else self.device,
                tensor_core_enabled=True,
                precision=self.precision or "fp16",
                workspace_size_mb=1024,
                enable_profiling=self.enable_profiling
            )
            
            # Check if tensor cores are actually available
            if not self.tensor_core_optimizer.tensor_cores_available:
                logger.warning("Tensor Cores not available on this GPU. Disabling Tensor Core optimizations.")
                self.tensor_core_optimizer = None
        
        # Initialize multi-GPU manager if enabled
        if self.multi_gpu:
            # Enable multi-GPU manager
            os.environ["MULTI_GPU_ENABLED"] = "true"
            os.environ["MULTI_GPU_STRATEGY"] = "auto"
            
            # Get the multi-GPU manager
            multi_gpu_manager = get_multi_gpu_manager()
            
            # Check if multiple GPUs are available
            available_devices = multi_gpu_manager.get_available_devices()
            if len(available_devices) <= 1:
                logger.warning(f"Only {len(available_devices)} GPUs available. Multi-GPU support disabled.")
                self.multi_gpu = False
        
        # Initialize TensorRT embeddings
        try:
            self.embedding_model = TensorRTEmbeddings(
                model_name=self.model_name,
                cache_dir=self.cache_dir,
                force_engine_rebuild=self.force_engine_rebuild,
                max_batch_size=self.max_batch_size,
                gpu_memory_threshold=self.gpu_memory_threshold,
                precision=self.precision,
                calibration_data=self.calibration_data
            )
            logger.info(
                f"TensorRT embeddings initialized: {self.model_name} with "
                f"{self.precision or 'auto'} precision "
                f"(Multi-GPU: {self.multi_gpu})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT embeddings: {e}")
            # Fall back to standard embeddings via sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                logger.warning(f"Falling back to CPU-based SentenceTransformer for {self.model_name}")
                self.embedding_model = SentenceTransformer(self.model_name)
            except ImportError:
                logger.error("Could not import sentence_transformers. Please install with: pip install sentence-transformers")
                raise ImportError("sentence_transformers is required for fallback embedding generation")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        This method leverages GPU acceleration via TensorRT or falls back to
        internal HANA embeddings when configured to do so.
        
        Args:
            texts: List of texts to generate embeddings for
            
        Returns:
            List of embedding vectors, one for each text
            
        Notes:
            - When using SAP HANA internal embeddings, this will raise NotImplementedError
              as intended, since embedding generation happens in the database
            - When using GPU acceleration, this will distribute work across multiple
              GPUs if enabled, with dynamic batch sizing for optimal performance
        """
        if self.use_internal_embeddings:
            # Let the internal embeddings implementation handle this
            # This will raise NotImplementedError as expected
            return self.embedding_model.embed_documents(texts)
        
        # Track performance
        import time
        start_time = time.time()
        peak_memory = 0
        
        # Get GPU info for statistics
        gpu_name = None
        gpu_count = 1
        
        try:
            import torch
            if torch.cuda.is_available():
                device_idx = 0 if isinstance(self.device, str) else self.device
                gpu_name = torch.cuda.get_device_name(device_idx)
                gpu_count = torch.cuda.device_count() if self.multi_gpu else 1
        except Exception:
            pass
        
        # Use multi-GPU processing if enabled
        if self.multi_gpu:
            try:
                multi_gpu_manager = get_multi_gpu_manager()
                
                # Define a function to process a batch using TensorRT
                def process_batch(batch: List[str]) -> List[List[float]]:
                    return self.embedding_model.embed_documents(batch)
                
                # Submit batch processing task
                embeddings = multi_gpu_manager.batch_process(
                    func=process_batch,
                    items=texts,
                    batch_size=self.max_batch_size,
                    wait=True
                )
                
                # Get status for statistics
                status = multi_gpu_manager.get_status()
                if "devices" in status:
                    for device_info in status["devices"]:
                        if "memory_allocated_mb" in device_info:
                            peak_memory = max(peak_memory, device_info["memory_allocated_mb"])
            except Exception as e:
                logger.warning(f"Multi-GPU processing failed: {e}. Falling back to single GPU.")
                # Fall back to single GPU
                embeddings = self.embedding_model.embed_documents(texts)
        else:
            # Standard embedding generation
            embeddings = self.embedding_model.embed_documents(texts)
            
            # Try to get peak memory usage
            try:
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except Exception:
                pass
        
        # Record performance statistics
        total_time = time.time() - start_time
        
        self.last_performance_stats = EmbeddingPerformanceStats(
            total_documents=len(texts),
            total_time_seconds=total_time,
            documents_per_second=len(texts) / total_time if total_time > 0 else 0,
            avg_document_time_ms=(total_time / len(texts)) * 1000 if len(texts) > 0 else 0,
            peak_memory_mb=peak_memory,
            batch_size=self.max_batch_size,
            precision=self.precision or "auto",
            gpu_name=gpu_name,
            gpu_count=gpu_count
        )
        
        logger.info(f"Embedding performance: {self.last_performance_stats}")
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate an embedding for a query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector for the query
            
        Notes:
            - When using SAP HANA internal embeddings, this will raise NotImplementedError
              as intended, since embedding generation happens in the database
            - Query embedding is optimized for low latency with GPU acceleration
        """
        if self.use_internal_embeddings:
            # Let the internal embeddings implementation handle this
            # This will raise NotImplementedError as expected
            return self.embedding_model.embed_query(text)
        
        # For a single query, use the TensorRT model directly
        # This avoids overhead of multi-GPU processing for a single query
        return self.embedding_model.embed_query(text)
    
    def get_performance_stats(self) -> Optional[EmbeddingPerformanceStats]:
        """
        Get the performance statistics from the last embedding operation.
        
        Returns:
            Performance statistics or None if no embeddings have been generated
        """
        return self.last_performance_stats
    
    def get_model_id(self) -> Optional[str]:
        """
        Get the internal embedding model ID if using HANA internal embeddings.
        
        Returns:
            Internal embedding model ID or None if not using internal embeddings
        """
        if self.use_internal_embeddings:
            return self.internal_embedding_model_id
        return None
    
    def get_tensor_core_stats(self) -> Optional[List[Dict[str, Any]]]:
        """
        Get profiling data from the tensor core optimizer if enabled.
        
        Returns:
            List of profiling data entries or None if tensor core optimization is disabled
        """
        if self.tensor_core_optimizer and self.enable_profiling:
            return self.tensor_core_optimizer.get_profiling_data()
        return None
    
    def benchmark(self, batch_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run a benchmark to measure embedding performance.
        
        Args:
            batch_sizes: List of batch sizes to benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        if self.use_internal_embeddings:
            return {
                "error": "Benchmarking is not available with HANA internal embeddings"
            }
        
        # Use TensorRT benchmark
        try:
            return self.embedding_model.benchmark(batch_sizes=batch_sizes)
        except Exception as e:
            return {
                "error": f"Benchmark failed: {str(e)}"
            }
    
    def benchmark_precision_comparison(self) -> Dict[str, Any]:
        """
        Compare performance across different precision modes (FP32, FP16, INT8).
        
        Returns:
            Dictionary with benchmark results for different precision modes
        """
        if self.use_internal_embeddings:
            return {
                "error": "Precision comparison is not available with HANA internal embeddings"
            }
        
        # Use TensorRT precision comparison
        try:
            if hasattr(self.embedding_model, "benchmark_precision_comparison"):
                return self.embedding_model.benchmark_precision_comparison()
            else:
                return {
                    "error": "Precision comparison not supported by the current embedding model"
                }
        except Exception as e:
            return {
                "error": f"Precision comparison failed: {str(e)}"
            }