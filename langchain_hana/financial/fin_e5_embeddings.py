"""
FinMTEB/Fin-E5 Financial Embedding Model Integration for SAP HANA Cloud

This module provides specialized financial domain embeddings using the FinMTEB/Fin-E5 models,
optimized for financial text analysis and retrieval applications with SAP HANA Cloud.

Key features:
- Optimized financial domain embeddings with FinMTEB/Fin-E5 models
- GPU acceleration with TensorRT for high-performance embedding generation
- Financial context enhancement for improved embedding quality
- Domain-specific caching for better performance
- Automatic calibration with financial terminology
- Specialized preprocessing for financial documents
"""

import logging
import time
import os
import json
import time
from typing import List, Dict, Optional, Union, Any, Tuple, Callable, Iterator
import numpy as np

import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from langchain_hana.embeddings import HanaEmbeddingsCache
from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings
from langchain_hana.gpu.tensorrt_diagnostics import (
    TensorRTDiagnostics, 
    with_tensorrt_diagnostics,
    try_import_tensorrt
)
from langchain_hana.gpu.memory_manager import (
    GPUMemoryManager,
    FinancialEmbeddingMemoryManager,
    with_memory_optimization
)
from langchain_hana.financial.caching import FinancialEmbeddingCache

# Configure logging
logger = logging.getLogger(__name__)

# Model IDs for financial domain embedding models from FinMTEB
FINANCIAL_EMBEDDING_MODELS = {
    "default": "FinMTEB/Fin-E5-small",  # Best balance of quality and performance
    "high_quality": "FinMTEB/Fin-E5",  # Highest quality, larger model
    "efficient": "FinLang/investopedia_embedding",  # Most efficient, good for limited resources
    "tone": "yiyanghkust/finbert-tone",  # Specialized for sentiment/tone analysis
    "financial_bert": "ProsusAI/finbert",  # Good for SEC filings, earnings reports
    "finance_base": "baconnier/Finance_embedding_large_en-V0.1"  # General finance embeddings
}

# Financial terminology for context enhancement
FINANCIAL_PREFIXES = {
    "general": "Financial context: ",
    "analysis": "Financial analysis: ",
    "report": "Financial report: ",
    "news": "Financial news: ",
    "forecast": "Financial forecast: ",
    "investment": "Investment context: "
}

class FinE5Embeddings(Embeddings):
    """
    Financial domain-specific embedding model using FinMTEB/Fin-E5 models.
    
    This class provides optimized embeddings for financial text, with GPU acceleration
    when available and automatic fallback to CPU when needed. It includes memory
    optimization techniques such as mixed precision inference and dynamic batching.
    
    Examples:
        ```python
        from langchain_hana.financial.fin_e5_embeddings import FinE5Embeddings
        
        # Use default Fin-E5-small model
        embeddings = FinE5Embeddings()
        
        # Use high-quality Fin-E5 model with GPU acceleration
        embeddings = FinE5Embeddings(
            model_type="high_quality", 
            device="cuda",
            use_fp16=True
        )
        
        # Use with financial context enhancement
        embeddings = FinE5Embeddings(
            model_type="default",
            add_financial_prefix=True,
            financial_prefix_type="analysis"
        )
        
        # Use with advanced memory management for large document batches
        embeddings = FinE5Embeddings(
            model_type="default",
            enable_memory_optimization=True,
            memory_optimization_level="aggressive"
        )
        
        # Embed documents
        vectors = embeddings.embed_documents(["Q1 financial results showed a 15% increase in revenue"])
        
        # Embed query
        query_vector = embeddings.embed_query("What was the revenue growth in Q1?")
        ```
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_type: str = "default",
        device: Optional[str] = None,
        use_fp16: bool = True,
        normalize_embeddings: bool = True,
        max_seq_length: int = 512,
        batch_size: int = 32,
        add_financial_prefix: bool = False,
        financial_prefix_type: str = "general",
        enable_caching: bool = True,
        cache_ttl: int = 3600,  # 1 hour cache TTL
        cache_max_size: int = 10000,
        cache_persist_path: Optional[str] = None,
        use_tensorrt: bool = False,
        tensorrt_precision: Optional[str] = None,
        multi_gpu: bool = False,
        enable_tensor_cores: bool = True,
        enable_memory_optimization: bool = False,
        memory_optimization_level: str = "balanced",
        adaptive_batch_size: bool = True,
    ):
        """
        Initialize the financial embedding model.
        
        Parameters
        ----------
        model_name : str, optional
            Specific model name to use (overrides model_type if provided)
        model_type : str, default='default'
            Type of financial model to use ('default', 'high_quality', 
            'efficient', 'tone', 'financial_bert', 'finance_base')
        device : str, optional
            Device to use ('cuda', 'cpu', or None for auto-detection)
        use_fp16 : bool, default=True
            Whether to use half-precision (FP16) for faster inference
        normalize_embeddings : bool, default=True
            Whether to L2-normalize the embeddings
        max_seq_length : int, default=512
            Maximum sequence length for the model
        batch_size : int, default=32
            Batch size for document embedding
        add_financial_prefix : bool, default=False
            Whether to add a prefix to improve financial context
        financial_prefix_type : str, default='general'
            Type of financial prefix to add ('general', 'analysis', 
            'report', 'news', 'forecast', 'investment')
        enable_caching : bool, default=True
            Whether to enable caching for improved performance
        cache_ttl : int, default=3600
            Time-to-live for cache entries in seconds (1 hour default)
        cache_max_size : int, default=10000
            Maximum number of entries in the cache
        cache_persist_path : str, optional
            Path to persist the cache (None for in-memory only)
        use_tensorrt : bool, default=False
            Whether to use TensorRT GPU acceleration (if available)
        tensorrt_precision : str, optional
            Precision for TensorRT ('fp32', 'fp16', 'int8', None for auto)
        multi_gpu : bool, default=False
            Whether to use multiple GPUs with TensorRT (if available)
        enable_tensor_cores : bool, default=True
            Whether to enable Tensor Core optimizations with TensorRT
        enable_memory_optimization : bool, default=False
            Whether to enable advanced memory optimization for large batches
        memory_optimization_level : str, default='balanced'
            Level of memory optimization ('conservative', 'balanced', 'aggressive')
        adaptive_batch_size : bool, default=True
            Whether to adaptively adjust batch size based on available memory
            
        Notes
        -----
        The model initialization automatically detects available hardware and
        applies appropriate optimizations. When using GPU acceleration, the 
        model will try to use CUDA if available, with FP16 precision enabled
        by default for better performance.
        
        When using financial context prefixes, different types are available
        for different domains of financial text, which can significantly
        improve embedding quality for domain-specific queries.
        
        Memory optimization is particularly useful for large document batches
        and environments with limited GPU memory. When enabled, it automatically
        manages GPU memory to prevent out-of-memory errors by dynamically adjusting
        batch sizes and proactively reclaiming unused memory.
        """
        # Set model name based on model_type if not explicitly provided
        if model_name is None:
            if model_type not in FINANCIAL_EMBEDDING_MODELS:
                raise ValueError(
                    f"Invalid model_type: {model_type}. "
                    f"Available types: {', '.join(FINANCIAL_EMBEDDING_MODELS.keys())}"
                )
            model_name = FINANCIAL_EMBEDDING_MODELS[model_type]
        
        # Store initialization parameters
        self.model_name = model_name
        self.model_type = model_type
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        
        # Financial context enhancement
        self.add_financial_prefix = add_financial_prefix
        if financial_prefix_type not in FINANCIAL_PREFIXES:
            raise ValueError(
                f"Invalid financial_prefix_type: {financial_prefix_type}. "
                f"Available types: {', '.join(FINANCIAL_PREFIXES.keys())}"
            )
        self.financial_prefix = FINANCIAL_PREFIXES[financial_prefix_type]
        
        # TensorRT acceleration parameters
        self.use_tensorrt = use_tensorrt
        self.tensorrt_precision = tensorrt_precision
        self.multi_gpu = multi_gpu
        self.enable_tensor_cores = enable_tensor_cores
        
        # Memory optimization parameters
        self.enable_memory_optimization = enable_memory_optimization
        self.memory_optimization_level = memory_optimization_level
        self.adaptive_batch_size = adaptive_batch_size
        
        # Initialize memory manager if enabled
        self.memory_manager = None
        if self.enable_memory_optimization and self.device == 'cuda':
            # Set memory optimization parameters based on level
            if memory_optimization_level == "conservative":
                max_memory_usage = 0.8
                min_free_memory_mb = 2048
                safety_factor = 0.7
            elif memory_optimization_level == "aggressive":
                max_memory_usage = 0.95
                min_free_memory_mb = 512
                safety_factor = 0.9
            else:  # balanced (default)
                max_memory_usage = 0.9
                min_free_memory_mb = 1024
                safety_factor = 0.8
            
            # Initialize financial memory manager
            self.memory_manager = FinancialEmbeddingMemoryManager(
                device=self.device,
                model_type=self.model_type,
                max_memory_usage_percent=max_memory_usage,
                min_free_memory_mb=min_free_memory_mb,
                safety_factor=safety_factor,
                enable_active_gc=True,
                enable_logging=True
            )
            logger.info(f"Memory optimization enabled with {memory_optimization_level} level")
        
        # Log configuration
        logger.info(f"Initializing FinE5Embeddings with model: {model_name}")
        logger.info(f"Device: {self.device}, FP16: {self.use_fp16}")
        if self.use_tensorrt:
            logger.info(f"TensorRT acceleration enabled with precision: {self.tensorrt_precision or 'auto'}")
            if self.multi_gpu:
                logger.info("Multi-GPU support enabled")
        
        # Initialize model based on configuration
        self._initialize_model()
        
        # Initialize cache if enabled
        self.enable_caching = enable_caching
        if enable_caching:
            self.embeddings_cache = FinancialEmbeddingCache(
                base_embeddings=self,
                ttl_seconds=cache_ttl,
                max_size=cache_max_size,
                persist_path=cache_persist_path,
                model_name=model_name.split('/')[-1]  # Use model name for cache identification
            )
            logger.info("Financial embedding cache initialized")
    
    def _initialize_model(self) -> None:
        """
        Initialize the embedding model with optimizations.
        
        This method handles the initialization of the underlying embedding model,
        including loading the model weights, applying hardware-specific optimizations,
        and setting up TensorRT acceleration if requested.
        
        Returns
        -------
        None
            The method initializes the model in-place.
            
        Raises
        ------
        RuntimeError
            If model initialization fails.
        """
        try:
            # Initialize with TensorRT if enabled
            if self.use_tensorrt:
                try:
                    # First check if TensorRT is available with diagnostics
                    tensorrt_available, _, diagnostics = try_import_tensorrt()
                    if not tensorrt_available:
                        logger.warning(
                            f"TensorRT not available: {diagnostics.get_summary() if diagnostics else 'Unknown error'}"
                            f"\nFalling back to standard model."
                        )
                        raise ImportError("TensorRT not available")
                    
                    start_time = time.time()
                    
                    # Log detailed environment information
                    env_diagnostics = TensorRTDiagnostics.run_diagnostics()
                    logger.info(f"TensorRT environment: {env_diagnostics.get_summary()}")
                    
                    # Create specialized financial calibration data if needed
                    calibration_data = None
                    if self.tensorrt_precision == "int8":
                        from langchain_hana.gpu.calibration_datasets import create_enhanced_calibration_dataset
                        logger.info("Creating specialized financial calibration dataset for INT8 quantization")
                        calibration_data = create_enhanced_calibration_dataset(
                            domains=["financial"],
                            count=100,
                            custom_file_path=None
                        )
                    
                    # Initialize TensorRT accelerated embeddings with diagnostics
                    @with_tensorrt_diagnostics
                    def initialize_tensorrt_embeddings():
                        return HanaTensorRTEmbeddings(
                            model_name=self.model_name,
                            precision=self.tensorrt_precision,
                            multi_gpu=self.multi_gpu,
                            max_batch_size=self.batch_size,
                            enable_tensor_cores=self.enable_tensor_cores,
                            calibration_domain="financial",
                            calibration_data=calibration_data
                        )
                    
                    logger.info(f"Initializing TensorRT accelerated embeddings with model: {self.model_name}")
                    self.model = initialize_tensorrt_embeddings()
                    
                    load_time = time.time() - start_time
                    logger.info(f"TensorRT accelerated model loaded in {load_time:.2f} seconds")
                    
                    # Skip the rest of initialization since HanaTensorRTEmbeddings handles it
                    self.embedding_dim = 384  # Default for Fin-E5 models
                    return
                except Exception as e:
                    # Run diagnostics on the error
                    diagnostics = TensorRTDiagnostics.diagnose_error(e)
                    
                    # Log detailed error information and recommendations
                    logger.warning(f"Failed to initialize TensorRT: {diagnostics.error_type} - {diagnostics.error_message}")
                    logger.warning(f"TensorRT diagnostics: {diagnostics.get_summary()}")
                    
                    if diagnostics.recommendations:
                        logger.info("Recommendations for TensorRT initialization:")
                        for i, recommendation in enumerate(diagnostics.recommendations, 1):
                            logger.info(f"{i}. {recommendation}")
                    
                    logger.info("Falling back to standard model")
            
            # Standard model initialization if TensorRT is not used or failed
            logger.info(f"Loading model {self.model_name} on {self.device}")
            start_time = time.time()
            
            # Load the model with optimizations
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.model.max_seq_length = self.max_seq_length
            
            # Apply mixed precision if enabled
            if self.use_fp16:
                self.model.half()  # Convert to FP16
            
            # Apply torch.compile if available (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device == 'cuda':
                try:
                    self.model.encode = torch.compile(
                        self.model.encode, 
                        mode="reduce-overhead"
                    )
                    logger.info("Applied torch.compile optimization")
                except Exception as e:
                    logger.warning(f"Failed to apply torch.compile: {str(e)}")
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")
            
            # Get embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def _add_financial_context(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Add financial context prefix to improve embedding quality.
        
        Args:
            texts: Text or list of texts to add context to
            
        Returns:
            Text or list of texts with context prefix added
        """
        if not self.add_financial_prefix:
            return texts
        
        if isinstance(texts, str):
            return f"{self.financial_prefix}{texts}"
        
        return [f"{self.financial_prefix}{text}" for text in texts]
    
    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts with proper error handling and memory optimization.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed
            
        Returns
        -------
        List[List[float]]
            List of embedding vectors
            
        Notes
        -----
        This method implements several optimization techniques:
        - Uses memory manager for large batches when enabled
        - Implements dynamic batch sizing based on available memory
        - Provides fallback to CPU when GPU memory is exhausted
        - Automatically reclaims GPU memory when needed
        """
        if not texts:
            return []
        
        # Add financial context if enabled
        texts = self._add_financial_context(texts)
        
        # Handle TensorRT model specially
        if self.use_tensorrt and hasattr(self.model, 'embed_documents'):
            return self.model.embed_documents(texts)
        
        # Use memory manager if enabled
        if self.enable_memory_optimization and self.device == 'cuda' and self.memory_manager is not None:
            try:
                with with_memory_optimization(self.memory_manager):
                    # Get optimal batch size if adaptive batching is enabled
                    effective_batch_size = self.batch_size
                    if self.adaptive_batch_size:
                        text_lengths = [len(text) for text in texts]
                        avg_length = sum(text_lengths) / max(1, len(text_lengths))
                        effective_batch_size = self.memory_manager.get_optimal_batch_size(
                            avg_token_length=avg_length,
                            current_batch_size=self.batch_size,
                            model_name=self.model_name
                        )
                        logger.debug(f"Adaptive batch size: {effective_batch_size} (original: {self.batch_size})")
                    
                    # Process texts in optimized batches
                    if len(texts) > effective_batch_size:
                        # Process in smaller batches
                        results = []
                        for i in range(0, len(texts), effective_batch_size):
                            batch_texts = texts[i:i+effective_batch_size]
                            batch_embeddings = self.model.encode(
                                batch_texts,
                                batch_size=effective_batch_size,
                                show_progress_bar=False,
                                normalize_embeddings=self.normalize_embeddings,
                                convert_to_numpy=True
                            )
                            results.extend(batch_embeddings.tolist())
                            # Proactively reclaim memory after each batch
                            self.memory_manager.reclaim_memory()
                        return results
                    else:
                        # Process all texts at once
                        embeddings = self.model.encode(
                            texts,
                            batch_size=effective_batch_size,
                            show_progress_bar=False,
                            normalize_embeddings=self.normalize_embeddings,
                            convert_to_numpy=True
                        )
                        return embeddings.tolist()
            except Exception as e:
                logger.warning(f"Memory-optimized embedding failed: {str(e)}")
                # Continue with standard approach as fallback
        
        # Standard approach without memory optimization
        try:
            # Try to embed with current settings
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True
            )
            return embeddings.tolist()
            
        except torch.cuda.OutOfMemoryError:
            # Handle CUDA OOM error with fallback strategy
            logger.warning("CUDA out of memory, falling back to CPU")
            
            # Save original device and switch to CPU
            original_device = self.device
            self.device = 'cpu'
            
            try:
                # Move model to CPU
                self.model.to('cpu')
                
                # Try again with CPU
                embeddings = self.model.encode(
                    texts,
                    batch_size=max(1, self.batch_size // 4),  # Reduce batch size
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True
                )
                
                # Return the embeddings
                result = embeddings.tolist()
                
                # Try to move back to original device
                if original_device == 'cuda' and torch.cuda.is_available():
                    try:
                        # Clear CUDA cache
                        torch.cuda.empty_cache()
                        self.model.to('cuda')
                        self.device = 'cuda'
                        logger.info("Successfully moved back to CUDA device")
                    except Exception as e:
                        logger.warning(f"Failed to move back to CUDA: {str(e)}")
                
                return result
                
            except Exception as e:
                logger.error(f"CPU fallback failed: {str(e)}")
                raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Parameters
        ----------
        texts : List[str]
            List of texts to embed
            
        Returns
        -------
        List[List[float]]
            List of embedding vectors
            
        Notes
        -----
        This method applies memory optimization when enabled and large batches
        are detected. For very large document sets, the method will automatically
        adjust batch sizes and manage memory to prevent OOM errors.
        """
        # If caching is enabled, use the cache
        if self.enable_caching:
            return self.embeddings_cache.embed_documents(texts)
        
        # For very large document sets, apply additional memory optimization
        if (self.enable_memory_optimization and self.device == 'cuda' and 
                self.memory_manager is not None and len(texts) > 1000):
            logger.info(f"Processing {len(texts)} documents with aggressive memory optimization")
            
            # For extremely large batches, process in chunks with memory reclamation
            chunk_size = 500  # Process 500 documents at a time
            results = []
            
            for i in range(0, len(texts), chunk_size):
                chunk_texts = texts[i:i+chunk_size]
                logger.debug(f"Processing chunk {i//chunk_size + 1}/{(len(texts)-1)//chunk_size + 1} ({len(chunk_texts)} documents)")
                
                # Process chunk with memory optimization
                chunk_results = self._batch_embed(chunk_texts)
                results.extend(chunk_results)
                
                # Force memory reclamation after each chunk
                if self.memory_manager is not None:
                    self.memory_manager.reclaim_memory(force=True)
                    # Force garbage collection
                    import gc
                    gc.collect()
                    if self.device == 'cuda':
                        torch.cuda.empty_cache()
            
            return results
        
        # For regular sized batches, use standard batch embedding
        return self._batch_embed(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text.
        
        Parameters
        ----------
        text : str
            Text to embed
            
        Returns
        -------
        List[float]
            Embedding vector
            
        Notes
        -----
        Query embedding uses memory optimization when enabled, which is particularly
        useful when working with long query texts or when multiple queries are processed
        in parallel. Memory optimization helps maintain system stability under high load.
        """
        # If caching is enabled, use the cache
        if self.enable_caching:
            return self.embeddings_cache.embed_query(text)
        
        # Add financial context if enabled
        text = self._add_financial_context(text)
        
        # Handle TensorRT model specially
        if self.use_tensorrt and hasattr(self.model, 'embed_query'):
            return self.model.embed_query(text)
        
        # Use memory optimization if enabled and on CUDA
        if self.enable_memory_optimization and self.device == 'cuda' and self.memory_manager is not None:
            try:
                with with_memory_optimization(self.memory_manager):
                    embedding = self.model.encode(
                        [text],
                        show_progress_bar=False,
                        normalize_embeddings=self.normalize_embeddings,
                        convert_to_numpy=True
                    )
                    return embedding[0].tolist()
            except Exception as e:
                logger.warning(f"Memory-optimized query embedding failed: {str(e)}")
                # Continue with standard approach as fallback
        
        # Standard approach
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        
        return embedding[0].tolist()
    
    def clear_gpu_memory(self) -> None:
        """
        Clear GPU memory to free up resources.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
            This method has no return value.
            
        Notes
        -----
        This method provides different memory clearing strategies based on whether
        memory optimization is enabled. With memory optimization, it uses the memory
        manager for a more thorough cleanup including active garbage collection.
        """
        if self.device == 'cuda':
            if self.enable_memory_optimization and self.memory_manager is not None:
                # Use memory manager for thorough cleanup
                self.memory_manager.reclaim_memory(force=True)
                logger.info("GPU memory cleared using memory manager")
            else:
                # Standard cleanup
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared")
            
            # Report memory status if available
            try:
                import gc
                gc.collect()  # Run garbage collection first
                
                if torch.cuda.is_available():
                    # Get current memory usage
                    allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
                    reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
                    
                    logger.info(f"GPU memory after clearing - Allocated: {allocated:.1f} MB, Reserved: {reserved:.1f} MB")
            except Exception as e:
                logger.debug(f"Error reporting memory status: {str(e)}")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        int
            Embedding dimension
            
        Notes
        -----
        For FinE5 models, this is typically 384 dimensions for the small model
        and 768 dimensions for the base model. This information is useful when
        designing vector databases and similarity search applications.
        """
        return self.embedding_dim
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing memory statistics
            
        Notes
        -----
        This method provides detailed GPU memory statistics when memory optimization
        is enabled. This information is useful for monitoring resource usage and
        tuning performance in production environments.
        """
        stats = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "memory_optimization_enabled": self.enable_memory_optimization,
        }
        
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                # Get basic CUDA memory stats
                stats.update({
                    "cuda_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                    "cuda_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                    "cuda_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
                    "cuda_device_name": torch.cuda.get_device_name(0),
                })
                
                # Get memory manager stats if available
                if self.enable_memory_optimization and self.memory_manager is not None:
                    manager_stats = self.memory_manager.get_stats()
                    stats.update({
                        "memory_manager": manager_stats
                    })
            except Exception as e:
                stats["error"] = str(e)
        
        return stats


class FinE5TensorRTEmbeddings(Embeddings):
    """
    TensorRT-accelerated Fin-E5 financial embeddings for high-performance applications.
    
    This class provides GPU-accelerated financial embeddings using TensorRT, optimized
    specifically for FinMTEB/Fin-E5 models with financial data.
    
    Example:
        ```python
        from langchain_hana.financial.fin_e5_embeddings import FinE5TensorRTEmbeddings
        
        # Create GPU-accelerated Fin-E5 embeddings
        embeddings = FinE5TensorRTEmbeddings(
            model_type="high_quality",
            precision="fp16",
            multi_gpu=True
        )
        
        # Embed financial documents
        embeddings.embed_documents(["Q1 revenue increased by 12% YoY"])
        ```
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_type: str = "default",
        precision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        multi_gpu: bool = False,
        max_batch_size: int = 32,
        force_engine_rebuild: bool = False,
        enable_tensor_cores: bool = True,
        add_financial_prefix: bool = False,
        financial_prefix_type: str = "general",
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        cache_max_size: int = 10000,
        cache_persist_path: Optional[str] = None,
    ):
        """
        Initialize TensorRT-accelerated Fin-E5 embeddings.
        
        Parameters
        ----------
        model_name : str, optional
            Specific model name to use (overrides model_type if provided)
        model_type : str, default='default'
            Type of financial model to use
        precision : str, optional
            Precision to use for TensorRT optimization ("fp32", "fp16", "int8")
        cache_dir : str, optional
            Directory to cache optimized TensorRT engines
        multi_gpu : bool, default=False
            Whether to use multiple GPUs if available
        max_batch_size : int, default=32
            Maximum batch size for embedding generation
        force_engine_rebuild : bool, default=False
            If True, forces rebuilding the TensorRT engine
        enable_tensor_cores : bool, default=True
            Whether to enable Tensor Core optimizations
        add_financial_prefix : bool, default=False
            Whether to add a prefix to improve financial context
        financial_prefix_type : str, default='general'
            Type of financial prefix to add
        enable_caching : bool, default=True
            Whether to enable caching for improved performance
        cache_ttl : int, default=3600
            Time-to-live for cache entries in seconds
        cache_max_size : int, default=10000
            Maximum number of entries in the cache
        cache_persist_path : str, optional
            Path to persist the cache (None for in-memory only)
            
        Notes
        -----
        TensorRT acceleration provides significant performance improvements for
        embedding generation, especially for large batches. This implementation
        supports INT8 quantization with specialized financial calibration datasets
        for optimal accuracy.
        
        When using precision="int8", the model automatically creates a financial
        domain-specific calibration dataset to ensure accurate quantization.
        """
        # Set model name based on model_type if not explicitly provided
        if model_name is None:
            if model_type not in FINANCIAL_EMBEDDING_MODELS:
                raise ValueError(
                    f"Invalid model_type: {model_type}. "
                    f"Available types: {', '.join(FINANCIAL_EMBEDDING_MODELS.keys())}"
                )
            model_name = FINANCIAL_EMBEDDING_MODELS[model_type]
        
        # Store configuration
        self.model_name = model_name
        self.model_type = model_type
        self.precision = precision
        self.cache_dir = cache_dir
        self.multi_gpu = multi_gpu
        self.max_batch_size = max_batch_size
        self.force_engine_rebuild = force_engine_rebuild
        self.enable_tensor_cores = enable_tensor_cores
        
        # Financial context enhancement
        self.add_financial_prefix = add_financial_prefix
        if financial_prefix_type not in FINANCIAL_PREFIXES:
            raise ValueError(
                f"Invalid financial_prefix_type: {financial_prefix_type}. "
                f"Available types: {', '.join(FINANCIAL_PREFIXES.keys())}"
            )
        self.financial_prefix = FINANCIAL_PREFIXES[financial_prefix_type]
        
        # Create specialized financial calibration data if needed
        self.calibration_data = None
        if precision == "int8":
            from langchain_hana.gpu.calibration_datasets import create_enhanced_calibration_dataset
            self.calibration_data = create_enhanced_calibration_dataset(
                domains=["financial"],
                count=100,
                custom_file_path=None
            )
        
        # Initialize TensorRT embeddings
        logger.info(f"Initializing TensorRT-accelerated Fin-E5 embeddings with model: {model_name}")
        
        try:
            # Create TensorRT accelerated embeddings
            self.tensorrt_embeddings = HanaTensorRTEmbeddings(
                model_name=self.model_name,
                precision=self.precision,
                cache_dir=self.cache_dir,
                multi_gpu=self.multi_gpu,
                max_batch_size=self.max_batch_size,
                force_engine_rebuild=self.force_engine_rebuild,
                enable_tensor_cores=self.enable_tensor_cores,
                calibration_domain="financial",
                calibration_data=self.calibration_data,
                enable_profiling=True
            )
            
            # Initialize cache if enabled
            self.enable_caching = enable_caching
            if enable_caching:
                self.embeddings_cache = FinancialEmbeddingCache(
                    base_embeddings=self,
                    ttl_seconds=cache_ttl,
                    max_size=cache_max_size,
                    persist_path=cache_persist_path,
                    model_name=model_name.split('/')[-1]  # Use model name for cache identification
                )
                logger.info("Financial embedding cache initialized")
            
            logger.info(f"TensorRT-accelerated Fin-E5 embeddings initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT-accelerated Fin-E5 embeddings: {str(e)}")
            raise RuntimeError(f"TensorRT initialization failed: {str(e)}")
    
    def _add_financial_context(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Add financial context prefix to improve embedding quality.
        
        Args:
            texts: Text or list of texts to add context to
            
        Returns:
            Text or list of texts with context prefix added
        """
        if not self.add_financial_prefix:
            return texts
        
        if isinstance(texts, str):
            return f"{self.financial_prefix}{texts}"
        
        return [f"{self.financial_prefix}{text}" for text in texts]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using TensorRT acceleration.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # If caching is enabled, use the cache
        if self.enable_caching:
            return self.embeddings_cache.embed_documents(texts)
        
        # Add financial context if enabled
        texts = self._add_financial_context(texts)
        
        # Use TensorRT for embedding
        return self.tensorrt_embeddings.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text using TensorRT acceleration.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # If caching is enabled, use the cache
        if self.enable_caching:
            return self.embeddings_cache.embed_query(text)
        
        # Add financial context if enabled
        text = self._add_financial_context(text)
        
        # Use TensorRT for embedding
        return self.tensorrt_embeddings.embed_query(text)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from the TensorRT embeddings.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = self.tensorrt_embeddings.get_performance_stats()
        if stats:
            return {
                "total_documents": stats.total_documents,
                "total_time_seconds": stats.total_time_seconds,
                "documents_per_second": stats.documents_per_second,
                "avg_document_time_ms": stats.avg_document_time_ms,
                "peak_memory_mb": stats.peak_memory_mb,
                "batch_size": stats.batch_size,
                "precision": stats.precision,
                "gpu_name": stats.gpu_name,
                "gpu_count": stats.gpu_count
            }
        return {}
    
    def benchmark(self, batch_sizes: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Run a benchmark to measure embedding performance.
        
        Args:
            batch_sizes: List of batch sizes to benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        return self.tensorrt_embeddings.benchmark(batch_sizes=batch_sizes)


# Factory function for creating financial embeddings
def create_financial_embeddings(
    model_type: str = "default",
    use_gpu: bool = True,
    use_tensorrt: bool = False,
    add_financial_prefix: bool = True,
    financial_prefix_type: str = "general",
    enable_caching: bool = True,
) -> Embeddings:
    """
    Create financial domain-specific embeddings with recommended settings.
    
    Parameters
    ----------
    model_type : str, default='default'
        Type of financial model to use
    use_gpu : bool, default=True
        Whether to use GPU acceleration if available
    use_tensorrt : bool, default=False
        Whether to use TensorRT acceleration (requires GPU)
    add_financial_prefix : bool, default=True
        Whether to add financial context prefix
    financial_prefix_type : str, default='general'
        Type of financial prefix to add
    enable_caching : bool, default=True
        Whether to enable caching
        
    Returns
    -------
    Embeddings
        Financial domain-specific embeddings instance
        
    Examples
    --------
    >>> from langchain_hana.financial import create_financial_embeddings
    >>> # Create standard Fin-E5 embeddings
    >>> embeddings = create_financial_embeddings(model_type="default")
    >>> 
    >>> # Create TensorRT-accelerated embeddings
    >>> embeddings = create_financial_embeddings(
    ...     model_type="high_quality",
    ...     use_gpu=True,
    ...     use_tensorrt=True
    ... )
    """
    # Check if GPU is available
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    # Use TensorRT if requested and GPU is available
    if use_tensorrt and device == 'cuda':
        try:
            # First check if TensorRT is available with diagnostics
            tensorrt_available, _, diagnostics = try_import_tensorrt()
            if not tensorrt_available:
                logger.warning(
                    f"TensorRT not available: {diagnostics.get_summary() if diagnostics else 'Unknown error'}"
                    f"\nFalling back to standard model."
                )
            else:
                # Run environment diagnostics
                env_diagnostics = TensorRTDiagnostics.run_diagnostics()
                logger.info(f"TensorRT environment: {env_diagnostics.get_summary()}")
                
                # Create TensorRT embeddings with enhanced diagnostics
                return FinE5TensorRTEmbeddings(
                    model_type=model_type,
                    precision="fp16",  # Use FP16 precision by default for best performance
                    multi_gpu=True,    # Use multiple GPUs if available
                    add_financial_prefix=add_financial_prefix,
                    financial_prefix_type=financial_prefix_type,
                    enable_caching=enable_caching
                )
        except Exception as e:
            # Run diagnostics on the error
            diagnostics = TensorRTDiagnostics.diagnose_error(e)
            
            # Log detailed error information and recommendations
            logger.warning(f"Failed to initialize TensorRT: {diagnostics.error_type} - {diagnostics.error_message}")
            logger.warning(f"TensorRT diagnostics: {diagnostics.get_summary()}")
            
            if diagnostics.recommendations:
                logger.info("Recommendations for TensorRT initialization:")
                for i, recommendation in enumerate(diagnostics.recommendations, 1):
                    logger.info(f"{i}. {recommendation}")
            
            logger.info("Falling back to standard model")
    
    # Use standard Fin-E5 embeddings
    return FinE5Embeddings(
        model_type=model_type,
        device=device,
        use_fp16=device == 'cuda',
        enable_caching=enable_caching,
        normalize_embeddings=True,
        add_financial_prefix=add_financial_prefix,
        financial_prefix_type=financial_prefix_type
    )