"""
Production-grade financial embedding models for SAP HANA Cloud integration.

This module provides enterprise-ready embedding models specifically optimized for
financial text and seamless integration with SAP HANA Cloud's vector capabilities.
"""

import os
import time
import json
import logging
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable

# Core LangChain interfaces
from langchain_core.embeddings import Embeddings

# Initialize module logger
logger = logging.getLogger(__name__)


class FinancialEmbeddings(Embeddings):
    """
    Production-grade financial domain embedding model.
    
    This embedding model is specifically designed for financial text processing
    in enterprise environments, with optimizations for performance, stability,
    and integration with SAP HANA Cloud.
    """
    
    def __init__(
        self,
        model_name: str = "FinMTEB/Fin-E5-small",
        device: Optional[str] = None,
        max_seq_length: int = 512,
        batch_size: int = 32,
        cache_folder: Optional[str] = None,
        normalize_embeddings: bool = True,
        use_fp16: bool = True,
        use_auth_token: Optional[str] = None,
        context_aware: bool = True,
        context_prefix: str = "Financial context: ",
        model_kwargs: Optional[Dict[str, Any]] = None,
        encode_kwargs: Optional[Dict[str, Any]] = None,
        enterprise_mode: bool = True,
    ):
        """
        Initialize the production-grade financial embedding model.
        
        Args:
            model_name: HuggingFace model name for financial embeddings
            device: Device to use (cuda, cpu, or None for auto-detection)
            max_seq_length: Maximum sequence length for the model
            batch_size: Batch size for document embedding
            cache_folder: Folder to cache downloaded models
            normalize_embeddings: Whether to normalize embeddings to unit length
            use_fp16: Whether to use FP16 for faster inference (if supported)
            use_auth_token: HuggingFace auth token for private models
            context_aware: Whether to add financial context to queries
            context_prefix: Prefix to add for financial context
            model_kwargs: Additional arguments for model initialization
            encode_kwargs: Additional arguments for the encode method
            enterprise_mode: Enable enterprise features (thread safety, monitoring)
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.cache_folder = cache_folder
        self.normalize_embeddings = normalize_embeddings
        self.use_fp16 = use_fp16
        self.use_auth_token = use_auth_token
        self.context_aware = context_aware
        self.context_prefix = context_prefix
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.enterprise_mode = enterprise_mode
        
        # Auto-detect device if not specified
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Track embedding count for monitoring
        self._embedding_count = 0
        self._total_tokens = 0
        self._total_embedding_time = 0
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model with enterprise optimizations."""
        try:
            # Import here to avoid dependency issues
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Initializing financial embedding model: {self.model_name}")
            start_time = time.time()
            
            # Load model with optimizations
            self.model = SentenceTransformer(
                model_name_or_path=self.model_name,
                device=self.device,
                cache_folder=self.cache_folder,
                use_auth_token=self.use_auth_token,
                **self.model_kwargs
            )
            
            # Set maximum sequence length
            self.model.max_seq_length = self.max_seq_length
            
            # Enable mixed precision if requested and supported
            if self.use_fp16 and self.device == 'cuda' and torch.cuda.is_available():
                self.model.half()
                logger.info("Enabled FP16 precision for faster inference")
            
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
            
            # Get embedding dimension
            self.embedding_dimension = self.model.get_sentence_embedding_dimension()
            
            # Log initialization metrics
            load_time = time.time() - start_time
            logger.info(
                f"Financial model initialized in {load_time:.2f}s "
                f"(dim={self.embedding_dimension}, device={self.device})"
            )
            
            # Add thread locking if in enterprise mode
            if self.enterprise_mode:
                import threading
                self._model_lock = threading.RLock()
                logger.info("Thread safety enabled for enterprise mode")
        
        except ImportError:
            raise ImportError(
                "The sentence-transformers package is required for financial embeddings. "
                "Please install it with `pip install sentence-transformers`."
            )
        except Exception as e:
            logger.error(f"Failed to initialize financial embedding model: {str(e)}")
            raise RuntimeError(f"Model initialization failed: {str(e)}")
    
    def _add_financial_context(self, text: str) -> str:
        """Add financial context to improve embedding quality."""
        if not self.context_aware:
            return text
        
        # Skip adding context if it's already present
        if text.startswith(self.context_prefix):
            return text
        
        return f"{self.context_prefix}{text}"
    
    def _embed_with_error_handling(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with comprehensive error handling and recovery."""
        if not texts:
            return []
        
        # Apply financial context enhancement
        if self.context_aware:
            texts = [self._add_financial_context(text) for text in texts]
        
        # Track token count (approximate)
        if self.enterprise_mode:
            tokens = sum(len(text.split()) * 1.3 for text in texts)
            self._total_tokens += tokens
        
        # Acquire thread lock in enterprise mode
        if self.enterprise_mode and hasattr(self, '_model_lock'):
            self._model_lock.acquire()
        
        try:
            start_time = time.time()
            
            # Core embedding generation
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                **self.encode_kwargs
            )
            
            # Track metrics in enterprise mode
            if self.enterprise_mode:
                embed_time = time.time() - start_time
                self._embedding_count += len(texts)
                self._total_embedding_time += embed_time
                
                # Log performance metrics periodically
                if self._embedding_count % 1000 < len(texts):
                    avg_time = self._total_embedding_time / self._embedding_count
                    logger.info(
                        f"Performance metrics: {self._embedding_count} embeddings generated, "
                        f"avg time: {avg_time*1000:.1f}ms per embedding, "
                        f"total tokens: {self._total_tokens:,}"
                    )
            
            return embeddings.tolist()
            
        except torch.cuda.OutOfMemoryError:
            # Handle GPU OOM with automatic fallback to CPU
            logger.warning("CUDA out of memory, falling back to CPU")
            original_device = self.device
            
            try:
                # Move model to CPU
                self.model.to('cpu')
                self.device = 'cpu'
                
                # Retry with smaller batch size on CPU
                reduced_batch = max(1, self.batch_size // 4)
                embeddings = self.model.encode(
                    texts,
                    batch_size=reduced_batch,
                    show_progress_bar=False,
                    normalize_embeddings=self.normalize_embeddings,
                    convert_to_numpy=True
                )
                
                # Try to move back to original device
                if original_device == 'cuda' and torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                        self.model.to('cuda')
                        self.device = 'cuda'
                    except Exception as e:
                        logger.warning(f"Failed to move back to CUDA: {str(e)}")
                
                return embeddings.tolist()
                
            except Exception as e:
                logger.error(f"CPU fallback failed: {str(e)}")
                raise RuntimeError(f"Embedding generation failed: {str(e)}")
        
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            raise RuntimeError(f"Embedding generation failed: {str(e)}")
            
        finally:
            # Release thread lock in enterprise mode
            if self.enterprise_mode and hasattr(self, '_model_lock'):
                self._model_lock.release()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self._embed_with_error_handling(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        embeddings = self._embed_with_error_handling([text])
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of generated embeddings."""
        return self.embedding_dimension
    
    def clear_gpu_memory(self) -> None:
        """Clear GPU memory to free up resources."""
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring."""
        if not self.enterprise_mode:
            return {}
        
        return {
            "embedding_count": self._embedding_count,
            "total_tokens": self._total_tokens,
            "total_embedding_time": self._total_embedding_time,
            "avg_time_per_embedding": (
                self._total_embedding_time / self._embedding_count 
                if self._embedding_count > 0 else 0
            ),
            "embeddings_per_second": (
                self._embedding_count / self._total_embedding_time 
                if self._total_embedding_time > 0 else 0
            ),
            "model_name": self.model_name,
            "device": self.device,
            "precision": "fp16" if self.use_fp16 and self.device == 'cuda' else "fp32",
            "dimension": self.embedding_dimension,
        }


# Factory function to create production-ready embeddings
def create_production_financial_embeddings(
    model_name: Optional[str] = None,
    quality_tier: str = "balanced",
    memory_tier: str = "auto",
    enterprise_mode: bool = True,
    **kwargs
) -> Embeddings:
    """
    Create production-ready financial embeddings with optimal configuration.
    
    Args:
        model_name: Custom model name (overrides quality_tier if provided)
        quality_tier: Quality tier ('high', 'balanced', 'efficient')
        memory_tier: Memory tier ('high', 'medium', 'low', 'auto')
        enterprise_mode: Enable enterprise features
        **kwargs: Additional arguments for FinancialEmbeddings
        
    Returns:
        Production-ready FinancialEmbeddings instance
    """
    # Define tier-based model mapping
    model_tiers = {
        "high": "FinMTEB/Fin-E5",  # High quality, high resource usage
        "balanced": "FinMTEB/Fin-E5-small",  # Good balance of quality and resources
        "efficient": "FinLang/investopedia_embedding",  # Lower resource usage
    }
    
    # Select model based on tier if not explicitly provided
    if model_name is None:
        if quality_tier not in model_tiers:
            raise ValueError(
                f"Invalid quality_tier: {quality_tier}. "
                f"Must be one of: {', '.join(model_tiers.keys())}"
            )
        model_name = model_tiers[quality_tier]
    
    # Configure memory settings
    use_fp16 = True
    batch_size = 32
    
    # Auto-detect memory tier if set to auto
    if memory_tier == "auto":
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 16:
                memory_tier = "high"
            elif gpu_memory >= 8:
                memory_tier = "medium"
            else:
                memory_tier = "low"
        else:
            memory_tier = "low"
    
    # Apply memory tier settings
    if memory_tier == "high":
        batch_size = 64
    elif memory_tier == "medium":
        batch_size = 32
    elif memory_tier == "low":
        batch_size = 16
        # If on low memory, adjust settings
        if "FinMTEB/Fin-E5" in model_name and model_name != "FinMTEB/Fin-E5-small":
            logger.warning(
                f"Selected model {model_name} may require significant resources. "
                f"Consider using 'FinMTEB/Fin-E5-small' for better performance on limited hardware."
            )
    
    # Create and return embeddings
    return FinancialEmbeddings(
        model_name=model_name,
        batch_size=batch_size,
        use_fp16=use_fp16,
        enterprise_mode=enterprise_mode,
        **kwargs
    )