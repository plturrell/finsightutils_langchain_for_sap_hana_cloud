"""Multi-GPU accelerated embedding services."""

import logging
import time
from typing import Dict, List, Optional, Union, Any, Callable

import numpy as np
from langchain_core.embeddings import Embeddings

import gpu_utils
from multi_gpu import get_gpu_manager

logger = logging.getLogger(__name__)


class MultiGPUEmbeddings(Embeddings):
    """
    Multi-GPU accelerated embeddings using sentence-transformers.
    
    This class distributes embedding generation across multiple GPUs
    for high-performance parallel processing.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the multi-GPU accelerated embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
            batch_size: Batch size for processing. If None, determines automatically.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.gpu_manager = get_gpu_manager()
        self.models = {}
        
        # Validate GPU support
        if not gpu_utils.is_torch_available():
            logger.warning("PyTorch CUDA not available. Falling back to CPU.")
        
        # Initialize models for each GPU
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize sentence-transformer models on each GPU."""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Initialize on CPU as fallback
            logger.info(f"Loading sentence-transformers model {self.model_name} on CPU")
            self.models["cpu"] = SentenceTransformer(self.model_name, device="cpu")
            
            # Initialize on each available GPU
            for device_name in self.gpu_manager.get_available_devices():
                try:
                    logger.info(f"Loading sentence-transformers model {self.model_name} on {device_name}")
                    device_id = self.gpu_manager.get_device_info(device_name)["id"]
                    self.models[device_name] = SentenceTransformer(
                        self.model_name, device=f"cuda:{device_id}"
                    )
                except Exception as e:
                    logger.error(f"Error loading model on {device_name}: {str(e)}")
        
        except ImportError:
            logger.error("sentence-transformers not installed. Please install it with pip.")
            raise ImportError(
                "sentence-transformers not installed. "
                "Please install it with pip: pip install sentence-transformers"
            )
    
    def _process_batch_on_device(
        self, texts: List[str], device_name: str
    ) -> np.ndarray:
        """
        Process a batch of texts on a specific device.
        
        Args:
            texts: List of texts to embed.
            device_name: Device to use for embedding.
            
        Returns:
            NumPy array of embeddings.
        """
        model = self.models.get(device_name, self.models["cpu"])
        
        embeddings = model.encode(
            texts,
            batch_size=32,  # Use fixed batch size for device-level processing
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed documents using multiple GPUs in parallel.
        
        Args:
            texts: List of documents to embed.
            
        Returns:
            List of embeddings.
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        # Process texts in parallel across GPUs
        embeddings = self.gpu_manager.process_batch(
            items=texts,
            process_fn=self._process_batch_on_device,
            batch_size=self.batch_size,
        )
        
        duration = time.time() - start_time
        logger.debug(f"Embedded {len(texts)} documents in {duration:.2f} seconds")
        
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query.
        
        Args:
            text: Query to embed.
            
        Returns:
            Embedding of the query.
        """
        embedding = self.embed_documents([text])[0]
        return embedding


class MultiGPUHybridEmbeddings(Embeddings):
    """
    Hybrid embeddings that can switch between SAP HANA internal embeddings
    and multi-GPU accelerated embeddings.
    
    This class provides multi-GPU acceleration for external embedding generation
    while still supporting SAP HANA's internal embedding capabilities.
    """
    
    def __init__(
        self,
        internal_embedding_model_id: str = "SAP_NEB.20240715",
        external_model_name: str = "all-MiniLM-L6-v2",
        use_internal: bool = True,
        batch_size: Optional[int] = None,
    ):
        """
        Initialize the hybrid embeddings.
        
        Args:
            internal_embedding_model_id: ID of the SAP HANA internal embedding model.
            external_model_name: Name of the external sentence-transformers model.
            use_internal: Whether to use internal embeddings by default.
            batch_size: Batch size for processing external embeddings.
        """
        self.use_internal = use_internal
        self.internal_embedding_model_id = internal_embedding_model_id
        self.batch_size = batch_size
        
        # Initialize internal embeddings
        from langchain_hana import HanaInternalEmbeddings
        self.internal_embeddings = HanaInternalEmbeddings(
            internal_embedding_model_id=internal_embedding_model_id
        )
        
        # Initialize external GPU-accelerated embeddings
        self.external_embeddings = MultiGPUEmbeddings(
            model_name=external_model_name,
            batch_size=batch_size,
        )
        
        logger.info(
            f"Multi-GPU hybrid embeddings initialized: "
            f"Using {'internal' if use_internal else 'external'} embeddings by default"
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
        logger.info(f"Embedding mode set to: {'internal' if use_internal else 'external'}")