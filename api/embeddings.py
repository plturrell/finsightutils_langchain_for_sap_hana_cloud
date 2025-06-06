"""GPU-accelerated embedding services."""

import logging
from typing import Dict, List, Optional, Union, Any

import numpy as np
from langchain_core.embeddings import Embeddings

import gpu_utils

logger = logging.getLogger(__name__)


class GPUAcceleratedEmbeddings(Embeddings):
    """
    GPU-accelerated embeddings using sentence-transformers.
    
    This class provides GPU acceleration for embedding generation
    when a compatible NVIDIA GPU is available.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the GPU-accelerated embeddings.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
            device: Device to use for computations ('cuda', 'cpu', or None for auto).
            batch_size: Batch size for processing.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        
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
            
            if self.device == "cuda":
                logger.info("Using GPU acceleration for embeddings")
            else:
                logger.info("Using CPU for embeddings")
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


class GPUHybridEmbeddings(Embeddings):
    """
    Hybrid embeddings that can switch between SAP HANA internal embeddings
    and GPU-accelerated embeddings.
    
    This class provides GPU acceleration for external embedding generation
    while still supporting SAP HANA's internal embedding capabilities.
    """
    
    def __init__(
        self,
        internal_embedding_model_id: str = "SAP_NEB.20240715",
        external_model_name: str = "all-MiniLM-L6-v2",
        use_internal: bool = True,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize the hybrid embeddings.
        
        Args:
            internal_embedding_model_id: ID of the SAP HANA internal embedding model.
            external_model_name: Name of the external sentence-transformers model.
            use_internal: Whether to use internal embeddings by default.
            device: Device to use for external embeddings.
            batch_size: Batch size for processing external embeddings.
        """
        self.use_internal = use_internal
        self.internal_embedding_model_id = internal_embedding_model_id
        self.external_model_name = external_model_name
        self.device = device
        self.batch_size = batch_size
        
        # Initialize internal embeddings
        from langchain_hana import HanaInternalEmbeddings
        self.internal_embeddings = HanaInternalEmbeddings(
            internal_embedding_model_id=internal_embedding_model_id
        )
        
        # Lazy initialization of external embeddings
        self._external_embeddings = None
        
        logger.info(
            f"Hybrid embeddings initialized: "
            f"Using {'internal' if use_internal else 'external'} embeddings by default"
        )
    
    @property
    def external_embeddings(self):
        """Lazy initialization of external embeddings to save resources when only using internal embeddings."""
        if self._external_embeddings is None:
            logger.info(f"Initializing external embeddings model {self.external_model_name}")
            self._external_embeddings = GPUAcceleratedEmbeddings(
                model_name=self.external_model_name,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._external_embeddings
    
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