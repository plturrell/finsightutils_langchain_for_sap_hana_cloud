"""
Financial domain-specific embedding models for SAP HANA Cloud integration.

This module provides specialized embedding models for financial domain text,
optimized for GPU acceleration and seamless integration with SAP HANA Cloud's
vector capabilities.

Key features:
- Domain-specific models for financial text understanding
- GPU acceleration with mixed precision and dynamic batching
- Error handling with automatic fallback mechanisms
- Efficient memory management for large-scale processing
- Semantic caching for improved performance on similar queries
"""

import logging
import time
import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Iterator

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from langchain_hana.embeddings import HanaEmbeddingsCache

logger = logging.getLogger(__name__)

# Model IDs for financial domain embedding models
FINANCIAL_EMBEDDING_MODELS = {
    "default": "FinMTEB/Fin-E5-small",  # Best balance of quality and performance
    "high_quality": "FinMTEB/Fin-E5",  # Highest quality, requires more resources
    "efficient": "FinLang/investopedia_embedding",  # Most efficient, good for limited resources
    "tone": "yiyanghkust/finbert-tone",  # Specialized for sentiment/tone analysis
    "financial_bert": "ProsusAI/finbert",  # Good for SEC filings, earnings reports
    "finance_base": "baconnier/Finance_embedding_large_en-V0.1"  # General finance embeddings
}


class FinancialEmbeddings(Embeddings):
    """
    Financial domain-specific embedding model using Hugging Face Sentence Transformers.
    
    This class provides optimized embeddings for financial text, with GPU acceleration
    when available and automatic fallback to CPU when needed. It includes memory
    optimization techniques such as mixed precision inference and dynamic batching.
    
    Examples:
        ```python
        from langchain_hana.finance_embeddings import FinancialEmbeddings
        
        # Use default financial model
        embeddings = FinancialEmbeddings()
        
        # Use specific financial model with GPU acceleration
        embeddings = FinancialEmbeddings(
            model_name="FinMTEB/Fin-E5", 
            device="cuda",
            use_fp16=True
        )
        
        # Embed documents
        vectors = embeddings.embed_documents(["Financial text to embed"])
        
        # Embed query
        query_vector = embeddings.embed_query("Financial query")
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
        financial_prefix: str = "Financial context: ",
        enable_caching: bool = True,
        cache_ttl: int = 3600,  # 1 hour cache TTL
        cache_max_size: int = 10000,
        cache_persist_path: Optional[str] = None,
    ):
        """
        Initialize the financial embedding model.
        
        Args:
            model_name: Specific model name to use (overrides model_type if provided)
            model_type: Type of financial model to use ('default', 'high_quality', 
                       'efficient', 'tone', 'financial_bert', 'finance_base')
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
            use_fp16: Whether to use half-precision (FP16) for faster inference
            normalize_embeddings: Whether to L2-normalize the embeddings
            max_seq_length: Maximum sequence length for the model
            batch_size: Batch size for document embedding
            add_financial_prefix: Whether to add a prefix to improve financial context
            financial_prefix: Prefix to add if add_financial_prefix is True
            enable_caching: Whether to enable caching for improved performance
            cache_ttl: Time-to-live for cache entries in seconds
            cache_max_size: Maximum number of entries in the cache
            cache_persist_path: Path to persist the cache (None for in-memory only)
        """
        # Set model name based on model_type if not explicitly provided
        if model_name is None:
            if model_type not in FINANCIAL_EMBEDDING_MODELS:
                raise ValueError(
                    f"Invalid model_type: {model_type}. "
                    f"Available types: {', '.join(FINANCIAL_EMBEDDING_MODELS.keys())}"
                )
            model_name = FINANCIAL_EMBEDDING_MODELS[model_type]
        
        # Determine device (use CUDA if available)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_fp16 = use_fp16 and self.device == 'cuda'
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.add_financial_prefix = add_financial_prefix
        self.financial_prefix = financial_prefix
        
        # Log configuration
        logger.info(f"Initializing FinancialEmbeddings with model: {model_name}")
        logger.info(f"Device: {self.device}, FP16: {self.use_fp16}")
        
        # Initialize model
        self._initialize_model(model_name)
        
        # Initialize cache if enabled
        self.enable_caching = enable_caching
        if enable_caching:
            self.embeddings_cache = HanaEmbeddingsCache(
                base_embeddings=self,
                ttl_seconds=cache_ttl,
                max_size=cache_max_size,
                persist_path=cache_persist_path,
            )
            logger.info("Embedding cache initialized")
    
    def _initialize_model(self, model_name: str) -> None:
        """
        Initialize the model with optimizations.
        
        Args:
            model_name: The name of the model to initialize
        """
        try:
            # Initialize the model
            logger.info(f"Loading model {model_name} on {self.device}")
            start_time = time.time()
            
            # Load the model with optimizations
            self.model = SentenceTransformer(model_name, device=self.device)
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
        Embed a batch of texts with proper error handling and fallback.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Add financial context if enabled
        texts = self._add_financial_context(texts)
        
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
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        # If caching is enabled, use the cache
        if self.enable_caching:
            return self.embeddings_cache.embed_documents(texts)
        
        # Process in dynamic batches based on available memory
        return self._batch_embed(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a query text.
        
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
        
        # Embed the query
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
        """
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.info("GPU memory cleared")
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        return self.embedding_dim


class FinancialEmbeddingsPipeline:
    """
    Pipeline for processing financial documents with optimized throughput.
    
    This class provides a pipeline for processing large volumes of financial documents
    with optimized throughput using dynamic batching and parallel processing.
    
    Examples:
        ```python
        from langchain_hana.finance_embeddings import FinancialEmbeddingsPipeline
        
        # Create pipeline
        pipeline = FinancialEmbeddingsPipeline(
            model_name="FinMTEB/Fin-E5",
            batch_size=64,
            max_workers=4
        )
        
        # Process documents
        embedded_docs = pipeline.process_documents(documents)
        
        # Store in SAP HANA
        pipeline.store_in_hana(embedded_docs, connection, table_name="FINANCIAL_DOCS")
        ```
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        model_type: str = "default",
        batch_size: int = 64,
        max_workers: int = 1,
        use_fp16: bool = True,
        max_seq_length: int = 512,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize the financial embeddings pipeline.
        
        Args:
            model_name: Specific model name to use (overrides model_type if provided)
            model_type: Type of financial model to use
            batch_size: Batch size for document processing
            max_workers: Maximum number of workers for parallel processing
            use_fp16: Whether to use half-precision (FP16) for faster inference
            max_seq_length: Maximum sequence length for the model
            normalize_embeddings: Whether to L2-normalize the embeddings
        """
        self.embeddings = FinancialEmbeddings(
            model_name=model_name,
            model_type=model_type,
            use_fp16=use_fp16,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            enable_caching=False,  # Disable caching for pipeline
        )
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process_documents(
        self, 
        documents: List[Dict[str, Any]],
        text_key: str = "text",
        embed_key: str = "embedding",
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process a list of documents, adding embeddings.
        
        Args:
            documents: List of document dictionaries
            text_key: Key for the text field in documents
            embed_key: Key to store the embedding in documents
            show_progress: Whether to show progress bar
            
        Returns:
            List of documents with embeddings added
        """
        if not documents:
            return []
        
        # Extract texts
        texts = [doc[text_key] for doc in documents]
        
        # Get embeddings
        if show_progress:
            logger.info(f"Embedding {len(texts)} documents in batches of {self.batch_size}")
            start_time = time.time()
        
        # Process in batches
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
            
            if show_progress and i % (self.batch_size * 10) == 0 and i > 0:
                progress = i / len(texts) * 100
                logger.info(f"Progress: {progress:.1f}% ({i}/{len(texts)} documents)")
        
        if show_progress:
            elapsed = time.time() - start_time
            rate = len(texts) / elapsed
            logger.info(f"Embedded {len(texts)} documents in {elapsed:.1f} seconds ({rate:.1f} docs/sec)")
        
        # Add embeddings to documents
        result = []
        for doc, embedding in zip(documents, embeddings):
            doc_copy = doc.copy()
            doc_copy[embed_key] = embedding
            result.append(doc_copy)
        
        return result
    
    def store_in_hana(
        self,
        documents: List[Dict[str, Any]],
        connection,
        table_name: str,
        text_key: str = "text",
        metadata_key: str = "metadata",
        embed_key: str = "embedding",
        batch_size: int = 100,
    ) -> int:
        """
        Store documents with embeddings in SAP HANA.
        
        Args:
            documents: List of document dictionaries with embeddings
            connection: SAP HANA connection
            table_name: Table name to store documents
            text_key: Key for the text field in documents
            metadata_key: Key for the metadata field in documents
            embed_key: Key for the embedding field in documents
            batch_size: Batch size for database inserts
            
        Returns:
            Number of documents stored
        """
        if not documents:
            return 0
        
        # Sanitize table name
        table_name = table_name.upper()
        
        # Check if table exists
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM SYS.TABLES WHERE TABLE_NAME = ?",
            (table_name,)
        )
        table_exists = cursor.fetchone()[0] > 0
        
        # Get embedding dimension from first document
        embedding_dim = len(documents[0][embed_key])
        
        # Create table if it doesn't exist
        if not table_exists:
            create_table_sql = f"""
            CREATE TABLE "{table_name}" (
                "ID" INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                "TEXT" NVARCHAR(5000),
                "METADATA" NCLOB,
                "EMBEDDING" REAL_VECTOR({embedding_dim})
            )
            """
            cursor.execute(create_table_sql)
            connection.commit()
            logger.info(f"Created table {table_name}")
        
        # Insert documents in batches
        total_inserted = 0
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            # Prepare batch data
            batch_data = []
            for doc in batch:
                text = doc.get(text_key, "")
                metadata = json.dumps(doc.get(metadata_key, {}))
                embedding = doc.get(embed_key, [])
                
                # Convert embedding to string format for HANA
                embedding_str = f"[{','.join(str(float(v)) for v in embedding)}]"
                
                batch_data.append((text, metadata, embedding_str))
            
            # Insert batch
            cursor.executemany(
                f'INSERT INTO "{table_name}" ("TEXT", "METADATA", "EMBEDDING") VALUES (?, ?, TO_REAL_VECTOR(?))',
                batch_data
            )
            connection.commit()
            
            total_inserted += len(batch)
            if i % (batch_size * 5) == 0 and i > 0:
                logger.info(f"Inserted {total_inserted}/{len(documents)} documents")
        
        cursor.close()
        logger.info(f"Successfully stored {total_inserted} documents in {table_name}")
        return total_inserted


# Factory function for creating financial embeddings
def create_financial_embeddings(
    model_type: str = "default",
    use_gpu: bool = True,
    use_fp16: bool = True,
    enable_caching: bool = True,
) -> Embeddings:
    """
    Create financial domain-specific embeddings with recommended settings.
    
    Args:
        model_type: Type of financial model to use
        use_gpu: Whether to use GPU acceleration if available
        use_fp16: Whether to use half-precision (FP16) for faster inference
        enable_caching: Whether to enable caching
        
    Returns:
        Financial domain-specific embeddings instance
    """
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    
    return FinancialEmbeddings(
        model_type=model_type,
        device=device,
        use_fp16=use_fp16 and device == 'cuda',
        enable_caching=enable_caching,
        normalize_embeddings=True,
        add_financial_prefix=True,  # Add financial context for better results
    )