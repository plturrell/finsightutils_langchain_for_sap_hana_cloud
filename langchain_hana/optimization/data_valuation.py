"""
Data Valuation using Reinforcement Learning (DVRL) integration for SAP HANA Cloud.

This module provides document importance scoring and data quality assessment for vector stores.
It helps identify the most valuable documents for retrieval and enables optimized storage.
"""

import logging
import os
import json
import numpy as np
from typing import Dict, List, Optional, Union, Any, Callable

# Import LangChain core components
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

# Set up logging
logger = logging.getLogger(__name__)

# Conditionally import DVRL components - allow for fallback if not available
try:
    import tensorflow as tf
    from dvrl import dvrl
    from dvrl.data_valuation import DataValuation
    HAS_DVRL = True
except ImportError:
    logger.warning("DVRL dependencies not found. Using fallback data valuation mechanism.")
    HAS_DVRL = False


class DVRLDataValuation:
    """
    Data Valuation using Reinforcement Learning for document importance scoring.
    
    This class evaluates document importance for vector search applications,
    helping identify which documents are most valuable for retrieval quality.
    
    Attributes:
        embedding_dimension: Dimension of the document embeddings
        data_valuation_model: DVRL model or fallback mechanism
        value_threshold: Threshold for considering documents valuable
        scoring_cache: Cache of document scores for efficiency
    """
    
    def __init__(
        self,
        embedding_dimension: int = 768,
        value_threshold: float = 0.5,
        cache_file: Optional[str] = None,
        scoring_model: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize the DVRL Data Valuation component.
        
        Args:
            embedding_dimension: Dimension of document embeddings
            value_threshold: Threshold for considering documents valuable (0.0-1.0)
            cache_file: Optional path to cache computed values
            scoring_model: Optional path to pre-trained scoring model
            use_gpu: Whether to use GPU for DVRL computation
        """
        self.embedding_dimension = embedding_dimension
        self.value_threshold = value_threshold
        self.scoring_cache = {}
        self.cache_file = cache_file
        
        # Load cache if file exists
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.scoring_cache = json.load(f)
                logger.info(f"Loaded {len(self.scoring_cache)} cached document values")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file: {e}")
        
        # Initialize DVRL if available
        if HAS_DVRL:
            try:
                # Configure device strategy for TensorFlow
                if use_gpu and tf.config.list_physical_devices('GPU'):
                    # Use GPU if available and requested
                    device_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                    logger.info("Using GPU for DVRL computation")
                else:
                    # Fall back to CPU
                    device_strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
                    logger.info("Using CPU for DVRL computation")
                
                # Create DVRL model
                self.data_valuation_model = self._create_dvrl_model(
                    embedding_dimension=embedding_dimension,
                    device_strategy=device_strategy,
                    scoring_model=scoring_model,
                )
                logger.info("DVRL model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize DVRL model: {e}")
                self.data_valuation_model = self._create_fallback_model()
        else:
            # Create fallback model if DVRL is not available
            self.data_valuation_model = self._create_fallback_model()
    
    def _create_dvrl_model(
        self,
        embedding_dimension: int,
        device_strategy: tf.distribute.Strategy,
        scoring_model: Optional[str] = None,
    ) -> Any:
        """
        Create a DVRL model for data valuation.
        
        Args:
            embedding_dimension: Dimension of document embeddings
            device_strategy: TensorFlow device strategy
            scoring_model: Optional path to pre-trained scoring model
            
        Returns:
            DVRL model instance
        """
        # Define predictor network architecture
        predictor_network = [
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
        
        # Define valuation network architecture
        valuation_network = [
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
        
        # Create DVRL model
        dvrl_model = dvrl.DVRL(
            embedding_dim=embedding_dimension,
            prediction_model=predictor_network,
            valuation_model=valuation_network,
            learning_rate=0.001,
            device_strategy=device_strategy
        )
        
        # Load pre-trained model if provided
        if scoring_model and os.path.exists(scoring_model):
            try:
                dvrl_model.load_weights(scoring_model)
                logger.info(f"Loaded pre-trained DVRL model from {scoring_model}")
            except Exception as e:
                logger.warning(f"Failed to load pre-trained model: {e}")
        
        return dvrl_model
    
    def _create_fallback_model(self) -> Callable:
        """
        Create a fallback model when DVRL is not available.
        
        Returns:
            Callable function for document scoring
        """
        logger.info("Using fallback document scoring mechanism")
        
        # Simple fallback scoring function based on text length and complexity
        def fallback_scoring_fn(
            documents: List[Dict[str, Any]],
            embeddings: Optional[np.ndarray] = None
        ) -> np.ndarray:
            scores = []
            
            for doc in documents:
                text = doc.get('text', '')
                metadata = doc.get('metadata', {})
                
                # Basic text quality signals
                length_score = min(len(text) / 1000, 1.0)  # Text length (capped at 1000 chars)
                
                # Structural complexity (word count / sentence count ratio)
                sentences = text.split('.')
                words = text.split()
                complexity_score = 0.5  # Default value
                if len(sentences) > 0:
                    words_per_sentence = len(words) / len(sentences)
                    complexity_score = min(words_per_sentence / 20, 1.0)  # Cap at 20 words/sentence
                
                # Metadata quality signals
                metadata_score = min(len(metadata) / 5, 1.0)  # Number of metadata fields (capped at 5)
                
                # Combined score
                score = 0.4 * length_score + 0.4 * complexity_score + 0.2 * metadata_score
                scores.append(score)
            
            return np.array(scores)
        
        return fallback_scoring_fn
    
    def compute_document_values(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None,
        performance_metric: Optional[List[float]] = None,
    ) -> List[float]:
        """
        Compute importance values for documents in a collection.
        
        Args:
            documents: List of LangChain documents
            embeddings: Optional pre-computed document embeddings
            performance_metric: Optional performance metric for each document
            
        Returns:
            List of document importance scores (0.0-1.0)
        """
        # Extract document IDs or compute hashes for caching
        doc_ids = []
        for doc in documents:
            doc_id = doc.metadata.get('id', None)
            if doc_id is None:
                # Compute hash of document content as ID
                doc_id = f"doc_{hash(doc.page_content)}"
            doc_ids.append(doc_id)
        
        # Check cache for existing values
        cached_values = []
        docs_to_compute = []
        doc_indices = []
        embeddings_to_compute = []
        
        for i, doc_id in enumerate(doc_ids):
            if doc_id in self.scoring_cache:
                cached_values.append((i, self.scoring_cache[doc_id]))
            else:
                docs_to_compute.append(documents[i])
                doc_indices.append(i)
                if embeddings:
                    embeddings_to_compute.append(embeddings[i])
        
        # If all values are cached, return them
        if not docs_to_compute:
            logger.info("All document values found in cache")
            # Sort by original document order
            sorted_values = [v for _, v in sorted(cached_values, key=lambda x: x[0])]
            return sorted_values
        
        # Prepare inputs for DVRL
        if HAS_DVRL:
            try:
                # Convert documents to the format expected by DVRL
                dvrl_inputs = []
                for i, doc in enumerate(docs_to_compute):
                    # Extract text and metadata
                    text = doc.page_content
                    metadata = doc.metadata
                    
                    # Create input dictionary
                    input_dict = {
                        'text': text,
                        'metadata': metadata,
                    }
                    
                    # Add embedding if available
                    if embeddings_to_compute:
                        input_dict['embedding'] = embeddings_to_compute[i]
                    
                    dvrl_inputs.append(input_dict)
                
                # Compute embeddings if not provided
                if not embeddings_to_compute and hasattr(self.data_valuation_model, 'compute_embeddings'):
                    # Extract text for embedding computation
                    texts = [doc.page_content for doc in docs_to_compute]
                    computed_embeddings = self.data_valuation_model.compute_embeddings(texts)
                    
                    # Add embeddings to inputs
                    for i, embedding in enumerate(computed_embeddings):
                        dvrl_inputs[i]['embedding'] = embedding
                
                # Compute document values using DVRL
                if isinstance(self.data_valuation_model, dvrl.DVRL):
                    # For DVRL model
                    computed_values = self.data_valuation_model.compute_data_values(
                        dvrl_inputs,
                        performance_metric=performance_metric
                    )
                else:
                    # For fallback model
                    computed_values = self.data_valuation_model(dvrl_inputs)
                
                # Ensure values are in range [0, 1]
                computed_values = np.clip(computed_values, 0.0, 1.0)
                
            except Exception as e:
                logger.error(f"Error computing document values with DVRL: {e}")
                # Use fallback scoring as a safety measure
                fallback_fn = self._create_fallback_model()
                computed_values = fallback_fn(dvrl_inputs)
        else:
            # Use fallback model
            dvrl_inputs = [
                {'text': doc.page_content, 'metadata': doc.metadata}
                for doc in docs_to_compute
            ]
            computed_values = self.data_valuation_model(dvrl_inputs)
        
        # Update cache with new values
        for i, doc_id in enumerate([doc_ids[idx] for idx in doc_indices]):
            self.scoring_cache[doc_id] = float(computed_values[i])
        
        # Save cache if file is specified
        if self.cache_file:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.scoring_cache, f)
                logger.info(f"Saved {len(self.scoring_cache)} document values to cache")
            except IOError as e:
                logger.warning(f"Failed to save cache file: {e}")
        
        # Combine cached and computed values
        all_values = [0.0] * len(documents)
        for i, value in cached_values:
            all_values[i] = value
        
        for comp_idx, orig_idx in enumerate(doc_indices):
            all_values[orig_idx] = float(computed_values[comp_idx])
        
        return all_values
    
    def filter_valuable_documents(
        self,
        documents: List[Document],
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[Document]:
        """
        Filter documents based on their computed value.
        
        Args:
            documents: List of LangChain documents
            threshold: Value threshold (0.0-1.0), uses instance default if None
            top_k: Optional number of top documents to retain
            embeddings: Optional pre-computed document embeddings
            
        Returns:
            List of valuable documents
        """
        if not documents:
            return []
        
        # Use instance threshold if not specified
        threshold = threshold if threshold is not None else self.value_threshold
        
        # Compute document values
        values = self.compute_document_values(documents, embeddings)
        
        # Filter based on threshold
        valuable_docs = []
        for i, (doc, value) in enumerate(zip(documents, values)):
            if value >= threshold:
                # Add value to metadata for reference
                doc.metadata['dvrl_value'] = value
                valuable_docs.append((value, i, doc))
        
        # Sort by value (descending)
        valuable_docs.sort(reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None and top_k > 0:
            valuable_docs = valuable_docs[:top_k]
        
        # Return documents in original order
        valuable_docs.sort(key=lambda x: x[1])
        return [doc for _, _, doc in valuable_docs]
    
    def optimize_vectorstore(
        self,
        vectorstore: VectorStore,
        threshold: Optional[float] = None,
        batch_size: int = 100,
        max_docs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Optimize a vector store by identifying and retaining valuable documents.
        
        Args:
            vectorstore: LangChain vector store to optimize
            threshold: Value threshold (0.0-1.0), uses instance default if None
            batch_size: Batch size for processing documents
            max_docs: Maximum number of documents to process
            
        Returns:
            Dict with optimization statistics
        """
        if not hasattr(vectorstore, "get_all_documents"):
            logger.error("Vector store does not support retrieving all documents")
            return {
                "success": False,
                "error": "Vector store does not support retrieving all documents",
            }
        
        try:
            # Get all documents from vector store
            all_docs = vectorstore.get_all_documents()
            
            # Limit documents if specified
            if max_docs is not None and max_docs > 0:
                all_docs = all_docs[:max_docs]
            
            total_docs = len(all_docs)
            logger.info(f"Retrieved {total_docs} documents from vector store")
            
            # Process documents in batches
            valuable_docs = []
            processed_docs = 0
            
            for i in range(0, total_docs, batch_size):
                batch_docs = all_docs[i:i+batch_size]
                batch_valuable = self.filter_valuable_documents(
                    batch_docs,
                    threshold=threshold
                )
                valuable_docs.extend(batch_valuable)
                processed_docs += len(batch_docs)
                logger.info(f"Processed {processed_docs}/{total_docs} documents")
            
            # Calculate statistics
            valuable_count = len(valuable_docs)
            reduction_percent = 100 * (1 - valuable_count / total_docs) if total_docs > 0 else 0
            
            logger.info(f"Identified {valuable_count} valuable documents ({reduction_percent:.2f}% reduction)")
            
            # Create new vector store with valuable documents if supported
            if hasattr(vectorstore, "from_documents"):
                # Get embedding model from original vector store
                embedding_model = getattr(vectorstore, "embedding", None)
                if embedding_model is None:
                    logger.warning("Could not retrieve embedding model from vector store")
                    return {
                        "success": True,
                        "valuable_docs": valuable_docs,
                        "total_docs": total_docs,
                        "valuable_count": valuable_count,
                        "reduction_percent": reduction_percent,
                    }
                
                # Get constructor parameters for optimized store
                constructor_params = {
                    k: getattr(vectorstore, k, None)
                    for k in ["table_name", "content_column", "metadata_column", "vector_column"]
                    if hasattr(vectorstore, k)
                }
                
                # Create optimized vector store - implementation will vary based on vector store type
                if hasattr(vectorstore, "connection"):
                    # For SQL-based vector stores like HanaDB
                    constructor_params["connection"] = getattr(vectorstore, "connection", None)
                    optimized_store = vectorstore.__class__.from_documents(
                        documents=valuable_docs,
                        embedding=embedding_model,
                        **constructor_params
                    )
                    
                    return {
                        "success": True,
                        "valuable_docs": valuable_docs,
                        "total_docs": total_docs,
                        "valuable_count": valuable_count,
                        "reduction_percent": reduction_percent,
                        "optimized_store": optimized_store,
                    }
                else:
                    # Generic fallback
                    logger.warning("Unsupported vector store type for optimization")
                    return {
                        "success": True,
                        "valuable_docs": valuable_docs,
                        "total_docs": total_docs,
                        "valuable_count": valuable_count,
                        "reduction_percent": reduction_percent,
                    }
            else:
                # Return statistics if creation of optimized store is not supported
                return {
                    "success": True,
                    "valuable_docs": valuable_docs,
                    "total_docs": total_docs,
                    "valuable_count": valuable_count,
                    "reduction_percent": reduction_percent,
                }
        
        except Exception as e:
            logger.error(f"Error optimizing vector store: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Example usage:
# data_valuation = DVRLDataValuation(embedding_dimension=768, value_threshold=0.7)
# valuable_docs = data_valuation.filter_valuable_documents(documents)
# optimization_result = data_valuation.optimize_vectorstore(vectorstore)