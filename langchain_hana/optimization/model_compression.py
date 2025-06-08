"""
Model compression and sparsification using state_of_sparsity.

This module provides:
1. Embedding model compression and sparsification
2. Reduced memory footprint for embedding models
3. Faster inference with minimal accuracy loss
"""

import logging
import os
import json
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple, Callable, Type

import numpy as np

# Import LangChain core components
from langchain_core.embeddings import Embeddings

# Set up logging
logger = logging.getLogger(__name__)

# Conditionally import state_of_sparsity - allow for fallback if not available
try:
    import tensorflow as tf
    from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
    from state_of_sparsity.pruning import masked_layer
    from state_of_sparsity.pruning import core as pruning_core
    HAS_SOS = True
except ImportError:
    logger.warning("state_of_sparsity dependencies not found. Using fallback compression mechanism.")
    HAS_SOS = False


class SparseEmbeddingModel:
    """
    Sparse embedding model with compression.
    
    This class applies model compression techniques to embedding models,
    reducing memory footprint and inference time with minimal accuracy loss.
    
    Attributes:
        compression_ratio: Target compression ratio
        pruning_strategy: Pruning strategy to use
        model_path: Path to the saved compressed model
    """
    
    def __init__(
        self,
        compression_ratio: float = 0.5,
        pruning_strategy: str = "magnitude",
        model_path: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize the SparseEmbeddingModel component.
        
        Args:
            compression_ratio: Target compression ratio (0.0-1.0)
            pruning_strategy: Pruning strategy ('magnitude', 'random', 'structured')
            model_path: Optional path to pre-trained compressed model
            use_gpu: Whether to use GPU for computation
        """
        self.compression_ratio = compression_ratio
        self.pruning_strategy = pruning_strategy
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.compressed_model = None
        self.original_model = None
        self.pruning_params = None
        
        # Set up TensorFlow device if available
        if HAS_SOS:
            try:
                # Configure device strategy for TensorFlow
                if use_gpu and tf.config.list_physical_devices('GPU'):
                    # Use GPU if available and requested
                    self.device_strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
                    logger.info("Using GPU for compression computation")
                else:
                    # Fall back to CPU
                    self.device_strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
                    logger.info("Using CPU for compression computation")
                
                # Set up pruning parameters
                self.pruning_params = self._setup_pruning_params()
                logger.info("Pruning parameters initialized successfully")
                
                # Load pre-trained model if provided
                if model_path and os.path.exists(model_path):
                    self._load_model(model_path)
                    logger.info(f"Loaded pre-trained compressed model from {model_path}")
            
            except Exception as e:
                logger.error(f"Failed to initialize compression model: {e}")
                self.compressed_model = None
        else:
            logger.warning("Using fallback compression mechanism")
    
    def _setup_pruning_params(self) -> Dict[str, Any]:
        """
        Set up pruning parameters based on strategy.
        
        Returns:
            Dictionary of pruning parameters
        """
        # Define pruning parameters based on strategy
        if self.pruning_strategy == "magnitude":
            return {
                "pruning_schedule": pruning_core.PolynomialDecayPruningSchedule(
                    initial_sparsity=0.0,
                    final_sparsity=self.compression_ratio,
                    begin_step=1000,
                    end_step=20000,
                    frequency=100,
                ),
                "block_size": (1, 1),
                "block_pooling_type": "AVG",
                "pruning_type": "magnitude",
            }
        
        elif self.pruning_strategy == "random":
            return {
                "pruning_schedule": pruning_core.ConstantSparsityPruningSchedule(
                    target_sparsity=self.compression_ratio,
                    begin_step=0,
                    frequency=100,
                ),
                "block_size": (1, 1),
                "block_pooling_type": "AVG",
                "pruning_type": "random",
            }
        
        elif self.pruning_strategy == "structured":
            return {
                "pruning_schedule": pruning_core.PolynomialDecayPruningSchedule(
                    initial_sparsity=0.0,
                    final_sparsity=self.compression_ratio,
                    begin_step=1000,
                    end_step=20000,
                    frequency=100,
                ),
                "block_size": (1, 8),  # Structured pruning uses blocks
                "block_pooling_type": "AVG",
                "pruning_type": "magnitude",
            }
        
        else:
            # Default to magnitude pruning
            return {
                "pruning_schedule": pruning_core.PolynomialDecayPruningSchedule(
                    initial_sparsity=0.0,
                    final_sparsity=self.compression_ratio,
                    begin_step=1000,
                    end_step=20000,
                    frequency=100,
                ),
                "block_size": (1, 1),
                "block_pooling_type": "AVG",
                "pruning_type": "magnitude",
            }
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a pre-trained compressed model.
        
        Args:
            model_path: Path to the model file
        """
        if not HAS_SOS:
            logger.warning("Cannot load compressed model without state_of_sparsity dependency")
            return
        
        try:
            # Load the compressed model
            self.compressed_model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    'MaskedDense': masked_layer.MaskedDense,
                    'MaskedConv2D': masked_layer.MaskedConv2D,
                    'PruneLowMagnitude': pruning_wrapper.PruneLowMagnitude,
                }
            )
            logger.info(f"Loaded compressed model from {model_path}")
            
            # Check sparsity
            sparsity = self._get_model_sparsity(self.compressed_model)
            logger.info(f"Loaded model has sparsity of {sparsity:.2%}")
            
        except Exception as e:
            logger.error(f"Error loading compressed model: {e}")
            self.compressed_model = None
    
    def _get_model_sparsity(self, model: tf.keras.Model) -> float:
        """
        Calculate the sparsity of a model.
        
        Args:
            model: TensorFlow model
            
        Returns:
            Sparsity ratio (0.0-1.0)
        """
        if not HAS_SOS:
            return 0.0
        
        try:
            total_params = 0
            zero_params = 0
            
            # Iterate through layers
            for layer in model.layers:
                if hasattr(layer, 'kernel'):
                    # For regular layers
                    weights = layer.kernel.numpy()
                    total_params += weights.size
                    zero_params += np.sum(weights == 0)
                elif isinstance(layer, pruning_wrapper.PruneLowMagnitude):
                    # For pruned layers
                    if hasattr(layer.layer, 'kernel'):
                        weights = layer.layer.kernel.numpy()
                        mask = layer.layer.pruning_vars[0].numpy()
                        total_params += weights.size
                        zero_params += np.sum(mask == 0)
            
            # Calculate sparsity
            if total_params > 0:
                return zero_params / total_params
            else:
                return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating model sparsity: {e}")
            return 0.0
    
    def compress_model(
        self,
        model: tf.keras.Model,
        dataset: Optional[tf.data.Dataset] = None,
        epochs: int = 10,
        save_path: Optional[str] = None,
    ) -> tf.keras.Model:
        """
        Compress a TensorFlow model using pruning.
        
        Args:
            model: TensorFlow model to compress
            dataset: Optional dataset for fine-tuning
            epochs: Number of fine-tuning epochs
            save_path: Optional path to save the compressed model
            
        Returns:
            Compressed TensorFlow model
        """
        if not HAS_SOS:
            logger.warning("Cannot compress model without state_of_sparsity dependency")
            return model
        
        try:
            # Store original model
            self.original_model = model
            
            # Clone model
            cloned_model = tf.keras.models.clone_model(model)
            cloned_model.set_weights(model.get_weights())
            
            # Apply pruning to dense layers
            pruned_model = self._apply_pruning(cloned_model)
            
            # Compile pruned model
            pruned_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=model.loss if hasattr(model, 'loss') else 'mse',
                metrics=model.metrics if hasattr(model, 'metrics') else None,
            )
            
            # Fine-tune if dataset is provided
            if dataset is not None:
                with self.device_strategy.scope():
                    pruned_model.fit(
                        dataset,
                        epochs=epochs,
                        callbacks=[
                            pruning_core.UpdatePruningStep(),
                            pruning_core.PruningSummaries(log_dir="/tmp/pruning_logs"),
                        ]
                    )
            
            # Apply final pruning
            final_model = pruning_core.strip_pruning(pruned_model)
            
            # Calculate sparsity
            sparsity = self._get_model_sparsity(final_model)
            logger.info(f"Compressed model has sparsity of {sparsity:.2%}")
            
            # Save compressed model if path is provided
            if save_path:
                final_model.save(save_path)
                logger.info(f"Saved compressed model to {save_path}")
            
            # Store compressed model
            self.compressed_model = final_model
            self.model_path = save_path
            
            return final_model
            
        except Exception as e:
            logger.error(f"Error compressing model: {e}")
            return model
    
    def _apply_pruning(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Apply pruning to a TensorFlow model.
        
        Args:
            model: TensorFlow model
            
        Returns:
            Pruned model
        """
        pruned_layers = []
        
        # Create new model with pruning applied
        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.Dense):
                # Apply pruning to dense layers
                pruned_layer = masked_layer.MaskedDense.from_layer(
                    layer,
                    **self.pruning_params,
                )
                pruned_layers.append(pruned_layer)
            elif isinstance(layer, tf.keras.layers.Conv2D):
                # Apply pruning to convolutional layers
                pruned_layer = masked_layer.MaskedConv2D.from_layer(
                    layer,
                    **self.pruning_params,
                )
                pruned_layers.append(pruned_layer)
            else:
                # Keep other layers as is
                pruned_layers.append(layer)
        
        # Create new model with pruned layers
        pruned_model = tf.keras.Sequential(pruned_layers)
        
        return pruned_model
    
    def compress_numpy_weights(
        self,
        weights: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compress weights using magnitude pruning.
        
        Args:
            weights: NumPy array of weights
            threshold: Optional magnitude threshold (uses compression_ratio if None)
            
        Returns:
            Tuple of (compressed_weights, mask)
        """
        # Determine threshold if not provided
        if threshold is None:
            # Sort weights by magnitude
            flat_weights = weights.flatten()
            sorted_weights = np.sort(np.abs(flat_weights))
            
            # Determine threshold index
            threshold_idx = int(len(sorted_weights) * self.compression_ratio)
            
            # Get threshold value
            if threshold_idx < len(sorted_weights):
                threshold = sorted_weights[threshold_idx]
            else:
                threshold = 0.0
        
        # Create mask based on threshold
        mask = np.abs(weights) > threshold
        
        # Apply mask to weights
        compressed_weights = weights * mask
        
        # Calculate sparsity
        sparsity = 1.0 - np.mean(mask)
        logger.info(f"Compressed weights with sparsity of {sparsity:.2%}")
        
        return compressed_weights, mask
    
    def create_sparse_embedding_layer(
        self,
        input_dim: int,
        output_dim: int,
        input_length: Optional[int] = None,
    ) -> tf.keras.layers.Layer:
        """
        Create a sparse embedding layer.
        
        Args:
            input_dim: Size of the vocabulary
            output_dim: Size of the embeddings
            input_length: Optional fixed input length
            
        Returns:
            TensorFlow embedding layer with sparsity
        """
        if not HAS_SOS:
            logger.warning("Cannot create sparse embedding layer without state_of_sparsity")
            return tf.keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim,
                input_length=input_length,
            )
        
        try:
            # Create regular embedding layer
            embedding_layer = tf.keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim,
                input_length=input_length,
            )
            
            # Apply pruning
            pruned_layer = masked_layer.MaskedEmbedding.from_layer(
                embedding_layer,
                **self.pruning_params,
            )
            
            return pruned_layer
            
        except Exception as e:
            logger.error(f"Error creating sparse embedding layer: {e}")
            return tf.keras.layers.Embedding(
                input_dim=input_dim,
                output_dim=output_dim,
                input_length=input_length,
            )


class SparseEmbeddingModel(Embeddings):
    """
    Sparse embedding model with compression for LangChain.
    
    This class wraps an embedding model with compression techniques,
    reducing memory footprint and inference time with minimal accuracy loss.
    
    Attributes:
        base_embeddings: Base embedding model to compress
        compression_ratio: Target compression ratio
        compression_strategy: Compression strategy to use
        model_path: Path to the saved compressed model
    """
    
    def __init__(
        self,
        base_embeddings: Embeddings,
        compression_ratio: float = 0.5,
        compression_strategy: str = "magnitude",
        cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize the SparseEmbeddingModel component.
        
        Args:
            base_embeddings: Base embedding model to compress
            compression_ratio: Target compression ratio (0.0-1.0)
            compression_strategy: Compression strategy ('magnitude', 'random', 'structured')
            cache_dir: Optional directory to cache embeddings
            model_path: Optional path to pre-trained compressed model
            use_gpu: Whether to use GPU for computation
        """
        self.base_embeddings = base_embeddings
        self.compression_ratio = compression_ratio
        self.compression_strategy = compression_strategy
        self.cache_dir = cache_dir
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.embedding_cache = {}
        self.compressed_weights = {}
        self.weight_masks = {}
        
        # Create cache directory if specified
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Initialize compressor if SOS is available
        if HAS_SOS:
            try:
                self.compressor = SparseEmbeddingModel(
                    compression_ratio=compression_ratio,
                    pruning_strategy=compression_strategy,
                    model_path=model_path,
                    use_gpu=use_gpu,
                )
                logger.info("SOS compressor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SOS compressor: {e}")
                self.compressor = None
        else:
            logger.warning("Using fallback compression mechanism")
            self.compressor = None
        
        # Load cached compressed weights if available
        self._load_cached_weights()
    
    def _load_cached_weights(self) -> None:
        """Load cached compressed weights if available."""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "compressed_weights.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Load compression metadata
                    self.compression_ratio = cache_data.get("compression_ratio", self.compression_ratio)
                    self.compression_strategy = cache_data.get("compression_strategy", self.compression_strategy)
                    
                    logger.info(f"Loaded compressed weights metadata from cache")
                except Exception as e:
                    logger.warning(f"Failed to load compressed weights from cache: {e}")
    
    def _save_cached_weights(self) -> None:
        """Save cached compressed weights if directory is specified."""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "compressed_weights.json")
            try:
                # Save compression metadata
                cache_data = {
                    "compression_ratio": self.compression_ratio,
                    "compression_strategy": self.compression_strategy,
                    "timestamp": str(os.path.getmtime(cache_file) if os.path.exists(cache_file) else 0),
                }
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                
                logger.info(f"Saved compressed weights metadata to cache")
            except Exception as e:
                logger.warning(f"Failed to save compressed weights to cache: {e}")
    
    def compress_embeddings(
        self,
        embeddings: List[List[float]],
    ) -> List[List[float]]:
        """
        Apply compression to embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Compressed embedding vectors
        """
        # Convert to numpy array
        embeddings_np = np.array(embeddings)
        
        # Generate a unique key for this embedding shape
        shape_key = f"shape_{embeddings_np.shape[1]}"
        
        # Check if we have a mask for this shape
        if shape_key not in self.weight_masks:
            # Create new mask and compressed weights
            if self.compressor is not None and HAS_SOS:
                try:
                    # Use SOS compressor
                    _, mask = self.compressor.compress_numpy_weights(embeddings_np)
                    self.weight_masks[shape_key] = mask
                except Exception as e:
                    logger.error(f"Error using SOS compressor: {e}")
                    # Fall back to simple thresholding
                    self.weight_masks[shape_key] = self._fallback_compression(embeddings_np)
            else:
                # Use fallback compression
                self.weight_masks[shape_key] = self._fallback_compression(embeddings_np)
        
        # Apply mask to embeddings
        mask = self.weight_masks[shape_key]
        compressed_embeddings_np = embeddings_np * mask
        
        # Convert back to list
        return compressed_embeddings_np.tolist()
    
    def _fallback_compression(self, embeddings_np: np.ndarray) -> np.ndarray:
        """
        Apply fallback compression using magnitude thresholding.
        
        Args:
            embeddings_np: NumPy array of embeddings
            
        Returns:
            Binary mask for compression
        """
        # Calculate threshold based on compression ratio
        flat_values = embeddings_np.flatten()
        sorted_values = np.sort(np.abs(flat_values))
        threshold_idx = int(len(sorted_values) * self.compression_ratio)
        
        if threshold_idx < len(sorted_values):
            threshold = sorted_values[threshold_idx]
        else:
            threshold = 0.0
        
        # Create mask
        mask = np.abs(embeddings_np) > threshold
        
        # Calculate sparsity
        sparsity = 1.0 - np.mean(mask)
        logger.info(f"Created compression mask with sparsity of {sparsity:.2%}")
        
        return mask
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate compressed embeddings for a list of documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of compressed embeddings
        """
        # Check cache first
        cached_embeddings = []
        texts_to_embed = []
        indices = []
        
        for i, text in enumerate(texts):
            # Generate cache key
            cache_key = f"doc_{hash(text)}"
            
            # Check if in memory cache
            if cache_key in self.embedding_cache:
                cached_embeddings.append((i, self.embedding_cache[cache_key]))
            # Check if in disk cache
            elif self.cache_dir:
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                if os.path.exists(cache_file):
                    try:
                        with open(cache_file, 'r') as f:
                            embedding = json.load(f)
                        cached_embeddings.append((i, embedding))
                        # Update memory cache
                        self.embedding_cache[cache_key] = embedding
                    except Exception:
                        # If cache file is corrupted, re-embed
                        texts_to_embed.append(text)
                        indices.append(i)
                else:
                    texts_to_embed.append(text)
                    indices.append(i)
            else:
                texts_to_embed.append(text)
                indices.append(i)
        
        # Return all from cache if possible
        if not texts_to_embed:
            # Sort by original order
            sorted_embeddings = [emb for _, emb in sorted(cached_embeddings, key=lambda x: x[0])]
            return sorted_embeddings
        
        # Generate base embeddings for remaining texts
        base_embeddings = self.base_embeddings.embed_documents(texts_to_embed)
        
        # Apply compression
        compressed_embeddings = self.compress_embeddings(base_embeddings)
        
        # Cache new embeddings
        for i, (idx, text) in enumerate(zip(indices, texts_to_embed)):
            # Generate cache key
            cache_key = f"doc_{hash(text)}"
            
            # Update memory cache
            self.embedding_cache[cache_key] = compressed_embeddings[i]
            
            # Update disk cache if enabled
            if self.cache_dir:
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(compressed_embeddings[i], f)
                except Exception as e:
                    logger.warning(f"Failed to write embedding to cache: {e}")
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        for i, emb in cached_embeddings:
            all_embeddings[i] = emb
        
        for i, (idx, emb) in enumerate(zip(indices, compressed_embeddings)):
            all_embeddings[idx] = emb
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate compressed embedding for a query text.
        
        Args:
            text: Query text
            
        Returns:
            Compressed embedding
        """
        # Check cache first
        cache_key = f"query_{hash(text)}"
        
        # Check if in memory cache
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Check if in disk cache
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        embedding = json.load(f)
                    # Update memory cache
                    self.embedding_cache[cache_key] = embedding
                    return embedding
                except Exception:
                    # If cache file is corrupted, re-embed
                    pass
        
        # Generate base embedding
        base_embedding = self.base_embeddings.embed_query(text)
        
        # Apply compression (for consistency with document embeddings)
        compressed_embedding = self.compress_embeddings([base_embedding])[0]
        
        # Cache embedding
        self.embedding_cache[cache_key] = compressed_embedding
        
        # Update disk cache if enabled
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                with open(cache_file, 'w') as f:
                    json.dump(compressed_embedding, f)
            except Exception as e:
                logger.warning(f"Failed to write embedding to cache: {e}")
        
        return compressed_embedding
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Get compression statistics.
        
        Returns:
            Dictionary with compression statistics
        """
        stats = {
            "compression_ratio": self.compression_ratio,
            "compression_strategy": self.compression_strategy,
            "compressed_shapes": {},
            "total_sparsity": 0.0,
        }
        
        # Calculate sparsity for each shape
        total_elements = 0
        total_nonzeros = 0
        
        for shape_key, mask in self.weight_masks.items():
            # Extract shape from key
            shape_match = shape_key.replace("shape_", "")
            
            # Calculate sparsity
            elements = mask.size
            nonzeros = np.sum(mask)
            sparsity = 1.0 - (nonzeros / elements)
            
            # Add to totals
            total_elements += elements
            total_nonzeros += nonzeros
            
            # Add to stats
            stats["compressed_shapes"][shape_key] = {
                "elements": int(elements),
                "nonzeros": int(nonzeros),
                "sparsity": float(sparsity),
            }
        
        # Calculate overall sparsity
        if total_elements > 0:
            stats["total_sparsity"] = 1.0 - (total_nonzeros / total_elements)
        
        # Add cache stats
        stats["cache_size"] = len(self.embedding_cache)
        stats["cache_dir"] = self.cache_dir
        
        return stats


# Example usage:
# base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# sparse_embeddings = SparseEmbeddingModel(
#     base_embeddings=base_embeddings,
#     compression_ratio=0.7,
#     compression_strategy="magnitude",
#     cache_dir="/tmp/sparse_embeddings",
# )
# embeddings = sparse_embeddings.embed_documents(["Sample text"])
# stats = sparse_embeddings.get_compression_stats()