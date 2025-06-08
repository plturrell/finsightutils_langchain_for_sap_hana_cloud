"""
Neural Additive Models (NAM) integration for interpretable embeddings.

This module provides:
1. Interpretable embedding generation using Neural Additive Models
2. Feature-level explanations for vector search results
3. Embedding visualization capabilities
"""

import logging
import os
import json
import time
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np

# Import LangChain core components
from langchain_core.embeddings import Embeddings

# Set up logging
logger = logging.getLogger(__name__)

# Conditionally import NAM components - allow for fallback if not available
try:
    import jax
    import jax.numpy as jnp
    import flax
    import neural_additive_models as nam
    from neural_additive_models.models import NAM, FeatureNN
    HAS_NAM = True
except ImportError:
    logger.warning("Neural Additive Models dependencies not found. Using fallback mechanism.")
    HAS_NAM = False


class NAMEmbeddings(Embeddings):
    """
    Interpretable embeddings using Neural Additive Models.
    
    This class generates interpretable embeddings that can be explained
    at the feature level, allowing users to understand which aspects of
    the text contribute most to search results.
    
    Attributes:
        base_embeddings: Base embedding model to generate initial vectors
        nam_model: Neural Additive Model for interpretable embeddings
        dimension: Output embedding dimension
        feature_names: Names of features for interpretability
        cache_dir: Directory to cache embeddings
    """
    
    def __init__(
        self,
        base_embeddings: Embeddings,
        dimension: int = 768,
        num_features: int = 128,
        feature_names: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize NAM Embeddings.
        
        Args:
            base_embeddings: Base embedding model to generate initial vectors
            dimension: Output embedding dimension
            num_features: Number of interpretable features
            feature_names: Optional names for features
            cache_dir: Optional directory to cache embeddings
            model_path: Optional path to pre-trained NAM model
            use_gpu: Whether to use GPU for computation
        """
        self.base_embeddings = base_embeddings
        self.dimension = dimension
        self.num_features = num_features
        self.feature_names = feature_names or [f"feature_{i}" for i in range(num_features)]
        self.cache_dir = cache_dir
        self.embedding_cache = {}
        
        # Create cache directory if specified
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        # Initialize NAM model
        if HAS_NAM:
            try:
                # Set up JAX device
                if use_gpu and len(jax.devices("gpu")) > 0:
                    jax_device = jax.devices("gpu")[0]
                    logger.info("Using GPU for NAM computation")
                else:
                    jax_device = jax.devices("cpu")[0]
                    logger.info("Using CPU for NAM computation")
                
                # Create NAM model
                self.nam_model = self._create_nam_model(
                    input_dim=dimension,
                    output_dim=num_features,
                    device=jax_device,
                )
                
                # Load pre-trained model if provided
                if model_path and os.path.exists(model_path):
                    self._load_model(model_path)
                    logger.info(f"Loaded pre-trained NAM model from {model_path}")
                
                logger.info("NAM model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize NAM model: {e}")
                self.nam_model = self._create_fallback_model()
        else:
            # Create fallback model if NAM is not available
            self.nam_model = self._create_fallback_model()
            logger.info("Using fallback embedding model")
    
    def _create_nam_model(
        self,
        input_dim: int,
        output_dim: int,
        device: Any,
    ) -> Any:
        """
        Create a Neural Additive Model for interpretable embeddings.
        
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output embeddings
            device: JAX device for computation
            
        Returns:
            NAM model instance
        """
        # Define feature network configuration
        feature_config = {
            "num_layers": 3,
            "hidden_dims": [64, 32, 16],
            "activation": "relu",
        }
        
        # Create feature networks
        feature_nns = []
        for i in range(input_dim):
            # Each feature gets its own neural network
            feature_nn = FeatureNN(
                num_layers=feature_config["num_layers"],
                hidden_dims=feature_config["hidden_dims"],
                activation=feature_config["activation"],
                dropout_rate=0.1,
                feature_idx=i,
            )
            feature_nns.append(feature_nn)
        
        # Create NAM model
        model = NAM(
            feature_nns=feature_nns,
            output_dim=output_dim,
            use_bias=True,
        )
        
        # Initialize model
        rng = jax.random.PRNGKey(42)
        dummy_input = jnp.zeros((1, input_dim))
        params = model.init(rng, dummy_input)
        
        # Return model and params
        return {
            "model": model,
            "params": params,
            "device": device,
        }
    
    def _create_fallback_model(self) -> Any:
        """
        Create a fallback model when NAM is not available.
        
        Returns:
            Fallback model for embedding transformation
        """
        # Simple fallback using linear transformation
        def fallback_transform(embeddings: np.ndarray) -> np.ndarray:
            # Generate a fixed random projection matrix (same for all calls)
            np.random.seed(42)
            if not hasattr(fallback_transform, "projection_matrix"):
                input_dim = embeddings.shape[1] if len(embeddings.shape) > 1 else len(embeddings)
                fallback_transform.projection_matrix = np.random.normal(
                    size=(input_dim, self.num_features)
                )
                # Normalize columns
                for j in range(self.num_features):
                    fallback_transform.projection_matrix[:, j] /= np.linalg.norm(
                        fallback_transform.projection_matrix[:, j]
                    )
            
            # Apply linear transformation
            if len(embeddings.shape) == 1:
                # Single embedding
                output = np.dot(embeddings, fallback_transform.projection_matrix)
            else:
                # Batch of embeddings
                output = np.dot(embeddings, fallback_transform.projection_matrix)
            
            return output
        
        return fallback_transform
    
    def _load_model(self, model_path: str) -> None:
        """
        Load a pre-trained NAM model.
        
        Args:
            model_path: Path to model file
        """
        if not HAS_NAM:
            logger.warning("Cannot load NAM model without Neural Additive Models dependency")
            return
        
        try:
            with open(model_path, 'rb') as f:
                model_data = flax.serialization.msgpack_restore(f.read())
            
            self.nam_model["params"] = model_data
            logger.info(f"Loaded NAM model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading NAM model: {e}")
    
    def _save_model(self, model_path: str) -> None:
        """
        Save the NAM model to a file.
        
        Args:
            model_path: Path to save model file
        """
        if not HAS_NAM:
            logger.warning("Cannot save NAM model without Neural Additive Models dependency")
            return
        
        try:
            model_data = flax.serialization.msgpack_serialize(self.nam_model["params"])
            with open(model_path, 'wb') as f:
                f.write(model_data)
            logger.info(f"Saved NAM model to {model_path}")
        except Exception as e:
            logger.error(f"Error saving NAM model: {e}")
    
    def _apply_nam_model(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Apply the NAM model to transform embeddings.
        
        Args:
            embeddings: Input embeddings (batch or single)
            
        Returns:
            Transformed embeddings
        """
        if not HAS_NAM:
            # Use fallback model
            return self.nam_model(embeddings)
        
        try:
            # Move to device
            device = self.nam_model["device"]
            model = self.nam_model["model"]
            params = self.nam_model["params"]
            
            # Convert to JAX array
            embeddings_jax = jnp.array(embeddings)
            
            # Apply model
            with jax.default_device(device):
                output = model.apply(params, embeddings_jax)
            
            # Convert back to numpy
            return np.array(output)
        except Exception as e:
            logger.error(f"Error applying NAM model: {e}")
            # Fall back to simple transformation
            fallback_model = self._create_fallback_model()
            return fallback_model(embeddings)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of interpretable embeddings
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
        
        # Apply NAM model for interpretability
        interpretable_embeddings = self._apply_nam_model(np.array(base_embeddings))
        
        # Convert to list of lists
        interpretable_embeddings = interpretable_embeddings.tolist()
        
        # Cache new embeddings
        for i, (idx, text) in enumerate(zip(indices, texts_to_embed)):
            # Generate cache key
            cache_key = f"doc_{hash(text)}"
            
            # Update memory cache
            self.embedding_cache[cache_key] = interpretable_embeddings[i]
            
            # Update disk cache if enabled
            if self.cache_dir:
                cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
                try:
                    with open(cache_file, 'w') as f:
                        json.dump(interpretable_embeddings[i], f)
                except Exception as e:
                    logger.warning(f"Failed to write embedding to cache: {e}")
        
        # Combine cached and new embeddings
        all_embeddings = [None] * len(texts)
        for i, emb in cached_embeddings:
            all_embeddings[i] = emb
        
        for i, (idx, emb) in enumerate(zip(indices, interpretable_embeddings)):
            all_embeddings[idx] = emb
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a query text.
        
        Args:
            text: Query text
            
        Returns:
            Interpretable embedding
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
        
        # Apply NAM model for interpretability
        interpretable_embedding = self._apply_nam_model(np.array(base_embedding))
        
        # Convert to list
        interpretable_embedding = interpretable_embedding.tolist()
        
        # Cache embedding
        self.embedding_cache[cache_key] = interpretable_embedding
        
        # Update disk cache if enabled
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
            try:
                with open(cache_file, 'w') as f:
                    json.dump(interpretable_embedding, f)
            except Exception as e:
                logger.warning(f"Failed to write embedding to cache: {e}")
        
        return interpretable_embedding
    
    def get_feature_importance(
        self,
        text: str,
        top_k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        Get feature importance scores for a given text.
        
        Args:
            text: Text to analyze
            top_k: Optional number of top features to return
            
        Returns:
            List of (feature_name, importance_score) tuples
        """
        # Generate embedding with base model
        base_embedding = self.base_embeddings.embed_query(text)
        
        if HAS_NAM:
            try:
                # Get NAM model
                model = self.nam_model["model"]
                params = self.nam_model["params"]
                device = self.nam_model["device"]
                
                # Convert to JAX array
                embedding_jax = jnp.array(base_embedding)
                
                # Get feature outputs
                with jax.default_device(device):
                    feature_outputs = model.apply_feature_nns(params, embedding_jax)
                
                # Calculate importance as magnitude of feature outputs
                feature_importance = np.abs(np.array(feature_outputs))
                
                # Normalize to sum to 1
                feature_importance = feature_importance / np.sum(feature_importance)
            except Exception as e:
                logger.error(f"Error calculating feature importance: {e}")
                # Fallback method
                feature_importance = self._fallback_feature_importance(base_embedding)
        else:
            # Fallback method for when NAM is not available
            feature_importance = self._fallback_feature_importance(base_embedding)
        
        # Match importance with feature names
        importance_pairs = list(zip(self.feature_names, feature_importance))
        
        # Sort by importance (descending)
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to top_k if specified
        if top_k is not None and top_k > 0:
            importance_pairs = importance_pairs[:top_k]
        
        return importance_pairs
    
    def _fallback_feature_importance(self, embedding: List[float]) -> np.ndarray:
        """
        Fallback method to calculate feature importance.
        
        Args:
            embedding: Base embedding
            
        Returns:
            Array of feature importance scores
        """
        # Convert embedding to numpy array
        embedding_np = np.array(embedding)
        
        # Create random but consistent projection
        np.random.seed(42)
        projection = np.random.normal(size=(len(embedding), self.num_features))
        
        # Project embedding to feature space
        feature_values = np.dot(embedding_np, projection)
        
        # Calculate importance as absolute values
        importance = np.abs(feature_values)
        
        # Normalize to sum to 1
        importance = importance / np.sum(importance)
        
        return importance
    
    def explain_similarity(
        self,
        query: str,
        document: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain why a document is similar to a query.
        
        Args:
            query: Query text
            document: Document text
            top_k: Number of top features to include in explanation
            
        Returns:
            Dictionary with explanation information
        """
        # Generate embeddings
        query_embedding = self.base_embeddings.embed_query(query)
        doc_embedding = self.base_embeddings.embed_documents([document])[0]
        
        if HAS_NAM:
            try:
                # Get NAM model
                model = self.nam_model["model"]
                params = self.nam_model["params"]
                device = self.nam_model["device"]
                
                # Convert to JAX arrays
                query_jax = jnp.array(query_embedding)
                doc_jax = jnp.array(doc_embedding)
                
                # Get feature outputs
                with jax.default_device(device):
                    query_features = model.apply_feature_nns(params, query_jax)
                    doc_features = model.apply_feature_nns(params, doc_jax)
                
                # Calculate feature-wise similarity
                feature_sim = jnp.multiply(query_features, doc_features)
                
                # Convert to numpy
                feature_sim = np.array(feature_sim)
                
                # Calculate overall similarity
                similarity_score = float(np.sum(feature_sim) / (
                    np.linalg.norm(query_features) * np.linalg.norm(doc_features)
                ))
                
                # Normalize feature similarities
                feature_importance = feature_sim / np.sum(np.abs(feature_sim))
            except Exception as e:
                logger.error(f"Error calculating similarity explanation: {e}")
                # Fallback method
                similarity_score, feature_importance = self._fallback_similarity(
                    query_embedding, doc_embedding
                )
        else:
            # Fallback method for when NAM is not available
            similarity_score, feature_importance = self._fallback_similarity(
                query_embedding, doc_embedding
            )
        
        # Match importance with feature names
        importance_pairs = list(zip(self.feature_names, feature_importance))
        
        # Sort by importance (descending)
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Get top and bottom features
        top_features = importance_pairs[:top_k]
        bottom_features = importance_pairs[-top_k:]
        
        return {
            "similarity_score": similarity_score,
            "top_matching_features": top_features,
            "least_matching_features": bottom_features,
            "query": query,
            "document": document,
        }
    
    def _fallback_similarity(
        self,
        query_embedding: List[float],
        doc_embedding: List[float],
    ) -> Tuple[float, np.ndarray]:
        """
        Fallback method to calculate similarity explanation.
        
        Args:
            query_embedding: Query embedding
            doc_embedding: Document embedding
            
        Returns:
            Tuple of (similarity_score, feature_importance_array)
        """
        # Convert embeddings to numpy arrays
        query_np = np.array(query_embedding)
        doc_np = np.array(doc_embedding)
        
        # Calculate cosine similarity
        similarity_score = float(np.dot(query_np, doc_np) / (
            np.linalg.norm(query_np) * np.linalg.norm(doc_np)
        ))
        
        # Create random but consistent projection
        np.random.seed(42)
        projection = np.random.normal(size=(len(query_embedding), self.num_features))
        
        # Project embeddings to feature space
        query_features = np.dot(query_np, projection)
        doc_features = np.dot(doc_np, projection)
        
        # Calculate feature-wise similarity
        feature_sim = np.multiply(query_features, doc_features)
        
        # Normalize
        feature_importance = feature_sim / np.sum(np.abs(feature_sim))
        
        return similarity_score, feature_importance
    
    def train(
        self,
        texts: List[str],
        similarity_labels: Optional[List[float]] = None,
        num_epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        model_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the NAM model for better interpretability.
        
        Args:
            texts: Training texts
            similarity_labels: Optional similarity labels for supervised training
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
            model_path: Optional path to save the trained model
            
        Returns:
            Dictionary with training information
        """
        if not HAS_NAM:
            logger.warning("Cannot train NAM model without Neural Additive Models dependency")
            return {
                "success": False,
                "error": "Neural Additive Models dependency not available",
            }
        
        try:
            # Generate base embeddings
            logger.info(f"Generating base embeddings for {len(texts)} texts")
            base_embeddings = self.base_embeddings.embed_documents(texts)
            
            # Convert to JAX arrays
            inputs = jnp.array(base_embeddings)
            
            # Create target labels if not provided
            if similarity_labels is None:
                # Use identity mapping as default training objective
                targets = inputs
            else:
                if len(similarity_labels) != len(texts):
                    raise ValueError("Number of labels must match number of texts")
                # Use provided labels
                targets = jnp.array(similarity_labels)
            
            # Get NAM model components
            model = self.nam_model["model"]
            params = self.nam_model["params"]
            device = self.nam_model["device"]
            
            # Set up optimizer
            import optax
            optimizer = optax.adam(learning_rate=learning_rate)
            opt_state = optimizer.init(params)
            
            # Define loss function
            def loss_fn(params, x, y):
                preds = model.apply(params, x)
                loss = jnp.mean(jnp.square(preds - y))
                return loss
            
            # Define update step
            @jax.jit
            def train_step(params, opt_state, x, y):
                loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss
            
            # Training loop
            logger.info(f"Training NAM model for {num_epochs} epochs")
            losses = []
            start_time = time.time()
            
            for epoch in range(num_epochs):
                # Shuffle data
                indices = np.random.permutation(len(inputs))
                
                # Process in batches
                epoch_losses = []
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_x = inputs[batch_indices]
                    batch_y = targets[batch_indices]
                    
                    # Update model
                    params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)
                    epoch_losses.append(float(loss))
                
                # Record average loss
                avg_loss = np.mean(epoch_losses)
                losses.append(avg_loss)
                
                # Log progress
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            # Update model parameters
            self.nam_model["params"] = params
            
            # Save model if path provided
            if model_path:
                self._save_model(model_path)
            
            # Return training information
            elapsed_time = time.time() - start_time
            return {
                "success": True,
                "num_epochs": num_epochs,
                "final_loss": float(losses[-1]),
                "training_time": elapsed_time,
                "losses": losses,
            }
        
        except Exception as e:
            logger.error(f"Error training NAM model: {e}")
            return {
                "success": False,
                "error": str(e),
            }


# Example usage:
# base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# interpretable_embeddings = NAMEmbeddings(base_embeddings, dimension=384, num_features=64)
# embeddings = interpretable_embeddings.embed_documents(["Sample text"])
# explanation = interpretable_embeddings.explain_similarity("query", "document")