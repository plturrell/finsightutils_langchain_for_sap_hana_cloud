"""
Optimized hyperparameters for embedding models using opt_list.

This module provides:
1. Optimized learning rates and hyperparameters from large-scale optimization studies
2. Hyperparameter adaptation based on hardware and dataset characteristics
3. Integration with TensorFlow and PyTorch models
"""

import logging
import os
import json
import tempfile
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Conditionally import opt_list components - allow for fallback if not available
try:
    from opt_list import opt_list
    HAS_OPT_LIST = True
except ImportError:
    logger.warning("opt_list dependencies not found. Using fallback optimization mechanism.")
    HAS_OPT_LIST = False

# Conditionally import ML frameworks for wider compatibility
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    logger.warning("TensorFlow not found. TensorFlow-specific optimizations unavailable.")
    HAS_TF = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    logger.warning("PyTorch not found. PyTorch-specific optimizations unavailable.")
    HAS_TORCH = False


class OptimizedHyperparameters:
    """
    Optimized hyperparameters for embedding models.
    
    This class provides access to optimized hyperparameters derived from
    large-scale optimization studies, and adapts them based on hardware
    and dataset characteristics.
    
    Attributes:
        parameter_cache: Cache of computed hyperparameters
        cache_file: Optional path to cache computed values
        framework: ML framework to use ('tensorflow', 'pytorch', or 'auto')
    """
    
    def __init__(
        self,
        cache_file: Optional[str] = None,
        framework: str = "auto",
    ):
        """
        Initialize the OptimizedHyperparameters component.
        
        Args:
            cache_file: Optional path to cache computed values
            framework: ML framework to use ('tensorflow', 'pytorch', or 'auto')
        """
        self.parameter_cache = {}
        self.cache_file = cache_file
        
        # Determine framework
        if framework == "auto":
            if HAS_TF:
                self.framework = "tensorflow"
            elif HAS_TORCH:
                self.framework = "pytorch"
            else:
                self.framework = "numpy"
                logger.warning("No ML framework found. Using numpy fallbacks.")
        else:
            self.framework = framework
        
        logger.info(f"Using {self.framework} as backend framework")
        
        # Load cache if file exists
        if cache_file and os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.parameter_cache = json.load(f)
                logger.info(f"Loaded {len(self.parameter_cache)} cached hyperparameters")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load cache file: {e}")
    
    def get_learning_rate(
        self,
        model_size: int,
        batch_size: int,
        dataset_size: Optional[int] = None,
        training_steps: Optional[int] = None,
        optimizer_type: str = "adam",
    ) -> float:
        """
        Get optimized learning rate based on model and data characteristics.
        
        Args:
            model_size: Number of parameters in the model
            batch_size: Batch size for training
            dataset_size: Optional size of the dataset
            training_steps: Optional number of training steps
            optimizer_type: Optimizer type (adam, sgd, etc.)
            
        Returns:
            Optimized learning rate
        """
        # Generate cache key
        cache_key = f"lr_{model_size}_{batch_size}_{dataset_size}_{training_steps}_{optimizer_type}"
        
        # Check cache
        if cache_key in self.parameter_cache:
            return self.parameter_cache[cache_key]
        
        # Use opt_list if available
        if HAS_OPT_LIST:
            try:
                # Calculate steps per epoch if dataset size is provided
                steps_per_epoch = None
                if dataset_size is not None and batch_size > 0:
                    steps_per_epoch = max(1, dataset_size // batch_size)
                
                # Get learning rate from opt_list
                learning_rate = opt_list.get_learning_rate(
                    model_size=model_size,
                    batch_size=batch_size,
                    steps_per_epoch=steps_per_epoch,
                    optimizer_name=optimizer_type,
                )
                
                # Cache result
                self.parameter_cache[cache_key] = learning_rate
                self._save_cache()
                
                return learning_rate
            
            except Exception as e:
                logger.error(f"Error getting learning rate from opt_list: {e}")
                # Fall back to default calculation
                learning_rate = self._default_learning_rate(
                    model_size, batch_size, dataset_size, training_steps, optimizer_type
                )
                
                # Cache result
                self.parameter_cache[cache_key] = learning_rate
                self._save_cache()
                
                return learning_rate
        else:
            # Use default calculation
            learning_rate = self._default_learning_rate(
                model_size, batch_size, dataset_size, training_steps, optimizer_type
            )
            
            # Cache result
            self.parameter_cache[cache_key] = learning_rate
            self._save_cache()
            
            return learning_rate
    
    def _default_learning_rate(
        self,
        model_size: int,
        batch_size: int,
        dataset_size: Optional[int],
        training_steps: Optional[int],
        optimizer_type: str,
    ) -> float:
        """
        Calculate default learning rate based on model and data characteristics.
        
        Args:
            model_size: Number of parameters in the model
            batch_size: Batch size for training
            dataset_size: Optional size of the dataset
            training_steps: Optional number of training steps
            optimizer_type: Optimizer type (adam, sgd, etc.)
            
        Returns:
            Default learning rate
        """
        # Base learning rates for different optimizers
        base_rates = {
            "adam": 0.001,
            "sgd": 0.01,
            "rmsprop": 0.0001,
            "adagrad": 0.01,
            "adadelta": 1.0,
        }
        
        # Get base rate for optimizer (default to adam)
        base_rate = base_rates.get(optimizer_type.lower(), 0.001)
        
        # Adjust for model size (larger models need smaller learning rates)
        size_factor = 1.0
        if model_size > 1e8:  # Very large model
            size_factor = 0.1
        elif model_size > 1e7:  # Large model
            size_factor = 0.3
        elif model_size > 1e6:  # Medium model
            size_factor = 0.5
        
        # Adjust for batch size (larger batches need larger learning rates)
        batch_factor = min(1.0, np.sqrt(batch_size / 32))
        
        # Combine factors
        learning_rate = base_rate * size_factor * batch_factor
        
        return learning_rate
    
    def get_optimizer(
        self,
        model_size: int,
        batch_size: int,
        dataset_size: Optional[int] = None,
        training_steps: Optional[int] = None,
    ) -> Any:
        """
        Get optimized optimizer based on model and data characteristics.
        
        Args:
            model_size: Number of parameters in the model
            batch_size: Batch size for training
            dataset_size: Optional size of the dataset
            training_steps: Optional number of training steps
            
        Returns:
            Optimizer object for the selected framework
        """
        # Get optimized learning rate
        learning_rate = self.get_learning_rate(
            model_size=model_size,
            batch_size=batch_size,
            dataset_size=dataset_size,
            training_steps=training_steps,
        )
        
        # Create optimizer based on framework
        if self.framework == "tensorflow" and HAS_TF:
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        elif self.framework == "pytorch" and HAS_TORCH:
            return torch.optim.Adam([], lr=learning_rate)
        
        else:
            # Return parameters for generic optimizer
            return {
                "optimizer_type": "adam",
                "learning_rate": learning_rate,
            }
    
    def get_batch_size(
        self,
        model_size: int,
        max_memory: Optional[int] = None,
        dataset_size: Optional[int] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
    ) -> int:
        """
        Get optimized batch size based on model and hardware characteristics.
        
        Args:
            model_size: Number of parameters in the model
            max_memory: Optional maximum memory in bytes
            dataset_size: Optional size of the dataset
            input_shape: Optional shape of input tensors
            
        Returns:
            Optimized batch size
        """
        # Generate cache key
        cache_key = f"bs_{model_size}_{max_memory}_{dataset_size}"
        
        # Add input shape to cache key if provided
        if input_shape is not None:
            cache_key += f"_{input_shape}"
        
        # Check cache
        if cache_key in self.parameter_cache:
            return self.parameter_cache[cache_key]
        
        # Determine max memory if not provided
        if max_memory is None:
            if HAS_TF:
                # Get available GPU memory
                gpus = tf.config.list_physical_devices('GPU')
                if gpus:
                    try:
                        gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                        max_memory = gpu_memory['free']
                    except Exception:
                        # Fall back to conservative estimate
                        max_memory = 4 * 1024 * 1024 * 1024  # 4 GB
                else:
                    # CPU memory (conservative estimate)
                    max_memory = 8 * 1024 * 1024 * 1024  # 8 GB
            elif HAS_TORCH:
                # Get available GPU memory
                if torch.cuda.is_available():
                    device = torch.cuda.current_device()
                    max_memory = torch.cuda.get_device_properties(device).total_memory
                else:
                    # CPU memory (conservative estimate)
                    max_memory = 8 * 1024 * 1024 * 1024  # 8 GB
            else:
                # Conservative estimate
                max_memory = 4 * 1024 * 1024 * 1024  # 4 GB
        
        # Calculate bytes per sample
        bytes_per_sample = 4  # Float32 by default
        if input_shape is not None:
            # Calculate size based on input shape
            sample_size = np.prod(input_shape)
            bytes_per_sample = sample_size * 4  # Assuming float32
        else:
            # Estimate based on model size
            bytes_per_sample = max(4, model_size // 1000)
        
        # Calculate maximum batch size based on memory
        max_batch_size = max(1, max_memory // (bytes_per_sample * 4))  # Factor of 4 for gradient storage
        
        # Limit to powers of 2 for better performance
        batch_size = 2 ** int(np.log2(max_batch_size))
        
        # Adjust based on dataset size if provided
        if dataset_size is not None:
            # Ensure batch size is not too large relative to dataset
            batch_size = min(batch_size, max(1, dataset_size // 10))
        
        # Ensure reasonable limits
        batch_size = max(1, min(batch_size, 1024))
        
        # Cache result
        self.parameter_cache[cache_key] = batch_size
        self._save_cache()
        
        return batch_size
    
    def get_embedding_parameters(
        self,
        embedding_dimension: int,
        vocabulary_size: int,
        max_sequence_length: int,
    ) -> Dict[str, Any]:
        """
        Get optimized parameters for embedding models.
        
        Args:
            embedding_dimension: Dimension of embeddings
            vocabulary_size: Size of vocabulary
            max_sequence_length: Maximum sequence length
            
        Returns:
            Dictionary of optimized parameters
        """
        # Generate cache key
        cache_key = f"emb_{embedding_dimension}_{vocabulary_size}_{max_sequence_length}"
        
        # Check cache
        if cache_key in self.parameter_cache:
            return self.parameter_cache[cache_key]
        
        # Calculate model size
        model_size = embedding_dimension * vocabulary_size
        
        # Use opt_list if available
        if HAS_OPT_LIST:
            try:
                # Get learning rate from opt_list
                learning_rate = opt_list.get_learning_rate(
                    model_size=model_size,
                    batch_size=32,  # Default batch size
                    optimizer_name="adam",
                )
                
                # Get hyperparameters
                params = {
                    "learning_rate": learning_rate,
                    "dropout_rate": min(0.1 + (model_size / 1e9), 0.5),  # Scale dropout with model size
                    "weight_decay": 0.01,
                    "embedding_dropout": min(0.05 + (model_size / 1e9), 0.3),
                    "layer_norm_epsilon": 1e-6,
                    "hidden_dimension": embedding_dimension * 4,
                }
                
            except Exception as e:
                logger.error(f"Error getting parameters from opt_list: {e}")
                # Fall back to default calculation
                params = self._default_embedding_parameters(
                    embedding_dimension, vocabulary_size, max_sequence_length
                )
        else:
            # Use default calculation
            params = self._default_embedding_parameters(
                embedding_dimension, vocabulary_size, max_sequence_length
            )
        
        # Cache result
        self.parameter_cache[cache_key] = params
        self._save_cache()
        
        return params
    
    def _default_embedding_parameters(
        self,
        embedding_dimension: int,
        vocabulary_size: int,
        max_sequence_length: int,
    ) -> Dict[str, Any]:
        """
        Calculate default parameters for embedding models.
        
        Args:
            embedding_dimension: Dimension of embeddings
            vocabulary_size: Size of vocabulary
            max_sequence_length: Maximum sequence length
            
        Returns:
            Dictionary of default parameters
        """
        # Calculate model size
        model_size = embedding_dimension * vocabulary_size
        
        # Scale learning rate based on model size
        if model_size < 1e6:
            learning_rate = 1e-3
        elif model_size < 1e8:
            learning_rate = 5e-4
        else:
            learning_rate = 1e-4
        
        # Scale dropout with model size
        dropout_rate = min(0.1 + (model_size / 1e9), 0.5)
        
        # Calculate hidden dimension
        hidden_dimension = embedding_dimension * 4
        
        # Return parameters
        return {
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "weight_decay": 0.01,
            "embedding_dropout": min(0.05 + (model_size / 1e9), 0.3),
            "layer_norm_epsilon": 1e-6,
            "hidden_dimension": hidden_dimension,
        }
    
    def get_training_schedule(
        self,
        model_size: int,
        dataset_size: int,
        batch_size: int,
        target_epochs: int = 10,
    ) -> Dict[str, Any]:
        """
        Get optimized training schedule.
        
        Args:
            model_size: Number of parameters in the model
            dataset_size: Size of the dataset
            batch_size: Batch size for training
            target_epochs: Target number of epochs
            
        Returns:
            Dictionary with training schedule parameters
        """
        # Generate cache key
        cache_key = f"sched_{model_size}_{dataset_size}_{batch_size}_{target_epochs}"
        
        # Check cache
        if cache_key in self.parameter_cache:
            return self.parameter_cache[cache_key]
        
        # Calculate steps per epoch
        steps_per_epoch = max(1, dataset_size // batch_size)
        
        # Total steps
        total_steps = steps_per_epoch * target_epochs
        
        # Warmup steps (10% of total steps, or 100, whichever is smaller)
        warmup_steps = min(100, total_steps // 10)
        
        # Get base learning rate
        base_lr = self.get_learning_rate(
            model_size=model_size,
            batch_size=batch_size,
            dataset_size=dataset_size,
        )
        
        # Learning rate schedule
        schedule = {
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "base_learning_rate": base_lr,
            "min_learning_rate": base_lr / 100,
            "warmup_learning_rate": base_lr / 10,
        }
        
        # Cache result
        self.parameter_cache[cache_key] = schedule
        self._save_cache()
        
        return schedule
    
    def create_learning_rate_schedule(
        self,
        schedule_params: Dict[str, Any],
    ) -> Any:
        """
        Create a learning rate schedule based on the framework.
        
        Args:
            schedule_params: Parameters for the schedule
            
        Returns:
            Learning rate schedule object for the selected framework
        """
        if self.framework == "tensorflow" and HAS_TF:
            return self._create_tf_lr_schedule(schedule_params)
        
        elif self.framework == "pytorch" and HAS_TORCH:
            return self._create_torch_lr_schedule(schedule_params)
        
        else:
            # Return a callable for generic scheduler
            return self._create_generic_lr_schedule(schedule_params)
    
    def _create_tf_lr_schedule(self, params: Dict[str, Any]) -> tf.keras.optimizers.schedules.LearningRateSchedule:
        """
        Create a TensorFlow learning rate schedule.
        
        Args:
            params: Schedule parameters
            
        Returns:
            TensorFlow learning rate schedule
        """
        if not HAS_TF:
            raise ImportError("TensorFlow is not available")
        
        # Extract parameters
        warmup_steps = params.get("warmup_steps", 100)
        total_steps = params.get("total_steps", 1000)
        base_lr = params.get("base_learning_rate", 0.001)
        min_lr = params.get("min_learning_rate", 0.00001)
        warmup_lr = params.get("warmup_learning_rate", 0.0001)
        
        # Create custom learning rate schedule
        class CustomLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
            def __init__(
                self,
                warmup_steps,
                total_steps,
                base_lr,
                min_lr,
                warmup_lr,
            ):
                super().__init__()
                self.warmup_steps = warmup_steps
                self.total_steps = total_steps
                self.base_lr = base_lr
                self.min_lr = min_lr
                self.warmup_lr = warmup_lr
            
            def __call__(self, step):
                # Warmup phase
                warmup_lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (
                    tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32)
                )
                
                # Decay phase
                decay_steps = self.total_steps - self.warmup_steps
                decay_factor = (self.base_lr - self.min_lr) / decay_steps
                decay_lr = self.base_lr - decay_factor * (
                    tf.cast(step, tf.float32) - tf.cast(self.warmup_steps, tf.float32)
                )
                
                # Combine phases
                lr = tf.where(
                    step < self.warmup_steps,
                    warmup_lr,
                    decay_lr,
                )
                
                # Ensure minimum learning rate
                lr = tf.maximum(self.min_lr, lr)
                
                return lr
            
            def get_config(self):
                return {
                    "warmup_steps": self.warmup_steps,
                    "total_steps": self.total_steps,
                    "base_lr": self.base_lr,
                    "min_lr": self.min_lr,
                    "warmup_lr": self.warmup_lr,
                }
        
        return CustomLRSchedule(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            base_lr=base_lr,
            min_lr=min_lr,
            warmup_lr=warmup_lr,
        )
    
    def _create_torch_lr_schedule(self, params: Dict[str, Any]) -> Callable:
        """
        Create a PyTorch learning rate schedule.
        
        Args:
            params: Schedule parameters
            
        Returns:
            Function for PyTorch LambdaLR
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is not available")
        
        # Extract parameters
        warmup_steps = params.get("warmup_steps", 100)
        total_steps = params.get("total_steps", 1000)
        base_lr = params.get("base_learning_rate", 0.001)
        min_lr = params.get("min_learning_rate", 0.00001)
        warmup_lr = params.get("warmup_learning_rate", 0.0001)
        
        # Create schedule function
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Warmup phase
                factor = warmup_lr / base_lr + (1.0 - warmup_lr / base_lr) * (
                    current_step / max(1, warmup_steps)
                )
            else:
                # Decay phase
                decay_steps = total_steps - warmup_steps
                decay_factor = (base_lr - min_lr) / base_lr / max(1, decay_steps)
                factor = 1.0 - decay_factor * (current_step - warmup_steps)
                factor = max(min_lr / base_lr, factor)
            
            return factor
        
        return lr_lambda
    
    def _create_generic_lr_schedule(self, params: Dict[str, Any]) -> Callable:
        """
        Create a generic learning rate schedule function.
        
        Args:
            params: Schedule parameters
            
        Returns:
            Function that computes learning rate for a given step
        """
        # Extract parameters
        warmup_steps = params.get("warmup_steps", 100)
        total_steps = params.get("total_steps", 1000)
        base_lr = params.get("base_learning_rate", 0.001)
        min_lr = params.get("min_learning_rate", 0.00001)
        warmup_lr = params.get("warmup_learning_rate", 0.0001)
        
        # Create schedule function
        def lr_schedule(step):
            if step < warmup_steps:
                # Warmup phase
                return warmup_lr + (base_lr - warmup_lr) * (step / max(1, warmup_steps))
            else:
                # Decay phase
                decay_steps = total_steps - warmup_steps
                decay_factor = (base_lr - min_lr) / max(1, decay_steps)
                return max(min_lr, base_lr - decay_factor * (step - warmup_steps))
        
        return lr_schedule
    
    def _save_cache(self) -> None:
        """Save cache to file if path is specified."""
        if self.cache_file:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.parameter_cache, f)
                logger.info(f"Saved {len(self.parameter_cache)} hyperparameters to cache")
            except IOError as e:
                logger.warning(f"Failed to save cache file: {e}")


# Example usage:
# optimizer = OptimizedHyperparameters(cache_file="hyperparams.json")
# learning_rate = optimizer.get_learning_rate(model_size=1e7, batch_size=32)
# batch_size = optimizer.get_batch_size(model_size=1e7, max_memory=8e9)
# embedding_params = optimizer.get_embedding_parameters(768, 50000, 512)
# schedule = optimizer.get_training_schedule(model_size=1e7, dataset_size=100000, batch_size=32)