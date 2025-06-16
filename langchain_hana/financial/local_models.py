"""
Local financial embedding models with download and fine-tuning capabilities.

This module provides functionality to download, cache, and fine-tune financial
embedding models locally for improved performance and offline usage.
"""

import os
import json
import logging
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple, Set

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# Default models directory
DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".hana_financial_models")


class LocalModelManager:
    """
    Manager for local financial embedding models.
    
    This class handles downloading, caching, and fine-tuning of financial embedding
    models for local usage, enabling offline operation and custom model training.
    """
    
    def __init__(
        self,
        models_dir: Optional[str] = None,
        auto_download: bool = True,
        use_huggingface_cache: bool = True,
        default_model: str = "FinMTEB/Fin-E5-small",
        model_aliases: Optional[Dict[str, str]] = None,
        check_updates: bool = True,
        update_interval: int = 86400,  # 24 hours
    ):
        """
        Initialize the local model manager.
        
        Args:
            models_dir: Directory to store models (None for default)
            auto_download: Whether to automatically download models
            use_huggingface_cache: Whether to use Hugging Face cache
            default_model: Default model to use
            model_aliases: Aliases for model names
            check_updates: Whether to check for model updates
            update_interval: Update check interval in seconds
        """
        self.models_dir = models_dir or DEFAULT_MODELS_DIR
        self.auto_download = auto_download
        self.use_huggingface_cache = use_huggingface_cache
        self.default_model = default_model
        self.check_updates = check_updates
        self.update_interval = update_interval
        
        # Initialize model aliases
        self.model_aliases = {
            "default": "FinMTEB/Fin-E5-small",
            "high_quality": "FinMTEB/Fin-E5",
            "efficient": "FinLang/investopedia_embedding",
            "tone": "yiyanghkust/finbert-tone",
            "financial_bert": "ProsusAI/finbert",
            "finance_base": "baconnier/Finance_embedding_large_en-V0.1",
        }
        
        # Update with custom aliases if provided
        if model_aliases:
            self.model_aliases.update(model_aliases)
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize models registry
        self.models_registry_path = os.path.join(self.models_dir, "models_registry.json")
        self.models_registry = self._load_models_registry()
        
        # Track downloaded models
        self.downloaded_models: Set[str] = set()
        
        # Detect already downloaded models
        self._detect_downloaded_models()
        
        # Check for updates if enabled
        if self.check_updates:
            self._check_model_updates()
    
    def _load_models_registry(self) -> Dict[str, Any]:
        """
        Load models registry from disk.
        
        Returns:
            Models registry
        """
        if os.path.exists(self.models_registry_path):
            try:
                with open(self.models_registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load models registry: {str(e)}")
        
        # Initialize empty registry
        return {
            "models": {},
            "aliases": self.model_aliases.copy(),
            "last_update_check": 0,
        }
    
    def _save_models_registry(self) -> None:
        """Save models registry to disk."""
        try:
            with open(self.models_registry_path, "w") as f:
                json.dump(self.models_registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save models registry: {str(e)}")
    
    def _detect_downloaded_models(self) -> None:
        """Detect already downloaded models."""
        # Add models from registry
        for model_name, model_info in self.models_registry.get("models", {}).items():
            model_path = model_info.get("path")
            if model_path and os.path.exists(model_path):
                self.downloaded_models.add(model_name)
        
        # Scan models directory for additional models
        if os.path.exists(self.models_dir):
            for entry in os.listdir(self.models_dir):
                entry_path = os.path.join(self.models_dir, entry)
                if os.path.isdir(entry_path):
                    # Check if it's a Hugging Face model directory
                    if os.path.exists(os.path.join(entry_path, "config.json")):
                        self.downloaded_models.add(entry)
        
        logger.info(f"Detected {len(self.downloaded_models)} downloaded models")
    
    def _check_model_updates(self) -> None:
        """Check for model updates."""
        import time
        
        # Check if it's time to check for updates
        last_check = self.models_registry.get("last_update_check", 0)
        if time.time() - last_check < self.update_interval:
            return
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Check for updates for each model
            for model_name in self.downloaded_models:
                if model_name.startswith("./") or os.path.isabs(model_name):
                    # Skip local paths
                    continue
                
                # Get model info from registry
                model_info = self.models_registry.get("models", {}).get(model_name, {})
                last_modified = model_info.get("last_modified", 0)
                
                # Check if model has been updated
                try:
                    # Get model info from Hugging Face
                    model_info = api.model_info(model_name)
                    new_last_modified = model_info.lastModified.timestamp()
                    
                    if new_last_modified > last_modified:
                        logger.info(f"Update available for model {model_name}")
                        
                        # Update registry
                        self.models_registry.setdefault("models", {})[model_name] = {
                            "last_modified": new_last_modified,
                            "last_checked": time.time(),
                            "update_available": True,
                        }
                except Exception as e:
                    logger.warning(f"Failed to check updates for model {model_name}: {str(e)}")
            
            # Update last check time
            self.models_registry["last_update_check"] = time.time()
            
            # Save registry
            self._save_models_registry()
            
        except ImportError:
            logger.warning("huggingface_hub not installed, skipping update check")
        except Exception as e:
            logger.warning(f"Failed to check for model updates: {str(e)}")
    
    def resolve_model_name(self, model_name: Optional[str]) -> str:
        """
        Resolve model name from alias or default.
        
        Args:
            model_name: Model name or alias
            
        Returns:
            Resolved model name
        """
        if not model_name:
            return self.default_model
        
        # Check if it's an alias
        return self.model_aliases.get(model_name, model_name)
    
    def download_model(
        self,
        model_name: str,
        force: bool = False,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> str:
        """
        Download a model from Hugging Face.
        
        Args:
            model_name: Model name or path
            force: Whether to force download even if already downloaded
            revision: Model revision
            token: Hugging Face token for private models
            
        Returns:
            Path to downloaded model
        """
        import time
        
        # Resolve model name
        model_name = self.resolve_model_name(model_name)
        
        # Check if model is already downloaded
        if not force and model_name in self.downloaded_models:
            logger.info(f"Model {model_name} already downloaded")
            return os.path.join(self.models_dir, model_name)
        
        # Local paths don't need downloading
        if model_name.startswith("./") or os.path.isabs(model_name):
            if os.path.exists(model_name):
                logger.info(f"Using local model at {model_name}")
                return model_name
            else:
                raise ValueError(f"Local model path {model_name} does not exist")
        
        try:
            # Import Hugging Face libraries
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Downloading model {model_name}...")
            start_time = time.time()
            
            # Set environment variables for Hugging Face
            if self.use_huggingface_cache:
                os.environ["HF_HOME"] = self.models_dir
                os.environ["TRANSFORMERS_CACHE"] = os.path.join(self.models_dir, "transformers")
                os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(self.models_dir, "sentence_transformers")
            
            # Set token if provided
            if token:
                os.environ["HF_TOKEN"] = token
            
            # Download model
            model_path = os.path.join(self.models_dir, model_name)
            
            # Create model directory
            os.makedirs(model_path, exist_ok=True)
            
            # Download model
            _ = SentenceTransformer(
                model_name,
                cache_folder=model_path,
                use_auth_token=token,
            )
            
            # Add to downloaded models
            self.downloaded_models.add(model_name)
            
            # Update registry
            self.models_registry.setdefault("models", {})[model_name] = {
                "path": model_path,
                "downloaded_at": time.time(),
                "last_modified": time.time(),
                "last_checked": time.time(),
                "update_available": False,
            }
            
            # Save registry
            self._save_models_registry()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Model {model_name} downloaded in {elapsed_time:.1f}s")
            
            return model_path
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import required libraries for model download: {str(e)}. "
                f"Please install sentence-transformers with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {str(e)}")
            raise
    
    def get_model_path(self, model_name: str) -> str:
        """
        Get path to downloaded model.
        
        Args:
            model_name: Model name or alias
            
        Returns:
            Path to model
        """
        # Resolve model name
        model_name = self.resolve_model_name(model_name)
        
        # Local paths don't need resolution
        if model_name.startswith("./") or os.path.isabs(model_name):
            if os.path.exists(model_name):
                return model_name
            else:
                raise ValueError(f"Local model path {model_name} does not exist")
        
        # Check if model is downloaded
        if model_name in self.downloaded_models:
            return os.path.join(self.models_dir, model_name)
        
        # Auto-download if enabled
        if self.auto_download:
            return self.download_model(model_name)
        
        raise ValueError(
            f"Model {model_name} is not downloaded and auto_download is disabled. "
            f"Please download the model first with download_model()."
        )
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all downloaded models.
        
        Returns:
            List of model information
        """
        models = []
        
        for model_name in self.downloaded_models:
            model_info = self.models_registry.get("models", {}).get(model_name, {})
            models.append({
                "name": model_name,
                "path": model_info.get("path", os.path.join(self.models_dir, model_name)),
                "downloaded_at": model_info.get("downloaded_at", 0),
                "update_available": model_info.get("update_available", False),
                "is_fine_tuned": model_info.get("is_fine_tuned", False),
                "aliases": [
                    alias for alias, target in self.model_aliases.items()
                    if target == model_name
                ],
            })
        
        return models


class FinancialDataset(Dataset):
    """Dataset for financial text data."""
    
    def __init__(self, texts: List[str], labels: Optional[List[Any]] = None):
        """
        Initialize the dataset.
        
        Args:
            texts: List of texts
            labels: Optional list of labels
        """
        self.texts = texts
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class ModelFineTuner:
    """Fine-tuner for financial embedding models."""
    
    def __init__(
        self,
        model_manager: LocalModelManager,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model_manager: Local model manager
            output_dir: Directory for fine-tuned models
        """
        self.model_manager = model_manager
        self.output_dir = output_dir or os.path.join(self.model_manager.models_dir, "fine_tuned")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fine_tune(
        self,
        base_model: str,
        train_texts: List[str],
        train_labels: Optional[List[Any]] = None,
        val_texts: Optional[List[str]] = None,
        val_labels: Optional[List[Any]] = None,
        output_model_name: Optional[str] = None,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        max_seq_length: int = 256,
        evaluation_steps: int = 100,
    ) -> str:
        """
        Fine-tune a model on financial data.
        
        Args:
            base_model: Base model name or path
            train_texts: Training texts
            train_labels: Training labels (optional)
            val_texts: Validation texts (optional)
            val_labels: Validation labels (optional)
            output_model_name: Name for fine-tuned model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            max_seq_length: Maximum sequence length
            evaluation_steps: Evaluation steps
            
        Returns:
            Path to fine-tuned model
        """
        try:
            # Import required libraries
            from sentence_transformers import SentenceTransformer, losses, InputExample
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
            
            # Resolve and download base model
            base_model_path = self.model_manager.get_model_path(base_model)
            
            # Create output model name if not provided
            if not output_model_name:
                base_name = os.path.basename(base_model_path)
                output_model_name = f"{base_name}-finetuned-financial"
            
            # Create output directory
            output_path = os.path.join(self.output_dir, output_model_name)
            os.makedirs(output_path, exist_ok=True)
            
            logger.info(f"Fine-tuning model {base_model} to {output_model_name}")
            
            # Load base model
            model = SentenceTransformer(base_model_path)
            model.max_seq_length = max_seq_length
            
            # Prepare training data
            train_examples = []
            
            if train_labels is not None:
                # Supervised learning
                for text, label in zip(train_texts, train_labels):
                    train_examples.append(InputExample(texts=[text], label=label))
                
                # Create appropriate loss function based on label type
                if isinstance(train_labels[0], float):
                    # Regression
                    train_loss = losses.CosineSimilarityLoss(model)
                else:
                    # Classification
                    train_loss = losses.SoftmaxLoss(
                        model=model,
                        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                        num_labels=len(set(train_labels)),
                    )
            else:
                # Unsupervised learning (multiple positive pairs)
                for i in range(0, len(train_texts), 2):
                    if i + 1 < len(train_texts):
                        train_examples.append(InputExample(texts=[train_texts[i], train_texts[i+1]]))
                
                train_loss = losses.MultipleNegativesRankingLoss(model)
            
            # Prepare validation data if provided
            evaluator = None
            if val_texts and val_labels:
                val_examples = []
                for text, label in zip(val_texts, val_labels):
                    val_examples.append(InputExample(texts=[text], label=label))
                
                evaluator = EmbeddingSimilarityEvaluator.from_input_examples(val_examples)
            
            # Train the model
            logger.info(f"Starting fine-tuning with {len(train_examples)} examples for {epochs} epochs")
            model.fit(
                train_objectives=[(train_examples, train_loss)],
                evaluator=evaluator,
                epochs=epochs,
                evaluation_steps=evaluation_steps,
                warmup_steps=int(len(train_examples) * 0.1),
                output_path=output_path,
                optimizer_params={"lr": learning_rate},
                batch_size=batch_size,
                show_progress_bar=True,
            )
            
            logger.info(f"Fine-tuning completed. Model saved to {output_path}")
            
            # Update model registry
            self.model_manager.downloaded_models.add(output_model_name)
            self.model_manager.models_registry.setdefault("models", {})[output_model_name] = {
                "path": output_path,
                "base_model": base_model,
                "is_fine_tuned": True,
                "created_at": time.time(),
                "train_examples": len(train_examples),
                "epochs": epochs,
            }
            
            # Save registry
            self.model_manager._save_models_registry()
            
            return output_path
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import required libraries for fine-tuning: {str(e)}. "
                f"Please install sentence-transformers with: pip install sentence-transformers"
            )
        except Exception as e:
            logger.error(f"Fine-tuning failed: {str(e)}")
            raise


# Factory function to create a local model manager
def create_local_model_manager(
    models_dir: Optional[str] = None,
    auto_download: bool = True,
    default_model: str = "FinMTEB/Fin-E5-small",
) -> LocalModelManager:
    """
    Create a local model manager.
    
    Args:
        models_dir: Directory to store models (None for default)
        auto_download: Whether to automatically download models
        default_model: Default model to use
        
    Returns:
        LocalModelManager instance
    """
    return LocalModelManager(
        models_dir=models_dir,
        auto_download=auto_download,
        default_model=default_model,
    )


# Factory function to create a model fine-tuner
def create_model_fine_tuner(
    model_manager: Optional[LocalModelManager] = None,
    models_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> ModelFineTuner:
    """
    Create a model fine-tuner.
    
    Args:
        model_manager: Local model manager (created if None)
        models_dir: Directory to store models (used if model_manager is None)
        output_dir: Directory for fine-tuned models
        
    Returns:
        ModelFineTuner instance
    """
    if model_manager is None:
        model_manager = create_local_model_manager(models_dir=models_dir)
    
    return ModelFineTuner(
        model_manager=model_manager,
        output_dir=output_dir,
    )