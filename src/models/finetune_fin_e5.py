#!/usr/bin/env python3
"""
Fine-tune FinMTEB/Fin-E5 model for specialized financial applications.

This script handles the entire fine-tuning process for the FinMTEB/Fin-E5 model,
including data preparation, training, evaluation, and model saving.
"""

import os
import sys
import json
import time
import logging
import argparse
import tempfile
from typing import List, Dict, Any, Optional, Tuple
import multiprocessing as mp
from contextlib import contextmanager

from langchain_hana.financial.local_models import (
    create_local_model_manager,
    create_model_fine_tuner,
)

# Configure logging
def setup_logging(log_level=None, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Log file path (default: finetune.log)
    """
    if log_level is None:
        log_level = os.environ.get("FINETUNE_LOG_LEVEL", "INFO").upper()
    
    # Convert string log level to numeric value
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Set up log file
    if log_file is None:
        log_file = os.environ.get("FINETUNE_LOG_FILE", "finetune.log")
    
    # Create log directory if needed
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    
    # Return logger for this module
    return logging.getLogger(__name__)

# Initialize logger
logger = setup_logging()

# Global variables for training progress tracking
_PROGRESS_FILE = os.path.join(tempfile.gettempdir(), "fin_e5_training_progress.json")
_METRICS_FILE = os.path.join(tempfile.gettempdir(), "fin_e5_training_metrics.json")

# Create process-safe queues for IPC
progress_queue = None
metrics_queue = None

@contextmanager
def training_progress_tracker(total_steps, metrics_labels=None):
    """Context manager to track training progress and metrics.
    
    Args:
        total_steps: Total number of training steps
        metrics_labels: Labels for metrics to track
    """
    # Initialize progress file
    progress_data = {
        "status": "initializing",
        "progress": 0.0,
        "step": 0,
        "total_steps": total_steps,
        "stage": "preparation",
        "started_at": time.time(),
        "estimated_completion": None,
        "messages": []
    }
    
    # Initialize metrics file
    metrics_data = {
        "loss": [],
        "evaluation": [],
        "similarity_scores": [],
        "batch_time": [],
        "learning_rate": []
    }
    
    # Add custom metrics if provided
    if metrics_labels:
        for label in metrics_labels:
            if label not in metrics_data:
                metrics_data[label] = []
    
    # Write initial data
    with open(_PROGRESS_FILE, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    with open(_METRICS_FILE, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    try:
        yield _update_progress, _update_metrics
    finally:
        # Update status to completed
        progress_data["status"] = "completed"
        progress_data["progress"] = 1.0
        progress_data["messages"].append("Training completed")
        
        with open(_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)

def _update_progress(step=None, progress=None, stage=None, message=None, reset=False):
    """Update training progress.
    
    Args:
        step: Current training step
        progress: Current progress (0.0-1.0)
        stage: Current training stage
        message: Message to add to progress
        reset: Whether to reset progress
    """
    try:
        # Read current progress
        if os.path.exists(_PROGRESS_FILE) and not reset:
            with open(_PROGRESS_FILE, 'r') as f:
                progress_data = json.load(f)
        else:
            progress_data = {
                "status": "running",
                "progress": 0.0,
                "step": 0,
                "total_steps": 1,
                "stage": "preparation",
                "started_at": time.time(),
                "estimated_completion": None,
                "messages": []
            }
        
        # Update values
        progress_data["status"] = "running"
        
        if step is not None:
            progress_data["step"] = step
            if progress_data["total_steps"] > 0:
                progress_data["progress"] = min(1.0, max(0.0, step / progress_data["total_steps"]))
        
        if progress is not None:
            progress_data["progress"] = min(1.0, max(0.0, progress))
        
        if stage is not None:
            progress_data["stage"] = stage
        
        if message is not None:
            progress_data["messages"].append(message)
            # Keep only the last 10 messages
            progress_data["messages"] = progress_data["messages"][-10:]
        
        # Calculate estimated completion time
        if progress_data["progress"] > 0:
            elapsed = time.time() - progress_data["started_at"]
            total_estimated = elapsed / progress_data["progress"]
            remaining = total_estimated - elapsed
            progress_data["estimated_completion"] = time.time() + remaining
        
        # Write updated progress
        with open(_PROGRESS_FILE, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        # Also send via queue if available
        global progress_queue
        if progress_queue is not None:
            progress_queue.put(progress_data)
    
    except Exception as e:
        logger.warning(f"Failed to update progress: {str(e)}")

def _update_metrics(metrics_dict):
    """Update training metrics.
    
    Args:
        metrics_dict: Dictionary of metrics to update
    """
    try:
        # Read current metrics
        if os.path.exists(_METRICS_FILE):
            with open(_METRICS_FILE, 'r') as f:
                metrics_data = json.load(f)
        else:
            metrics_data = {
                "loss": [],
                "evaluation": [],
                "similarity_scores": [],
                "batch_time": [],
                "learning_rate": []
            }
        
        # Update metrics
        for key, value in metrics_dict.items():
            if key in metrics_data:
                if isinstance(value, list):
                    metrics_data[key].extend(value)
                else:
                    metrics_data[key].append(value)
        
        # Write updated metrics
        with open(_METRICS_FILE, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Also send via queue if available
        global metrics_queue
        if metrics_queue is not None:
            metrics_queue.put(metrics_dict)
    
    except Exception as e:
        logger.warning(f"Failed to update metrics: {str(e)}")

def initialize_ipc(with_queue=True):
    """Initialize IPC mechanisms if needed.
    
    Args:
        with_queue: Whether to create process queues
    """
    global progress_queue, metrics_queue
    
    if with_queue:
        # Create queues
        progress_queue = mp.Queue()
        metrics_queue = mp.Queue()
        
        return progress_queue, metrics_queue
    
    return None, None


def prepare_training_data(args) -> Tuple[List[str], Optional[List[Any]], List[str], Optional[List[Any]]]:
    """
    Prepare training and validation data from input files.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels)
    """
    train_texts = []
    train_labels = []
    val_texts = []
    val_labels = []
    
    # Load training data
    logger.info(f"Loading training data from {args.train_file}")
    with open(args.train_file, 'r') as f:
        train_data = json.load(f)
    
    # Process training data based on format
    if args.training_format == "pairs":
        # Format: [{"text1": "...", "text2": "...", "score": 0.92}, ...]
        for item in train_data:
            train_texts.append(item["text1"])
            train_texts.append(item["text2"])
            if "score" in item:
                train_labels.append(item["score"])
                train_labels.append(item["score"])
    
    elif args.training_format == "documents":
        # Format: [{"content": "...", "metadata": {...}}, ...]
        for item in train_data:
            train_texts.append(item["content"])
            if "label" in item:
                train_labels.append(item["label"])
    
    else:  # simple
        # Format: ["text1", "text2", ...]
        train_texts = train_data
        train_labels = None
    
    # Load validation data if provided
    if args.val_file:
        logger.info(f"Loading validation data from {args.val_file}")
        with open(args.val_file, 'r') as f:
            val_data = json.load(f)
        
        # Process validation data based on format
        if args.training_format == "pairs":
            for item in val_data:
                val_texts.append(item["text1"])
                val_texts.append(item["text2"])
                if "score" in item:
                    val_labels.append(item["score"])
                    val_labels.append(item["score"])
        
        elif args.training_format == "documents":
            for item in val_data:
                val_texts.append(item["content"])
                if "label" in item:
                    val_labels.append(item["label"])
        
        else:  # simple
            val_texts = val_data
            val_labels = None
    
    # Validate data
    if len(train_texts) == 0:
        raise ValueError("No training texts found in input file")
    
    if train_labels and len(train_labels) != len(train_texts):
        raise ValueError(f"Number of labels ({len(train_labels)}) doesn't match number of texts ({len(train_texts)})")
    
    if val_labels and len(val_labels) != len(val_texts):
        raise ValueError(f"Number of validation labels ({len(val_labels)}) doesn't match number of validation texts ({len(val_texts)})")
    
    logger.info(f"Prepared {len(train_texts)} training texts and {len(val_texts)} validation texts")
    
    return train_texts, train_labels, val_texts, val_labels


def download_base_model(args) -> str:
    """
    Download the base model for fine-tuning.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to downloaded model
    """
    logger.info(f"Downloading base model: {args.base_model}")
    
    # Create model manager
    model_manager = create_local_model_manager(
        models_dir=args.models_dir,
        auto_download=True,
        default_model=args.base_model,
    )
    
    # Download model
    model_path = model_manager.download_model(
        model_name=args.base_model,
        force=args.force_download,
    )
    
    logger.info(f"Base model downloaded to: {model_path}")
    return model_path


def fine_tune_model(args, model_path: str, train_texts: List[str], train_labels: Optional[List[Any]],
                   val_texts: List[str], val_labels: Optional[List[Any]]) -> str:
    """
    Fine-tune the model on provided data.
    
    Args:
        args: Command line arguments
        model_path: Path to base model
        train_texts: Training texts
        train_labels: Training labels (optional)
        val_texts: Validation texts
        val_labels: Validation labels (optional)
        
    Returns:
        Path to fine-tuned model
    """
    logger.info("Initializing fine-tuning process")
    
    # Create model manager
    model_manager = create_local_model_manager(
        models_dir=args.models_dir,
        auto_download=False,  # Already downloaded
    )
    
    # Create fine-tuner
    fine_tuner = create_model_fine_tuner(
        model_manager=model_manager,
        output_dir=args.output_dir,
    )
    
    # Configure fine-tuning parameters
    logger.info(f"Fine-tuning with: epochs={args.epochs}, batch_size={args.batch_size}, learning_rate={args.learning_rate}")
    
    # Calculate total steps for progress tracking
    num_train_examples = len(train_texts)
    steps_per_epoch = num_train_examples // args.batch_size
    if num_train_examples % args.batch_size > 0:
        steps_per_epoch += 1
    total_steps = steps_per_epoch * args.epochs
    
    # Fine-tune the model with progress tracking
    start_time = time.time()
    
    # Custom callback to track progress
    class ProgressCallback:
        def __init__(self, update_progress, update_metrics, total_steps):
            self.update_progress = update_progress
            self.update_metrics = update_metrics
            self.total_steps = total_steps
            self.current_step = 0
            self.current_epoch = 0
            self.batch_times = []
            self.last_batch_start = None
            self.stages = [
                "Understanding financial terms and concepts",
                "Learning financial relationships and context",
                "Refining financial pattern recognition",
                "Optimizing financial comprehension"
            ]
            
        def on_train_start(self, **kwargs):
            self.update_progress(
                step=0,
                progress=0.0,
                stage=self.stages[0],
                message="Starting fine-tuning process"
            )
            
        def on_epoch_start(self, epoch, **kwargs):
            self.current_epoch = epoch
            stage_idx = min(epoch, len(self.stages) - 1)
            self.update_progress(
                progress=epoch / args.epochs,
                stage=self.stages[stage_idx],
                message=f"Starting epoch {epoch+1}/{args.epochs}"
            )
            self.last_batch_start = time.time()
            
        def on_step_end(self, step, **kwargs):
            self.current_step += 1
            overall_step = (self.current_epoch * steps_per_epoch) + step
            
            # Calculate batch time
            if self.last_batch_start:
                batch_time = time.time() - self.last_batch_start
                self.batch_times.append(batch_time)
                self.last_batch_start = time.time()
                
                # Update metrics with batch time
                self.update_metrics({"batch_time": batch_time})
            
            # Update progress every few steps to avoid file I/O overhead
            if step % 10 == 0 or step == steps_per_epoch - 1:
                progress = min(1.0, overall_step / self.total_steps)
                stage_idx = min(self.current_epoch, len(self.stages) - 1)
                
                self.update_progress(
                    step=overall_step,
                    progress=progress,
                    stage=self.stages[stage_idx]
                )
                
        def on_evaluation_end(self, scores, **kwargs):
            # Update metrics with evaluation scores
            self.update_metrics({
                "evaluation": scores,
                "similarity_scores": scores.get("cosine_similarity", {}).get("scores", [])
            })
            
            # Add evaluation message
            if "cosine_similarity" in scores:
                pearson = scores["cosine_similarity"].get("pearson", 0)
                spearman = scores["cosine_similarity"].get("spearman", 0)
                self.update_progress(
                    message=f"Evaluation: Pearson={pearson:.4f}, Spearman={spearman:.4f}"
                )
            
        def on_train_end(self, **kwargs):
            # Calculate average batch time
            avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
            
            self.update_progress(
                progress=1.0,
                stage="Completed",
                message=f"Training completed in {time.time() - start_time:.1f}s, avg batch: {avg_batch_time:.3f}s"
            )
            
        def on_batch_end(self, batch, loss, **kwargs):
            # Update metrics with loss
            self.update_metrics({"loss": float(loss)})
            
            # Update progress message occasionally
            if batch % 50 == 0:
                self.update_progress(
                    message=f"Batch {batch}: loss={float(loss):.4f}"
                )
    
    # Use context manager for progress tracking
    with training_progress_tracker(total_steps) as (update_progress, update_metrics):
        # Initialize callback
        progress_callback = ProgressCallback(update_progress, update_metrics, total_steps)
        
        # Register hooks with sentence-transformers
        # We'll use monkey patching to inject our callback
        try:
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.SentenceTransformer import SentenceTransformer as STModel
            
            # Save original methods
            original_fit = STModel.fit
            
            # Define patched methods
            def patched_fit(self, *args, **kwargs):
                # Call our hooks
                if hasattr(self, '_progress_callback'):
                    self._progress_callback.on_train_start()
                
                # Add our custom callback to the kwargs
                if 'callback' not in kwargs and hasattr(self, '_progress_callback'):
                    kwargs['callback'] = lambda model, score, epoch, steps: (
                        self._progress_callback.on_evaluation_end(score)
                    )
                
                # Patch the epoch callback
                original_epoch_callback = kwargs.get('epoch_callback')
                
                def patched_epoch_callback(model, outputs, epoch):
                    # Call original callback if exists
                    if original_epoch_callback:
                        original_epoch_callback(model, outputs, epoch)
                    
                    # Call our hook
                    if hasattr(model, '_progress_callback'):
                        model._progress_callback.on_epoch_start(epoch)
                
                kwargs['epoch_callback'] = patched_epoch_callback
                
                # Patch the update callback
                original_update_callback = kwargs.get('update_callback')
                
                def patched_update_callback(model, batch, loss, step):
                    # Call original callback if exists
                    if original_update_callback:
                        original_update_callback(model, batch, loss, step)
                    
                    # Call our hooks
                    if hasattr(model, '_progress_callback'):
                        model._progress_callback.on_batch_end(step, loss)
                        model._progress_callback.on_step_end(step)
                
                kwargs['update_callback'] = patched_update_callback
                
                # Call original method
                result = original_fit(self, *args, **kwargs)
                
                # Call our hooks
                if hasattr(self, '_progress_callback'):
                    self._progress_callback.on_train_end()
                
                return result
            
            # Apply monkey patch
            STModel.fit = patched_fit
            
            # Create and fine-tune model
            model = SentenceTransformer(model_path)
            model._progress_callback = progress_callback
            
            # Fine-tune the model
            update_progress(stage="Loading model and data", message="Preparing for fine-tuning")
            
            tuned_model_path = fine_tuner.fine_tune(
                base_model=model_path,
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts if val_texts else None,
                val_labels=val_labels if val_labels else None,
                output_model_name=args.output_model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_seq_length=args.max_seq_length,
                evaluation_steps=args.evaluation_steps,
            )
            
            # Restore original method
            STModel.fit = original_fit
            
        except ImportError:
            # Fall back to original approach if patching fails
            update_progress(stage="Loading model and data", message="Preparing for fine-tuning")
            
            tuned_model_path = fine_tuner.fine_tune(
                base_model=model_path,
                train_texts=train_texts,
                train_labels=train_labels,
                val_texts=val_texts if val_texts else None,
                val_labels=val_labels if val_labels else None,
                output_model_name=args.output_model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                max_seq_length=args.max_seq_length,
                evaluation_steps=args.evaluation_steps,
            )
            
            # Update progress manually in fallback mode
            for epoch in range(args.epochs):
                stage_idx = min(epoch, len(progress_callback.stages) - 1)
                update_progress(
                    progress=(epoch + 1) / args.epochs,
                    stage=progress_callback.stages[stage_idx],
                    message=f"Completed epoch {epoch+1}/{args.epochs}"
                )
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Fine-tuning completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Fine-tuned model saved to: {tuned_model_path}")
    
    return tuned_model_path


def test_model(args, model_path: str) -> None:
    """
    Test the fine-tuned model on sample queries.
    
    Args:
        args: Command line arguments
        model_path: Path to fine-tuned model
    """
    logger.info("Testing fine-tuned model")
    
    try:
        # Import necessary libraries
        from sentence_transformers import SentenceTransformer
        import numpy as np
        
        # Load the model
        model = SentenceTransformer(model_path)
        
        # Sample financial queries
        sample_queries = [
            "What were the quarterly earnings results?",
            "How has market volatility affected investment performance?",
            "What are the key financial risks in the current economic environment?",
            "What regulatory changes are impacting the financial sector?",
            "What investment opportunities have positive growth potential?",
        ]
        
        # Embed queries
        logger.info("Generating embeddings for sample queries")
        embeddings = model.encode(sample_queries, convert_to_numpy=True)
        
        # Calculate similarities between queries
        logger.info("Calculating similarities between queries")
        similarities = []
        for i in range(len(sample_queries)):
            for j in range(i+1, len(sample_queries)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarities.append((sample_queries[i], sample_queries[j], similarity))
        
        # Sort similarities
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Display similarities
        logger.info("Query similarities (most similar to least similar):")
        for query1, query2, score in similarities:
            logger.info(f"- {score:.4f}: \"{query1}\" <-> \"{query2}\"")
        
        # Calculate average embedding dimension
        avg_norm = np.mean([np.linalg.norm(emb) for emb in embeddings])
        
        logger.info(f"Model information:")
        logger.info(f"- Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"- Average embedding norm: {avg_norm:.4f}")
        
    except ImportError as e:
        logger.error(f"Import error during testing: {e}")
    except Exception as e:
        logger.error(f"Error testing model: {e}")


def validate_args(args):
    """
    Validate command line arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Validated arguments
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Train file
    if not os.path.exists(args.train_file):
        raise ValueError(f"Training file not found: {args.train_file}")
    
    # Validation file
    if args.val_file and not os.path.exists(args.val_file):
        raise ValueError(f"Validation file not found: {args.val_file}")
    
    # Training format
    if args.training_format not in ["pairs", "documents", "simple"]:
        raise ValueError(f"Invalid training format: {args.training_format}")
    
    # Epochs
    if args.epochs < 1:
        raise ValueError(f"Epochs must be at least 1, got {args.epochs}")
    
    # Batch size
    if args.batch_size < 1:
        raise ValueError(f"Batch size must be at least 1, got {args.batch_size}")
    
    # Learning rate
    if args.learning_rate <= 0:
        raise ValueError(f"Learning rate must be positive, got {args.learning_rate}")
    
    # Max sequence length
    if args.max_seq_length < 1:
        raise ValueError(f"Max sequence length must be at least 1, got {args.max_seq_length}")
    
    # Evaluation steps
    if args.evaluation_steps < 1:
        raise ValueError(f"Evaluation steps must be at least 1, got {args.evaluation_steps}")
    
    # Progress and metrics files
    if args.progress_file:
        progress_dir = os.path.dirname(os.path.abspath(args.progress_file))
        if not os.path.exists(progress_dir):
            raise ValueError(f"Directory for progress file does not exist: {progress_dir}")
    
    if args.metrics_file:
        metrics_dir = os.path.dirname(os.path.abspath(args.metrics_file))
        if not os.path.exists(metrics_dir):
            raise ValueError(f"Directory for metrics file does not exist: {metrics_dir}")
    
    # Visualization mode
    if args.visualization_mode not in ["full", "minimal", "none"]:
        raise ValueError(f"Invalid visualization mode: {args.visualization_mode}")
    
    return args


def main(args):
    """Main function."""
    try:
        # Validate arguments
        args = validate_args(args)
        logger.info("Arguments validated successfully")
        logger.info(f"Training with parameters: epochs={args.epochs}, batch_size={args.batch_size}, "
                    f"learning_rate={args.learning_rate}, max_seq_length={args.max_seq_length}")
        
        # Initialize IPC if requested
        if args.enable_ipc:
            initialize_ipc(with_queue=True)
            logger.info("IPC initialized with multiprocessing queues")
        
        # Clean up existing progress files
        for file_path in [_PROGRESS_FILE, _METRICS_FILE]:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"Cleaned up existing file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up file {file_path}: {e}")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {args.output_dir}")
        
        # Create models directory if it doesn't exist
        os.makedirs(args.models_dir, exist_ok=True)
        logger.debug(f"Ensured models directory exists: {args.models_dir}")
        
        # Prepare training data
        logger.info("Preparing training data...")
        train_texts, train_labels, val_texts, val_labels = prepare_training_data(args)
        logger.info(f"Prepared {len(train_texts)} training texts and {len(val_texts)} validation texts")
        
        # Download base model
        logger.info(f"Downloading base model: {args.base_model}")
        model_path = download_base_model(args)
        logger.info(f"Base model downloaded to: {model_path}")
        
        # Fine-tune model
        logger.info("Starting fine-tuning process...")
        tuned_model_path = fine_tune_model(
            args, model_path, train_texts, train_labels, val_texts, val_labels
        )
        logger.info(f"Fine-tuning completed. Model saved to: {tuned_model_path}")
        
        # Test model
        if not args.skip_test:
            logger.info("Testing fine-tuned model...")
            test_model(args, tuned_model_path)
            logger.info("Model testing completed")
        else:
            logger.info("Model testing skipped")
        
        # Save model path to file for easy reference
        model_path_file = "fin_e5_tuned_model_path.txt"
        with open(model_path_file, "w") as f:
            f.write(tuned_model_path)
        logger.info(f"Model path saved to {model_path_file}")
        
        # Save metadata about the training process
        training_metadata = {
            "base_model": args.base_model,
            "train_file": args.train_file,
            "val_file": args.val_file if args.val_file else None,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "completed_at": time.time(),
            "output_model_path": tuned_model_path,
            "progress_file": _PROGRESS_FILE,
            "metrics_file": _METRICS_FILE,
            "visualization_mode": args.visualization_mode,
        }
        
        metadata_file = os.path.join(args.output_dir, "training_metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(training_metadata, f, indent=2)
        logger.info(f"Training metadata saved to {metadata_file}")
        
        logger.info(f"Fine-tuning completed successfully. Model path saved to {model_path_file}")
        logger.info(f"To use this model with the financial system, run:")
        logger.info(f"  ./run_fin_e5.sh query --model-name \"{tuned_model_path}\" --input-file queries.json")
        
        return tuned_model_path
        
    except ValueError as e:
        # Handle validation errors
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        # Handle other errors
        logger.error(f"Error in fine-tuning: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune FinMTEB/Fin-E5 model")
    
    # Data parameters
    data_group = parser.add_argument_group("Data Parameters")
    data_group.add_argument("--train-file", required=True, help="Training data file (JSON)")
    data_group.add_argument("--val-file", help="Validation data file (JSON)")
    data_group.add_argument(
        "--training-format",
        default="pairs",
        choices=["pairs", "documents", "simple"],
        help="Training data format"
    )
    
    # Model parameters
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--base-model", default="FinMTEB/Fin-E5", help="Base model to fine-tune")
    model_group.add_argument("--models-dir", default="./financial_models", help="Models directory")
    model_group.add_argument("--output-dir", default="./fine_tuned_models", help="Output directory for fine-tuned models")
    model_group.add_argument("--output-model-name", default="FinMTEB-Fin-E5-custom", help="Name for fine-tuned model")
    model_group.add_argument("--force-download", action="store_true", help="Force download of base model")
    
    # Training parameters
    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    training_group.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    training_group.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate")
    training_group.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    training_group.add_argument("--evaluation-steps", type=int, default=100, help="Evaluation steps")
    
    # Testing parameters
    testing_group = parser.add_argument_group("Testing Parameters")
    testing_group.add_argument("--skip-test", action="store_true", help="Skip model testing")
    
    # Visualization and IPC parameters
    viz_group = parser.add_argument_group("Visualization Parameters")
    viz_group.add_argument("--enable-ipc", action="store_true", help="Enable IPC mechanisms for visualization")
    viz_group.add_argument("--progress-file", help="Path to progress file (default: temp directory)")
    viz_group.add_argument("--metrics-file", help="Path to metrics file (default: temp directory)")
    viz_group.add_argument("--visualization-mode", choices=["full", "minimal", "none"], default="full", 
                      help="Visualization detail level")
    
    # Logging parameters
    logging_group = parser.add_argument_group("Logging Parameters")
    logging_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                         default=os.environ.get("FINETUNE_LOG_LEVEL", "INFO").upper(),
                         help="Logging level")
    logging_group.add_argument("--log-file", default=os.environ.get("FINETUNE_LOG_FILE", "finetune.log"),
                         help="Path to log file")
    
    # Performance parameters
    perf_group = parser.add_argument_group("Performance Parameters")
    perf_group.add_argument("--num-workers", type=int, default=4, 
                       help="Number of worker processes for data loading")
    perf_group.add_argument("--chunk-size", type=int, default=1000,
                       help="Chunk size for processing large datasets")
    perf_group.add_argument("--use-fp16", action="store_true",
                       help="Use mixed precision training (fp16)")
    
    args = parser.parse_args()
    
    # Set up logging with specified parameters
    logger = setup_logging(args.log_level, args.log_file)
    
    # Update progress and metrics files if provided
    if args.progress_file:
        global _PROGRESS_FILE
        _PROGRESS_FILE = args.progress_file
    
    if args.metrics_file:
        global _METRICS_FILE
        _METRICS_FILE = args.metrics_file
    
    # Run main function
    main(args)