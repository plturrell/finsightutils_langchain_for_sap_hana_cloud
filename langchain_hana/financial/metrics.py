"""
Financial model metrics collection and evaluation module.

This module provides tools for collecting, calculating, and visualizing
metrics related to financial embedding models and their performance.
"""

import os
import json
import time
import logging
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and stores metrics for financial models with optimizations for large datasets."""
    
    def __init__(
        self,
        metrics_file: Optional[str] = None,
        metrics_prefix: str = "fin_metrics",
        auto_save: bool = True,
        max_items_per_metric: int = 10000,
        chunk_size: int = 1000,
        buffer_flush_threshold: int = 50,
    ):
        """
        Initialize metrics collector.
        
        Args:
            metrics_file: Path to metrics file (None for auto-generated temp file)
            metrics_prefix: Prefix for auto-generated metrics files
            auto_save: Whether to automatically save metrics on update
            max_items_per_metric: Maximum number of items to keep per metric
            chunk_size: Size of chunks for processing large datasets
            buffer_flush_threshold: Number of updates before flushing buffer to disk
        """
        self.metrics_file = metrics_file or os.path.join(
            tempfile.gettempdir(), f"{metrics_prefix}_{int(time.time())}.json"
        )
        self.auto_save = auto_save
        self.max_items_per_metric = max_items_per_metric
        self.chunk_size = chunk_size
        self.buffer_flush_threshold = buffer_flush_threshold
        
        # In-memory metrics storage
        self.metrics: Dict[str, List[Any]] = {
            "loss": [],
            "similarity": [],
            "relevance": [],
            "execution_time": [],
            "batch_times": [],
            "memory_usage": [],
            "precision": [],
            "recall": [],
            "f1_score": [],
            "training_steps": [],
            "custom": {},
        }
        
        # Buffer for updates to reduce disk I/O
        self.update_buffer: Dict[str, List[Any]] = {}
        self.buffer_count = 0
        
        # Statistics for large metrics that exceed max_items_per_metric
        self.statistics: Dict[str, Dict[str, Any]] = {}
        
        # Mutex for thread safety
        self._lock = threading.Lock()
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(self.metrics_file)), exist_ok=True)
        
        # Initialize metrics file
        self.save_metrics()
        
        logger.debug(f"Metrics collector initialized with file: {self.metrics_file}")
    
    def update_metric(self, metric_name: str, value: Any, step: Optional[int] = None) -> None:
        """
        Update a specific metric with optimizations for large datasets.
        
        Args:
            metric_name: Name of the metric to update
            value: Value to add
            step: Optional training step
        """
        with self._lock:
            # Add to buffer first
            if metric_name not in self.update_buffer:
                self.update_buffer[metric_name] = []
            
            if isinstance(value, list):
                self.update_buffer[metric_name].extend(value)
            else:
                self.update_buffer[metric_name].append(value)
            
            # Track step in buffer if provided
            if step is not None:
                if "training_steps" not in self.update_buffer:
                    self.update_buffer["training_steps"] = []
                
                if isinstance(value, list):
                    self.update_buffer["training_steps"].extend([step] * len(value))
                else:
                    self.update_buffer["training_steps"].append(step)
            
            # Increment buffer count
            self.buffer_count += 1
            
            # Flush buffer if threshold reached
            if self.auto_save and self.buffer_count >= self.buffer_flush_threshold:
                self._flush_buffer()
                
                # Reset buffer count
                self.buffer_count = 0
    
    def _flush_buffer(self) -> None:
        """Flush update buffer to metrics."""
        # Apply buffered updates to metrics
        for metric_name, values in self.update_buffer.items():
            if not values:
                continue
                
            if metric_name == "custom":
                # Special handling for custom metrics
                continue
            
            # Check if this is a custom metric
            if metric_name not in self.metrics:
                # This is a custom metric
                if metric_name not in self.metrics["custom"]:
                    self.metrics["custom"][metric_name] = []
                
                target = self.metrics["custom"][metric_name]
            else:
                target = self.metrics[metric_name]
            
            # Check if adding these values would exceed max_items_per_metric
            if len(target) + len(values) > self.max_items_per_metric:
                # Update statistics instead of storing all values
                self._update_statistics(metric_name, values, target)
                
                # Store only the most recent values
                excess = len(target) + len(values) - self.max_items_per_metric
                if excess < len(target):
                    # Keep some existing values plus new values
                    target[:-excess] = []
                    target.extend(values)
                else:
                    # Keep only the most recent values
                    target.clear()
                    target.extend(values[-self.max_items_per_metric:])
            else:
                # Just append values
                target.extend(values)
        
        # Clear buffer
        self.update_buffer.clear()
        
        # Save metrics if auto_save is enabled
        if self.auto_save:
            self.save_metrics()
    
    def _update_statistics(self, metric_name: str, new_values: List[Any], existing_values: List[Any]) -> None:
        """
        Update running statistics for a metric with too many values.
        
        Args:
            metric_name: Name of the metric
            new_values: New values to incorporate
            existing_values: Existing values for this metric
        """
        if not all(isinstance(v, (int, float)) for v in new_values):
            # Can't compute statistics for non-numeric values
            return
        
        # Get or initialize statistics
        if metric_name not in self.statistics:
            if existing_values and all(isinstance(v, (int, float)) for v in existing_values):
                # Initialize with existing values
                self.statistics[metric_name] = {
                    "count": len(existing_values),
                    "mean": float(np.mean(existing_values)),
                    "sum": float(np.sum(existing_values)),
                    "sum_squares": float(np.sum(np.square(existing_values))),
                    "min": float(np.min(existing_values)),
                    "max": float(np.max(existing_values)),
                }
            else:
                # Initialize with zeros
                self.statistics[metric_name] = {
                    "count": 0,
                    "mean": 0.0,
                    "sum": 0.0,
                    "sum_squares": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                }
        
        # Convert to numpy array for efficient calculations
        values_array = np.array(new_values, dtype=np.float64)
        
        # Update statistics using Welford's online algorithm
        new_count = len(new_values)
        old_count = self.statistics[metric_name]["count"]
        total_count = old_count + new_count
        
        # Update min and max
        self.statistics[metric_name]["min"] = min(
            self.statistics[metric_name]["min"],
            float(np.min(values_array))
        )
        self.statistics[metric_name]["max"] = max(
            self.statistics[metric_name]["max"],
            float(np.max(values_array))
        )
        
        # Update sum
        self.statistics[metric_name]["sum"] += float(np.sum(values_array))
        
        # Update sum of squares
        self.statistics[metric_name]["sum_squares"] += float(np.sum(np.square(values_array)))
        
        # Update mean
        self.statistics[metric_name]["mean"] = self.statistics[metric_name]["sum"] / total_count
        
        # Update count
        self.statistics[metric_name]["count"] = total_count
    
    def update_metrics(self, metrics_dict: Dict[str, Any], step: Optional[int] = None) -> None:
        """
        Update multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metrics to update
            step: Optional training step
        """
        with self._lock:
            for metric_name, value in metrics_dict.items():
                self.update_metric(metric_name, value, step=step)
    
    def get_metric(self, metric_name: str, max_items: Optional[int] = None) -> List[Any]:
        """
        Get values for a specific metric.
        
        Args:
            metric_name: Name of the metric to retrieve
            max_items: Maximum number of items to return (None for all)
            
        Returns:
            List of metric values
        """
        # Flush buffer to ensure we have latest data
        with self._lock:
            self._flush_buffer()
            
            if metric_name in self.metrics:
                values = self.metrics[metric_name]
            elif metric_name in self.metrics["custom"]:
                values = self.metrics["custom"][metric_name]
            else:
                return []
            
            if max_items is not None and len(values) > max_items:
                return values[-max_items:]
            
            return values
    
    def get_latest_metric(self, metric_name: str) -> Optional[Any]:
        """
        Get the most recent value for a specific metric.
        
        Args:
            metric_name: Name of the metric to retrieve
            
        Returns:
            Most recent metric value
        """
        # Check buffer first for latest value
        with self._lock:
            if metric_name in self.update_buffer and self.update_buffer[metric_name]:
                return self.update_buffer[metric_name][-1]
            
            # If not in buffer, check metrics
            values = self.get_metric(metric_name, max_items=1)
            if values:
                return values[-1]
            
            return None
    
    def save_metrics(self) -> None:
        """Save metrics to file with optimizations for large datasets."""
        with self._lock:
            try:
                # Flush buffer to ensure all updates are applied
                self._flush_buffer()
                
                # Add statistics to output
                output_metrics = self.metrics.copy()
                output_metrics["statistics"] = self.statistics
                
                # Write metrics in chunks to avoid memory issues
                with open(self.metrics_file, "w") as f:
                    # Start JSON object
                    f.write("{\n")
                    
                    # Write each metric as a separate chunk
                    first_item = True
                    for key, values in output_metrics.items():
                        if not first_item:
                            f.write(",\n")
                        first_item = False
                        
                        if isinstance(values, dict):
                            # Handle nested dictionaries (like custom metrics)
                            f.write(f'  "{key}": {json.dumps(values)}')
                        else:
                            # Handle lists
                            f.write(f'  "{key}": {json.dumps(values)}')
                    
                    # End JSON object
                    f.write("\n}")
                
                logger.debug(f"Metrics saved to {self.metrics_file}")
            except Exception as e:
                logger.warning(f"Failed to save metrics to {self.metrics_file}: {str(e)}")
    
    def load_metrics(self) -> Dict[str, List[Any]]:
        """
        Load metrics from file with optimizations for large datasets.
        
        Returns:
            Dictionary of metrics
        """
        with self._lock:
            if os.path.exists(self.metrics_file):
                try:
                    # Load metrics in chunks to handle large files
                    with open(self.metrics_file, "r") as f:
                        loaded_metrics = json.load(f)
                    
                    # Extract statistics if present
                    if "statistics" in loaded_metrics:
                        self.statistics = loaded_metrics.pop("statistics")
                    
                    # Update metrics
                    self.metrics.update(loaded_metrics)
                    
                    logger.debug(f"Metrics loaded from {self.metrics_file}")
                except Exception as e:
                    logger.warning(f"Failed to load metrics from {self.metrics_file}: {str(e)}")
            
            return self.metrics
    
    def clear_metrics(self) -> None:
        """Clear all metrics."""
        with self._lock:
            self.metrics = {
                "loss": [],
                "similarity": [],
                "relevance": [],
                "execution_time": [],
                "batch_times": [],
                "memory_usage": [],
                "precision": [],
                "recall": [],
                "f1_score": [],
                "training_steps": [],
                "custom": {},
            }
            self.statistics = {}
            self.update_buffer.clear()
            self.buffer_count = 0
            
            # Save empty metrics
            if self.auto_save:
                self.save_metrics()
    
    def calculate_summary(self) -> Dict[str, Any]:
        """
        Calculate summary statistics for metrics.
        
        Returns:
            Dictionary of summary statistics
        """
        with self._lock:
            # Flush buffer to ensure all updates are applied
            self._flush_buffer()
            
            summary = {}
            
            # Include pre-computed statistics
            for metric_name, stats in self.statistics.items():
                if metric_name not in summary:
                    summary[metric_name] = {}
                
                # Copy statistics
                summary[metric_name].update(stats)
                
                # Calculate standard deviation from sum of squares
                if stats["count"] > 0:
                    variance = (stats["sum_squares"] / stats["count"]) - (stats["mean"] ** 2)
                    summary[metric_name]["std"] = float(np.sqrt(max(0, variance)))
            
            # Process standard metrics not in statistics
            for metric_name, values in self.metrics.items():
                if metric_name == "custom":
                    # Handle custom metrics separately
                    continue
                
                if metric_name in summary:
                    # Already processed via statistics
                    continue
                
                if values and all(isinstance(v, (int, float)) for v in values):
                    summary[metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "latest": float(values[-1]) if values else None,
                        "count": len(values),
                    }
            
            # Process custom metrics
            summary["custom"] = {}
            for metric_name, values in self.metrics["custom"].items():
                if f"custom.{metric_name}" in summary:
                    # Already processed via statistics
                    summary["custom"][metric_name] = summary.pop(f"custom.{metric_name}")
                    continue
                
                if values and all(isinstance(v, (int, float)) for v in values):
                    summary["custom"][metric_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "latest": float(values[-1]) if values else None,
                        "count": len(values),
                    }
            
            return summary


class FinancialModelEvaluator:
    """Evaluates financial embedding models using various metrics."""
    
    def __init__(
        self,
        metrics_collector: Optional[MetricsCollector] = None,
        evaluation_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize model evaluator.
        
        Args:
            metrics_collector: Metrics collector (creates new one if None)
            evaluation_data: Evaluation data
        """
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.evaluation_data = evaluation_data or {}
        self.reference_embeddings = {}
    
    def set_reference_model(self, model_name: str, embeddings: Dict[str, np.ndarray]) -> None:
        """
        Set reference embeddings from a model.
        
        Args:
            model_name: Name of the reference model
            embeddings: Dictionary mapping text to embeddings
        """
        self.reference_embeddings[model_name] = embeddings
    
    def calculate_semantic_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Cosine similarity score
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embeddings1)
        norm2 = np.linalg.norm(embeddings2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embeddings1, embeddings2) / (norm1 * norm2))
    
    def calculate_retrieval_metrics(
        self,
        relevant_docs: List[str],
        retrieved_docs: List[str],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for retrieval.
        
        Args:
            relevant_docs: List of relevant document IDs
            retrieved_docs: List of retrieved document IDs
            k: Number of top documents to consider
            
        Returns:
            Dictionary of retrieval metrics
        """
        # Truncate retrieved docs to top k
        retrieved_top_k = retrieved_docs[:k]
        
        # Calculate true positives
        true_positives = len(set(relevant_docs) & set(retrieved_top_k))
        
        # Calculate precision, recall, F1
        precision = true_positives / len(retrieved_top_k) if retrieved_top_k else 0.0
        recall = true_positives / len(relevant_docs) if relevant_docs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        
        # Update metrics
        self.metrics_collector.update_metrics({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "relevant_count": len(relevant_docs),
            "retrieved_count": len(retrieved_docs),
            "true_positives": true_positives,
        }
    
    def evaluate_model_improvement(
        self,
        base_model_results: Dict[str, Any],
        tuned_model_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Calculate improvement metrics between base and tuned models.
        
        Args:
            base_model_results: Base model evaluation results
            tuned_model_results: Tuned model evaluation results
            
        Returns:
            Dictionary of improvement metrics
        """
        improvements = {}
        
        # Calculate improvements for numeric metrics
        for metric in ["precision", "recall", "f1_score", "execution_time"]:
            if metric in base_model_results and metric in tuned_model_results:
                base_value = base_model_results[metric]
                tuned_value = tuned_model_results[metric]
                
                if metric == "execution_time":
                    # For time metrics, lower is better
                    improvement = (base_value - tuned_value) / base_value if base_value > 0 else 0.0
                else:
                    # For other metrics, higher is better
                    improvement = (tuned_value - base_value) / base_value if base_value > 0 else 0.0
                
                improvements[f"{metric}_improvement"] = improvement * 100.0  # as percentage
        
        # Calculate semantic similarity improvement if available
        if "similarity" in base_model_results and "similarity" in tuned_model_results:
            base_similarity = base_model_results["similarity"]
            tuned_similarity = tuned_model_results["similarity"]
            similarity_improvement = (tuned_similarity - base_similarity) / base_similarity if base_similarity > 0 else 0.0
            improvements["similarity_improvement"] = similarity_improvement * 100.0  # as percentage
        
        # Track improvements
        for metric, value in improvements.items():
            self.metrics_collector.update_metric(metric, value)
        
        return improvements
    
    def evaluate_model_on_queries(
        self,
        model: Any,  # SentenceTransformer
        queries: List[str],
        relevant_docs: Dict[str, List[str]],
        doc_embeddings: Dict[str, np.ndarray],
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Evaluate a model on a set of queries.
        
        Args:
            model: The model to evaluate
            queries: List of query strings
            relevant_docs: Dictionary mapping queries to relevant doc IDs
            doc_embeddings: Dictionary mapping doc IDs to embeddings
            k: Number of documents to retrieve
            
        Returns:
            Evaluation results
        """
        results = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "execution_time": [],
            "query_results": {},
        }
        
        for query in queries:
            # Time the query embedding
            start_time = time.time()
            query_embedding = model.encode(query, convert_to_numpy=True)
            encoding_time = time.time() - start_time
            
            # Calculate similarity with all documents
            similarities = {}
            for doc_id, doc_embedding in doc_embeddings.items():
                similarity = self.calculate_semantic_similarity(query_embedding, doc_embedding)
                similarities[doc_id] = similarity
            
            # Sort by similarity
            sorted_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            retrieved_docs = [doc_id for doc_id, _ in sorted_docs[:k]]
            
            # Calculate retrieval metrics
            if query in relevant_docs:
                metrics = self.calculate_retrieval_metrics(relevant_docs[query], retrieved_docs, k)
                results["precision"].append(metrics["precision"])
                results["recall"].append(metrics["recall"])
                results["f1_score"].append(metrics["f1_score"])
            
            results["execution_time"].append(encoding_time)
            
            # Store query results
            results["query_results"][query] = {
                "retrieved_docs": retrieved_docs,
                "execution_time": encoding_time,
            }
        
        # Calculate averages
        for metric in ["precision", "recall", "f1_score", "execution_time"]:
            if results[metric]:
                results[f"avg_{metric}"] = sum(results[metric]) / len(results[metric])
            else:
                results[f"avg_{metric}"] = 0.0
        
        # Update metrics collector
        self.metrics_collector.update_metrics({
            "precision": results["avg_precision"],
            "recall": results["avg_recall"],
            "f1_score": results["avg_f1_score"],
            "execution_time": results["avg_execution_time"],
        })
        
        return results


# Factory function to create metrics collector
def create_metrics_collector(
    metrics_file: Optional[str] = None,
    metrics_prefix: str = "fin_metrics",
) -> MetricsCollector:
    """
    Create a metrics collector.
    
    Args:
        metrics_file: Path to metrics file (None for auto-generated)
        metrics_prefix: Prefix for auto-generated metrics files
        
    Returns:
        MetricsCollector instance
    """
    return MetricsCollector(
        metrics_file=metrics_file,
        metrics_prefix=metrics_prefix,
    )


# Factory function to create model evaluator
def create_model_evaluator(
    metrics_collector: Optional[MetricsCollector] = None,
    evaluation_data: Optional[Dict[str, Any]] = None,
) -> FinancialModelEvaluator:
    """
    Create a model evaluator.
    
    Args:
        metrics_collector: Metrics collector (creates new one if None)
        evaluation_data: Evaluation data
        
    Returns:
        FinancialModelEvaluator instance
    """
    return FinancialModelEvaluator(
        metrics_collector=metrics_collector,
        evaluation_data=evaluation_data,
    )