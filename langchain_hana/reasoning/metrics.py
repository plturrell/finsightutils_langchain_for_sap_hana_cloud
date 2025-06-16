"""
Information preservation metrics module for measuring transformation quality.

This module provides tools for measuring how well information is preserved
through vector transformations, identifying information loss, and evaluating
the quality of embeddings.
"""

import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Set
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class InformationPreservationMetrics:
    """
    Measures information preservation through transformations.
    
    Provides tools for quantifying how well information is preserved
    as it transforms through the vector pipeline.
    """
    
    def __init__(self):
        """Initialize information preservation metrics."""
        pass
    
    def measure_text_to_vector_preservation(
        self,
        original_texts: List[str],
        vectors: List[List[float]],
        queries: Optional[List[str]] = None,
        ground_truth_indices: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Measure information preservation from text to vectors.
        
        Args:
            original_texts: Original text documents
            vectors: Vector embeddings of the documents
            queries: Optional query texts for retrieval evaluation
            ground_truth_indices: Optional ground truth indices for queries
            
        Returns:
            Metrics of information preservation
        """
        if len(original_texts) != len(vectors):
            raise ValueError("Number of texts and vectors must match")
        
        if queries and not ground_truth_indices:
            raise ValueError("Ground truth indices must be provided with queries")
        
        if ground_truth_indices and len(queries) != len(ground_truth_indices):
            raise ValueError("Number of queries and ground truth sets must match")
        
        # Basic metrics
        num_documents = len(original_texts)
        vector_dimension = len(vectors[0]) if vectors else 0
        
        # Convert vectors to numpy arrays
        vectors_np = np.array(vectors)
        
        # Calculate vector statistics
        vector_norms = np.linalg.norm(vectors_np, axis=1)
        mean_norm = float(np.mean(vector_norms))
        std_norm = float(np.std(vector_norms))
        min_norm = float(np.min(vector_norms))
        max_norm = float(np.max(vector_norms))
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(vectors_np)
        np.fill_diagonal(similarities, 0)  # Exclude self-similarities
        
        mean_similarity = float(np.mean(similarities))
        max_similarity = float(np.max(similarities))
        min_similarity = float(np.min(similarities[similarities > 0]) if np.any(similarities > 0) else 0)
        
        # Document length vs. vector norm correlation
        document_lengths = [len(text) for text in original_texts]
        length_norm_correlation = float(np.corrcoef(document_lengths, vector_norms)[0, 1])
        
        # Create results
        results = {
            "basic_metrics": {
                "num_documents": num_documents,
                "vector_dimension": vector_dimension,
                "mean_document_length": float(np.mean(document_lengths)),
                "std_document_length": float(np.std(document_lengths)),
            },
            "vector_metrics": {
                "mean_norm": mean_norm,
                "std_norm": std_norm,
                "min_norm": min_norm,
                "max_norm": max_norm,
                "mean_similarity": mean_similarity,
                "max_similarity": max_similarity,
                "min_similarity": min_similarity,
            },
            "correlation_metrics": {
                "length_norm_correlation": length_norm_correlation,
            },
        }
        
        # Add retrieval metrics if queries are provided
        if queries and ground_truth_indices:
            retrieval_metrics = self._calculate_retrieval_metrics(
                vectors_np, queries, ground_truth_indices
            )
            results["retrieval_metrics"] = retrieval_metrics
        
        return results
    
    def measure_vector_to_vector_preservation(
        self,
        original_vectors: List[List[float]],
        transformed_vectors: List[List[float]],
    ) -> Dict[str, Any]:
        """
        Measure information preservation from vectors to transformed vectors.
        
        Args:
            original_vectors: Original vector embeddings
            transformed_vectors: Transformed vector embeddings
            
        Returns:
            Metrics of information preservation
        """
        if len(original_vectors) != len(transformed_vectors):
            raise ValueError("Number of original and transformed vectors must match")
        
        # Convert to numpy arrays
        original_np = np.array(original_vectors)
        transformed_np = np.array(transformed_vectors)
        
        # Calculate direct correspondences
        cosine_scores = []
        euclidean_distances = []
        
        for i in range(len(original_vectors)):
            orig = original_np[i]
            trans = transformed_np[i]
            
            # Normalize vectors for cosine similarity
            orig_norm = np.linalg.norm(orig)
            trans_norm = np.linalg.norm(trans)
            
            if orig_norm > 0 and trans_norm > 0:
                cos_sim = np.dot(orig, trans) / (orig_norm * trans_norm)
                cosine_scores.append(float(cos_sim))
            
            # Euclidean distance
            eucl_dist = np.linalg.norm(orig - trans)
            euclidean_distances.append(float(eucl_dist))
        
        # Calculate mean, min, max for metrics
        mean_cosine = float(np.mean(cosine_scores)) if cosine_scores else 0.0
        min_cosine = float(np.min(cosine_scores)) if cosine_scores else 0.0
        max_cosine = float(np.max(cosine_scores)) if cosine_scores else 0.0
        
        mean_euclidean = float(np.mean(euclidean_distances)) if euclidean_distances else 0.0
        min_euclidean = float(np.min(euclidean_distances)) if euclidean_distances else 0.0
        max_euclidean = float(np.max(euclidean_distances)) if euclidean_distances else 0.0
        
        # Calculate neighborhood preservation
        neighborhood_metrics = self._calculate_neighborhood_preservation(original_np, transformed_np)
        
        # Calculate dimension-wise correlation
        dimension_correlations = self._calculate_dimension_correlations(original_np, transformed_np)
        
        return {
            "direct_correspondence": {
                "mean_cosine_similarity": mean_cosine,
                "min_cosine_similarity": min_cosine,
                "max_cosine_similarity": max_cosine,
                "mean_euclidean_distance": mean_euclidean,
                "min_euclidean_distance": min_euclidean,
                "max_euclidean_distance": max_euclidean,
            },
            "neighborhood_preservation": neighborhood_metrics,
            "dimension_correlations": dimension_correlations,
        }
    
    def measure_semantic_preservation(
        self,
        original_texts: List[str],
        embedded_vectors: List[List[float]],
        semantic_pairs: List[Tuple[int, int, float]],
    ) -> Dict[str, Any]:
        """
        Measure semantic information preservation.
        
        Args:
            original_texts: Original text documents
            embedded_vectors: Vector embeddings of the documents
            semantic_pairs: List of (index1, index2, semantic_similarity) tuples
            
        Returns:
            Metrics of semantic preservation
        """
        if len(original_texts) != len(embedded_vectors):
            raise ValueError("Number of texts and vectors must match")
        
        # Convert to numpy arrays
        vectors_np = np.array(embedded_vectors)
        
        # Calculate vector similarities for semantic pairs
        vector_similarities = []
        semantic_similarities = []
        
        for idx1, idx2, semantic_sim in semantic_pairs:
            if idx1 >= len(vectors_np) or idx2 >= len(vectors_np):
                continue
            
            vec1 = vectors_np[idx1]
            vec2 = vectors_np[idx2]
            
            # Calculate cosine similarity between vectors
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                vector_sim = np.dot(vec1, vec2) / (norm1 * norm2)
                vector_similarities.append(float(vector_sim))
                semantic_similarities.append(float(semantic_sim))
        
        # Calculate correlation between semantic and vector similarities
        if len(vector_similarities) > 1 and len(semantic_similarities) > 1:
            correlation = float(np.corrcoef(vector_similarities, semantic_similarities)[0, 1])
        else:
            correlation = 0.0
        
        # Calculate mean absolute error
        if vector_similarities and semantic_similarities:
            mae = float(np.mean(np.abs(np.array(vector_similarities) - np.array(semantic_similarities))))
        else:
            mae = 0.0
        
        # Calculate metrics for different semantic similarity ranges
        range_metrics = self._calculate_range_metrics(vector_similarities, semantic_similarities)
        
        return {
            "overall_metrics": {
                "correlation": correlation,
                "mean_absolute_error": mae,
                "num_pairs": len(vector_similarities),
            },
            "range_metrics": range_metrics,
        }
    
    def measure_task_performance(
        self,
        vectors: List[List[float]],
        task_type: str,
        ground_truth: List[Any],
        predictions: List[Any],
    ) -> Dict[str, Any]:
        """
        Measure task-specific performance of vectors.
        
        Args:
            vectors: Vector embeddings
            task_type: Type of task ('classification', 'clustering', 'retrieval')
            ground_truth: Ground truth labels or indices
            predictions: Predicted labels or indices
            
        Returns:
            Task-specific performance metrics
        """
        if len(ground_truth) != len(predictions):
            raise ValueError("Number of ground truth and predictions must match")
        
        if task_type == "classification":
            return self._calculate_classification_metrics(ground_truth, predictions)
        elif task_type == "clustering":
            return self._calculate_clustering_metrics(vectors, ground_truth, predictions)
        elif task_type == "retrieval":
            return self._calculate_retrieval_performance(vectors, ground_truth, predictions)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _calculate_retrieval_metrics(
        self,
        vectors: np.ndarray,
        queries: List[str],
        ground_truth_indices: List[List[int]],
        k_values: List[int] = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """
        Calculate retrieval metrics for vector embeddings.
        
        Args:
            vectors: Vector embeddings
            queries: Query texts
            ground_truth_indices: Ground truth relevant indices for each query
            k_values: K values for precision and recall calculation
            
        Returns:
            Retrieval metrics
        """
        # This is a simplified implementation
        # A full implementation would use query embeddings and search
        # For now, return placeholder metrics
        metrics = {
            "precision": {},
            "recall": {},
            "ndcg": {},
            "mean_reciprocal_rank": 0.0,
        }
        
        for k in k_values:
            metrics["precision"][f"p@{k}"] = 0.0
            metrics["recall"][f"r@{k}"] = 0.0
            metrics["ndcg"][f"ndcg@{k}"] = 0.0
        
        return metrics
    
    def _calculate_neighborhood_preservation(
        self,
        original_vectors: np.ndarray,
        transformed_vectors: np.ndarray,
        k_values: List[int] = [5, 10, 20],
    ) -> Dict[str, float]:
        """
        Calculate neighborhood preservation metrics.
        
        Args:
            original_vectors: Original vector embeddings
            transformed_vectors: Transformed vector embeddings
            k_values: K values for neighborhood calculation
            
        Returns:
            Neighborhood preservation metrics
        """
        # Calculate pairwise distances
        original_distances = cosine_similarity(original_vectors)
        transformed_distances = cosine_similarity(transformed_vectors)
        
        # Convert similarities to distances
        original_distances = 1 - original_distances
        transformed_distances = 1 - transformed_distances
        
        metrics = {}
        
        for k in k_values:
            k = min(k, original_vectors.shape[0] - 1)
            
            # Calculate k-nearest neighbors for each point
            original_neighbors = np.argsort(original_distances, axis=1)[:, 1:k+1]
            transformed_neighbors = np.argsort(transformed_distances, axis=1)[:, 1:k+1]
            
            # Calculate neighborhood preservation
            preservation_scores = []
            
            for i in range(len(original_vectors)):
                orig_neighbors = set(original_neighbors[i])
                trans_neighbors = set(transformed_neighbors[i])
                
                # Calculate Jaccard similarity
                intersection = len(orig_neighbors.intersection(trans_neighbors))
                union = len(orig_neighbors.union(trans_neighbors))
                
                preservation = intersection / union if union > 0 else 0.0
                preservation_scores.append(preservation)
            
            metrics[f"k{k}_preservation"] = float(np.mean(preservation_scores))
        
        return metrics
    
    def _calculate_dimension_correlations(
        self,
        original_vectors: np.ndarray,
        transformed_vectors: np.ndarray,
        max_dimensions: int = 10,
    ) -> Dict[str, float]:
        """
        Calculate dimension-wise correlations.
        
        Args:
            original_vectors: Original vector embeddings
            transformed_vectors: Transformed vector embeddings
            max_dimensions: Maximum number of dimensions to analyze
            
        Returns:
            Dimension correlation metrics
        """
        # Determine minimum dimension to analyze
        min_dim = min(
            original_vectors.shape[1],
            transformed_vectors.shape[1],
            max_dimensions
        )
        
        # Calculate correlation for each dimension
        dimension_corrs = []
        
        for i in range(min_dim):
            orig_dim = original_vectors[:, i]
            trans_dim = transformed_vectors[:, i]
            
            corr = float(np.corrcoef(orig_dim, trans_dim)[0, 1])
            dimension_corrs.append(corr)
        
        # Calculate statistics
        mean_corr = float(np.mean(dimension_corrs))
        min_corr = float(np.min(dimension_corrs))
        max_corr = float(np.max(dimension_corrs))
        
        return {
            "mean_dimension_correlation": mean_corr,
            "min_dimension_correlation": min_corr,
            "max_dimension_correlation": max_corr,
            "dimension_correlations": dimension_corrs[:max_dimensions],
        }
    
    def _calculate_range_metrics(
        self,
        vector_similarities: List[float],
        semantic_similarities: List[float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for different semantic similarity ranges.
        
        Args:
            vector_similarities: Vector cosine similarities
            semantic_similarities: Semantic similarities
            
        Returns:
            Metrics for different similarity ranges
        """
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        range_metrics = {}
        
        for start, end in ranges:
            range_name = f"{start:.1f}-{end:.1f}"
            
            # Get pairs in this semantic similarity range
            indices = [
                i for i, sim in enumerate(semantic_similarities)
                if start <= sim < end
            ]
            
            if indices:
                vec_sims = [vector_similarities[i] for i in indices]
                sem_sims = [semantic_similarities[i] for i in indices]
                
                # Calculate metrics for this range
                mae = float(np.mean(np.abs(np.array(vec_sims) - np.array(sem_sims))))
                
                if len(vec_sims) > 1 and len(sem_sims) > 1:
                    corr = float(np.corrcoef(vec_sims, sem_sims)[0, 1])
                else:
                    corr = 0.0
                
                range_metrics[range_name] = {
                    "count": len(indices),
                    "mean_absolute_error": mae,
                    "correlation": corr,
                    "mean_vector_similarity": float(np.mean(vec_sims)),
                    "mean_semantic_similarity": float(np.mean(sem_sims)),
                }
            else:
                range_metrics[range_name] = {
                    "count": 0,
                    "mean_absolute_error": 0.0,
                    "correlation": 0.0,
                    "mean_vector_similarity": 0.0,
                    "mean_semantic_similarity": 0.0,
                }
        
        return range_metrics
    
    def _calculate_classification_metrics(
        self,
        ground_truth: List[Any],
        predictions: List[Any],
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics.
        
        Args:
            ground_truth: Ground truth labels
            predictions: Predicted labels
            
        Returns:
            Classification metrics
        """
        # Calculate accuracy
        correct = sum(1 for g, p in zip(ground_truth, predictions) if g == p)
        accuracy = correct / len(ground_truth) if ground_truth else 0.0
        
        # Get unique classes
        unique_classes = set(ground_truth)
        
        # Calculate per-class metrics
        class_metrics = {}
        
        for cls in unique_classes:
            # Calculate precision
            true_positives = sum(1 for g, p in zip(ground_truth, predictions) if g == cls and p == cls)
            predicted_positives = sum(1 for p in predictions if p == cls)
            
            precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
            
            # Calculate recall
            actual_positives = sum(1 for g in ground_truth if g == cls)
            recall = true_positives / actual_positives if actual_positives > 0 else 0.0
            
            # Calculate F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[str(cls)] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": actual_positives,
            }
        
        # Calculate macro and weighted averages
        macro_precision = sum(m["precision"] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0.0
        macro_recall = sum(m["recall"] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0.0
        macro_f1 = sum(m["f1"] for m in class_metrics.values()) / len(class_metrics) if class_metrics else 0.0
        
        total_support = sum(m["support"] for m in class_metrics.values())
        weighted_precision = sum(m["precision"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0.0
        weighted_recall = sum(m["recall"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0.0
        weighted_f1 = sum(m["f1"] * m["support"] for m in class_metrics.values()) / total_support if total_support > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "per_class": class_metrics,
            "macro_avg": {
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
            },
            "weighted_avg": {
                "precision": weighted_precision,
                "recall": weighted_recall,
                "f1": weighted_f1,
            },
        }
    
    def _calculate_clustering_metrics(
        self,
        vectors: np.ndarray,
        ground_truth: List[Any],
        predictions: List[Any],
    ) -> Dict[str, Any]:
        """
        Calculate clustering metrics.
        
        Args:
            vectors: Vector embeddings
            ground_truth: Ground truth cluster assignments
            predictions: Predicted cluster assignments
            
        Returns:
            Clustering metrics
        """
        # This is a simplified implementation
        # A full implementation would use scikit-learn metrics
        
        # Get unique clusters
        true_clusters = set(ground_truth)
        pred_clusters = set(predictions)
        
        # Calculate cluster statistics
        true_cluster_stats = {
            str(cls): {
                "count": sum(1 for g in ground_truth if g == cls),
                "percentage": sum(1 for g in ground_truth if g == cls) / len(ground_truth) * 100,
            }
            for cls in true_clusters
        }
        
        pred_cluster_stats = {
            str(cls): {
                "count": sum(1 for p in predictions if p == cls),
                "percentage": sum(1 for p in predictions if p == cls) / len(predictions) * 100,
            }
            for cls in pred_clusters
        }
        
        # Calculate simple metrics
        matches = 0
        total_pairs = 0
        
        for i in range(len(ground_truth)):
            for j in range(i+1, len(ground_truth)):
                same_true_cluster = ground_truth[i] == ground_truth[j]
                same_pred_cluster = predictions[i] == predictions[j]
                
                if same_true_cluster == same_pred_cluster:
                    matches += 1
                
                total_pairs += 1
        
        rand_index = matches / total_pairs if total_pairs > 0 else 0.0
        
        return {
            "true_clusters": len(true_clusters),
            "predicted_clusters": len(pred_clusters),
            "true_cluster_stats": true_cluster_stats,
            "predicted_cluster_stats": pred_cluster_stats,
            "rand_index": rand_index,
        }
    
    def _calculate_retrieval_performance(
        self,
        vectors: np.ndarray,
        ground_truth: List[List[int]],
        predictions: List[List[int]],
    ) -> Dict[str, Any]:
        """
        Calculate retrieval performance metrics.
        
        Args:
            vectors: Vector embeddings
            ground_truth: Ground truth relevant indices for each query
            predictions: Predicted relevant indices for each query
            
        Returns:
            Retrieval performance metrics
        """
        precisions = []
        recalls = []
        f1_scores = []
        
        for gt, pred in zip(ground_truth, predictions):
            gt_set = set(gt)
            pred_set = set(pred)
            
            # Calculate precision
            precision = len(gt_set.intersection(pred_set)) / len(pred_set) if pred_set else 0.0
            precisions.append(precision)
            
            # Calculate recall
            recall = len(gt_set.intersection(pred_set)) / len(gt_set) if gt_set else 0.0
            recalls.append(recall)
            
            # Calculate F1
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)
        
        # Calculate mean metrics
        mean_precision = float(np.mean(precisions)) if precisions else 0.0
        mean_recall = float(np.mean(recalls)) if recalls else 0.0
        mean_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0
        
        return {
            "mean_precision": mean_precision,
            "mean_recall": mean_recall,
            "mean_f1": mean_f1,
            "num_queries": len(ground_truth),
        }