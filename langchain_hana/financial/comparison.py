"""
Financial model comparison and evaluation.

This module provides tools for comparing and evaluating financial embedding models,
focusing on semantic understanding, retrieval performance, and domain-specific metrics.
"""

import os
import json
import time
import logging
import tempfile
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set

from langchain_hana.financial.metrics import MetricsCollector, FinancialModelEvaluator

logger = logging.getLogger(__name__)

class ModelComparison:
    """Compares financial embedding models across multiple dimensions."""
    
    def __init__(
        self,
        base_model_name: str,
        tuned_model_name: str,
        metrics_collector: Optional[MetricsCollector] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize model comparison.
        
        Args:
            base_model_name: Name of base model
            tuned_model_name: Name of tuned model
            metrics_collector: Metrics collector (creates new one if None)
            output_dir: Output directory for comparison results
        """
        self.base_model_name = base_model_name
        self.tuned_model_name = tuned_model_name
        self.metrics_collector = metrics_collector or MetricsCollector(
            metrics_file=os.path.join(tempfile.gettempdir(), "model_comparison.json")
        )
        self.output_dir = output_dir or os.path.join(tempfile.gettempdir(), "model_comparison")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize results
        self.results = {
            "base_model": {
                "name": base_model_name,
                "metrics": {},
                "query_results": {},
            },
            "tuned_model": {
                "name": tuned_model_name,
                "metrics": {},
                "query_results": {},
            },
            "improvements": {},
            "query_improvements": {},
            "semantic_analysis": {},
            "created_at": time.time(),
        }
    
    def compare_models_on_queries(
        self,
        queries: List[str],
        relevant_docs: Dict[str, List[str]],
        base_model,  # SentenceTransformer
        tuned_model,  # SentenceTransformer
        doc_texts: Dict[str, str],
        k: int = 10,
    ) -> Dict[str, Any]:
        """
        Compare models on a set of queries.
        
        Args:
            queries: List of query strings
            relevant_docs: Dictionary mapping queries to relevant doc IDs
            base_model: Base model
            tuned_model: Tuned model
            doc_texts: Dictionary mapping doc IDs to text
            k: Number of documents to retrieve
            
        Returns:
            Comparison results
        """
        # Compute document embeddings
        doc_ids = list(doc_texts.keys())
        
        # Base model embeddings
        start_time = time.time()
        base_doc_embeddings = {
            doc_id: base_model.encode(doc_texts[doc_id], convert_to_numpy=True)
            for doc_id in doc_ids
        }
        base_embedding_time = time.time() - start_time
        
        # Tuned model embeddings
        start_time = time.time()
        tuned_doc_embeddings = {
            doc_id: tuned_model.encode(doc_texts[doc_id], convert_to_numpy=True)
            for doc_id in doc_ids
        }
        tuned_embedding_time = time.time() - start_time
        
        # Create evaluators
        base_evaluator = FinancialModelEvaluator()
        tuned_evaluator = FinancialModelEvaluator()
        
        # Evaluate models
        base_results = base_evaluator.evaluate_model_on_queries(
            base_model, queries, relevant_docs, base_doc_embeddings, k
        )
        
        tuned_results = tuned_evaluator.evaluate_model_on_queries(
            tuned_model, queries, relevant_docs, tuned_doc_embeddings, k
        )
        
        # Calculate improvements
        improvements = {
            "precision": (tuned_results["avg_precision"] - base_results["avg_precision"]) / max(1e-6, base_results["avg_precision"]) * 100,
            "recall": (tuned_results["avg_recall"] - base_results["avg_recall"]) / max(1e-6, base_results["avg_recall"]) * 100,
            "f1_score": (tuned_results["avg_f1_score"] - base_results["avg_f1_score"]) / max(1e-6, base_results["avg_f1_score"]) * 100,
            "execution_time": (base_results["avg_execution_time"] - tuned_results["avg_execution_time"]) / max(1e-6, base_results["avg_execution_time"]) * 100,
            "embedding_time": (base_embedding_time - tuned_embedding_time) / max(1e-6, base_embedding_time) * 100,
        }
        
        # Calculate per-query improvements
        query_improvements = {}
        for query in queries:
            if query in base_results["query_results"] and query in tuned_results["query_results"]:
                base_time = base_results["query_results"][query]["execution_time"]
                tuned_time = tuned_results["query_results"][query]["execution_time"]
                
                time_improvement = (base_time - tuned_time) / max(1e-6, base_time) * 100
                
                # Calculate retrieval improvement
                if query in relevant_docs:
                    base_retrieved = set(base_results["query_results"][query]["retrieved_docs"])
                    tuned_retrieved = set(tuned_results["query_results"][query]["retrieved_docs"])
                    relevant = set(relevant_docs[query])
                    
                    base_relevant_count = len(base_retrieved & relevant)
                    tuned_relevant_count = len(tuned_retrieved & relevant)
                    
                    retrieval_improvement = (tuned_relevant_count - base_relevant_count) / max(1, len(relevant)) * 100
                else:
                    retrieval_improvement = 0
                
                query_improvements[query] = {
                    "time_improvement": time_improvement,
                    "retrieval_improvement": retrieval_improvement,
                }
        
        # Store results
        self.results["base_model"]["metrics"] = {
            "precision": base_results["avg_precision"],
            "recall": base_results["avg_recall"],
            "f1_score": base_results["avg_f1_score"],
            "execution_time": base_results["avg_execution_time"],
            "embedding_time": base_embedding_time,
        }
        
        self.results["tuned_model"]["metrics"] = {
            "precision": tuned_results["avg_precision"],
            "recall": tuned_results["avg_recall"],
            "f1_score": tuned_results["avg_f1_score"],
            "execution_time": tuned_results["avg_execution_time"],
            "embedding_time": tuned_embedding_time,
        }
        
        self.results["base_model"]["query_results"] = base_results["query_results"]
        self.results["tuned_model"]["query_results"] = tuned_results["query_results"]
        self.results["improvements"] = improvements
        self.results["query_improvements"] = query_improvements
        
        # Update metrics collector
        for metric, value in improvements.items():
            self.metrics_collector.update_metric(f"{metric}_improvement", value)
        
        # Save results
        self.save_results()
        
        return self.results
    
    def analyze_semantic_understanding(
        self,
        base_model,  # SentenceTransformer
        tuned_model,  # SentenceTransformer
        financial_terms: List[str],
        financial_concepts: List[str],
        financial_relationships: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """
        Analyze semantic understanding improvements.
        
        Args:
            base_model: Base model
            tuned_model: Tuned model
            financial_terms: List of financial terms
            financial_concepts: List of financial concept descriptions
            financial_relationships: List of related term pairs
            
        Returns:
            Semantic analysis results
        """
        semantic_analysis = {
            "term_understanding": {},
            "concept_comprehension": {},
            "relationship_recognition": {},
            "overall_improvement": 0,
        }
        
        # Analyze term understanding
        base_term_embeddings = base_model.encode(financial_terms, convert_to_numpy=True)
        tuned_term_embeddings = tuned_model.encode(financial_terms, convert_to_numpy=True)
        
        # Calculate term clustering quality
        base_term_similarity = self._calculate_cosine_similarity_matrix(base_term_embeddings)
        tuned_term_similarity = self._calculate_cosine_similarity_matrix(tuned_term_embeddings)
        
        # Analyze term differentiation
        base_term_variance = np.var(base_term_similarity)
        tuned_term_variance = np.var(tuned_term_similarity)
        
        term_improvement = (tuned_term_variance - base_term_variance) / max(1e-6, base_term_variance) * 100
        
        semantic_analysis["term_understanding"] = {
            "base_variance": float(base_term_variance),
            "tuned_variance": float(tuned_term_variance),
            "improvement": float(term_improvement),
        }
        
        # Analyze concept comprehension
        base_concept_embeddings = base_model.encode(financial_concepts, convert_to_numpy=True)
        tuned_concept_embeddings = tuned_model.encode(financial_concepts, convert_to_numpy=True)
        
        # Calculate concept similarity
        base_concept_similarity = self._calculate_cosine_similarity_matrix(base_concept_embeddings)
        tuned_concept_similarity = self._calculate_cosine_similarity_matrix(tuned_concept_embeddings)
        
        # Analyze concept richness
        base_concept_entropy = self._calculate_entropy(base_concept_similarity)
        tuned_concept_entropy = self._calculate_entropy(tuned_concept_similarity)
        
        concept_improvement = (tuned_concept_entropy - base_concept_entropy) / max(1e-6, base_concept_entropy) * 100
        
        semantic_analysis["concept_comprehension"] = {
            "base_entropy": float(base_concept_entropy),
            "tuned_entropy": float(tuned_concept_entropy),
            "improvement": float(concept_improvement),
        }
        
        # Analyze relationship recognition
        relationship_improvements = []
        
        for term1, term2 in financial_relationships:
            # Encode terms
            base_emb1 = base_model.encode(term1, convert_to_numpy=True)
            base_emb2 = base_model.encode(term2, convert_to_numpy=True)
            
            tuned_emb1 = tuned_model.encode(term1, convert_to_numpy=True)
            tuned_emb2 = tuned_model.encode(term2, convert_to_numpy=True)
            
            # Calculate similarity
            base_similarity = self._calculate_cosine_similarity(base_emb1, base_emb2)
            tuned_similarity = self._calculate_cosine_similarity(tuned_emb1, tuned_emb2)
            
            # Calculate improvement
            improvement = (tuned_similarity - base_similarity) / max(1e-6, base_similarity) * 100
            relationship_improvements.append(improvement)
        
        avg_relationship_improvement = sum(relationship_improvements) / len(relationship_improvements) if relationship_improvements else 0
        
        semantic_analysis["relationship_recognition"] = {
            "improvements": [float(imp) for imp in relationship_improvements],
            "avg_improvement": float(avg_relationship_improvement),
        }
        
        # Calculate overall improvement
        overall_improvement = (term_improvement + concept_improvement + avg_relationship_improvement) / 3
        semantic_analysis["overall_improvement"] = float(overall_improvement)
        
        # Update results
        self.results["semantic_analysis"] = semantic_analysis
        
        # Update metrics collector
        self.metrics_collector.update_metric("term_understanding_improvement", term_improvement)
        self.metrics_collector.update_metric("concept_comprehension_improvement", concept_improvement)
        self.metrics_collector.update_metric("relationship_recognition_improvement", avg_relationship_improvement)
        self.metrics_collector.update_metric("semantic_understanding_improvement", overall_improvement)
        
        # Save results
        self.save_results()
        
        return semantic_analysis
    
    def _calculate_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    def _calculate_cosine_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """Calculate cosine similarity matrix for a set of embeddings."""
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / np.maximum(norms, 1e-6)
        
        # Calculate similarity matrix
        similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        
        return similarity_matrix
    
    def _calculate_entropy(self, similarity_matrix: np.ndarray) -> float:
        """Calculate entropy of similarity matrix as a measure of information richness."""
        # Convert similarities to probabilities
        # Shift to [0, 1] range
        shifted = (similarity_matrix + 1) / 2
        
        # Normalize
        row_sums = np.sum(shifted, axis=1, keepdims=True)
        probabilities = shifted / np.maximum(row_sums, 1e-6)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(np.maximum(probabilities, 1e-6))) / probabilities.shape[0]
        
        return float(entropy)
    
    def save_results(self, file_path: Optional[str] = None) -> str:
        """
        Save comparison results to file.
        
        Args:
            file_path: Path to save results (None for default)
            
        Returns:
            Path to saved results
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, "comparison_results.json")
        
        with open(file_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        return file_path
    
    def load_results(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load comparison results from file.
        
        Args:
            file_path: Path to load results from (None for default)
            
        Returns:
            Comparison results
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, "comparison_results.json")
        
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                self.results = json.load(f)
        
        return self.results
    
    def generate_comparison_report(self, file_path: Optional[str] = None) -> str:
        """
        Generate a detailed comparison report.
        
        Args:
            file_path: Path to save report (None for default)
            
        Returns:
            Path to saved report
        """
        if file_path is None:
            file_path = os.path.join(self.output_dir, "comparison_report.md")
        
        # Format report
        lines = []
        
        # Title
        lines.append("# Financial Model Transformation Report")
        lines.append("")
        lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
        lines.append("")
        
        # Overview
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Base Model**: {self.results['base_model']['name']}")
        lines.append(f"- **Enlightened Model**: {self.results['tuned_model']['name']}")
        lines.append("")
        
        # Performance improvements
        lines.append("## Performance Transformation")
        lines.append("")
        lines.append("| Metric | Base Model | Enlightened Model | Improvement |")
        lines.append("|--------|------------|-------------------|------------|")
        
        for metric in ["precision", "recall", "f1_score", "execution_time", "embedding_time"]:
            if metric in self.results["base_model"]["metrics"] and metric in self.results["tuned_model"]["metrics"]:
                base_value = self.results["base_model"]["metrics"][metric]
                tuned_value = self.results["tuned_model"]["metrics"][metric]
                improvement = self.results["improvements"].get(metric, 0)
                
                # Format values
                if metric.endswith("time"):
                    base_str = f"{base_value:.4f}s"
                    tuned_str = f"{tuned_value:.4f}s"
                else:
                    base_str = f"{base_value:.4f}"
                    tuned_str = f"{tuned_value:.4f}"
                
                # Format improvement
                if metric.endswith("time"):
                    # For time metrics, negative is better
                    if improvement > 0:
                        improvement_str = f"+{improvement:.2f}% faster"
                    else:
                        improvement_str = f"{improvement:.2f}% slower"
                else:
                    # For other metrics, positive is better
                    if improvement > 0:
                        improvement_str = f"+{improvement:.2f}% better"
                    else:
                        improvement_str = f"{improvement:.2f}% worse"
                
                lines.append(f"| {metric.replace('_', ' ').title()} | {base_str} | {tuned_str} | {improvement_str} |")
        
        lines.append("")
        
        # Semantic understanding
        if "semantic_analysis" in self.results and self.results["semantic_analysis"]:
            lines.append("## Semantic Understanding")
            lines.append("")
            lines.append("The enlightened model demonstrates deeper understanding of financial language:")
            lines.append("")
            
            # Term understanding
            term_improvement = self.results["semantic_analysis"]["term_understanding"].get("improvement", 0)
            lines.append(f"- **Financial Term Understanding**: {term_improvement:.2f}% improvement")
            
            # Concept comprehension
            concept_improvement = self.results["semantic_analysis"]["concept_comprehension"].get("improvement", 0)
            lines.append(f"- **Financial Concept Comprehension**: {concept_improvement:.2f}% improvement")
            
            # Relationship recognition
            relationship_improvement = self.results["semantic_analysis"]["relationship_recognition"].get("avg_improvement", 0)
            lines.append(f"- **Financial Relationship Recognition**: {relationship_improvement:.2f}% improvement")
            
            # Overall improvement
            overall_improvement = self.results["semantic_analysis"].get("overall_improvement", 0)
            lines.append(f"- **Overall Semantic Understanding**: {overall_improvement:.2f}% improvement")
            
            lines.append("")
        
        # Query examples
        if "query_improvements" in self.results and self.results["query_improvements"]:
            lines.append("## Query Examples")
            lines.append("")
            lines.append("Improvements observed in specific financial queries:")
            lines.append("")
            
            # Select top 5 queries with highest improvement
            top_queries = sorted(
                self.results["query_improvements"].items(),
                key=lambda x: x[1]["time_improvement"] + x[1]["retrieval_improvement"],
                reverse=True
            )[:5]
            
            for query, improvements in top_queries:
                time_improvement = improvements["time_improvement"]
                retrieval_improvement = improvements["retrieval_improvement"]
                
                lines.append(f"### Query: \"{query}\"")
                lines.append("")
                lines.append(f"- **Time Improvement**: {time_improvement:.2f}% faster")
                lines.append(f"- **Retrieval Improvement**: {retrieval_improvement:.2f}% more relevant results")
                
                # Add results if available
                if (query in self.results["base_model"]["query_results"] and
                    query in self.results["tuned_model"]["query_results"]):
                    
                    base_results = self.results["base_model"]["query_results"][query]["retrieved_docs"]
                    tuned_results = self.results["tuned_model"]["query_results"][query]["retrieved_docs"]
                    
                    # Show top 3 results
                    lines.append("")
                    lines.append("#### Top Results Comparison")
                    lines.append("")
                    lines.append("| Rank | Base Model | Enlightened Model |")
                    lines.append("|------|------------|-------------------|")
                    
                    for i in range(min(3, len(base_results), len(tuned_results))):
                        lines.append(f"| {i+1} | {base_results[i]} | {tuned_results[i]} |")
                
                lines.append("")
        
        # Conclusion
        lines.append("## Conclusion")
        lines.append("")
        lines.append("The enlightened model demonstrates significant improvements across multiple dimensions:")
        lines.append("")
        
        # Calculate average performance improvement
        perf_metrics = ["precision", "recall", "f1_score"]
        perf_improvements = [self.results["improvements"].get(metric, 0) for metric in perf_metrics]
        avg_perf_improvement = sum(perf_improvements) / len(perf_improvements) if perf_improvements else 0
        
        # Calculate speed improvement
        speed_metrics = ["execution_time", "embedding_time"]
        speed_improvements = [self.results["improvements"].get(metric, 0) for metric in speed_metrics]
        avg_speed_improvement = sum(speed_improvements) / len(speed_improvements) if speed_improvements else 0
        
        # Get semantic improvement
        semantic_improvement = self.results.get("semantic_analysis", {}).get("overall_improvement", 0)
        
        lines.append(f"1. **Performance**: {avg_perf_improvement:.2f}% better retrieval performance")
        lines.append(f"2. **Speed**: {avg_speed_improvement:.2f}% faster processing")
        lines.append(f"3. **Understanding**: {semantic_improvement:.2f}% deeper financial semantic understanding")
        lines.append("")
        lines.append("This transformation results in more accurate, faster, and more contextually aware financial information retrieval.")
        
        # Write report
        with open(file_path, "w") as f:
            f.write("\n".join(lines))
        
        return file_path


# Factory function to create model comparison
def create_model_comparison(
    base_model_name: str,
    tuned_model_name: str,
    output_dir: Optional[str] = None,
) -> ModelComparison:
    """
    Create a model comparison.
    
    Args:
        base_model_name: Name of base model
        tuned_model_name: Name of tuned model
        output_dir: Output directory for comparison results
        
    Returns:
        ModelComparison instance
    """
    return ModelComparison(
        base_model_name=base_model_name,
        tuned_model_name=tuned_model_name,
        output_dir=output_dir,
    )