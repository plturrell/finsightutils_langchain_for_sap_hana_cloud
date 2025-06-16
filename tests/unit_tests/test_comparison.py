"""Unit tests for the comparison module."""

import os
import json
import tempfile
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Import the module to test
from langchain_hana.financial.comparison import (
    ModelComparison,
    create_model_comparison,
)


class TestModelComparison(unittest.TestCase):
    """Tests for the ModelComparison class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, "comparison_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create comparison object
        self.comparison = ModelComparison(
            base_model_name="base_model",
            tuned_model_name="tuned_model",
            output_dir=self.output_dir,
        )
        
        # Create mock models
        self.base_model = MagicMock()
        self.tuned_model = MagicMock()

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up temp files
        if os.path.exists(self.output_dir):
            for file in os.listdir(self.output_dir):
                os.unlink(os.path.join(self.output_dir, file))
            os.rmdir(self.output_dir)
        
        os.rmdir(self.temp_dir)

    def test_compare_models_on_queries(self):
        """Test comparing models on queries."""
        # Set up mock models to return specific embeddings
        def mock_encode(text, **kwargs):
            if text == "query1":
                return np.array([1.0, 0.0, 0.0])
            elif text == "query2":
                return np.array([0.0, 1.0, 0.0])
            elif text.startswith("doc1"):
                return np.array([0.9, 0.1, 0.0])
            elif text.startswith("doc2"):
                return np.array([0.1, 0.9, 0.0])
            elif text.startswith("doc3"):
                return np.array([0.5, 0.5, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0])
        
        self.base_model.encode.side_effect = mock_encode
        
        # Tuned model returns slightly better embeddings
        def tuned_mock_encode(text, **kwargs):
            if text == "query1":
                return np.array([0.95, 0.05, 0.0])
            elif text == "query2":
                return np.array([0.05, 0.95, 0.0])
            elif text.startswith("doc1"):
                return np.array([0.95, 0.05, 0.0])
            elif text.startswith("doc2"):
                return np.array([0.05, 0.95, 0.0])
            elif text.startswith("doc3"):
                return np.array([0.5, 0.5, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0])
        
        self.tuned_model.encode.side_effect = tuned_mock_encode
        
        # Test data
        queries = ["query1", "query2"]
        relevant_docs = {
            "query1": ["doc1", "doc3"],
            "query2": ["doc2", "doc4"]
        }
        doc_texts = {
            "doc1": "Document 1 content",
            "doc2": "Document 2 content",
            "doc3": "Document 3 content",
            "doc4": "Document 4 content",
            "doc5": "Document 5 content"
        }
        
        # Compare models
        results = self.comparison.compare_models_on_queries(
            queries=queries,
            relevant_docs=relevant_docs,
            base_model=self.base_model,
            tuned_model=self.tuned_model,
            doc_texts=doc_texts,
            k=2,
        )
        
        # Check that results were generated
        self.assertIn("base_model", results)
        self.assertIn("tuned_model", results)
        self.assertIn("improvements", results)
        self.assertIn("query_improvements", results)
        
        # Check specific improvements
        self.assertIn("execution_time", results["improvements"])
        self.assertIn("precision", results["improvements"])
        self.assertIn("recall", results["improvements"])
        self.assertIn("f1_score", results["improvements"])
        
        # Check query improvements
        self.assertIn("query1", results["query_improvements"])
        self.assertIn("query2", results["query_improvements"])
        self.assertIn("time_improvement", results["query_improvements"]["query1"])
        self.assertIn("retrieval_improvement", results["query_improvements"]["query1"])

    def test_analyze_semantic_understanding(self):
        """Test analyzing semantic understanding."""
        # Set up mock models to return specific embeddings
        def mock_encode(text, **kwargs):
            if "market" in text:
                return np.array([0.9, 0.1, 0.0])
            elif "credit" in text:
                return np.array([0.1, 0.9, 0.0])
            elif "risk" in text:
                return np.array([0.5, 0.5, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0])
        
        self.base_model.encode.side_effect = mock_encode
        
        # Tuned model returns slightly better embeddings
        def tuned_mock_encode(text, **kwargs):
            if "market" in text:
                return np.array([0.95, 0.05, 0.0])
            elif "credit" in text:
                return np.array([0.05, 0.95, 0.0])
            elif "risk" in text:
                return np.array([0.5, 0.5, 0.0])
            else:
                return np.array([0.0, 0.0, 1.0])
        
        self.tuned_model.encode.side_effect = tuned_mock_encode
        
        # Test data
        financial_terms = [
            "market risk",
            "credit risk",
            "operational risk"
        ]
        
        financial_concepts = [
            "Market risk involves price fluctuations",
            "Credit risk involves default possibilities",
            "Operational risk involves internal failures"
        ]
        
        financial_relationships = [
            ("market risk", "volatility"),
            ("credit risk", "default"),
            ("operational risk", "process")
        ]
        
        # Analyze semantic understanding
        results = self.comparison.analyze_semantic_understanding(
            base_model=self.base_model,
            tuned_model=self.tuned_model,
            financial_terms=financial_terms,
            financial_concepts=financial_concepts,
            financial_relationships=financial_relationships,
        )
        
        # Check that results were generated
        self.assertIn("term_understanding", results)
        self.assertIn("concept_comprehension", results)
        self.assertIn("relationship_recognition", results)
        self.assertIn("overall_improvement", results)
        
        # Check specific improvements
        self.assertIn("base_variance", results["term_understanding"])
        self.assertIn("tuned_variance", results["term_understanding"])
        self.assertIn("improvement", results["term_understanding"])
        
        self.assertIn("base_entropy", results["concept_comprehension"])
        self.assertIn("tuned_entropy", results["concept_comprehension"])
        self.assertIn("improvement", results["concept_comprehension"])
        
        self.assertIn("improvements", results["relationship_recognition"])
        self.assertIn("avg_improvement", results["relationship_recognition"])
        
        # Check overall improvement
        self.assertGreaterEqual(results["overall_improvement"], 0.0)

    def test_save_and_load_results(self):
        """Test saving and loading comparison results."""
        # Create sample results
        self.comparison.results = {
            "base_model": {
                "name": "base_model",
                "metrics": {
                    "precision": 0.7,
                    "recall": 0.6
                }
            },
            "tuned_model": {
                "name": "tuned_model",
                "metrics": {
                    "precision": 0.8,
                    "recall": 0.7
                }
            },
            "improvements": {
                "precision": 14.28,
                "recall": 16.67
            }
        }
        
        # Save results
        file_path = self.comparison.save_results()
        
        # Check that file was created
        self.assertTrue(os.path.exists(file_path))
        
        # Create a new comparison object
        new_comparison = ModelComparison(
            base_model_name="base_model",
            tuned_model_name="tuned_model",
            output_dir=self.output_dir,
        )
        
        # Load results
        loaded_results = new_comparison.load_results(file_path)
        
        # Check that results were loaded correctly
        self.assertEqual(loaded_results["base_model"]["name"], "base_model")
        self.assertEqual(loaded_results["tuned_model"]["name"], "tuned_model")
        self.assertAlmostEqual(loaded_results["improvements"]["precision"], 14.28)
        self.assertAlmostEqual(loaded_results["improvements"]["recall"], 16.67)

    def test_generate_comparison_report(self):
        """Test generating a comparison report."""
        # Create sample results
        self.comparison.results = {
            "base_model": {
                "name": "base_model",
                "metrics": {
                    "precision": 0.7,
                    "recall": 0.6,
                    "f1_score": 0.65,
                    "execution_time": 0.2
                },
                "query_results": {
                    "query1": {
                        "execution_time": 0.2,
                        "retrieved_docs": ["doc1", "doc3"]
                    },
                    "query2": {
                        "execution_time": 0.3,
                        "retrieved_docs": ["doc2", "doc4"]
                    }
                }
            },
            "tuned_model": {
                "name": "tuned_model",
                "metrics": {
                    "precision": 0.8,
                    "recall": 0.7,
                    "f1_score": 0.75,
                    "execution_time": 0.15
                },
                "query_results": {
                    "query1": {
                        "execution_time": 0.15,
                        "retrieved_docs": ["doc1", "doc2"]
                    },
                    "query2": {
                        "execution_time": 0.25,
                        "retrieved_docs": ["doc2", "doc5"]
                    }
                }
            },
            "improvements": {
                "precision": 14.28,
                "recall": 16.67,
                "f1_score": 15.38,
                "execution_time": 25.0
            },
            "query_improvements": {
                "query1": {
                    "time_improvement": 25.0,
                    "retrieval_improvement": 20.0
                },
                "query2": {
                    "time_improvement": 16.67,
                    "retrieval_improvement": 15.0
                }
            },
            "semantic_analysis": {
                "overall_improvement": 25.0
            }
        }
        
        # Generate report
        report_path = self.comparison.generate_comparison_report()
        
        # Check that report was created
        self.assertTrue(os.path.exists(report_path))
        
        # Check report contents
        with open(report_path, "r") as f:
            content = f.read()
        
        # Check that key information is in the report
        self.assertIn("# Financial Model Transformation Report", content)
        self.assertIn("## Overview", content)
        self.assertIn("## Performance Transformation", content)
        self.assertIn("| Metric | Base Model | Enlightened Model | Improvement |", content)
        self.assertIn("base_model", content)
        self.assertIn("tuned_model", content)
        self.assertIn("## Query Examples", content)
        self.assertIn("## Conclusion", content)

    def test_calculate_cosine_similarity(self):
        """Test calculating cosine similarity."""
        # Create test embeddings
        embedding1 = np.array([1.0, 0.0, 0.0])
        embedding2 = np.array([0.0, 1.0, 0.0])
        embedding3 = np.array([1.0, 1.0, 0.0])
        
        # Calculate similarities
        sim12 = self.comparison._calculate_cosine_similarity(embedding1, embedding2)
        sim13 = self.comparison._calculate_cosine_similarity(embedding1, embedding3)
        sim23 = self.comparison._calculate_cosine_similarity(embedding2, embedding3)
        
        # Check similarities
        self.assertAlmostEqual(sim12, 0.0)
        self.assertAlmostEqual(sim13, 1.0 / np.sqrt(2))
        self.assertAlmostEqual(sim23, 1.0 / np.sqrt(2))
        
        # Test with zero vectors
        zero_vec = np.array([0.0, 0.0, 0.0])
        zero_sim = self.comparison._calculate_cosine_similarity(zero_vec, embedding1)
        self.assertEqual(zero_sim, 0.0)

    def test_calculate_cosine_similarity_matrix(self):
        """Test calculating cosine similarity matrix."""
        # Create test embeddings
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ])
        
        # Calculate similarity matrix
        sim_matrix = self.comparison._calculate_cosine_similarity_matrix(embeddings)
        
        # Check similarity matrix shape
        self.assertEqual(sim_matrix.shape, (3, 3))
        
        # Check diagonal elements (self-similarity)
        self.assertAlmostEqual(sim_matrix[0, 0], 1.0)
        self.assertAlmostEqual(sim_matrix[1, 1], 1.0)
        self.assertAlmostEqual(sim_matrix[2, 2], 1.0)
        
        # Check off-diagonal elements
        self.assertAlmostEqual(sim_matrix[0, 1], 0.0)
        self.assertAlmostEqual(sim_matrix[0, 2], 1.0 / np.sqrt(2))
        self.assertAlmostEqual(sim_matrix[1, 2], 1.0 / np.sqrt(2))

    def test_calculate_entropy(self):
        """Test calculating entropy of similarity matrix."""
        # Create test similarity matrix (already normalized to [0, 1])
        sim_matrix = np.array([
            [1.0, 0.5, 0.2],
            [0.5, 1.0, 0.8],
            [0.2, 0.8, 1.0]
        ])
        
        # Calculate entropy
        entropy = self.comparison._calculate_entropy(sim_matrix)
        
        # Entropy should be positive
        self.assertGreater(entropy, 0.0)

    def test_factory_function(self):
        """Test the factory function for creating a model comparison."""
        # Create comparison using factory function
        comparison = create_model_comparison(
            base_model_name="base_model",
            tuned_model_name="tuned_model",
            output_dir=self.output_dir,
        )
        
        # Check that comparison was created correctly
        self.assertIsInstance(comparison, ModelComparison)
        self.assertEqual(comparison.base_model_name, "base_model")
        self.assertEqual(comparison.tuned_model_name, "tuned_model")
        self.assertEqual(comparison.output_dir, self.output_dir)


if __name__ == "__main__":
    unittest.main()