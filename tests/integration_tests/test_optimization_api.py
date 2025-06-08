"""
Integration tests for optimization API endpoints.

This module tests:
1. Data valuation API
2. Interpretable embeddings API
3. Optimized hyperparameters API
4. Model compression API
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Skip tests if FastAPI or TestClient are not available
HAS_FASTAPI = True
try:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
except ImportError:
    HAS_FASTAPI = False


# Test suite for optimization API endpoints
@unittest.skipIf(not HAS_FASTAPI, "FastAPI and TestClient not available")
class TestOptimizationAPI(unittest.TestCase):
    """Test suite for optimization API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Import API application
        from api.core.main import app
        cls.client = TestClient(app)
    
    def test_data_valuation_endpoint(self):
        """Test data valuation endpoint."""
        # Create test request data
        request_data = {
            "documents": [
                {"page_content": "This is a high quality document about finance.", 
                 "metadata": {"id": "doc_1", "category": "finance"}},
                {"page_content": "Short text.", 
                 "metadata": {"id": "doc_2", "category": "other"}},
            ],
            "threshold": 0.7,
            "top_k": 1
        }
        
        # Mock the DVRLDataValuation class
        with patch("api.routes.optimization.DVRLDataValuation") as mock_dvrl:
            # Setup mock return values
            mock_instance = mock_dvrl.return_value
            mock_instance.compute_document_values.return_value = [0.8, 0.4]
            mock_instance.filter_valuable_documents.return_value = [
                MagicMock(page_content="This is a high quality document about finance.",
                         metadata={"id": "doc_1", "category": "finance", "dvrl_value": 0.8})
            ]
            
            # Make API request
            response = self.client.post("/optimization/data-valuation", json=request_data)
            
            # Check response
            self.assertEqual(response.status_code, 200)
            response_data = response.json()
            self.assertEqual(len(response_data["document_values"]), 2)
            self.assertEqual(len(response_data["valuable_documents"]), 1)
            self.assertEqual(response_data["total_documents"], 2)
            self.assertEqual(response_data["valuable_count"], 1)
    
    def test_explain_similarity_endpoint(self):
        """Test similarity explanation endpoint."""
        # Create test request data
        request_data = {
            "query": "Tell me about finance",
            "document": "This document discusses financial markets and investment strategies.",
            "top_k": 2
        }
        
        # Mock the NAMEmbeddings class
        with patch("api.routes.optimization.NAMEmbeddings") as mock_nam:
            # Setup mock return values
            mock_instance = mock_nam.return_value
            mock_instance.explain_similarity.return_value = {
                "similarity_score": 0.75,
                "top_matching_features": [("feature_1", 0.4), ("feature_2", 0.3)],
                "least_matching_features": [("feature_3", -0.1), ("feature_4", -0.05)],
                "query": "Tell me about finance",
                "document": "This document discusses financial markets and investment strategies."
            }
            
            # Make API request
            response = self.client.post("/optimization/explain-similarity", json=request_data)
            
            # Check response
            self.assertEqual(response.status_code, 200)
            response_data = response.json()
            self.assertAlmostEqual(response_data["similarity_score"], 0.75)
            self.assertEqual(len(response_data["top_matching_features"]), 2)
            self.assertEqual(len(response_data["least_matching_features"]), 2)
    
    def test_optimized_hyperparameters_endpoint(self):
        """Test optimized hyperparameters endpoint."""
        # Create test request data
        request_data = {
            "model_size": 10000000,
            "batch_size": 32,
            "dataset_size": 50000,
            "embedding_dimension": 768,
            "vocabulary_size": 30000,
            "max_sequence_length": 512
        }
        
        # Mock the OptimizedHyperparameters class
        with patch("api.routes.optimization.OptimizedHyperparameters") as mock_opt:
            # Setup mock return values
            mock_instance = mock_opt.return_value
            mock_instance.get_learning_rate.return_value = 0.0003
            mock_instance.get_batch_size.return_value = 64
            mock_instance.get_embedding_parameters.return_value = {
                "dropout_rate": 0.2,
                "weight_decay": 0.01,
                "hidden_dimension": 3072
            }
            mock_instance.get_training_schedule.return_value = {
                "warmup_steps": 100,
                "total_steps": 15000,
                "base_learning_rate": 0.0003
            }
            
            # Make API request
            response = self.client.post("/optimization/optimized-hyperparameters", json=request_data)
            
            # Check response
            self.assertEqual(response.status_code, 200)
            response_data = response.json()
            self.assertEqual(response_data["learning_rate"], 0.0003)
            self.assertEqual(response_data["batch_size"], 64)
            self.assertIn("dropout_rate", response_data["embedding_parameters"])
            self.assertIn("warmup_steps", response_data["training_schedule"])
    
    def test_compressed_embeddings_endpoint(self):
        """Test compressed embeddings endpoint."""
        # Create test request data
        request_data = {
            "texts": [
                "This is the first document to embed.",
                "This is the second document to embed."
            ],
            "compression_ratio": 0.7,
            "compression_strategy": "magnitude"
        }
        
        # Mock the SparseEmbeddingModel class
        with patch("api.routes.optimization.SparseEmbeddingModel") as mock_sparse:
            # Setup mock return values
            mock_instance = mock_sparse.return_value
            mock_instance.embed_documents.return_value = [
                [0.1, 0.0, 0.3, 0.0, 0.5],
                [0.0, 0.2, 0.0, 0.4, 0.0]
            ]
            mock_instance.get_compression_stats.return_value = {
                "compression_ratio": 0.7,
                "compression_strategy": "magnitude",
                "total_sparsity": 0.6,
                "compressed_shapes": {
                    "shape_5": {
                        "elements": 10,
                        "nonzeros": 4,
                        "sparsity": 0.6
                    }
                },
                "cache_size": 2,
                "cache_dir": None
            }
            
            # Make API request
            response = self.client.post("/optimization/compressed-embeddings", json=request_data)
            
            # Check response
            self.assertEqual(response.status_code, 200)
            response_data = response.json()
            self.assertEqual(len(response_data["embeddings"]), 2)
            self.assertEqual(len(response_data["embeddings"][0]), 5)
            self.assertIn("compression_ratio", response_data["compression_stats"])
            self.assertIn("total_sparsity", response_data["compression_stats"])


# Run tests
if __name__ == "__main__":
    unittest.main()