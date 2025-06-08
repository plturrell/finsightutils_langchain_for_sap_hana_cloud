"""
Tests for optimization components in the SAP HANA Cloud LangChain integration.

This module tests:
1. Data valuation with DVRL
2. Interpretable embeddings with Neural Additive Models 
3. Optimized hyperparameters with opt_list
4. Model compression with state_of_sparsity
"""

import os
import unittest
import numpy as np
from unittest.mock import patch, MagicMock, Mock
from typing import List

from langchain_core.documents import Document

# Skip tests if optimization dependencies aren't available
HAS_OPTIMIZATION = True
try:
    from langchain_hana.optimization.data_valuation import DVRLDataValuation
    from langchain_hana.optimization.interpretable_embeddings import NAMEmbeddings
    from langchain_hana.optimization.hyperparameters import OptimizedHyperparameters
    from langchain_hana.optimization.model_compression import SparseEmbeddingModel
except ImportError:
    HAS_OPTIMIZATION = False


# Create a dummy embedding model for testing
class DummyEmbeddings:
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # Generate deterministic but unique embeddings for each text
        return [
            np.array([hash(text) % 100, len(text) % 50, 0.5]).astype(float) / 100
            for text in texts
        ]
    
    def embed_query(self, text: str) -> List[float]:
        # Generate deterministic embedding for query
        return np.array([hash(text) % 100, len(text) % 50, 0.5]).astype(float) / 100


# Test suite for optimization components
@unittest.skipIf(not HAS_OPTIMIZATION, "Optimization dependencies not available")
class TestOptimizationComponents(unittest.TestCase):
    """Test suite for optimization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create dummy documents
        self.documents = [
            Document(page_content="This is a high quality document about finance.", 
                     metadata={"id": "doc_1", "category": "finance"}),
            Document(page_content="Short text.", 
                     metadata={"id": "doc_2", "category": "other"}),
            Document(page_content="This document contains relevant information about the topic.", 
                     metadata={"id": "doc_3", "category": "general"}),
        ]
        
        # Create dummy embedding model
        self.embedding_model = DummyEmbeddings()
        
        # Create temp directory for cache
        os.makedirs("temp_test_cache", exist_ok=True)
    
    def tearDown(self):
        """Clean up resources."""
        # Clean up temp files
        import shutil
        if os.path.exists("temp_test_cache"):
            shutil.rmtree("temp_test_cache")
    
    def test_data_valuation(self):
        """Test data valuation component."""
        with patch("langchain_hana.optimization.data_valuation.HAS_DVRL", False):
            # Test with fallback mechanism
            data_valuation = DVRLDataValuation(
                embedding_dimension=3,
                value_threshold=0.5,
                cache_file="temp_test_cache/values.json",
            )
            
            # Test document value computation
            values = data_valuation.compute_document_values(self.documents)
            
            # Check basic properties
            self.assertEqual(len(values), len(self.documents))
            self.assertTrue(all(0 <= v <= 1 for v in values))
            
            # Test filtering
            valuable_docs = data_valuation.filter_valuable_documents(
                self.documents,
                threshold=0.5,
            )
            
            # Check filtering results
            self.assertLessEqual(len(valuable_docs), len(self.documents))
    
    def test_interpretable_embeddings(self):
        """Test interpretable embeddings component."""
        with patch("langchain_hana.optimization.interpretable_embeddings.HAS_NAM", False):
            # Test with fallback mechanism
            interpretable_embeddings = NAMEmbeddings(
                base_embeddings=self.embedding_model,
                dimension=3,
                num_features=2,
                cache_dir="temp_test_cache",
            )
            
            # Test document embedding
            embeddings = interpretable_embeddings.embed_documents(
                [doc.page_content for doc in self.documents]
            )
            
            # Check basic properties
            self.assertEqual(len(embeddings), len(self.documents))
            self.assertEqual(len(embeddings[0]), 2)  # num_features
            
            # Test query embedding
            query_embedding = interpretable_embeddings.embed_query("test query")
            self.assertEqual(len(query_embedding), 2)  # num_features
            
            # Test similarity explanation
            explanation = interpretable_embeddings.explain_similarity(
                "test query",
                self.documents[0].page_content,
                top_k=1,
            )
            
            # Check explanation structure
            self.assertIn("similarity_score", explanation)
            self.assertIn("top_matching_features", explanation)
            self.assertIn("least_matching_features", explanation)
    
    def test_optimized_hyperparameters(self):
        """Test optimized hyperparameters component."""
        with patch("langchain_hana.optimization.hyperparameters.HAS_OPT_LIST", False):
            # Test with fallback mechanism
            optimizer = OptimizedHyperparameters(
                cache_file="temp_test_cache/hyperparams.json",
            )
            
            # Test learning rate optimization
            lr = optimizer.get_learning_rate(
                model_size=1000000,
                batch_size=32,
            )
            
            # Check basic properties
            self.assertTrue(0 < lr < 1)
            
            # Test batch size optimization
            batch_size = optimizer.get_batch_size(
                model_size=1000000,
            )
            
            # Check basic properties
            self.assertTrue(batch_size > 0)
            
            # Test embedding parameters
            params = optimizer.get_embedding_parameters(
                embedding_dimension=768,
                vocabulary_size=30000,
                max_sequence_length=512,
            )
            
            # Check parameters structure
            self.assertIn("learning_rate", params)
            self.assertIn("dropout_rate", params)
            self.assertIn("hidden_dimension", params)
    
    def test_model_compression(self):
        """Test model compression component."""
        with patch("langchain_hana.optimization.model_compression.HAS_SOS", False):
            # Test with fallback mechanism
            compressed_embeddings = SparseEmbeddingModel(
                base_embeddings=self.embedding_model,
                compression_ratio=0.5,
                cache_dir="temp_test_cache",
            )
            
            # Test document embedding
            embeddings = compressed_embeddings.embed_documents(
                [doc.page_content for doc in self.documents]
            )
            
            # Check basic properties
            self.assertEqual(len(embeddings), len(self.documents))
            
            # Test query embedding
            query_embedding = compressed_embeddings.embed_query("test query")
            
            # Check compression stats
            stats = compressed_embeddings.get_compression_stats()
            self.assertIn("compression_ratio", stats)
            self.assertIn("total_sparsity", stats)


# Run tests
if __name__ == "__main__":
    unittest.main()