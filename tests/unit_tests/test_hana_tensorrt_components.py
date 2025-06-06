"""
Unit tests for SAP HANA Cloud TensorRT GPU acceleration components.

These tests verify the functionality of GPU-accelerated embeddings and vectorstore
implementations for SAP HANA Cloud integration.
"""

import unittest
import pytest
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from unittest.mock import patch, MagicMock, Mock

# Import components to test
from langchain_hana.gpu.hana_tensorrt_embeddings import HanaTensorRTEmbeddings
from langchain_hana.gpu.hana_tensorrt_vectorstore import HanaTensorRTVectorStore
from langchain_hana.gpu.vector_serialization import (
    serialize_vector, 
    deserialize_vector,
    serialize_vectors_batch,
    deserialize_vectors_batch
)


class TestHanaTensorRTEmbeddings(unittest.TestCase):
    """Tests for HanaTensorRTEmbeddings class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock CUDA availability
        self.cuda_patcher = patch('torch.cuda.is_available', return_value=True)
        self.mock_cuda_available = self.cuda_patcher.start()
        
        # Mock TensorRT embeddings creation
        self.tensorrt_patcher = patch('langchain_hana.gpu.tensorrt_embeddings.TensorRTEmbeddings')
        self.mock_tensorrt_embeddings = self.tensorrt_patcher.start()
        
        # Mock embedding model with expected output
        self.embedding_dim = 384
        self.mock_model = MagicMock()
        self.mock_model.embed_documents.return_value = [
            [0.1] * self.embedding_dim, 
            [0.2] * self.embedding_dim
        ]
        self.mock_tensorrt_embeddings.return_value = self.mock_model
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.cuda_patcher.stop()
        self.tensorrt_patcher.stop()
    
    def test_initialization(self):
        """Test initialization with default parameters."""
        embeddings = HanaTensorRTEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Verify TensorRTEmbeddings was initialized
        self.mock_tensorrt_embeddings.assert_called_once()
        
        # Verify model attributes
        self.assertEqual(embeddings.model_name, "sentence-transformers/all-MiniLM-L6-v2")
    
    def test_embed_documents(self):
        """Test embedding documents."""
        embeddings = HanaTensorRTEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32
        )
        
        texts = ["Document 1", "Document 2"]
        result = embeddings.embed_documents(texts)
        
        # Verify embedding shape and values
        self.assertEqual(len(result), len(texts))
        self.assertEqual(len(result[0]), self.embedding_dim)
        self.assertEqual(result, self.mock_model.embed_documents.return_value)
    
    def test_embed_query(self):
        """Test embedding a query."""
        embeddings = HanaTensorRTEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.mock_model.embed_query.return_value = [0.3] * self.embedding_dim
        
        query = "Test query"
        result = embeddings.embed_query(query)
        
        # Verify embedding shape and call
        self.assertEqual(len(result), self.embedding_dim)
        self.mock_model.embed_query.assert_called_once_with(query)
    
    def test_precision_modes(self):
        """Test different precision modes."""
        for precision in ["fp32", "fp16", "int8"]:
            embeddings = HanaTensorRTEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                precision=precision
            )
            
            # Verify precision is passed correctly
            self.assertEqual(embeddings.precision, precision)
    
    def test_multi_gpu_support(self):
        """Test multi-GPU support."""
        # Mock multiple GPUs
        with patch('torch.cuda.device_count', return_value=2):
            embeddings = HanaTensorRTEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                multi_gpu=True
            )
            
            # Verify multi-GPU is enabled
            self.assertTrue(embeddings.multi_gpu)
            
            # Mock embed_documents to verify multi-GPU logic
            texts = ["Doc 1", "Doc 2", "Doc 3", "Doc 4"]
            result = embeddings.embed_documents(texts)
            
            # Verify result matches expected output
            self.assertEqual(len(result), len(texts))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required for this test")
class TestVectorSerialization:
    """Tests for vector serialization utilities."""
    
    def test_serialize_deserialize_fp32(self):
        """Test serialization and deserialization with float32 precision."""
        # Create a sample vector
        vector = np.random.rand(384).astype(np.float32)
        
        # Serialize and deserialize
        serialized = serialize_vector(vector, precision="float32")
        deserialized = deserialize_vector(serialized)
        
        # Verify
        assert isinstance(serialized, bytes)
        assert np.allclose(vector, deserialized, atol=1e-5)
    
    def test_serialize_deserialize_fp16(self):
        """Test serialization and deserialization with float16 precision."""
        # Create a sample vector
        vector = np.random.rand(384).astype(np.float32)
        
        # Serialize and deserialize
        serialized = serialize_vector(vector, precision="float16")
        deserialized = deserialize_vector(serialized)
        
        # Verify - allow slightly higher tolerance due to precision loss
        assert isinstance(serialized, bytes)
        assert np.allclose(vector, deserialized, atol=1e-3)
        
        # Verify size reduction
        fp32_size = len(serialize_vector(vector, precision="float32"))
        fp16_size = len(serialized)
        assert fp16_size < fp32_size
    
    def test_batch_serialization(self):
        """Test batch serialization and deserialization."""
        # Create sample vectors
        vectors = [np.random.rand(384).astype(np.float32) for _ in range(10)]
        
        # Serialize and deserialize batch
        serialized_batch = serialize_vectors_batch(vectors, precision="float16")
        deserialized_batch = deserialize_vectors_batch(serialized_batch)
        
        # Verify
        assert len(deserialized_batch) == len(vectors)
        for original, deserialized in zip(vectors, deserialized_batch):
            assert np.allclose(original, deserialized, atol=1e-3)


class TestHanaTensorRTVectorStore(unittest.TestCase):
    """Tests for HanaTensorRTVectorStore class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock database connection
        self.mock_conn = MagicMock()
        
        # Mock cursor and execution
        self.mock_cursor = MagicMock()
        self.mock_conn.cursor.return_value = self.mock_cursor
        
        # Mock embeddings provider
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_documents.return_value = [
            [0.1] * 384, 
            [0.2] * 384
        ]
        self.mock_embeddings.embed_query.return_value = [0.3] * 384
        
    def test_initialization(self):
        """Test initialization with default parameters."""
        vectorstore = HanaTensorRTVectorStore(
            connection=self.mock_conn,
            embedding=self.mock_embeddings,
            table_name="TEST_VECTORS"
        )
        
        # Verify attributes
        self.assertEqual(vectorstore.table_name, "TEST_VECTORS")
        self.assertEqual(vectorstore.connection, self.mock_conn)
    
    def test_add_texts(self):
        """Test adding texts to the vectorstore."""
        vectorstore = HanaTensorRTVectorStore(
            connection=self.mock_conn,
            embedding=self.mock_embeddings,
            table_name="TEST_VECTORS",
            batch_size=32
        )
        
        # Mock executemany to return IDs
        self.mock_cursor.executemany.return_value = None
        self.mock_cursor.fetchall.return_value = [[1], [2]]
        
        texts = ["Document 1", "Document 2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        
        ids = vectorstore.add_texts(texts, metadatas)
        
        # Verify embeddings were generated
        self.mock_embeddings.embed_documents.assert_called_once_with(texts)
        
        # Verify records were inserted
        self.mock_cursor.executemany.assert_called()
        
        # Verify IDs were returned
        self.assertEqual(len(ids), len(texts))
    
    def test_similarity_search(self):
        """Test similarity search."""
        vectorstore = HanaTensorRTVectorStore(
            connection=self.mock_conn,
            embedding=self.mock_embeddings,
            table_name="TEST_VECTORS"
        )
        
        # Mock query results
        mock_results = [
            ["content1", 0.95, '{"source": "test1"}'],
            ["content2", 0.85, '{"source": "test2"}']
        ]
        self.mock_cursor.execute.return_value = None
        self.mock_cursor.fetchall.return_value = mock_results
        
        query = "Test query"
        results = vectorstore.similarity_search(query, k=2)
        
        # Verify query embedding was generated
        self.mock_embeddings.embed_query.assert_called_once_with(query)
        
        # Verify query was executed
        self.mock_cursor.execute.assert_called_once()
        
        # Verify results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].page_content, "content1")
        self.assertEqual(results[0].metadata["source"], "test1")
    
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        vectorstore = HanaTensorRTVectorStore(
            connection=self.mock_conn,
            embedding=self.mock_embeddings,
            table_name="TEST_VECTORS",
            enable_performance_monitoring=True
        )
        
        # Set up mock query results
        mock_results = [
            ["content1", 0.95, '{"source": "test1"}'],
            ["content2", 0.85, '{"source": "test2"}']
        ]
        self.mock_cursor.fetchall.return_value = mock_results
        
        # Execute a search to generate metrics
        vectorstore.similarity_search("test query")
        
        # Get and verify metrics
        metrics = vectorstore.get_performance_metrics()
        
        # Verify metrics structure
        self.assertIn("embedding_generation", metrics)
        self.assertIn("database_operations", metrics)
        self.assertIn("total_queries", metrics)
        
        # Clear metrics
        vectorstore.clear_performance_metrics()
        cleared_metrics = vectorstore.get_performance_metrics()
        self.assertEqual(cleared_metrics["total_queries"], 0)


if __name__ == "__main__":
    unittest.main()