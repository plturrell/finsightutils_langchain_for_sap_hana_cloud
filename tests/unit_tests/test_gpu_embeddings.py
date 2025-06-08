"""Unit tests for GPU embeddings functionality."""

import os
import unittest
from unittest.mock import patch, MagicMock

import pytest
import numpy as np

from langchain_hana.embeddings import MultiGPUEmbeddings, HanaTensorRTMultiGPUEmbeddings, CacheConfig, EmbeddingCache


class TestEmbeddingCache(unittest.TestCase):
    """Test the EmbeddingCache class."""

    def test_cache_initialization(self):
        """Test that the cache initializes correctly."""
        config = CacheConfig(
            enabled=True,
            max_size=100,
            ttl_seconds=3600,
        )
        cache = EmbeddingCache(config)
        
        # Cache should start empty
        self.assertEqual(len(cache.cache), 0)
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)
    
    def test_cache_get_put(self):
        """Test the basic get/put functionality."""
        config = CacheConfig(enabled=True, max_size=100)
        cache = EmbeddingCache(config)
        
        # Cache miss on non-existent key
        result = cache.get("test_key")
        self.assertIsNone(result)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 0)
        
        # Cache put
        test_vector = [0.1, 0.2, 0.3]
        cache.put("test_key", test_vector)
        
        # Cache hit
        result = cache.get("test_key")
        self.assertEqual(result, test_vector)
        self.assertEqual(cache.misses, 1)
        self.assertEqual(cache.hits, 1)
    
    def test_cache_ttl(self):
        """Test that TTL expiration works correctly."""
        config = CacheConfig(enabled=True, max_size=100, ttl_seconds=0.1)  # 100ms TTL
        cache = EmbeddingCache(config)
        
        # Add item to cache
        cache.put("test_key", [0.1, 0.2, 0.3])
        
        # Should be a hit immediately
        self.assertEqual(cache.get("test_key"), [0.1, 0.2, 0.3])
        
        # Wait for TTL to expire
        import time
        time.sleep(0.2)
        
        # Should be a miss after expiration
        self.assertIsNone(cache.get("test_key"))
        self.assertEqual(cache.misses, 2)  # Initial miss + expired miss
        self.assertEqual(cache.hits, 1)
    
    def test_cache_max_size(self):
        """Test that the cache enforces max size limit."""
        config = CacheConfig(enabled=True, max_size=2)
        cache = EmbeddingCache(config)
        
        # Add items to cache
        cache.put("key1", [0.1])
        cache.put("key2", [0.2])
        
        # Both should be hits
        self.assertEqual(cache.get("key1"), [0.1])
        self.assertEqual(cache.get("key2"), [0.2])
        
        # Add a third item, which should evict the oldest (key1)
        cache.put("key3", [0.3])
        
        # key1 should be evicted
        self.assertIsNone(cache.get("key1"))
        
        # key2 and key3 should still be in cache
        self.assertEqual(cache.get("key2"), [0.2])
        self.assertEqual(cache.get("key3"), [0.3])


@pytest.mark.skipif(not os.environ.get("TEST_GPU"), reason="GPU tests disabled")
class TestMultiGPUEmbeddings:
    """Test the MultiGPUEmbeddings class with mocks."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create mock embeddings model
        self.mock_embeddings = MagicMock()
        self.mock_embeddings.embed_documents.return_value = [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
        self.mock_embeddings.embed_query.return_value = [0.7, 0.8, 0.9]
        
        # Create mock GPU manager
        self.mock_gpu_manager = MagicMock()
        self.mock_gpu_manager.submit_task.return_value = "task-123"
        self.mock_gpu_manager.wait_for_task.return_value = MagicMock(
            success=True, 
            result=[[0.1, 0.2, 0.3]], 
            error=None
        )
        self.mock_gpu_manager.batch_process.return_value = [
            [[0.1, 0.2, 0.3]],
            [[0.4, 0.5, 0.6]]
        ]
        
    def test_embed_query(self):
        """Test the embed_query method."""
        embeddings = MultiGPUEmbeddings(
            base_embeddings=self.mock_embeddings,
            gpu_manager=self.mock_gpu_manager,
            enable_caching=False,
        )
        
        # Test embed_query
        result = embeddings.embed_query("test query")
        
        # Verify result
        assert result == [0.1, 0.2, 0.3]
        self.mock_gpu_manager.submit_task.assert_called_once()
    
    def test_embed_documents(self):
        """Test the embed_documents method."""
        embeddings = MultiGPUEmbeddings(
            base_embeddings=self.mock_embeddings,
            gpu_manager=self.mock_gpu_manager,
            batch_size=1,
            enable_caching=False,
        )
        
        # Test embed_documents
        result = embeddings.embed_documents(["doc1", "doc2"])
        
        # Verify results
        assert len(result) == 2
        self.mock_gpu_manager.batch_process.assert_called_once()
    
    def test_embed_documents_with_cache(self):
        """Test embed_documents with caching."""
        embeddings = MultiGPUEmbeddings(
            base_embeddings=self.mock_embeddings,
            gpu_manager=self.mock_gpu_manager,
            batch_size=1,
            enable_caching=True,
        )
        
        # First call should cache results
        result1 = embeddings.embed_documents(["doc1", "doc2"])
        
        # Reset mock to verify it's not called again
        self.mock_gpu_manager.batch_process.reset_mock()
        
        # Second call with same docs should use cache
        result2 = embeddings.embed_documents(["doc1", "doc2"])
        
        # Verify results match and batch_process was not called again
        assert result1 == result2
        self.mock_gpu_manager.batch_process.assert_not_called()
    
    def test_normalize_vector(self):
        """Test vector normalization."""
        embeddings = MultiGPUEmbeddings(
            base_embeddings=self.mock_embeddings,
            normalize_embeddings=True,
        )
        
        # Test normalization
        vector = [1.0, 2.0, 2.0]  # Length = 3.0
        result = embeddings._normalize_vector(vector)
        
        # Verify normalized result (should have length 1.0)
        expected = [1/3, 2/3, 2/3]
        np.testing.assert_almost_equal(result, expected, decimal=6)
        
        # Test with zero vector
        zero_vector = [0.0, 0.0, 0.0]
        result = embeddings._normalize_vector(zero_vector)
        np.testing.assert_almost_equal(result, zero_vector, decimal=6)


@pytest.mark.skipif(not os.environ.get("TEST_GPU"), reason="GPU tests disabled")
class TestTensorRTEmbeddings:
    """Test TensorRT embeddings with mocks."""
    
    @patch("langchain_hana.gpu.tensorrt_embeddings.get_available_gpus")
    @patch("langchain_hana.gpu.tensorrt_embeddings.TORCH_AVAILABLE", True)
    def test_gpu_info(self, mock_get_gpus):
        """Test the GPU info gathering functionality."""
        from langchain_hana.gpu.tensorrt_embeddings import GPUInfo
        
        # Create a mock GPU
        gpu_info = GPUInfo(index=0, name="Tesla T4", memory_total=16000, 
                         memory_free=8000, compute_capability="7.5")
        
        # Test memory_usage_percent
        assert gpu_info.memory_usage_percent == 50.0
        
        # Test string representation
        assert "Tesla T4" in str(gpu_info)
        assert "Memory: 8000/16000 MB" in str(gpu_info)
        assert "Compute: 7.5" in str(gpu_info)


@patch("torch.cuda.is_available", return_value=False)
def test_cpu_fallback(mock_cuda):
    """Test that CPU fallback works when GPU is not available."""
    # This test is complicated in a unit test since it requires
    # SentenceTransformer, but we can at least verify the code path
    
    with patch("langchain_hana.gpu.tensorrt_embeddings.HAS_GPU_DEPENDENCIES", False):
        # Test will be marked as skipped since we can't mock SentenceTransformer easily
        # but we'll create a framework for the test
        with pytest.raises(ImportError):
            from langchain_hana.gpu.tensorrt_embeddings import TensorRTEmbeddings
            TensorRTEmbeddings(model_name="all-MiniLM-L6-v2")


# Mock setup for testing TensorRT optimized embeddings
class MockTensorRTEmbeddings:
    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]
    
    def embed_query(self, text):
        return [0.1, 0.2, 0.3]


# Integration-style test for tensor core optimization
@patch("langchain_hana.gpu.TensorRTEmbeddings", MockTensorRTEmbeddings)
@patch("langchain_hana.gpu.TensorCoreOptimizer")
def test_tensor_core_optimization(mock_optimizer):
    """Test the tensor core optimization logic."""
    # Setup mock optimizer
    mock_optimizer_instance = MagicMock()
    mock_optimizer_instance.is_supported.return_value = True
    mock_optimizer_instance.optimize_embeddings.return_value = MockTensorRTEmbeddings()
    mock_optimizer.return_value = mock_optimizer_instance
    
    # Test HanaTensorRTMultiGPUEmbeddings with tensor core optimization
    from langchain_hana.embeddings import HanaTensorRTMultiGPUEmbeddings
    
    with patch("langchain_hana.embeddings.TORCH_AVAILABLE", True):
        embeddings = HanaTensorRTMultiGPUEmbeddings(
            model_name="all-MiniLM-L6-v2",
            enable_tensor_cores=True
        )
        
        # Verify tensor core optimization was attempted
        mock_optimizer_instance.is_supported.assert_called_once()
        mock_optimizer_instance.optimize_embeddings.assert_called_once()