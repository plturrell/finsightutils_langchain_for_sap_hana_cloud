"""
Extended tests for batch processing functionality.

This module contains additional unit tests for the dynamic batch processor
implementation in langchain_hana.gpu.batch_processor, focusing on features like
memory profiling, error recovery, and embedding caching.
"""

import unittest
from unittest.mock import MagicMock, patch
import time
import random

import numpy as np

from langchain_hana.gpu.batch_processor import (
    ModelMemoryProfile,
    GPUMemoryInfo,
    BatchProcessingStats,
    DynamicBatchProcessor,
    EmbeddingBatchProcessor
)


class TestModelMemoryProfile(unittest.TestCase):
    """Tests for the ModelMemoryProfile class."""
    
    def test_auto_calculate_memory_per_item(self):
        """Test auto-calculation of memory per item."""
        # Test with float32 (4 bytes per element)
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            dtype="float32"
        )
        
        # Check that memory_per_item_kb is calculated based on embedding_dim
        # 768 dim * 4 bytes * 4 (activation multiplier) = 12,288 bytes
        # Plus token memory (128 * 2 * 2 = 512 bytes)
        # Total: 12,800 bytes * 1.2 (safety) = 15,360 bytes = ~15KB
        self.assertGreater(profile.memory_per_item_kb, 12)
        
        # Test with float16 (2 bytes per element)
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            dtype="float16"
        )
        
        # Check that memory_per_item_kb is calculated based on embedding_dim
        # 768 dim * 2 bytes * 4 (activation multiplier) = 6,144 bytes
        # Plus token memory (128 * 2 * 2 = 512 bytes)
        # Total: 6,656 bytes * 1.2 (safety) = 7,987 bytes = ~8KB
        self.assertGreater(profile.memory_per_item_kb, 6)
        self.assertLess(profile.memory_per_item_kb, 10)
    
    def test_manual_memory_per_item(self):
        """Test manually specified memory per item."""
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            memory_per_item_kb=50,
            dtype="float32"
        )
        
        self.assertEqual(profile.memory_per_item_kb, 50)
    
    def test_estimate_batch_memory(self):
        """Test memory estimation for different batch sizes."""
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            memory_per_item_kb=50,
            dtype="float32"
        )
        
        # Test with different batch sizes
        self.assertEqual(profile.estimate_batch_memory_mb(1), 100)  # Base memory only
        self.assertEqual(profile.estimate_batch_memory_mb(10), 100 + (50 * 10) // 1024)  # Base + 10 items
        self.assertEqual(profile.estimate_batch_memory_mb(100), 100 + (50 * 100) // 1024)  # Base + 100 items
    
    def test_max_batch_size(self):
        """Test calculation of maximum batch size that fits in memory."""
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            memory_per_item_kb=50,
            dtype="float32"
        )
        
        # Test with different available memory amounts
        # Available memory = 200MB, safety_factor = 0.8
        # Safe memory = 200 * 0.8 = 160MB
        # Memory for batch = 160 - 100 = 60MB = 61,440KB
        # Max batch = 61,440 / 50 = 1,228
        self.assertEqual(profile.max_batch_size(200, 0.8), 1228)
        
        # Test edge cases
        self.assertEqual(profile.max_batch_size(0, 0.8), 1)  # Minimum batch size
        self.assertEqual(profile.max_batch_size(100, 0.8), 1)  # No memory for batch
        self.assertEqual(profile.max_batch_size(101, 0.8), 1)  # Minimal memory for batch


class TestGPUMemoryInfo(unittest.TestCase):
    """Tests for the GPUMemoryInfo class."""
    
    def test_utilization_percent(self):
        """Test calculation of memory utilization percentage."""
        memory_info = GPUMemoryInfo(
            total_memory_mb=1000,
            free_memory_mb=400,
            used_memory_mb=600
        )
        
        self.assertEqual(memory_info.utilization_percent, 60.0)
    
    def test_available_for_allocation(self):
        """Test calculation of memory available for allocation."""
        memory_info = GPUMemoryInfo(
            total_memory_mb=1000,
            free_memory_mb=400,
            used_memory_mb=600
        )
        
        # Apply 5% safety margin: 400 - (1000 * 0.05) = 400 - 50 = 350
        self.assertEqual(memory_info.available_for_allocation_mb, 350)
        
        # Edge case: free memory less than safety margin
        memory_info = GPUMemoryInfo(
            total_memory_mb=1000,
            free_memory_mb=10,
            used_memory_mb=990
        )
        
        # 10 - (1000 * 0.05) = 10 - 50 = -40, but clamped to 0
        self.assertEqual(memory_info.available_for_allocation_mb, 0)


class TestDynamicBatchProcessor(unittest.TestCase):
    """Tests for the DynamicBatchProcessor class."""
    
    @patch('langchain_hana.gpu.batch_processor.DynamicBatchProcessor._get_gpu_memory_info')
    def test_initialization_with_auto_batch_size(self, mock_get_memory):
        """Test initialization with automatic batch size determination."""
        # Mock GPU memory info
        mock_memory_info = GPUMemoryInfo(
            total_memory_mb=1000,
            free_memory_mb=500,
            used_memory_mb=500
        )
        mock_get_memory.return_value = mock_memory_info
        
        # Create a model profile
        model_profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            memory_per_item_kb=50,
            dtype="float32"
        )
        
        # Create processor without specifying initial_batch_size
        processor = DynamicBatchProcessor(
            processing_fn=lambda x: [i * 2 for i in x],
            model_profile=model_profile,
            min_batch_size=1,
            max_batch_size=100
        )
        
        # Expected batch size from: profile.max_batch_size(500, 0.8)
        # = (500 * 0.8 - 100) * 1024 / 50 = ~6,553
        # Clamped to max_batch_size = 100
        self.assertEqual(processor.batch_size, 100)
    
    @patch('langchain_hana.gpu.batch_processor.DynamicBatchProcessor._get_gpu_memory_info')
    def test_initialization_with_manual_batch_size(self, mock_get_memory):
        """Test initialization with manually specified batch size."""
        # Mock GPU memory info (not used in this case)
        mock_get_memory.return_value = None
        
        # Create a model profile
        model_profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            memory_per_item_kb=50,
            dtype="float32"
        )
        
        # Create processor with specified initial_batch_size
        processor = DynamicBatchProcessor(
            processing_fn=lambda x: [i * 2 for i in x],
            model_profile=model_profile,
            initial_batch_size=20,
            min_batch_size=1,
            max_batch_size=100
        )
        
        # Batch size should be the specified value
        self.assertEqual(processor.batch_size, 20)
    
    def test_process_with_empty_input(self):
        """Test processing with empty input."""
        model_profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=10,
            memory_per_item_kb=10,
            dtype="float32"
        )
        
        processor = DynamicBatchProcessor(
            processing_fn=lambda x: [i * 2 for i in x],
            model_profile=model_profile,
            initial_batch_size=10,
            min_batch_size=1,
            max_batch_size=100
        )
        
        # Process empty input
        results, stats = processor.process([])
        
        # Check results
        self.assertEqual(results, [])
        
        # Check stats
        self.assertEqual(stats.total_items, 0)
        self.assertEqual(stats.total_batches, 0)
        self.assertEqual(stats.total_time, 0)
    
    def test_basic_processing(self):
        """Test basic batch processing functionality."""
        model_profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=10,
            memory_per_item_kb=10,
            dtype="float32"
        )
        
        # Create batch processor
        processor = DynamicBatchProcessor(
            processing_fn=lambda x: [i * 2 for i in x],
            model_profile=model_profile,
            initial_batch_size=10,
            min_batch_size=1,
            max_batch_size=100
        )
        
        # Process a list of items
        items = list(range(50))
        results, stats = processor.process(items)
        
        # Check results
        self.assertEqual(len(results), 50)
        self.assertEqual(results, [i * 2 for i in items])
        
        # Check stats
        self.assertEqual(stats.total_items, 50)
        self.assertTrue(stats.total_batches > 0)
        self.assertTrue(stats.total_time > 0)
        self.assertEqual(stats.initial_batch_size, 10)
    
    @patch('torch.cuda.OutOfMemoryError', create=True)
    @patch('langchain_hana.gpu.batch_processor.DynamicBatchProcessor._clear_gpu_cache')
    def test_oom_recovery(self, mock_clear_cache, mock_oom_error):
        """Test recovery from OOM errors."""
        model_profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=10,
            memory_per_item_kb=10,
            dtype="float32"
        )
        
        # Create a processing function that simulates OOM errors
        oom_count = 0
        def process_batch_with_oom(batch):
            nonlocal oom_count
            
            # Simulate OOM for large batches (first 2 times only)
            if len(batch) > 5 and oom_count < 2:
                oom_count += 1
                raise RuntimeError("CUDA out of memory")
            
            # Otherwise process normally
            return [x * 2 for x in batch]
        
        # Create batch processor
        processor = DynamicBatchProcessor(
            processing_fn=process_batch_with_oom,
            model_profile=model_profile,
            initial_batch_size=10,  # Start with a batch size that will trigger OOM
            min_batch_size=1,
            max_batch_size=100,
            oom_recovery_factor=0.5  # Reduce batch size by half on OOM
        )
        
        # Process a list of items
        items = list(range(50))
        results, stats = processor.process(items)
        
        # Check results
        self.assertEqual(len(results), 50)
        self.assertEqual(results, [x * 2 for x in items])
        
        # Check stats
        self.assertEqual(stats.oom_events, 2)
        self.assertEqual(stats.batch_size_adjustments, 2)
        self.assertTrue(stats.final_batch_size <= 5)  # Should be reduced to <= 5


class TestEmbeddingBatchProcessor(unittest.TestCase):
    """Tests for the EmbeddingBatchProcessor class."""
    
    def test_embedding_generation(self):
        """Test embedding generation with batch processing."""
        # Create a simple embedding function
        def embed_batch(batch):
            # Simulate embedding generation
            time.sleep(0.01 * len(batch))  # Simulate processing time
            return [list(np.ones(768) * hash(text) % 100) for text in batch]
        
        # Create embedding batch processor
        processor = EmbeddingBatchProcessor(
            embedding_fn=embed_batch,
            model_name="test-model",
            embedding_dim=768,
            initial_batch_size=5,
            min_batch_size=1,
            max_batch_size=10,
            dtype="float32",
            enable_caching=True
        )
        
        # Generate embeddings
        texts = [f"Text {i}" for i in range(20)]
        embeddings, stats = processor.embed_documents(texts)
        
        # Check results
        self.assertEqual(len(embeddings), 20)
        self.assertEqual(len(embeddings[0]), 768)
        
        # Test caching by embedding the same texts again
        texts_with_duplicates = texts + texts[:5]  # Add some duplicates
        embeddings2, stats2 = processor.embed_documents(texts_with_duplicates)
        
        # Check that cached embeddings are used
        self.assertEqual(len(embeddings2), 25)
        for i in range(20):
            self.assertEqual(embeddings2[i], embeddings[i])  # Should be identical
    
    def test_single_query_embedding(self):
        """Test embedding generation for a single query."""
        # Create a simple embedding function
        def embed_batch(batch):
            # Simulate embedding generation
            time.sleep(0.01 * len(batch))  # Simulate processing time
            return [list(np.ones(768) * hash(text) % 100) for text in batch]
        
        # Create embedding batch processor
        processor = EmbeddingBatchProcessor(
            embedding_fn=embed_batch,
            model_name="test-model",
            embedding_dim=768,
            initial_batch_size=5,
            min_batch_size=1,
            max_batch_size=10,
            dtype="float32",
            enable_caching=True
        )
        
        # Generate embedding for a single query
        query = "This is a test query"
        embedding = processor.embed_query(query)
        
        # Check result
        self.assertEqual(len(embedding), 768)
        
        # Test caching by embedding the same query again
        embedding2 = processor.embed_query(query)
        
        # Check that cached embedding is used
        self.assertEqual(embedding2, embedding)
    
    def test_embedding_caching(self):
        """Test that embedding caching works correctly."""
        # Create a mock embedding function that tracks calls
        call_count = 0
        def embed_batch(batch):
            nonlocal call_count
            call_count += 1
            return [list(np.ones(768) * hash(text) % 100) for text in batch]
        
        # Create embedding batch processor with caching enabled
        processor = EmbeddingBatchProcessor(
            embedding_fn=embed_batch,
            model_name="test-model",
            embedding_dim=768,
            initial_batch_size=5,
            min_batch_size=1,
            max_batch_size=10,
            dtype="float32",
            enable_caching=True
        )
        
        # Generate embeddings for same texts multiple times
        text = "This is a test"
        
        # First call should generate embedding
        embedding1 = processor.embed_query(text)
        self.assertEqual(call_count, 1)
        
        # Second call should use cache
        embedding2 = processor.embed_query(text)
        self.assertEqual(call_count, 1)  # Call count shouldn't increase
        self.assertEqual(embedding1, embedding2)
        
        # Create new processor with caching disabled
        processor_no_cache = EmbeddingBatchProcessor(
            embedding_fn=embed_batch,
            model_name="test-model",
            embedding_dim=768,
            initial_batch_size=5,
            min_batch_size=1,
            max_batch_size=10,
            dtype="float32",
            enable_caching=False
        )
        
        # Call count should increase for each call with caching disabled
        embedding3 = processor_no_cache.embed_query(text)
        self.assertEqual(call_count, 2)
        
        embedding4 = processor_no_cache.embed_query(text)
        self.assertEqual(call_count, 3)


if __name__ == "__main__":
    unittest.main()