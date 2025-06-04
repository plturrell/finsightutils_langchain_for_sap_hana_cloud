"""
Tests for the GPU batch processor module.

This module contains unit tests for the dynamic batch processor implementation.
"""

import unittest
from typing import List
import time
import random

import numpy as np

from langchain_hana.gpu.batch_processor import (
    DynamicBatchProcessor,
    EmbeddingBatchProcessor,
    ModelMemoryProfile,
    BatchProcessingStats,
    GPUMemoryInfo
)

class TestModelMemoryProfile(unittest.TestCase):
    """Tests for the ModelMemoryProfile class."""
    
    def test_estimate_batch_memory(self):
        """Test memory estimation for different batch sizes."""
        # Create a model profile
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            memory_per_item_kb=50,
            dtype="float32"
        )
        
        # Test memory estimation
        self.assertEqual(profile.estimate_batch_memory_mb(1), 100)  # Base memory only
        self.assertEqual(profile.estimate_batch_memory_mb(10), 100 + (50 * 10) // 1024)  # Base + 10 items
        self.assertEqual(profile.estimate_batch_memory_mb(100), 100 + (50 * 100) // 1024)  # Base + 100 items
    
    def test_max_batch_size(self):
        """Test max batch size calculation."""
        # Create a model profile
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=100,
            memory_per_item_kb=50,
            dtype="float32"
        )
        
        # Test max batch size calculation
        # Available memory = 200MB, safety_factor = 0.8
        # Safe memory = 200 * 0.8 = 160MB
        # Memory for batch = 160 - 100 = 60MB = 61,440KB
        # Max batch = 61,440 / 50 = 1,228
        self.assertEqual(profile.max_batch_size(200, 0.8), 1228)
        
        # Edge cases
        self.assertEqual(profile.max_batch_size(0, 0.8), 1)  # Minimum batch size
        self.assertEqual(profile.max_batch_size(100, 0.8), 1)  # No memory for batch
        self.assertEqual(profile.max_batch_size(101, 0.8), 1)  # Minimal memory for batch


class TestDynamicBatchProcessor(unittest.TestCase):
    """Tests for the DynamicBatchProcessor class."""
    
    def test_basic_processing(self):
        """Test basic batch processing functionality."""
        # Create a simple processing function
        def process_batch(batch: List[int]) -> List[int]:
            return [x * 2 for x in batch]
        
        # Create a model profile
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=10,
            memory_per_item_kb=10,
            dtype="float32"
        )
        
        # Create batch processor
        processor = DynamicBatchProcessor(
            processing_fn=process_batch,
            model_profile=profile,
            initial_batch_size=10,
            min_batch_size=1,
            max_batch_size=100
        )
        
        # Process a list of items
        items = list(range(50))
        results, stats = processor.process(items)
        
        # Check results
        self.assertEqual(len(results), 50)
        self.assertEqual(results, [x * 2 for x in items])
        
        # Check stats
        self.assertEqual(stats.total_items, 50)
        self.assertTrue(stats.total_batches > 0)
        self.assertTrue(stats.total_time > 0)
        self.assertEqual(stats.initial_batch_size, 10)
    
    def test_empty_input(self):
        """Test processing with empty input."""
        # Create a simple processing function
        def process_batch(batch: List[int]) -> List[int]:
            return [x * 2 for x in batch]
        
        # Create a model profile
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=10,
            memory_per_item_kb=10,
            dtype="float32"
        )
        
        # Create batch processor
        processor = DynamicBatchProcessor(
            processing_fn=process_batch,
            model_profile=profile,
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


class TestEmbeddingBatchProcessor(unittest.TestCase):
    """Tests for the EmbeddingBatchProcessor class."""
    
    def test_embedding_generation(self):
        """Test embedding generation with batch processing."""
        # Create a simple embedding function
        def embed_batch(batch: List[str]) -> List[List[float]]:
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
        def embed_batch(batch: List[str]) -> List[List[float]]:
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


class TestOOMRecovery(unittest.TestCase):
    """Tests for OOM recovery in batch processing."""
    
    def test_oom_recovery(self):
        """Test recovery from simulated OOM errors."""
        # Create a processing function that simulates OOM errors
        oom_count = 0
        
        def process_batch_with_oom(batch: List[int]) -> List[int]:
            nonlocal oom_count
            
            # Simulate OOM for large batches (first 2 times only)
            if len(batch) > 5 and oom_count < 2:
                oom_count += 1
                raise RuntimeError("CUDA out of memory")
            
            # Otherwise process normally
            return [x * 2 for x in batch]
        
        # Create a model profile
        profile = ModelMemoryProfile(
            model_name="test-model",
            embedding_dim=768,
            base_memory_mb=10,
            memory_per_item_kb=10,
            dtype="float32"
        )
        
        # Create batch processor
        processor = DynamicBatchProcessor(
            processing_fn=process_batch_with_oom,
            model_profile=profile,
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


if __name__ == "__main__":
    unittest.main()