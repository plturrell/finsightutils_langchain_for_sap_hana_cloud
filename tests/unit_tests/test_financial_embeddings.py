"""
Comprehensive tests for financial embeddings modules.

This module provides thorough testing for the financial embedding implementations,
including edge cases, performance degradation scenarios, and memory constraints.
"""

import os
import pytest
import numpy as np
import torch
import tempfile
from unittest.mock import patch, MagicMock

from langchain_hana.financial.fin_e5_embeddings import (
    FinE5Embeddings,
    FinE5TensorRTEmbeddings,
    create_financial_embeddings,
    FINANCIAL_EMBEDDING_MODELS
)
from langchain_hana.financial.caching import FinancialEmbeddingCache


class TestFinE5Embeddings:
    """Test suite for FinE5Embeddings class."""
    
    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mocked SentenceTransformer."""
        with patch('langchain_hana.financial.fin_e5_embeddings.SentenceTransformer') as mock:
            # Mock encode method
            mock.return_value.encode.return_value = np.random.rand(3, 384)
            mock.return_value.get_sentence_embedding_dimension.return_value = 384
            yield mock
    
    @pytest.fixture
    def embeddings(self, mock_sentence_transformer):
        """Create a FinE5Embeddings instance with mocked model."""
        return FinE5Embeddings(
            model_type="default",
            device="cpu",
            enable_caching=False
        )
    
    def test_initialization_with_default_parameters(self, mock_sentence_transformer):
        """Test initialization with default parameters."""
        embeddings = FinE5Embeddings()
        
        assert embeddings.model_name == FINANCIAL_EMBEDDING_MODELS["default"]
        assert embeddings.model_type == "default"
        assert embeddings.device in ["cuda", "cpu"]
        assert embeddings.normalize_embeddings is True
        assert embeddings.batch_size == 32
        assert embeddings.max_seq_length == 512
        assert embeddings.add_financial_prefix is False
        assert embeddings.enable_caching is True
        
        # Verify model initialization
        mock_sentence_transformer.assert_called_once()
    
    def test_initialization_with_custom_parameters(self, mock_sentence_transformer):
        """Test initialization with custom parameters."""
        embeddings = FinE5Embeddings(
            model_type="high_quality",
            device="cpu",
            use_fp16=False,
            normalize_embeddings=False,
            max_seq_length=256,
            batch_size=16,
            add_financial_prefix=True,
            financial_prefix_type="analysis",
            enable_caching=False
        )
        
        assert embeddings.model_name == FINANCIAL_EMBEDDING_MODELS["high_quality"]
        assert embeddings.model_type == "high_quality"
        assert embeddings.device == "cpu"
        assert embeddings.use_fp16 is False
        assert embeddings.normalize_embeddings is False
        assert embeddings.batch_size == 16
        assert embeddings.max_seq_length == 256
        assert embeddings.add_financial_prefix is True
        assert embeddings.financial_prefix == "Financial analysis: "
        assert embeddings.enable_caching is False
        
        # Verify model initialization
        mock_sentence_transformer.assert_called_once_with(
            FINANCIAL_EMBEDDING_MODELS["high_quality"], 
            device="cpu"
        )
    
    def test_initialization_with_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(ValueError, match="Invalid model_type"):
            FinE5Embeddings(model_type="invalid_model_type")
    
    def test_initialization_with_invalid_financial_prefix_type(self):
        """Test initialization with invalid financial prefix type."""
        with pytest.raises(ValueError, match="Invalid financial_prefix_type"):
            FinE5Embeddings(
                add_financial_prefix=True,
                financial_prefix_type="invalid_prefix"
            )
    
    def test_add_financial_context_single_text(self, embeddings):
        """Test adding financial context to a single text."""
        # Set up
        embeddings.add_financial_prefix = True
        embeddings.financial_prefix = "Financial context: "
        
        # Test
        text = "Revenue increased by 15%"
        result = embeddings._add_financial_context(text)
        
        # Verify
        assert result == "Financial context: Revenue increased by 15%"
    
    def test_add_financial_context_multiple_texts(self, embeddings):
        """Test adding financial context to multiple texts."""
        # Set up
        embeddings.add_financial_prefix = True
        embeddings.financial_prefix = "Financial context: "
        
        # Test
        texts = ["Revenue increased by 15%", "EBITDA margin improved to 28%"]
        result = embeddings._add_financial_context(texts)
        
        # Verify
        assert result == [
            "Financial context: Revenue increased by 15%",
            "Financial context: EBITDA margin improved to 28%"
        ]
    
    def test_add_financial_context_disabled(self, embeddings):
        """Test that financial context is not added when disabled."""
        # Set up
        embeddings.add_financial_prefix = False
        
        # Test
        text = "Revenue increased by 15%"
        result = embeddings._add_financial_context(text)
        
        # Verify
        assert result == text
    
    def test_embed_documents(self, embeddings, mock_sentence_transformer):
        """Test embedding multiple documents."""
        # Set up
        texts = ["Revenue increased by 15%", "EBITDA margin improved to 28%"]
        mock_embeddings = np.random.rand(2, 384)
        embeddings.model.encode.return_value = mock_embeddings
        
        # Test
        result = embeddings.embed_documents(texts)
        
        # Verify
        embeddings.model.encode.assert_called_once_with(
            texts,
            batch_size=embeddings.batch_size,
            show_progress_bar=False,
            normalize_embeddings=embeddings.normalize_embeddings,
            convert_to_numpy=True
        )
        assert len(result) == 2
        assert len(result[0]) == 384
    
    def test_embed_query(self, embeddings, mock_sentence_transformer):
        """Test embedding a single query."""
        # Set up
        text = "What was the revenue growth?"
        mock_embeddings = np.random.rand(1, 384)
        embeddings.model.encode.return_value = mock_embeddings
        
        # Test
        result = embeddings.embed_query(text)
        
        # Verify
        embeddings.model.encode.assert_called_once_with(
            [text],
            show_progress_bar=False,
            normalize_embeddings=embeddings.normalize_embeddings,
            convert_to_numpy=True
        )
        assert len(result) == 384
    
    def test_batch_embed_empty_list(self, embeddings):
        """Test embedding an empty list."""
        # Test
        result = embeddings._batch_embed([])
        
        # Verify
        assert result == []
        embeddings.model.encode.assert_not_called()
    
    def test_batch_embed_with_financial_prefix(self, embeddings):
        """Test that batch embed adds financial prefix when enabled."""
        # Set up
        embeddings.add_financial_prefix = True
        embeddings.financial_prefix = "Financial context: "
        texts = ["Revenue increased by 15%"]
        mock_embeddings = np.random.rand(1, 384)
        embeddings.model.encode.return_value = mock_embeddings
        
        # Test
        result = embeddings._batch_embed(texts)
        
        # Verify
        embeddings.model.encode.assert_called_once()
        args, _ = embeddings.model.encode.call_args
        assert args[0] == ["Financial context: Revenue increased by 15%"]
    
    @patch('torch.cuda.OutOfMemoryError', create=True)
    @patch('torch.cuda.empty_cache')
    def test_batch_embed_cuda_oom_fallback(self, mock_empty_cache, mock_oom, embeddings):
        """Test fallback to CPU when CUDA out of memory error occurs."""
        # Set up
        embeddings.device = 'cuda'
        texts = ["Revenue increased by 15%", "EBITDA margin improved to 28%"]
        
        # Make first call raise OOM, second call succeed
        mock_embeddings = np.random.rand(2, 384)
        embeddings.model.encode.side_effect = [
            torch.cuda.OutOfMemoryError("CUDA out of memory"),
            mock_embeddings
        ]
        
        # Test
        result = embeddings._batch_embed(texts)
        
        # Verify
        assert embeddings.model.encode.call_count == 2
        assert embeddings.model.to.call_count == 2  # First to CPU, then back to CUDA
        mock_empty_cache.assert_called_once()
        assert len(result) == 2
        assert len(result[0]) == 384
    
    def test_get_embedding_dimension(self, embeddings):
        """Test getting the embedding dimension."""
        # Set up
        embeddings.embedding_dim = 384
        
        # Test
        result = embeddings.get_embedding_dimension()
        
        # Verify
        assert result == 384
    
    @patch('torch.cuda.empty_cache')
    def test_clear_gpu_memory(self, mock_empty_cache, embeddings):
        """Test clearing GPU memory."""
        # Set up
        embeddings.device = 'cuda'
        
        # Test
        embeddings.clear_gpu_memory()
        
        # Verify
        mock_empty_cache.assert_called_once()
    
    @patch('torch.cuda.empty_cache')
    def test_clear_gpu_memory_cpu_only(self, mock_empty_cache, embeddings):
        """Test that GPU memory is not cleared when using CPU."""
        # Set up
        embeddings.device = 'cpu'
        
        # Test
        embeddings.clear_gpu_memory()
        
        # Verify
        mock_empty_cache.assert_not_called()


class TestFinE5TensorRTEmbeddings:
    """Test suite for FinE5TensorRTEmbeddings class."""
    
    @pytest.fixture
    def mock_hana_tensorrt_embeddings(self):
        """Create a mocked HanaTensorRTEmbeddings."""
        with patch('langchain_hana.financial.fin_e5_embeddings.HanaTensorRTEmbeddings') as mock:
            # Mock embedding methods
            mock.return_value.embed_documents.return_value = [[0.1] * 384] * 3
            mock.return_value.embed_query.return_value = [0.1] * 384
            mock.return_value.get_performance_stats.return_value = MagicMock(
                total_documents=100,
                total_time_seconds=1.0,
                documents_per_second=100.0,
                avg_document_time_ms=10.0,
                peak_memory_mb=1000,
                batch_size=32,
                precision="fp16",
                gpu_name="Tesla T4",
                gpu_count=1
            )
            yield mock
    
    @pytest.fixture
    def embeddings(self, mock_hana_tensorrt_embeddings):
        """Create a FinE5TensorRTEmbeddings instance with mocked model."""
        return FinE5TensorRTEmbeddings(
            model_type="default",
            enable_caching=False
        )
    
    def test_initialization(self, mock_hana_tensorrt_embeddings):
        """Test initialization with default parameters."""
        embeddings = FinE5TensorRTEmbeddings()
        
        assert embeddings.model_name == FINANCIAL_EMBEDDING_MODELS["default"]
        assert embeddings.model_type == "default"
        assert embeddings.precision is None
        assert embeddings.multi_gpu is False
        assert embeddings.max_batch_size == 32
        assert embeddings.add_financial_prefix is False
        assert embeddings.enable_caching is True
        
        # Verify TensorRT initialization
        mock_hana_tensorrt_embeddings.assert_called_once()
    
    def test_initialization_with_custom_parameters(self, mock_hana_tensorrt_embeddings):
        """Test initialization with custom parameters."""
        embeddings = FinE5TensorRTEmbeddings(
            model_type="high_quality",
            precision="fp16",
            multi_gpu=True,
            max_batch_size=64,
            add_financial_prefix=True,
            financial_prefix_type="report",
            enable_caching=False
        )
        
        assert embeddings.model_name == FINANCIAL_EMBEDDING_MODELS["high_quality"]
        assert embeddings.model_type == "high_quality"
        assert embeddings.precision == "fp16"
        assert embeddings.multi_gpu is True
        assert embeddings.max_batch_size == 64
        assert embeddings.add_financial_prefix is True
        assert embeddings.financial_prefix == "Financial report: "
        assert embeddings.enable_caching is False
        
        # Verify TensorRT initialization
        mock_hana_tensorrt_embeddings.assert_called_once()
    
    def test_initialization_with_int8_calibration(self, mock_hana_tensorrt_embeddings):
        """Test initialization with INT8 calibration."""
        with patch('langchain_hana.financial.fin_e5_embeddings.create_enhanced_calibration_dataset') as mock_calibration:
            mock_calibration.return_value = ["sample text"] * 100
            
            embeddings = FinE5TensorRTEmbeddings(
                model_type="default",
                precision="int8"
            )
            
            # Verify calibration dataset creation
            mock_calibration.assert_called_once_with(
                domains=["financial"],
                count=100,
                custom_file_path=None
            )
            
            # Verify TensorRT initialization with calibration data
            mock_hana_tensorrt_embeddings.assert_called_once()
            _, kwargs = mock_hana_tensorrt_embeddings.call_args
            assert kwargs["calibration_domain"] == "financial"
            assert kwargs["calibration_data"] == mock_calibration.return_value
    
    def test_initialization_failure_handling(self, mock_hana_tensorrt_embeddings):
        """Test handling of initialization failures."""
        # Make initialization fail
        mock_hana_tensorrt_embeddings.side_effect = RuntimeError("TensorRT initialization failed")
        
        # Test
        with pytest.raises(RuntimeError, match="TensorRT initialization failed"):
            FinE5TensorRTEmbeddings()
    
    def test_embed_documents(self, embeddings):
        """Test embedding multiple documents."""
        # Set up
        texts = ["Revenue increased by 15%", "EBITDA margin improved to 28%", "EPS was $2.50"]
        
        # Test
        result = embeddings.embed_documents(texts)
        
        # Verify
        embeddings.tensorrt_embeddings.embed_documents.assert_called_once_with(texts)
        assert len(result) == 3
        assert len(result[0]) == 384
    
    def test_embed_documents_with_financial_prefix(self, embeddings):
        """Test embedding documents with financial prefix."""
        # Set up
        embeddings.add_financial_prefix = True
        embeddings.financial_prefix = "Financial context: "
        texts = ["Revenue increased by 15%", "EBITDA margin improved to 28%"]
        
        # Test
        result = embeddings.embed_documents(texts)
        
        # Verify
        expected_texts = [
            "Financial context: Revenue increased by 15%",
            "Financial context: EBITDA margin improved to 28%"
        ]
        embeddings.tensorrt_embeddings.embed_documents.assert_called_once_with(expected_texts)
    
    def test_embed_query(self, embeddings):
        """Test embedding a single query."""
        # Set up
        text = "What was the revenue growth?"
        
        # Test
        result = embeddings.embed_query(text)
        
        # Verify
        embeddings.tensorrt_embeddings.embed_query.assert_called_once_with(text)
        assert len(result) == 384
    
    def test_embed_query_with_financial_prefix(self, embeddings):
        """Test embedding a query with financial prefix."""
        # Set up
        embeddings.add_financial_prefix = True
        embeddings.financial_prefix = "Financial context: "
        text = "What was the revenue growth?"
        
        # Test
        result = embeddings.embed_query(text)
        
        # Verify
        expected_text = "Financial context: What was the revenue growth?"
        embeddings.tensorrt_embeddings.embed_query.assert_called_once_with(expected_text)
    
    def test_get_performance_stats(self, embeddings):
        """Test getting performance statistics."""
        # Test
        stats = embeddings.get_performance_stats()
        
        # Verify
        embeddings.tensorrt_embeddings.get_performance_stats.assert_called_once()
        assert "total_documents" in stats
        assert "total_time_seconds" in stats
        assert "documents_per_second" in stats
        assert "avg_document_time_ms" in stats
        assert "peak_memory_mb" in stats
        assert "batch_size" in stats
        assert "precision" in stats
        assert "gpu_name" in stats
        assert "gpu_count" in stats
    
    def test_benchmark(self, embeddings):
        """Test benchmarking functionality."""
        # Set up
        embeddings.tensorrt_embeddings.benchmark.return_value = {
            "batch_size_results": {
                "8": {"throughput": 100.0, "latency_ms": 10.0},
                "16": {"throughput": 150.0, "latency_ms": 8.0},
                "32": {"throughput": 200.0, "latency_ms": 6.0}
            },
            "optimal_batch_size": 32,
            "peak_memory_mb": 1000,
            "gpu_utilization": 0.8
        }
        
        # Test
        result = embeddings.benchmark(batch_sizes=[8, 16, 32])
        
        # Verify
        embeddings.tensorrt_embeddings.benchmark.assert_called_once_with(batch_sizes=[8, 16, 32])
        assert "batch_size_results" in result
        assert "optimal_batch_size" in result
        assert result["optimal_batch_size"] == 32


class TestFinancialEmbeddingCache:
    """Test suite for FinancialEmbeddingCache class."""
    
    @pytest.fixture
    def mock_embeddings(self):
        """Create a mock embeddings instance."""
        embeddings = MagicMock()
        embeddings.embed_documents.return_value = [[0.1] * 384] * 3
        embeddings.embed_query.return_value = [0.1] * 384
        return embeddings
    
    @pytest.fixture
    def cache(self, mock_embeddings):
        """Create a FinancialEmbeddingCache instance."""
        from langchain_hana.financial.caching import FinancialEmbeddingCache
        return FinancialEmbeddingCache(
            base_embeddings=mock_embeddings,
            ttl_seconds=3600,
            max_size=100
        )
    
    def test_initialization(self, mock_embeddings):
        """Test initialization with default parameters."""
        from langchain_hana.financial.caching import FinancialEmbeddingCache
        
        cache = FinancialEmbeddingCache(
            base_embeddings=mock_embeddings
        )
        
        assert cache.base_embeddings == mock_embeddings
        assert cache.ttl_seconds == 3600
        assert cache.max_size == 10000
        assert cache.persist_path is None
        assert len(cache.query_cache) == 0
        assert len(cache.document_cache) == 0
        assert len(cache.category_ttls) == 5  # news, report, analysis, market_data, default
    
    def test_initialization_with_custom_parameters(self, mock_embeddings):
        """Test initialization with custom parameters."""
        from langchain_hana.financial.caching import FinancialEmbeddingCache
        
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = os.path.join(temp_dir, "cache.pkl")
            
            cache = FinancialEmbeddingCache(
                base_embeddings=mock_embeddings,
                ttl_seconds=7200,
                max_size=500,
                persist_path=cache_file,
                model_name="test-model"
            )
            
            assert cache.base_embeddings == mock_embeddings
            assert cache.ttl_seconds == 7200
            assert cache.max_size == 500
            assert cache.persist_path == cache_file
            assert cache.model_name == "test-model"
    
    def test_get_category(self, cache):
        """Test categorization of financial text."""
        # Test news category
        assert cache._get_category("Breaking news: Company XYZ announces new CEO") == "news"
        
        # Test report category
        assert cache._get_category("Q1 2023 Earnings Report: Revenue increased by 15%") == "report"
        
        # Test analysis category
        assert cache._get_category("Financial Analysis: Market trends indicate growth") == "analysis"
        
        # Test market data category
        assert cache._get_category("Stock market prices show volatility") == "market_data"
        
        # Test default category
        assert cache._get_category("General text without specific indicators") == "default"
    
    def test_get_ttl_for_category(self, cache):
        """Test getting TTL for different categories."""
        # News category (1 hour)
        assert cache._get_ttl_for_category("news") == 3600
        
        # Report category (1 week)
        assert cache._get_ttl_for_category("report") == 86400 * 7
        
        # Analysis category (1 day)
        assert cache._get_ttl_for_category("analysis") == 86400
        
        # Market data category (30 minutes)
        assert cache._get_ttl_for_category("market_data") == 1800
        
        # Default category (from initialization)
        assert cache._get_ttl_for_category("default") == 3600
        
        # Unknown category (falls back to default)
        assert cache._get_ttl_for_category("unknown") == 3600
    
    def test_normalize_financial_text(self, cache):
        """Test normalization of financial text."""
        # Test percentage normalization
        assert "10 percent" in cache._normalize_financial_text("10%")
        
        # Test abbreviation normalization
        assert "quarter 1" in cache._normalize_financial_text("Q1 results")
        assert "fiscal year" in cache._normalize_financial_text("FY 2023 performance")
        assert "year over year" in cache._normalize_financial_text("YoY growth")
        assert "earnings before interest taxes depreciation amortization" in cache._normalize_financial_text("EBITDA margin")
        assert "earnings per share" in cache._normalize_financial_text("EPS was $2.50")
    
    def test_generate_cache_key(self, cache):
        """Test generation of cache keys."""
        # Test basic key generation
        key1 = cache._generate_cache_key("Revenue increased by 15%")
        assert isinstance(key1, str)
        
        # Test normalization in key generation
        key2 = cache._generate_cache_key("Revenue increased by 15 percent")
        key3 = cache._generate_cache_key("Revenue increased by 15%")
        assert key2 == key3  # Keys should match after normalization
        
        # Test abbreviation normalization in key generation
        key4 = cache._generate_cache_key("Q1 results")
        key5 = cache._generate_cache_key("quarter 1 results")
        assert key4 == key5  # Keys should match after abbreviation normalization
    
    def test_embed_query_cache_hit(self, cache, mock_embeddings):
        """Test query embedding with cache hit."""
        # Set up
        query = "What was the revenue growth?"
        cache_key = cache._generate_cache_key(query)
        vector = [0.2] * 384
        cache.query_cache[cache_key] = (vector, time.time(), "default")
        
        # Test
        result = cache.embed_query(query)
        
        # Verify
        assert result == vector
        mock_embeddings.embed_query.assert_not_called()
        assert cache.query_hits == 1
        assert cache.query_misses == 0
        assert cache.cache_usage_by_category["default"] == 1
    
    def test_embed_query_cache_miss(self, cache, mock_embeddings):
        """Test query embedding with cache miss."""
        # Set up
        query = "What was the revenue growth?"
        vector = [0.1] * 384
        mock_embeddings.embed_query.return_value = vector
        
        # Test
        result = cache.embed_query(query)
        
        # Verify
        assert result == vector
        mock_embeddings.embed_query.assert_called_once_with(query)
        assert cache.query_hits == 0
        assert cache.query_misses == 1
        
        # Check that the result was cached
        cache_key = cache._generate_cache_key(query)
        assert cache_key in cache.query_cache
        assert cache.query_cache[cache_key][0] == vector
    
    def test_embed_query_cache_expired(self, cache, mock_embeddings):
        """Test query embedding with expired cache entry."""
        # Set up
        query = "What was the revenue growth?"
        cache_key = cache._generate_cache_key(query)
        vector1 = [0.2] * 384
        vector2 = [0.3] * 384
        
        # Add expired entry to cache (1 hour ago + 1 second)
        import time
        cache.query_cache[cache_key] = (vector1, time.time() - 3601, "default")
        
        # Set return value for new embedding
        mock_embeddings.embed_query.return_value = vector2
        
        # Test
        result = cache.embed_query(query)
        
        # Verify
        assert result == vector2  # Should get new vector, not cached one
        mock_embeddings.embed_query.assert_called_once_with(query)
        assert cache.query_hits == 0
        assert cache.query_misses == 1
        
        # Check that the cache was updated
        assert cache.query_cache[cache_key][0] == vector2
    
    def test_embed_documents_mixed_cache(self, cache, mock_embeddings):
        """Test document embedding with mixed cache hits and misses."""
        # Set up
        texts = [
            "Revenue increased by 15%",
            "EBITDA margin improved to 28%",
            "EPS was $2.50"
        ]
        
        # Add second text to cache
        cache_key2 = cache._generate_cache_key(texts[1])
        vector2 = [0.2] * 384
        cache.document_cache[cache_key2] = (vector2, time.time(), "default")
        
        # Set return value for new embeddings
        vectors = [[0.1] * 384, [0.3] * 384]  # For texts 0 and 2
        mock_embeddings.embed_documents.return_value = vectors
        
        # Test
        result = cache.embed_documents(texts)
        
        # Verify
        assert len(result) == 3
        assert result[1] == vector2  # Should get cached vector for text 1
        assert result[0] == vectors[0]  # Should get new vector for text 0
        assert result[2] == vectors[1]  # Should get new vector for text 2
        
        # Check that the embeddings method was called with only the non-cached texts
        mock_embeddings.embed_documents.assert_called_once()
        call_texts = mock_embeddings.embed_documents.call_args[0][0]
        assert len(call_texts) == 2
        assert call_texts[0] == texts[0]
        assert call_texts[1] == texts[2]
        
        # Check cache statistics
        assert cache.document_hits == 1
        assert cache.document_misses == 2
        assert cache.cache_usage_by_category["default"] == 1
    
    def test_clear_cache(self, cache):
        """Test clearing the cache."""
        # Set up
        query = "What was the revenue growth?"
        cache_key = cache._generate_cache_key(query)
        vector = [0.2] * 384
        
        # Add entries to cache
        cache.query_cache[cache_key] = (vector, time.time(), "default")
        cache.document_cache[cache_key] = (vector, time.time(), "default")
        
        # Update stats
        cache.query_hits = 5
        cache.query_misses = 3
        cache.document_hits = 7
        cache.document_misses = 2
        cache.cache_usage_by_category["default"] = 12
        
        # Test
        cache.clear_cache()
        
        # Verify
        assert len(cache.query_cache) == 0
        assert len(cache.document_cache) == 0
        assert cache.query_hits == 0
        assert cache.query_misses == 0
        assert cache.document_hits == 0
        assert cache.document_misses == 0
        assert cache.cache_usage_by_category["default"] == 0
    
    def test_persist_and_load_cache(self, cache, mock_embeddings):
        """Test persisting and loading cache to/from disk."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set up
            cache_file = os.path.join(temp_dir, "cache.pkl")
            cache.persist_path = cache_file
            
            # Add entries to cache
            query = "What was the revenue growth?"
            cache_key = cache._generate_cache_key(query)
            vector = [0.2] * 384
            
            cache.query_cache[cache_key] = (vector, time.time(), "default")
            cache.document_cache[cache_key] = (vector, time.time(), "report")
            
            # Update stats
            cache.query_hits = 5
            cache.query_misses = 3
            cache.document_hits = 7
            cache.document_misses = 2
            cache.cache_usage_by_category["default"] = 5
            cache.cache_usage_by_category["report"] = 7
            
            # Test persist
            cache._persist_cache()
            
            # Clear cache
            cache.clear_cache()
            assert len(cache.query_cache) == 0
            assert len(cache.document_cache) == 0
            
            # Test load
            cache._load_cache()
            
            # Verify
            assert len(cache.query_cache) == 1
            assert len(cache.document_cache) == 1
            assert cache_key in cache.query_cache
            assert cache_key in cache.document_cache
            assert cache.query_cache[cache_key][0] == vector
            assert cache.document_cache[cache_key][0] == vector
            assert cache.query_hits == 5
            assert cache.query_misses == 3
            assert cache.document_hits == 7
            assert cache.document_misses == 2
            assert cache.cache_usage_by_category["default"] == 5
            assert cache.cache_usage_by_category["report"] == 7
    
    def test_get_stats(self, cache):
        """Test getting cache statistics."""
        # Set up
        cache.query_hits = 5
        cache.query_misses = 5
        cache.document_hits = 15
        cache.document_misses = 5
        cache.cache_usage_by_category["default"] = 10
        cache.cache_usage_by_category["report"] = 5
        cache.cache_usage_by_category["news"] = 5
        
        # Test
        stats = cache.get_stats()
        
        # Verify
        assert stats["query_cache_size"] == 0
        assert stats["document_cache_size"] == 0
        assert stats["max_size"] == 100
        assert stats["ttl_seconds"] == 3600
        assert stats["query_hits"] == 5
        assert stats["query_misses"] == 5
        assert stats["query_hit_rate"] == 0.5  # 5 / (5 + 5)
        assert stats["document_hits"] == 15
        assert stats["document_misses"] == 5
        assert stats["document_hit_rate"] == 0.75  # 15 / (15 + 5)
        assert stats["persistence_enabled"] is False
        
        # Check category statistics
        assert stats["cache_usage_by_category"]["default"] == 10
        assert stats["cache_usage_by_category"]["report"] == 5
        assert stats["cache_usage_by_category"]["news"] == 5
        
        # Check category distribution
        assert stats["category_distribution"]["default"] == 0.5  # 10 / 20
        assert stats["category_distribution"]["report"] == 0.25  # 5 / 20
        assert stats["category_distribution"]["news"] == 0.25  # 5 / 20


class TestCreateFinancialEmbeddings:
    """Test suite for create_financial_embeddings factory function."""
    
    @pytest.fixture
    def mock_torch(self):
        """Create a mocked torch module."""
        with patch('langchain_hana.financial.fin_e5_embeddings.torch') as mock:
            # Mock CUDA availability
            mock.cuda.is_available.return_value = True
            yield mock
    
    @pytest.fixture
    def mock_fin_e5_embeddings(self):
        """Create a mocked FinE5Embeddings class."""
        with patch('langchain_hana.financial.fin_e5_embeddings.FinE5Embeddings') as mock:
            yield mock
    
    @pytest.fixture
    def mock_fin_e5_tensorrt_embeddings(self):
        """Create a mocked FinE5TensorRTEmbeddings class."""
        with patch('langchain_hana.financial.fin_e5_embeddings.FinE5TensorRTEmbeddings') as mock:
            yield mock
    
    def test_create_with_defaults(self, mock_torch, mock_fin_e5_embeddings):
        """Test creating embeddings with default parameters."""
        # Test
        result = create_financial_embeddings()
        
        # Verify
        mock_fin_e5_embeddings.assert_called_once_with(
            model_type="default",
            device="cuda",
            use_fp16=True,
            enable_caching=True,
            normalize_embeddings=True,
            add_financial_prefix=True,
            financial_prefix_type="general"
        )
        assert result == mock_fin_e5_embeddings.return_value
    
    def test_create_with_cpu(self, mock_torch, mock_fin_e5_embeddings):
        """Test creating embeddings with CPU."""
        # Test
        result = create_financial_embeddings(use_gpu=False)
        
        # Verify
        mock_fin_e5_embeddings.assert_called_once_with(
            model_type="default",
            device="cpu",
            use_fp16=False,
            enable_caching=True,
            normalize_embeddings=True,
            add_financial_prefix=True,
            financial_prefix_type="general"
        )
        assert result == mock_fin_e5_embeddings.return_value
    
    def test_create_with_tensorrt(self, mock_torch, mock_fin_e5_tensorrt_embeddings):
        """Test creating embeddings with TensorRT."""
        # Test
        result = create_financial_embeddings(use_tensorrt=True)
        
        # Verify
        mock_fin_e5_tensorrt_embeddings.assert_called_once_with(
            model_type="default",
            precision="fp16",
            multi_gpu=True,
            add_financial_prefix=True,
            financial_prefix_type="general",
            enable_caching=True
        )
        assert result == mock_fin_e5_tensorrt_embeddings.return_value
    
    def test_create_with_tensorrt_fallback(self, mock_torch, mock_fin_e5_embeddings, mock_fin_e5_tensorrt_embeddings):
        """Test fallback to standard embeddings when TensorRT initialization fails."""
        # Make TensorRT initialization fail
        mock_fin_e5_tensorrt_embeddings.side_effect = RuntimeError("TensorRT initialization failed")
        
        # Test
        result = create_financial_embeddings(use_tensorrt=True)
        
        # Verify
        mock_fin_e5_tensorrt_embeddings.assert_called_once()
        mock_fin_e5_embeddings.assert_called_once_with(
            model_type="default",
            device="cuda",
            use_fp16=True,
            enable_caching=True,
            normalize_embeddings=True,
            add_financial_prefix=True,
            financial_prefix_type="general"
        )
        assert result == mock_fin_e5_embeddings.return_value
    
    def test_create_with_custom_parameters(self, mock_torch, mock_fin_e5_embeddings):
        """Test creating embeddings with custom parameters."""
        # Test
        result = create_financial_embeddings(
            model_type="high_quality",
            add_financial_prefix=False,
            enable_caching=False
        )
        
        # Verify
        mock_fin_e5_embeddings.assert_called_once_with(
            model_type="high_quality",
            device="cuda",
            use_fp16=True,
            enable_caching=False,
            normalize_embeddings=True,
            add_financial_prefix=False,
            financial_prefix_type="general"
        )
        assert result == mock_fin_e5_embeddings.return_value


# Performance tests

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestPerformance:
    """Performance tests for financial embeddings."""
    
    @pytest.fixture
    def sample_texts(self):
        """Generate sample financial texts."""
        return [
            f"Company XYZ reported Q{i % 4 + 1} 2023 revenue of ${i * 10} million, "
            f"up {i + 5}% year-over-year. EBITDA margin improved to {20 + i * 0.5}%, "
            f"while operating expenses decreased by {i % 5 + 1}%."
            for i in range(100)
        ]
    
    @pytest.mark.slow
    def test_batch_size_impact(self, sample_texts):
        """Test impact of batch size on performance."""
        # Skip if running in CI environment
        if os.environ.get("CI") == "true":
            pytest.skip("Skipping performance test in CI environment")
        
        try:
            embeddings = FinE5Embeddings(
                model_type="default",
                device="cuda",
                use_fp16=True,
                normalize_embeddings=True,
                enable_caching=False
            )
            
            # Test different batch sizes
            batch_sizes = [1, 4, 8, 16, 32, 64]
            results = {}
            
            for batch_size in batch_sizes:
                embeddings.batch_size = batch_size
                
                # Measure performance
                start_time = time.time()
                _ = embeddings.embed_documents(sample_texts)
                end_time = time.time()
                
                results[batch_size] = end_time - start_time
            
            # Verify that larger batch sizes generally improve performance
            # We expect diminishing returns and potential degradation at very large sizes
            optimal_batch_size = min(results, key=results.get)
            assert optimal_batch_size > 1, "Batching should improve performance"
            
            # Log results
            for batch_size, duration in sorted(results.items()):
                print(f"Batch size {batch_size}: {duration:.4f} seconds")
            print(f"Optimal batch size: {optimal_batch_size}")
            
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Could not initialize embeddings for performance test: {str(e)}")
    
    @pytest.mark.slow
    def test_memory_scaling(self, sample_texts):
        """Test memory scaling with input size."""
        # Skip if running in CI environment
        if os.environ.get("CI") == "true":
            pytest.skip("Skipping performance test in CI environment")
        
        try:
            # Skip if memory tracking not available
            pytest.importorskip("pynvml")
            from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
            
            nvmlInit()
            device_handle = nvmlDeviceGetHandleByIndex(0)
            
            embeddings = FinE5Embeddings(
                model_type="default",
                device="cuda",
                use_fp16=True,
                normalize_embeddings=True,
                enable_caching=False
            )
            
            # Test different text counts
            text_counts = [10, 50, 100, 200, 500]
            memory_usage = {}
            
            for count in text_counts:
                # Clear GPU memory
                torch.cuda.empty_cache()
                
                # Measure baseline memory
                info = nvmlDeviceGetMemoryInfo(device_handle)
                baseline_memory = info.used
                
                # Embed texts
                texts = sample_texts[:count]
                _ = embeddings.embed_documents(texts)
                
                # Measure memory after embedding
                info = nvmlDeviceGetMemoryInfo(device_handle)
                final_memory = info.used
                
                memory_usage[count] = (final_memory - baseline_memory) / (1024 * 1024)  # MB
            
            # Log results
            for count, memory in sorted(memory_usage.items()):
                print(f"Text count {count}: {memory:.2f} MB")
            
            # Verify that memory usage scales reasonably
            if len(memory_usage) >= 2:
                counts = sorted(memory_usage.keys())
                # Memory scaling should be sub-linear due to batching
                ratio1 = memory_usage[counts[-1]] / memory_usage[counts[0]]
                ratio2 = counts[-1] / counts[0]
                assert ratio1 < ratio2, "Memory scaling should be sub-linear"
            
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Could not initialize memory tracking for test: {str(e)}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])