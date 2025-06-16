"""
Tests for enhanced GPU-accelerated embeddings.

This module contains additional unit tests for the GPU acceleration components,
focusing on the API's embedding providers and registry functionality.
"""

import unittest
from unittest.mock import MagicMock, patch, Mock
import numpy as np

# Import the embedding classes
from api.embeddings.embedding_providers import (
    EmbeddingProvider,
    BaseEmbeddingProvider,
    EmbeddingProviderRegistry
)
from api.embeddings.embeddings import (
    GPUAcceleratedEmbeddings,
    GPUHybridEmbeddings,
    DefaultEmbeddingProvider
)

class TestGPUAcceleratedEmbeddings(unittest.TestCase):
    """Tests for the GPU-accelerated embeddings class."""
    
    @patch('api.embeddings.embeddings.gpu_utils')
    @patch('api.embeddings.embeddings.SentenceTransformer')
    def test_init_gpu_available(self, mock_sentence_transformer, mock_gpu_utils):
        """Test initialization when GPU is available."""
        # Setup
        mock_gpu_utils.is_torch_available.return_value = True
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        
        # Execute
        embeddings = GPUAcceleratedEmbeddings(model_name="test-model")
        
        # Assert
        self.assertEqual(embeddings.device, "cuda")
        self.assertEqual(embeddings.model_name, "test-model")
        mock_sentence_transformer.assert_called_once_with("test-model", device="cuda")
    
    @patch('api.embeddings.embeddings.gpu_utils')
    @patch('api.embeddings.embeddings.SentenceTransformer')
    def test_init_gpu_not_available(self, mock_sentence_transformer, mock_gpu_utils):
        """Test initialization when GPU is not available."""
        # Setup
        mock_gpu_utils.is_torch_available.return_value = False
        mock_model = MagicMock()
        mock_sentence_transformer.return_value = mock_model
        
        # Execute
        embeddings = GPUAcceleratedEmbeddings(model_name="test-model")
        
        # Assert
        self.assertEqual(embeddings.device, "cpu")
        mock_sentence_transformer.assert_called_once_with("test-model", device="cpu")
    
    @patch('api.embeddings.embeddings.gpu_utils')
    @patch('api.embeddings.embeddings.SentenceTransformer')
    def test_embed_documents(self, mock_sentence_transformer, mock_gpu_utils):
        """Test embedding multiple documents."""
        # Setup
        mock_gpu_utils.is_torch_available.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_sentence_transformer.return_value = mock_model
        
        # Execute
        embeddings = GPUAcceleratedEmbeddings(model_name="test-model")
        result = embeddings.embed_documents(["text1", "text2"])
        
        # Assert
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        mock_model.encode.assert_called_once_with(
            ["text1", "text2"], 
            batch_size=32, 
            show_progress_bar=False,
            convert_to_numpy=True
        )
    
    @patch('api.embeddings.embeddings.gpu_utils')
    @patch('api.embeddings.embeddings.SentenceTransformer')
    def test_embed_query(self, mock_sentence_transformer, mock_gpu_utils):
        """Test embedding a single query."""
        # Setup
        mock_gpu_utils.is_torch_available.return_value = True
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 2.0]])
        mock_sentence_transformer.return_value = mock_model
        
        # Execute
        embeddings = GPUAcceleratedEmbeddings(model_name="test-model")
        result = embeddings.embed_query("query text")
        
        # Assert
        self.assertEqual(result, [1.0, 2.0])
        mock_model.encode.assert_called_once_with(
            ["query text"], 
            batch_size=32, 
            show_progress_bar=False,
            convert_to_numpy=True
        )

class TestGPUHybridEmbeddings(unittest.TestCase):
    """Tests for the hybrid embeddings class."""
    
    @patch('api.embeddings.embeddings.HanaInternalEmbeddings')
    def test_init(self, mock_hana_embeddings):
        """Test initialization."""
        # Setup
        mock_internal = MagicMock()
        mock_hana_embeddings.return_value = mock_internal
        
        # Execute
        embeddings = GPUHybridEmbeddings(
            internal_embedding_model_id="test-model-id",
            external_model_name="test-external-model",
            use_internal=True
        )
        
        # Assert
        self.assertEqual(embeddings.internal_embedding_model_id, "test-model-id")
        self.assertEqual(embeddings.external_model_name, "test-external-model")
        self.assertTrue(embeddings.use_internal)
        mock_hana_embeddings.assert_called_once_with(internal_embedding_model_id="test-model-id")
        self.assertIsNone(embeddings._external_embeddings)
    
    @patch('api.embeddings.embeddings.HanaInternalEmbeddings')
    def test_external_embeddings_lazy_init(self, mock_hana_embeddings):
        """Test lazy initialization of external embeddings."""
        # Setup
        mock_internal = MagicMock()
        mock_hana_embeddings.return_value = mock_internal
        
        with patch('api.embeddings.embeddings.GPUAcceleratedEmbeddings') as mock_gpu_embeddings:
            mock_external = MagicMock()
            mock_gpu_embeddings.return_value = mock_external
            
            # Execute
            embeddings = GPUHybridEmbeddings(
                internal_embedding_model_id="test-model-id",
                external_model_name="test-external-model"
            )
            
            # Assert before accessing property
            self.assertIsNone(embeddings._external_embeddings)
            mock_gpu_embeddings.assert_not_called()
            
            # Access property to trigger lazy init
            external = embeddings.external_embeddings
            
            # Assert after accessing property
            self.assertIsNotNone(embeddings._external_embeddings)
            mock_gpu_embeddings.assert_called_once_with(
                model_name="test-external-model",
                device=None,
                batch_size=32
            )
    
    @patch('api.embeddings.embeddings.HanaInternalEmbeddings')
    def test_embed_documents_internal(self, mock_hana_embeddings):
        """Test embedding documents using internal embeddings."""
        # Setup
        mock_internal = MagicMock()
        mock_internal.embed_documents.return_value = [[1.0, 2.0], [3.0, 4.0]]
        mock_hana_embeddings.return_value = mock_internal
        
        # Execute
        embeddings = GPUHybridEmbeddings(use_internal=True)
        result = embeddings.embed_documents(["text1", "text2"])
        
        # Assert
        self.assertEqual(result, [[1.0, 2.0], [3.0, 4.0]])
        mock_internal.embed_documents.assert_called_once_with(["text1", "text2"])

class TestEmbeddingProviderRegistry(unittest.TestCase):
    """Tests for the embedding provider registry."""
    
    def test_registry_get_provider(self):
        """Test getting a provider from the registry."""
        # Setup
        mock_provider_class = Mock()
        mock_provider = Mock()
        mock_provider_class.return_value = mock_provider
        
        # Register provider
        EmbeddingProviderRegistry.register("test_provider", mock_provider_class)
        
        # Execute
        provider = EmbeddingProviderRegistry.get_provider("test_provider", model_name="test-model")
        
        # Assert
        self.assertEqual(provider, mock_provider)
        mock_provider_class.assert_called_once_with(model_name="test-model")
    
    def test_registry_get_available_providers(self):
        """Test getting available providers."""
        # Setup - ensure the basic providers are registered
        EmbeddingProviderRegistry.register("gpu", GPUAcceleratedEmbeddings)
        EmbeddingProviderRegistry.register("hybrid", GPUHybridEmbeddings)
        
        # Execute
        providers = EmbeddingProviderRegistry.get_available_providers()
        
        # Assert
        self.assertIn("gpu", providers)
        self.assertIn("hybrid", providers)
        self.assertIn("test_provider", providers)  # From previous test
    
    def test_registry_invalid_provider(self):
        """Test error when requesting an invalid provider."""
        # Execute and Assert
        with self.assertRaises(ValueError):
            EmbeddingProviderRegistry.get_provider("nonexistent_provider")

class TestEmbeddingProviderFactory(unittest.TestCase):
    """Tests for the embedding provider factory function."""
    
    @patch('api.embeddings.__init__.EmbeddingProviderRegistry')
    def test_get_embedding_provider(self, mock_registry):
        """Test the get_embedding_provider factory function."""
        # Setup
        mock_provider = MagicMock()
        mock_registry.get_provider.return_value = mock_provider
        
        # Import the factory function
        from api.embeddings import get_embedding_provider
        
        # Execute
        provider = get_embedding_provider("gpu", model_name="test-model")
        
        # Assert
        self.assertEqual(provider, mock_provider)
        mock_registry.get_provider.assert_called_once_with("gpu", model_name="test-model")
    
    @patch('api.embeddings.__init__.EmbeddingProviderRegistry')
    def test_get_embedding_provider_default(self, mock_registry):
        """Test the get_embedding_provider factory function with default provider."""
        # Setup
        mock_provider = MagicMock()
        mock_registry.get_provider.return_value = mock_provider
        
        # Import the factory function
        from api.embeddings import get_embedding_provider
        
        # Execute
        provider = get_embedding_provider()  # Default is "hybrid"
        
        # Assert
        self.assertEqual(provider, mock_provider)
        mock_registry.get_provider.assert_called_once_with("hybrid")

class TestTensorRTIntegration(unittest.TestCase):
    """Tests for TensorRT integration."""
    
    @patch('api.embeddings.embeddings_tensorrt.TensorRTEmbedding')
    def test_tensorrt_provider_registration(self, mock_tensorrt):
        """Test that TensorRT provider is registered."""
        # Setup
        mock_tensorrt_instance = MagicMock()
        mock_tensorrt.return_value = mock_tensorrt_instance
        
        # Import with patching to ensure TensorRTEmbedding is available
        with patch.dict('sys.modules', {'api.embeddings.embeddings_tensorrt': MagicMock()}):
            # Reset provider registry to force re-registration
            EmbeddingProviderRegistry._providers = {}
            
            # Re-import to trigger registration
            from api.embeddings import get_embedding_provider
            
            # Register manually since we've mocked the import
            EmbeddingProviderRegistry.register("tensorrt", mock_tensorrt)
            
            # Execute
            provider = get_embedding_provider("tensorrt", model_name="test-model")
            
            # Assert
            self.assertEqual(provider, mock_tensorrt_instance)
            mock_tensorrt.assert_called_once_with(model_name="test-model")

if __name__ == "__main__":
    unittest.main()