"""
Embedding provider interface definitions.

This module defines the interface for embedding providers and implements
the dependency injection pattern to eliminate circular imports between
the API and core library code.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, List, Optional, Protocol, runtime_checkable, Any, Type

import numpy as np
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# Interface definitions
@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the interface for embedding providers."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents."""
        ...
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        ...
    
    def get_model_id(self) -> str:
        """Get the model ID."""
        ...


class EmbeddingProviderRegistry:
    """
    Registry for embedding provider implementations.
    
    This class uses the provider pattern to eliminate circular imports
    between API and core library code. It maintains a registry of embedding
    provider implementations that can be instantiated on demand.
    """
    
    _providers: Dict[str, Type[Embeddings]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[Embeddings]) -> None:
        """
        Register an embedding provider.
        
        Args:
            name: Name of the provider
            provider_class: Class implementing the EmbeddingProvider interface
        """
        cls._providers[name] = provider_class
        logger.info(f"Registered embedding provider: {name}")
    
    @classmethod
    def get_provider(cls, name: str, **kwargs) -> Embeddings:
        """
        Get an instance of the specified provider.
        
        Args:
            name: Name of the provider
            **kwargs: Arguments to pass to the provider constructor
            
        Returns:
            An instance of the provider
            
        Raises:
            ValueError: If the provider is not registered
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Embedding provider '{name}' not registered. "
                f"Available providers: {available}"
            )
        
        provider_class = cls._providers[name]
        return provider_class(**kwargs)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """
        Get a list of available provider names.
        
        Returns:
            List of provider names
        """
        return list(cls._providers.keys())


# Base embedding provider implementations
class BaseEmbeddingProvider(Embeddings):
    """Base class for embedding providers."""
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the base embedding provider.
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
    
    def get_model_id(self) -> str:
        """
        Get the model ID.
        
        Returns:
            Model ID
        """
        return self.model_name