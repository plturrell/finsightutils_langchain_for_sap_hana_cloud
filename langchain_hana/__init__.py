"""
LangChain Integration for SAP HANA Cloud

This package provides a seamless integration between LangChain and SAP HANA Cloud's
vector capabilities, allowing you to leverage SAP HANA Cloud's powerful in-memory
database for vector search and retrieval.

Features:
- Vector store integration with SAP HANA Cloud
- Support for various embedding models
- Financial domain-specific embeddings with Fin-E5 models
- GPU acceleration for high-throughput embedding generation
- Connection management utilities
- Caching for improved performance
- Enterprise-grade reliability features
"""

from importlib import metadata

# Import core components
from langchain_hana.vectorstore import HanaVectorStore, DistanceStrategy
from langchain_hana.embeddings import HanaInternalEmbeddings, HanaEmbeddingsCache
from langchain_hana.connection import create_connection, test_connection, close_connection

# Import financial domain-specific components
from langchain_hana.financial import (
    FinE5Embeddings,
    FinE5TensorRTEmbeddings,
    create_financial_embeddings,
    FinancialEmbeddingCache,
    FINANCIAL_EMBEDDING_MODELS
)

# Import GPU optimization components
from langchain_hana.gpu import (
    HanaTensorRTEmbeddings,
    HanaTensorRTVectorStore,
    TensorRTDiagnostics
)

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available
    __version__ = "0.1.0"
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    # Vector Store
    "HanaVectorStore",
    "DistanceStrategy",
    
    # Embeddings
    "HanaInternalEmbeddings",
    "HanaEmbeddingsCache",
    
    # Connection Utilities
    "create_connection",
    "test_connection",
    "close_connection",
    
    # Financial Domain-Specific Components
    "FinE5Embeddings",
    "FinE5TensorRTEmbeddings",
    "create_financial_embeddings",
    "FinancialEmbeddingCache",
    "FINANCIAL_EMBEDDING_MODELS",
    
    # GPU Optimization Components
    "HanaTensorRTEmbeddings",
    "HanaTensorRTVectorStore",
    "TensorRTDiagnostics",
    
    # Version
    "__version__",
]