"""
LangChain Integration for SAP HANA Cloud
========================================

Production-grade integration between LangChain and SAP HANA Cloud's vector database
capabilities for building robust AI applications.

Copyright (c) 2023 FinSights AP
"""

from langchain_hana_integration.vectorstore import SAP_HANA_VectorStore
from langchain_hana_integration.embeddings import HanaOptimizedEmbeddings
from langchain_hana_integration.connection import create_connection_pool
from langchain_hana_integration.exceptions import (
    ConnectionError,
    DatabaseError,
    ConfigurationError,
    VectorOperationError
)

__version__ = "1.0.0"