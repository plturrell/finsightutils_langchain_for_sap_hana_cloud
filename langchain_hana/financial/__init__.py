"""
Production-grade financial embeddings and vector store integration for SAP HANA Cloud.

This package provides enterprise-ready components for integrating financial domain-specific
embeddings with SAP HANA Cloud's vector capabilities, optimized for production use.

Components:
- FinE5Embeddings: FinMTEB/Fin-E5 financial domain-specific embedding models
- FinE5TensorRTEmbeddings: GPU-accelerated financial embeddings with TensorRT
- FinancialEmbeddingCache: Domain-aware caching with financial context optimizations
- FinancialEmbeddingSystem: Complete, production-ready system for financial embeddings
- FinancialEmbeddings: Domain-specific embedding model for financial text
- GPUOptimizer: Production-grade GPU optimization for embedding models
- FinancialVectorStore: Enterprise-ready vector store for financial embeddings
- DistributedCache: High-performance caching system for embeddings and results
- LocalModelManager: Manager for downloading and using local embedding models
- ModelFineTuner: Fine-tuning capabilities for customizing embedding models

Usage:
    ```python
    # Production-ready system (recommended):
    from langchain_hana.financial import create_financial_system
    
    # Create complete, production-ready system
    system = create_financial_system(
        host="your-host.hanacloud.ondemand.com",
        port=443,
        user="your-user",
        password="your-password",
        quality_tier="balanced",
        table_name="FINANCIAL_DOCUMENTS"
    )
    
    # Add documents
    system.add_documents(documents)
    
    # Search with automatic caching
    results = system.similarity_search(
        query="What are the risks mentioned in the quarterly report?",
        filter={"document_type": "quarterly_report"}
    )
    
    # Get performance metrics
    metrics = system.get_metrics()
    
    # Check system health
    health = system.health_check()
    
    # Graceful shutdown
    system.shutdown()
    ```
    
    # Using FinMTEB/Fin-E5 financial embeddings:
    ```python
    from langchain_hana.financial import FinE5Embeddings, create_financial_embeddings
    from langchain_hana.vectorstores import HanaDB
    from hdbcli import dbapi
    
    # Create connection
    connection = dbapi.connect(
        address="your-host.hanacloud.ondemand.com",
        port=443,
        user="your-user",
        password="your-password"
    )
    
    # Create financial embeddings model
    embeddings = create_financial_embeddings(
        model_type="high_quality",  # Use Fin-E5 model
        use_gpu=True,
        add_financial_prefix=True
    )
    
    # Create vector store
    vector_store = HanaDB(
        connection=connection,
        embedding=embeddings,
        table_name="FINANCIAL_DOCUMENTS"
    )
    
    # Add documents
    vector_store.add_texts(financial_texts)
    
    # Search
    results = vector_store.similarity_search(
        query="What was the Q1 revenue growth?",
        filter={"document_type": "quarterly_report"}
    )
    ```
    
    # Using local models and fine-tuning:
    ```python
    from langchain_hana.financial import create_local_model_manager, create_model_fine_tuner
    
    # Create local model manager
    model_manager = create_local_model_manager(
        models_dir="./financial_models",
        default_model="FinMTEB/Fin-E5-small"
    )
    
    # Download a model
    model_path = model_manager.download_model("FinMTEB/Fin-E5-small")
    
    # Fine-tune the model
    fine_tuner = create_model_fine_tuner(model_manager=model_manager)
    tuned_model_path = fine_tuner.fine_tune(
        base_model="FinMTEB/Fin-E5-small",
        train_texts=financial_texts,
        epochs=3
    )
    
    # Use the fine-tuned model with SAP HANA
    system = create_financial_system(
        host="your-host.hanacloud.ondemand.com",
        port=443,
        user="your-user",
        password="your-password",
        model_name=tuned_model_path
    )
    ```
"""

# Original imports
from langchain_hana.financial.embeddings import (
    FinancialEmbeddings,
    create_production_financial_embeddings,
)

from langchain_hana.financial.gpu_optimization import (
    GPUOptimizer,
    optimize_for_gpu,
)

from langchain_hana.financial.vector_store import (
    FinancialVectorStore,
    create_financial_vector_store,
)

from langchain_hana.financial.caching import (
    DistributedCache,
    FinancialQueryCache,
    create_query_cache,
)

from langchain_hana.financial.production import (
    FinancialEmbeddingSystem,
    create_financial_system,
)

from langchain_hana.financial.local_models import (
    LocalModelManager,
    ModelFineTuner,
    create_local_model_manager,
    create_model_fine_tuner,
)

# New imports
from langchain_hana.financial.fin_e5_embeddings import (
    FinE5Embeddings, 
    FinE5TensorRTEmbeddings,
    create_financial_embeddings,
    FINANCIAL_EMBEDDING_MODELS
)

from langchain_hana.financial.caching import FinancialEmbeddingCache

__all__ = [
    # Production-ready system
    "FinancialEmbeddingSystem",
    "create_financial_system",
    
    # Embedding models
    "FinancialEmbeddings",
    "create_production_financial_embeddings",
    
    # GPU optimization
    "GPUOptimizer",
    "optimize_for_gpu",
    
    # Vector store
    "FinancialVectorStore",
    "create_financial_vector_store",
    
    # Caching
    "DistributedCache",
    "FinancialQueryCache",
    "create_query_cache",
    
    # Local models and fine-tuning
    "LocalModelManager",
    "ModelFineTuner",
    "create_local_model_manager",
    "create_model_fine_tuner",
    
    # New components
    "FinE5Embeddings",
    "FinE5TensorRTEmbeddings",
    "FinancialEmbeddingCache",
    "create_financial_embeddings",
    "FINANCIAL_EMBEDDING_MODELS"
]