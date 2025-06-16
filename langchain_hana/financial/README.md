# Production-Grade Financial Embeddings for SAP HANA Cloud

This module provides enterprise-ready components for integrating financial domain-specific embeddings with SAP HANA Cloud's vector capabilities, optimized for production use.

## Key Components

### 1. Domain-Specific Financial Embeddings

The `FinancialEmbeddings` class provides optimized embeddings for financial text, with built-in support for various financial embedding models:

- **FinMTEB/Fin-E5**: High-quality financial embeddings (7B parameters)
- **FinMTEB/Fin-E5-small**: Balanced performance and resource usage
- **FinLang/investopedia_embedding**: Efficient embeddings optimized for financial terminology
- **yiyanghkust/finbert-tone**: Specialized for sentiment and tone analysis in financial documents
- **ProsusAI/finbert**: Optimized for SEC filings and financial reports

```python
from langchain_hana.financial import create_production_financial_embeddings

# Create production-ready financial embeddings
embeddings = create_production_financial_embeddings(
    quality_tier="balanced",  # "high", "balanced", or "efficient"
    memory_tier="auto",       # "high", "medium", "low", or "auto"
    enterprise_mode=True
)

# Embed a financial query
query_vector = embeddings.embed_query("What are the risks mentioned in the quarterly report?")
```

### 2. Production-Grade GPU Optimization

The `GPUOptimizer` class provides enterprise-ready GPU optimization for embedding models:

- **Mixed precision inference** (FP16/BF16) for 2-3x speedup
- **Dynamic batch sizing** based on available GPU memory
- **Tensor Core optimization** for modern NVIDIA GPUs
- **Error recovery and fallback** mechanisms for fault tolerance
- **Memory management** with automatic cleanup

```python
from langchain_hana.financial import GPUOptimizer

# Create GPU optimizer
optimizer = GPUOptimizer(
    device_id=0,                # CUDA device ID (None for auto)
    memory_fraction=0.9,        # Fraction of GPU memory to use
    use_mixed_precision=True,   # Enable mixed precision
    precision_type="fp16",      # "fp16", "bf16", or "int8"
    enable_tensor_cores=True    # Enable Tensor Core optimizations
)

# Optimize model
optimized_model = optimizer.optimize_model(model)
```

### 3. Enterprise-Ready Vector Store

The `FinancialVectorStore` class provides a high-performance vector store for financial embeddings:

- **Seamless integration** with SAP HANA Cloud
- **HNSW indexing** for fast similarity search
- **Bulk operations** for efficient data loading
- **Transaction support** for data consistency
- **Connection health monitoring** with automatic recovery
- **Performance metrics** for monitoring and optimization

```python
from langchain_hana.financial import create_financial_vector_store
from hdbcli import dbapi

# Create SAP HANA connection
connection = dbapi.connect(
    address="your-host.hanacloud.ondemand.com",
    port=443,
    user="your-user",
    password="your-password"
)

# Create financial vector store
vector_store = create_financial_vector_store(
    connection=connection,
    embedding_model="FinMTEB/Fin-E5-small",
    table_name="FINANCIAL_DOCUMENTS",
    create_hnsw_index=True,
    enterprise_mode=True
)

# Add documents
vector_store.add_documents(documents)

# Search with metadata filtering
results = vector_store.similarity_search(
    query="What are the risks mentioned in the quarterly report?",
    filter={"document_type": "quarterly_report", "year": 2025}
)
```

### 4. Advanced Caching System

The module includes a high-performance, distributed caching system for embeddings and query results:

- **Multi-tiered storage** (memory, disk, Redis)
- **Semantic caching** for similar queries
- **Time-based expiration** with automatic cleanup
- **Cache eviction policies** (LRU, LFU, FIFO)
- **Performance metrics** for monitoring
- **Thread-safety** for concurrent access

```python
from langchain_hana.financial import create_query_cache

# Create query cache
query_cache = create_query_cache(
    cache_dir="./cache",
    redis_url="redis://localhost:6379/0",  # Optional
    ttl_hours=24,
    semantic_threshold=0.92,
    enable_cross_user=False
)

# Check cache for query result
cached_result = query_cache.get_query_result(
    query="What are the risks mentioned in the quarterly report?",
    user_id="user123",
    filter_params={"document_type": "quarterly_report"}
)

if cached_result:
    print("Cache hit!")
else:
    # Perform search
    results = vector_store.similarity_search(...)
    
    # Cache result
    query_cache.set_query_result(
        query="What are the risks mentioned in the quarterly report?",
        result=results,
        user_id="user123",
        filter_params={"document_type": "quarterly_report"}
    )
```

## Integration with LangChain

All components are fully compatible with LangChain's interfaces:

- `FinancialEmbeddings` implements the `Embeddings` interface
- `FinancialVectorStore` integrates with `HanaDB` and the `VectorStore` interface
- All components support LangChain's callback system for monitoring

## Production Features

- **Thread-safety** for concurrent access
- **Error handling and recovery** for robust operation
- **Performance monitoring** with detailed metrics
- **Memory management** to prevent resource leaks
- **Fault tolerance** with automatic fallback mechanisms
- **Scalable architecture** for high-volume workloads

## Hardware Requirements

- **Minimum**: 8GB RAM, 4 CPU cores, 4GB GPU VRAM (if using GPU)
- **Recommended**: 16GB RAM, 8 CPU cores, 16GB GPU VRAM (NVIDIA T4 or better)
- **Enterprise**: 32GB RAM, 16 CPU cores, 24-80GB GPU VRAM (NVIDIA A10/A100)