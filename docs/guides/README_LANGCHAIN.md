# LangChain Integration for SAP HANA Cloud

This package provides a seamless integration between LangChain and SAP HANA Cloud's vector capabilities, allowing you to leverage SAP HANA Cloud's powerful in-memory database for vector search and retrieval.

## Features

- **SAP HANA Cloud Vector Store**: Store and retrieve vectors using SAP HANA Cloud's vector capabilities
- **Efficient Connection Management**: Utilities for connecting to SAP HANA Cloud and managing connections
- **Versatile Embedding Options**:
  - Support for external embedding models (HuggingFace, OpenAI, etc.)
  - Support for SAP HANA Cloud's internal embedding functionality
  - Financial domain-specific embeddings with FinMTEB/Fin-E5 models
  - Caching layer for improved performance
- **Advanced Search Capabilities**:
  - Similarity search with various distance metrics (Cosine, Euclidean)
  - Maximal Marginal Relevance (MMR) search for diverse results
  - Rich metadata filtering
- **Performance Optimization**:
  - HNSW indexing for fast similarity search
  - Batch processing for efficient document insertion
  - Caching for repeated queries
  - TensorRT GPU acceleration for high-throughput embedding generation

## Installation

```bash
pip install langchain-hana
```

## Prerequisites

- SAP HANA Cloud instance with vector capabilities
- Python 3.8+
- `hdbcli` Python package (SAP HANA Cloud client)
- LangChain

## Quick Start

```python
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_hana.vectorstore import HanaVectorStore

# Initialize embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create vector store
vector_store = HanaVectorStore(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password",
    embedding=embeddings,
    table_name="MY_VECTORS",
    create_table=True  # Create table if it doesn't exist
)

# Add documents
docs = ["Document 1", "Document 2", "Document 3"]
metadata = [{"source": "wiki"}, {"source": "web"}, {"source": "book"}]
vector_store.add_texts(docs, metadata)

# Search
results = vector_store.similarity_search("your query", k=3)
for doc in results:
    print(doc.page_content, doc.metadata)
```

## Using SAP HANA Cloud's Internal Embeddings

If your SAP HANA Cloud instance supports internal embeddings, you can leverage them for improved performance:

```python
from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.vectorstore import HanaVectorStore

# Initialize with SAP HANA's internal embedding model ID
embeddings = HanaInternalEmbeddings(model_id="SAP_NEB.20240301")  # Use your model ID

# Create vector store with internal embeddings
vector_store = HanaVectorStore(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password",
    embedding=embeddings,
    table_name="MY_VECTORS",
    create_table=True
)

# When similarity_search is called, embedding generation happens
# directly in the database for maximum performance
results = vector_store.similarity_search("What is SAP HANA?")
```

## Financial Domain Embeddings

This integration includes specialized support for financial domain embeddings using the FinMTEB/Fin-E5 models, which are specifically fine-tuned for financial text understanding.

### Key Features

- **Domain-Specific Models**: Optimized for financial terminology and concepts
- **Contextual Enhancement**: Automatic addition of financial context for improved relevance
- **Performance Optimization**: GPU acceleration, caching, and memory management
- **Enterprise Reliability**: Error handling, fallbacks, and thread safety
- **Monitoring and Metrics**: Performance tracking and diagnostics
- **Customization Options**: Various models for different quality/performance tradeoffs

### Basic Usage

```python
from langchain_hana import create_financial_embeddings, HanaVectorStore
from langchain_hana.connection import create_connection

# Connect to SAP HANA Cloud
connection = create_connection(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password"
)

# Create financial embeddings
embeddings = create_financial_embeddings(
    model_type="high_quality",  # Use Fin-E5 model
    use_gpu=True,
    add_financial_prefix=True
)

# Create vector store
vector_store = HanaVectorStore(
    connection=connection,
    embedding=embeddings,
    table_name="FINANCIAL_DOCUMENTS",
    create_table=True  # Create table if it doesn't exist
)

# Add financial documents
financial_texts = [
    "Q1 2025 Financial Results: Company XYZ reported revenue of $1.2 billion, up 15% year-over-year.",
    "Market volatility remains elevated due to geopolitical tensions and inflationary pressures.",
    "The board approved a quarterly dividend of $0.45 per share."
]

metadata = [
    {"type": "earnings_report", "quarter": "Q1", "year": 2025},
    {"type": "risk_report", "date": "2025-04-15"},
    {"type": "dividend_announcement", "date": "2025-04-15"}
]

vector_store.add_texts(financial_texts, metadatas=metadata)

# Create HNSW index for faster searches
vector_store.create_hnsw_index()

# Search
results = vector_store.similarity_search(
    query="What was the Q1 revenue growth?",
    filter={"type": "earnings_report"}
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print("---")
```

### Available Financial Models

This integration provides access to multiple specialized financial embedding models:

| Model Type | Model Name | Description | Dimensions | Best For |
|------------|------------|-------------|------------|----------|
| **default** | FinMTEB/Fin-E5-small | Best balance of quality and performance | 384 | General financial text |
| **high_quality** | FinMTEB/Fin-E5 | Highest quality, larger model | 768 | Critical financial analysis |
| **efficient** | FinLang/investopedia_embedding | Most efficient, good for limited resources | 384 | High-volume processing |
| **tone** | yiyanghkust/finbert-tone | Specialized for sentiment/tone analysis | 768 | Sentiment analysis |
| **financial_bert** | ProsusAI/finbert | Good for SEC filings, earnings reports | 768 | Regulatory documents |
| **finance_base** | baconnier/Finance_embedding_large_en-V0.1 | General finance embeddings | 1024 | Comprehensive coverage |

### Financial Context Enhancement

For improved retrieval accuracy, the integration automatically adds financial context to documents and queries:

```python
from langchain_hana import FinE5Embeddings

# Create embeddings with financial context enhancement
embeddings = FinE5Embeddings(
    model_type="default",
    add_financial_prefix=True,
    financial_prefix_type="analysis"  # Options: "general", "analysis", "report", "news", "forecast", "investment"
)

# The embedding process will automatically add context like:
# "Financial analysis: What was the Q1 revenue growth?"
```

### Advanced Memory Management

For production deployments handling large volumes of financial documents:

```python
from langchain_hana import FinE5Embeddings

# Create embeddings with advanced memory management
embeddings = FinE5Embeddings(
    model_type="high_quality",
    device="cuda",
    enable_memory_optimization=True,
    memory_optimization_level="balanced",  # Options: "conservative", "balanced", "aggressive"
    adaptive_batch_size=True  # Automatically adjusts batch size based on document length
)

# Process thousands of documents efficiently
large_document_set = [...]  # List of thousands of financial documents
embeddings.embed_documents(large_document_set)  # Memory-optimized processing
```

### GPU Acceleration with TensorRT

For high-throughput embedding generation, you can use TensorRT-accelerated embeddings:

```python
from langchain_hana import FinE5TensorRTEmbeddings, HanaVectorStore

# Create TensorRT-accelerated financial embeddings
embeddings = FinE5TensorRTEmbeddings(
    model_type="default",
    precision="fp16",  # Options: "fp32", "fp16", "int8"
    multi_gpu=True,
    add_financial_prefix=True,
    enable_tensor_cores=True  # Use NVIDIA Tensor Cores for faster processing
)

# Create vector store
vector_store = HanaVectorStore(
    connection=connection,
    embedding=embeddings,
    table_name="FINANCIAL_DOCUMENTS"
)

# Run performance benchmark
benchmark_results = embeddings.benchmark(batch_sizes=[16, 32, 64])
print(f"Performance: {benchmark_results['documents_per_second']} docs/sec")
```

### Financial Embedding Caching

For improved performance with frequently accessed financial content:

```python
from langchain_hana import FinE5Embeddings, FinancialEmbeddingCache

# Create base embeddings model
base_embeddings = FinE5Embeddings(model_type="default")

# Create financial domain-specific cache with specialized TTLs
cached_embeddings = FinancialEmbeddingCache(
    base_embeddings=base_embeddings,
    ttl_seconds=3600,  # Default TTL (1 hour)
    max_size=10000,
    persist_path="/path/to/financial_cache.pkl",
    model_name="fin-e5"
)

# Different financial content types have different cache TTLs:
# - News: 1 hour (default for rapidly changing content)
# - Reports: 1 week (more stable content)
# - Analysis: 1 day (moderately stable content)
# - Market data: 30 minutes (very volatile content)
```

### Complete Financial RAG System

For a production-ready financial RAG system:

```python
from langchain_hana.financial import create_financial_system
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Create complete, production-ready financial system
system = create_financial_system(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password",
    quality_tier="balanced",
    table_name="FINANCIAL_DOCUMENTS"
)

# Add documents
system.add_documents(financial_documents)

# Create a custom template for financial analysis
template = """
You are a financial analyst assistant. Answer the user's question based on the retrieved financial information.

CONTEXT:
{context}

QUESTION:
{question}

FINANCIAL ANALYSIS:
"""
prompt = PromptTemplate.from_template(template)

# Create a retrieval chain
llm = ChatOpenAI(temperature=0, model="gpt-4o")
qa_chain = system.create_rag_chain(llm=llm, prompt=prompt)

# Run question answering
response = qa_chain.invoke("What are the financial risks mentioned in the quarterly report?")
print(response)

# Get system performance metrics
metrics = system.get_metrics()
print(f"System performance: {metrics['queries_per_second']} queries/sec")
```

## Advanced Usage

### Connection Management

```python
from langchain_hana.connection import create_connection, test_connection

# Create connection
connection = create_connection(
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password",
)

# Test connection
connection_valid, info = test_connection(connection)
if connection_valid:
    print(f"Connected to SAP HANA Cloud {info.get('version')}")
    print(f"Current schema: {info.get('current_schema')}")
else:
    print(f"Connection test failed: {info.get('error')}")
```

### Embedding Caching

```python
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_hana.embeddings import HanaEmbeddingsCache

# Create base embeddings model
base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create cached embeddings with 1-hour TTL and disk persistence
cached_embeddings = HanaEmbeddingsCache(
    base_embeddings=base_embeddings,
    ttl_seconds=3600,  # 1 hour cache lifetime
    max_size=10000,
    persist_path="/path/to/cache.pkl"  # Optional: persist cache to disk
)

# Use like any other embeddings model
embedding = cached_embeddings.embed_query("Hello, world!")
```

### Financial Domain-Specific Caching

```python
from langchain_hana.financial import FinE5Embeddings, FinancialEmbeddingCache

# Create financial embeddings model
base_embeddings = FinE5Embeddings(model_type="high_quality")

# Create financial domain-specific cache with category-aware TTLs
cached_embeddings = FinancialEmbeddingCache(
    base_embeddings=base_embeddings,
    ttl_seconds=3600,
    max_size=10000,
    persist_path="/path/to/financial_cache.pkl",
    model_name="fin-e5"
)

# Different financial content types have different cache TTLs:
# - News: 1 hour
# - Reports: 1 week
# - Analysis: 1 day
# - Market data: 30 minutes

# Use like any other embeddings model
embedding = cached_embeddings.embed_query("What was the Q1 revenue?")
```

### Metadata Filtering

```python
# Search with simple filter
results = vector_store.similarity_search(
    "your query",
    filter={"category": "database"}
)

# Search with complex filter
results = vector_store.similarity_search(
    "your query",
    filter={
        "$and": [
            {"category": "database"},
            {"year": {"$gt": 2020}},
            {"$or": [
                {"tags": {"$contains": "cloud"}},
                {"tags": {"$contains": "vector"}}
            ]}
        ]
    }
)
```

### Maximal Marginal Relevance (MMR) Search

```python
# Search with MMR to ensure diverse results
results = vector_store.max_marginal_relevance_search(
    "your query",
    k=5,  # Number of results to return
    fetch_k=20,  # Number of results to fetch before applying MMR
    lambda_mult=0.5  # Diversity factor (0.0 = max diversity, 1.0 = max relevance)
)
```

### Performance Optimization with HNSW Index

```python
# Create HNSW index for fast similarity search
vector_store.create_hnsw_index(
    m=16,                  # Number of connections per node
    ef_construction=128,   # Index building quality parameter
    ef_search=64           # Search quality parameter
)
```

## Example Scripts

- `examples/hana_vectorstore_basics.py`: Fundamentals of the HANA Vector Store implementation
- `examples/langchain_hana_quickstart.py`: Basic usage of LangChain with SAP HANA Cloud
- `examples/financial_embeddings_example.py`: Demonstration of financial domain embeddings
- `examples/langchain_hana_rag_example.py`: Building a RAG system with LangChain and SAP HANA Cloud
- `examples/financial_rag_example.py`: Financial domain-specific RAG with FinE5 embeddings
- `examples/real_time_financial_analysis.py`: Real-time financial data analysis system

For comprehensive implementation details, see the `LANGCHAIN_INTEGRATION_GUIDE.md` file.

To run the examples:

```bash
# Vector store basics example
python examples/hana_vectorstore_basics.py --config_file config/connection.json

# Basic LangChain integration example
python examples/langchain_hana_quickstart.py

# Financial embeddings example
python examples/financial_embeddings_example.py --config_file config/connection.json

# RAG example
python examples/langchain_hana_rag_example.py --config config/connection.json

# Financial RAG example
python examples/financial_rag_example.py --config_file config/connection.json

# Real-time financial analysis example
python examples/real_time_financial_analysis.py --config_file config/connection.json --run_time 300
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.