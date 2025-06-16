# Configuration Guide for LangChain SAP HANA Cloud Integration

This guide provides recommended configurations and sensible defaults for integrating LangChain with SAP HANA Cloud.

## Quick Start with Sensible Defaults

```python
from langchain_hana import HanaDB
from langchain_openai import OpenAIEmbeddings
from hdbcli import dbapi

# 1. Create a connection to SAP HANA Cloud
connection = dbapi.connect(
    address="your-hana-hostname.hanacloud.ondemand.com",
    port=443,
    user="your_user",
    password="your_password",
    encrypt=True,  # Always enable encryption for production
    sslValidateCertificate=True
)

# 2. Initialize with sensible defaults
vectorstore = HanaDB(
    connection=connection,
    embedding=OpenAIEmbeddings(),  # Use your preferred embedding model
    # All other parameters use sensible defaults
)
```

## Recommended Configurations for Different Use Cases

### 1. Production-Ready Configuration

```python
vectorstore = HanaDB(
    connection=connection,
    embedding=OpenAIEmbeddings(),
    distance_strategy=DistanceStrategy.COSINE,  # Most compatible option
    table_name="PRODUCTION_EMBEDDINGS",  # Descriptive table name
    vector_column_type="REAL_VECTOR",  # Most compatible vector type
    specific_metadata_columns=["category", "source", "timestamp"]  # Performance optimization
)

# Create HNSW index for faster searches
vectorstore.create_hnsw_index(
    m=64,  # Balanced for accuracy and performance
    ef_construction=200,  # Higher value = better accuracy, slower indexing
    ef_search=100  # Higher value = better accuracy, slower queries
)
```

### 2. Memory-Optimized Configuration

```python
vectorstore = HanaDB(
    connection=connection,
    embedding=OpenAIEmbeddings(dimensions=384),  # Smaller dimension embeddings
    distance_strategy=DistanceStrategy.COSINE,
    table_name="MEMORY_OPTIMIZED_EMBEDDINGS",
    vector_column_type="HALF_VECTOR",  # Uses half the storage (if supported)
    vector_column_length=384  # Fixed length for memory optimization
)

# Create HNSW index optimized for memory efficiency
vectorstore.create_hnsw_index(
    m=32,  # Lower M value uses less memory
    ef_construction=100,
    ef_search=50
)
```

### 3. Performance-Optimized Configuration

```python
from langchain_hana import HanaInternalEmbeddings

# Using HANA's internal embeddings for maximum performance
vectorstore = HanaDB(
    connection=connection,
    embedding=HanaInternalEmbeddings(internal_embedding_model_id="SAP_NEB.20240715"),
    distance_strategy=DistanceStrategy.COSINE,
    table_name="PERFORMANCE_OPTIMIZED_EMBEDDINGS",
    specific_metadata_columns=["category", "source", "timestamp"]
)

# Create high-performance HNSW index
vectorstore.create_hnsw_index(
    m=100,  # Higher M value improves search performance
    ef_construction=300,
    ef_search=200
)
```

## Parameter Reference with Recommendations

| Parameter | Description | Default | Recommendation |
|-----------|-------------|---------|---------------|
| `connection` | HANA database connection | Required | Use connection pooling in production |
| `embedding` | Embedding model | Required | OpenAIEmbeddings for quality, HanaInternalEmbeddings for performance |
| `distance_strategy` | Distance measure | COSINE | COSINE for most use cases, EUCLIDEAN for specific ML applications |
| `table_name` | Database table name | "EMBEDDINGS" | Use descriptive names for your use case |
| `content_column` | Column for text content | "VEC_TEXT" | Default is sufficient for most cases |
| `metadata_column` | Column for metadata | "VEC_META" | Default is sufficient for most cases |
| `vector_column` | Column for vectors | "VEC_VECTOR" | Default is sufficient for most cases |
| `vector_column_length` | Fixed length for vectors | -1 (dynamic) | Set to your embedding dimension for performance |
| `vector_column_type` | Vector data type | "REAL_VECTOR" | "REAL_VECTOR" for compatibility, "HALF_VECTOR" for storage optimization |
| `specific_metadata_columns` | Metadata for dedicated columns | None | Use for frequently filtered metadata fields |

## Understanding Distance Strategies

* **COSINE** (`DistanceStrategy.COSINE`): 
  * Measures similarity based on the angle between vectors
  * Range: 0 to 1 (higher is more similar)
  * Best for: Most text embedding models, semantic search
  * Recommended for: General-purpose applications

* **EUCLIDEAN** (`DistanceStrategy.EUCLIDEAN_DISTANCE`):
  * Measures similarity based on straight-line distance between vectors
  * Range: 0 to infinity (lower is more similar)
  * Best for: Some machine learning applications
  * Recommended for: Specialized use cases

## Vector Column Type Recommendations

* **REAL_VECTOR**:
  * 32-bit floating-point precision
  * Available in: HANA Cloud QRC 1/2024+
  * Best for: Maximum compatibility and precision
  * Recommended for: Most applications

* **HALF_VECTOR**:
  * 16-bit floating-point precision
  * Available in: HANA Cloud QRC 2/2025+
  * Best for: Storage optimization (half the storage space)
  * Recommended for: Large-scale deployments where storage is a concern

## Performance Optimization Tips

1. **Use specific metadata columns** for fields you frequently filter on:
   ```python
   vectorstore = HanaDB(
       connection=connection,
       embedding=embedding,
       specific_metadata_columns=["category", "source", "date"]
   )
   ```

2. **Create HNSW indexes** for faster similarity search:
   ```python
   vectorstore.create_hnsw_index()
   ```

3. **Use batch operations** for adding multiple documents:
   ```python
   vectorstore.add_texts(texts, metadatas)
   ```

4. **Consider internal embeddings** for higher throughput:
   ```python
   from langchain_hana import HanaInternalEmbeddings
   
   internal_emb = HanaInternalEmbeddings(internal_embedding_model_id="SAP_NEB.20240715")
   vectorstore = HanaDB(connection=connection, embedding=internal_emb)
   ```

## Error Message Reference

Common error messages and their solutions:

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| "Vector type 'X' is not available on your SAP HANA Cloud instance" | Your HANA instance doesn't support the vector type | Use 'REAL_VECTOR' or upgrade your HANA instance |
| "Internal embedding model ID cannot be None" | Missing model ID for internal embeddings | Specify a valid model ID or use external embeddings |
| "Column X does not exist" | Table structure mismatch | Check table structure and column names |
| "Unsupported distance_strategy" | Invalid distance strategy value | Use DistanceStrategy.COSINE or DistanceStrategy.EUCLIDEAN_DISTANCE |
| "Invalid filter operator" | Incorrect filter syntax | Check filter syntax and use supported operators |

## Next Steps

- See the [Security Guide](security_guide.md) for production deployment security recommendations
- Explore [Advanced Features](advanced_features.md) for additional capabilities