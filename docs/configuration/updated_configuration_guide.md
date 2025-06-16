# Configuration Guide for SAP HANA Cloud LangChain Integration

This guide provides comprehensive information about configuring the SAP HANA Cloud LangChain integration to optimize its performance, reliability, and functionality based on your specific requirements.

## Configuration File Locations

Configuration options can be set in multiple ways, with the following precedence (highest to lowest):

1. **Environment Variables**: Highest precedence, overrides all other settings
2. **Command-line Arguments**: For tools and CLI usage
3. **Configuration Files**: Main configuration source for most deployments
4. **Default Values**: Used when no configuration is explicitly provided

The main configuration files are:

- **API Configuration**: `/config.py` (main configuration file)
- **Environment Files**: `.env` (for local development) or `vercel.json` (for Vercel deployment)
- **Docker Compose**: `docker-compose.yml` (for containerized deployments)
- **Kubernetes**: `kubernetes/config.yaml` (for Kubernetes deployments)

## Core Configuration Options

### Connection Settings

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `HANA_HOST` | `HANA_HOST` | `"localhost"` | SAP HANA Cloud host address |
| `HANA_PORT` | `HANA_PORT` | `443` | SAP HANA Cloud port number |
| `HANA_USER` | `HANA_USER` | `""` | SAP HANA Cloud username |
| `HANA_PASSWORD` | `HANA_PASSWORD` | `""` | SAP HANA Cloud password |
| `HANA_SCHEMA` | `HANA_SCHEMA` | `"ML_DATA"` | Default schema for vector operations |
| `HANA_TABLE` | `HANA_TABLE` | `"VECTOR_STORE"` | Default table for vector operations |
| `HANA_USE_SSL` | `HANA_USE_SSL` | `True` | Enable SSL for connection |
| `HANA_VALIDATE_CERT` | `HANA_VALIDATE_CERT` | `True` | Validate SSL certificate |

**Example .env file:**
```
HANA_HOST=your-hana-hostname.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your-username
HANA_PASSWORD=your-password
HANA_SCHEMA=ML_DATA
HANA_TABLE=VECTOR_STORE
```

### Connection Pooling

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `DB_POOL_SIZE` | `DB_POOL_SIZE` | `10` | Maximum number of connections in the pool |
| `DB_POOL_TIMEOUT` | `DB_POOL_TIMEOUT` | `30` | Timeout for getting a connection from the pool (seconds) |
| `DB_POOL_RECYCLE` | `DB_POOL_RECYCLE` | `3600` | Time after which a connection is recycled (seconds) |
| `DB_MAX_OVERFLOW` | `DB_MAX_OVERFLOW` | `20` | Maximum number of connections that can be created beyond the pool size |

**Example configuration in config.py:**
```python
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
DB_POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))
DB_POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))
DB_MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
```

### Retry and Error Handling

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `RETRY_ATTEMPTS` | `RETRY_ATTEMPTS` | `3` | Number of retry attempts for transient errors |
| `RETRY_BACKOFF_FACTOR` | `RETRY_BACKOFF_FACTOR` | `1.5` | Exponential backoff factor between retries |
| `RETRY_STATUS_CODES` | `RETRY_STATUS_CODES` | `[500, 502, 503, 504]` | HTTP status codes to retry on |
| `RETRY_MAX_TIMEOUT` | `RETRY_MAX_TIMEOUT` | `60` | Maximum timeout for a retry operation (seconds) |
| `ERROR_VERBOSITY` | `ERROR_VERBOSITY` | `"standard"` | Error verbosity level (`"minimal"`, `"standard"`, `"detailed"`) |

**Example configuration in config.py:**
```python
RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
RETRY_BACKOFF_FACTOR = float(os.getenv("RETRY_BACKOFF_FACTOR", "1.5"))
RETRY_STATUS_CODES = [int(x) for x in os.getenv("RETRY_STATUS_CODES", "500,502,503,504").split(",")]
RETRY_MAX_TIMEOUT = int(os.getenv("RETRY_MAX_TIMEOUT", "60"))
ERROR_VERBOSITY = os.getenv("ERROR_VERBOSITY", "standard")
```

### GPU Acceleration

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `GPU_ACCELERATION_ENABLED` | `GPU_ACCELERATION_ENABLED` | `True` | Enable GPU acceleration globally |
| `GPU_DEVICE_ID` | `GPU_DEVICE_ID` | `0` | CUDA device ID to use |
| `TENSORRT_ENABLED` | `TENSORRT_ENABLED` | `True` | Enable TensorRT optimization |
| `TENSORRT_CACHE_DIR` | `TENSORRT_CACHE_DIR` | `"./tensorrt_cache"` | Directory to store TensorRT engine cache |
| `PRECISION` | `PRECISION` | `"fp16"` | Precision for tensor operations (`"fp32"`, `"fp16"`, or `"int8"`) |
| `EMBEDDING_BATCH_SIZE` | `EMBEDDING_BATCH_SIZE` | `64` | Default batch size for embedding generation |
| `MAX_BATCH_SIZE` | `MAX_BATCH_SIZE` | `128` | Maximum batch size limit |
| `AUTO_MIXED_PRECISION` | `AUTO_MIXED_PRECISION` | `True` | Enable automatic mixed precision |

**Example configuration in config.py:**
```python
GPU_ACCELERATION_ENABLED = os.getenv("GPU_ACCELERATION_ENABLED", "True").lower() in ("true", "1", "yes")
GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))
TENSORRT_ENABLED = os.getenv("TENSORRT_ENABLED", "True").lower() in ("true", "1", "yes")
TENSORRT_CACHE_DIR = os.getenv("TENSORRT_CACHE_DIR", "./tensorrt_cache")
PRECISION = os.getenv("PRECISION", "fp16")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "128"))
AUTO_MIXED_PRECISION = os.getenv("AUTO_MIXED_PRECISION", "True").lower() in ("true", "1", "yes")
```

### Embedding Models

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `DEFAULT_EMBEDDING_MODEL` | `DEFAULT_EMBEDDING_MODEL` | `"all-MiniLM-L6-v2"` | Default embedding model to use |
| `EMBEDDING_DIMENSION` | `EMBEDDING_DIMENSION` | `384` | Dimension of embeddings |
| `USE_INTERNAL_EMBEDDINGS` | `USE_INTERNAL_EMBEDDINGS` | `False` | Use SAP HANA's internal embedding function |
| `INTERNAL_EMBEDDING_MODEL_ID` | `INTERNAL_EMBEDDING_MODEL_ID` | `"SAP_NEB.20240715"` | Model ID for internal embeddings |
| `EMBEDDING_MODELS` | N/A | See below | Dictionary of available embedding models |

**Available embedding models configuration:**
```python
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "path": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "General purpose embedding model (384 dimensions)"
    },
    "all-mpnet-base-v2": {
        "path": "sentence-transformers/all-mpnet-base-v2",
        "dimension": 768,
        "description": "High-quality embedding model (768 dimensions)"
    },
    "multi-qa-MiniLM-L6": {
        "path": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "dimension": 384,
        "description": "Question-answering optimized model (384 dimensions)"
    },
    "hkunlp/instructor-large": {
        "path": "hkunlp/instructor-large",
        "dimension": 768,
        "description": "Instruction-tuned embedding model (768 dimensions)"
    }
}
```

### Caching

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `CACHE_ENABLED` | `CACHE_ENABLED` | `True` | Enable result caching |
| `CACHE_TYPE` | `CACHE_TYPE` | `"memory"` | Cache type (`"memory"`, `"redis"`, or `"disk"`) |
| `CACHE_TTL` | `CACHE_TTL` | `3600` | Time-to-live for cache entries (seconds) |
| `CACHE_MAX_SIZE` | `CACHE_MAX_SIZE` | `1000` | Maximum number of items in the cache |
| `REDIS_URL` | `REDIS_URL` | `"redis://localhost:6379"` | Redis URL for Redis cache |
| `DISK_CACHE_DIR` | `DISK_CACHE_DIR` | `"./cache"` | Directory for disk cache |

**Example configuration in config.py:**
```python
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "True").lower() in ("true", "1", "yes")
CACHE_TYPE = os.getenv("CACHE_TYPE", "memory")
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
CACHE_MAX_SIZE = int(os.getenv("CACHE_MAX_SIZE", "1000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DISK_CACHE_DIR = os.getenv("DISK_CACHE_DIR", "./cache")
```

### Vector Store Configuration

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `VECTOR_TABLE_SCHEMA` | `VECTOR_TABLE_SCHEMA` | See below | Schema definition for vector table |
| `DEFAULT_VECTOR_TYPE` | `DEFAULT_VECTOR_TYPE` | `"REAL_VECTOR"` | Vector type (`"REAL_VECTOR"` or `"HALF_VECTOR"`) |
| `DEFAULT_TEXT_TYPE` | `DEFAULT_TEXT_TYPE` | `"TEXT"` | Text column type (`"TEXT"` or `"NVARCHAR"`) |
| `CREATE_INDEX_ON_ADD` | `CREATE_INDEX_ON_ADD` | `False` | Automatically create HNSW index when adding documents |
| `DEFAULT_INDEX_PARAMS` | N/A | See below | Default parameters for HNSW index |

**Vector table schema configuration:**
```python
VECTOR_TABLE_SCHEMA = {
    "id": "VARCHAR(256)",
    "content": "TEXT",
    "embedding": "REAL_VECTOR(384)",
    "metadata": "NCLOB",
    "source": "VARCHAR(1024)",
    "created_at": "TIMESTAMP"
}
```

**Default HNSW index parameters:**
```python
DEFAULT_INDEX_PARAMS = {
    "index_type": "HNSW",
    "distance_function": "COSINE_DISTANCE",
    "dimension": 384,
    "ef_construction": 128,
    "m": 16,
    "max_connections": 64,
    "ef_search": 64
}
```

### API and Server Configuration

| Option | Environment Variable | Default | Description |
|--------|---------------------|---------|-------------|
| `API_PORT` | `API_PORT` | `8000` | Port for API server |
| `API_HOST` | `API_HOST` | `"0.0.0.0"` | Host address to bind API server |
| `API_WORKERS` | `API_WORKERS` | `4` | Number of worker processes |
| `API_TIMEOUT` | `API_TIMEOUT` | `120` | Request timeout (seconds) |
| `DEBUG_MODE` | `DEBUG_MODE` | `False` | Enable debug mode |
| `ALLOWED_ORIGINS` | `ALLOWED_ORIGINS` | `["*"]` | CORS allowed origins |
| `LOG_LEVEL` | `LOG_LEVEL` | `"INFO"` | Logging level |
| `LOG_FORMAT` | `LOG_FORMAT` | `"%(asctime)s - %(name)s - %(levelname)s - %(message)s"` | Log format string |

**Example configuration in config.py:**
```python
API_PORT = int(os.getenv("API_PORT", "8000"))
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_WORKERS = int(os.getenv("API_WORKERS", "4"))
API_TIMEOUT = int(os.getenv("API_TIMEOUT", "120"))
DEBUG_MODE = os.getenv("DEBUG_MODE", "False").lower() in ("true", "1", "yes")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
```

## Advanced Configuration

### Dynamic Batch Processing

The dynamic batch processor automatically adjusts batch sizes based on available GPU memory and processing performance. You can tune its behavior with these settings:

```python
# Dynamic batch processing configuration
DYNAMIC_BATCH_ENABLED = True
DYNAMIC_BATCH_INITIAL_SIZE = 64
DYNAMIC_BATCH_MIN_SIZE = 1
DYNAMIC_BATCH_MAX_SIZE = 256
DYNAMIC_BATCH_GROWTH_FACTOR = 1.5
DYNAMIC_BATCH_SHRINK_FACTOR = 0.5
DYNAMIC_BATCH_MEMORY_SAFETY_FACTOR = 0.8
DYNAMIC_BATCH_GROWTH_THRESHOLD = 0.85  # Grow batch if GPU utilization > 85%
DYNAMIC_BATCH_OOM_RECOVERY_ENABLED = True
```

### Telemetry and Observability

Configure the telemetry and observability features:

```python
# Telemetry configuration
TELEMETRY_ENABLED = True
OPENTELEMETRY_ENABLED = False  # Set to True to enable OpenTelemetry integration
METRICS_COLLECTION_INTERVAL = 15  # seconds
PROFILING_ENABLED = False  # Enable detailed performance profiling
HEALTH_CHECK_INTERVAL = 60  # seconds
PERFORMANCE_MONITORING_ENABLED = True
```

### Authentication and Security

Configure authentication and security options:

```python
# Authentication configuration
AUTH_ENABLED = True
AUTH_TYPE = "api_key"  # Options: "api_key", "oauth", "none"
API_KEY_HEADER = "X-API-Key"
JWT_SECRET = os.getenv("JWT_SECRET", "")
JWT_ALGORITHM = "HS256"
TOKEN_EXPIRATION_MINUTES = 60
OAUTH_CLIENT_ID = os.getenv("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.getenv("OAUTH_CLIENT_SECRET", "")
OAUTH_TOKEN_URL = os.getenv("OAUTH_TOKEN_URL", "")
```

## Docker Deployment Configuration

For Docker deployments, environment variables can be configured in the `docker-compose.yml` file:

```yaml
version: '3'
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - HANA_HOST=${HANA_HOST}
      - HANA_PORT=${HANA_PORT}
      - HANA_USER=${HANA_USER}
      - HANA_PASSWORD=${HANA_PASSWORD}
      - HANA_SCHEMA=${HANA_SCHEMA}
      - GPU_ACCELERATION_ENABLED=true
      - TENSORRT_ENABLED=true
      - PRECISION=fp16
      - EMBEDDING_BATCH_SIZE=64
      - CACHE_ENABLED=true
      - API_WORKERS=4
      - LOG_LEVEL=INFO
    volumes:
      - ./tensorrt_cache:/app/tensorrt_cache
      - ./cache:/app/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Kubernetes Deployment Configuration

For Kubernetes deployments, configuration is managed through ConfigMaps and Secrets:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: hana-langchain-config
data:
  HANA_SCHEMA: "ML_DATA"
  HANA_TABLE: "VECTOR_STORE"
  GPU_ACCELERATION_ENABLED: "true"
  TENSORRT_ENABLED: "true"
  PRECISION: "fp16"
  EMBEDDING_BATCH_SIZE: "64"
  API_WORKERS: "4"
  LOG_LEVEL: "INFO"
  CACHE_ENABLED: "true"
  CACHE_TYPE: "redis"
  REDIS_URL: "redis://redis:6379"
---
apiVersion: v1
kind: Secret
metadata:
  name: hana-langchain-secrets
type: Opaque
data:
  HANA_HOST: <base64-encoded-value>
  HANA_PORT: <base64-encoded-value>
  HANA_USER: <base64-encoded-value>
  HANA_PASSWORD: <base64-encoded-value>
  JWT_SECRET: <base64-encoded-value>
```

## Recommended Configurations for Different Use Cases

### Production-Ready Configuration

```python
from langchain_hana.vectorstores import HanaVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = HanaVectorStore.create_connection(
    host="your-hana-hostname.hanacloud.ondemand.com",
    port=443,
    user="your_user",
    password="your_password",
    schema="ML_DATA",
    table="PRODUCTION_EMBEDDINGS",
    embedding=OpenAIEmbeddings(),
    vector_type="REAL_VECTOR",
    distance_function="COSINE_DISTANCE",
    specific_metadata_columns=["category", "source", "timestamp"]
)

# Create HNSW index for faster searches
vectorstore.create_hnsw_index(
    m=64,  # Balanced for accuracy and performance
    ef_construction=200,  # Higher value = better accuracy, slower indexing
    ef_search=100  # Higher value = better accuracy, slower queries
)
```

### Memory-Optimized Configuration

```python
from langchain_hana.vectorstores import HanaVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = HanaVectorStore.create_connection(
    host="your-hana-hostname.hanacloud.ondemand.com",
    port=443,
    user="your_user",
    password="your_password",
    schema="ML_DATA",
    table="MEMORY_OPTIMIZED_EMBEDDINGS",
    embedding=OpenAIEmbeddings(dimensions=384),  # Smaller dimension embeddings
    vector_type="HALF_VECTOR",  # Uses half the storage (if supported)
    vector_dimension=384,  # Fixed length for memory optimization
    distance_function="COSINE_DISTANCE"
)

# Create HNSW index optimized for memory efficiency
vectorstore.create_hnsw_index(
    m=32,  # Lower M value uses less memory
    ef_construction=100,
    ef_search=50
)
```

### Performance-Optimized Configuration

```python
from langchain_hana.vectorstores import HanaVectorStore
from langchain_hana.embeddings import HanaEmbeddings

# Using HANA's internal embeddings for maximum performance
vectorstore = HanaVectorStore.create_connection(
    host="your-hana-hostname.hanacloud.ondemand.com",
    port=443,
    user="your_user",
    password="your_password",
    schema="ML_DATA",
    table="PERFORMANCE_OPTIMIZED_EMBEDDINGS",
    embedding=HanaEmbeddings(
        host="your-hana-hostname.hanacloud.ondemand.com",
        port=443,
        user="your_user",
        password="your_password",
        use_internal=True,
        internal_embedding_model_id="SAP_NEB.20240715"
    ),
    distance_function="COSINE_DISTANCE",
    specific_metadata_columns=["category", "source", "timestamp"]
)

# Create high-performance HNSW index
vectorstore.create_hnsw_index(
    m=100,  # Higher M value improves search performance
    ef_construction=300,
    ef_search=200
)
```

## Configuration File Templates

### Sample .env File

```
# Database Connection
HANA_HOST=your-hana-hostname.hanacloud.ondemand.com
HANA_PORT=443
HANA_USER=your-username
HANA_PASSWORD=your-password
HANA_SCHEMA=ML_DATA
HANA_TABLE=VECTOR_STORE

# GPU Configuration
GPU_ACCELERATION_ENABLED=true
GPU_DEVICE_ID=0
TENSORRT_ENABLED=true
PRECISION=fp16
EMBEDDING_BATCH_SIZE=64

# API Configuration
API_PORT=8000
API_HOST=0.0.0.0
API_WORKERS=4
DEBUG_MODE=false
LOG_LEVEL=INFO

# Caching
CACHE_ENABLED=true
CACHE_TYPE=memory
CACHE_TTL=3600

# Auth
AUTH_ENABLED=false
```

### Sample Vercel Configuration (vercel.json)

```json
{
  "env": {
    "HANA_HOST": "your-hana-hostname.hanacloud.ondemand.com",
    "HANA_PORT": "443",
    "HANA_USER": "your-username",
    "HANA_PASSWORD": "@hana-password",
    "HANA_SCHEMA": "ML_DATA",
    "GPU_ACCELERATION_ENABLED": "true",
    "TENSORRT_ENABLED": "true",
    "EMBEDDING_BATCH_SIZE": "32",
    "LOG_LEVEL": "INFO",
    "CACHE_ENABLED": "true",
    "CACHE_TYPE": "memory"
  },
  "builds": [
    {
      "src": "api/vercel_app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "api/vercel_app.py"
    }
  ]
}
```

## Best Practices

### Production Deployment

For production deployments, we recommend:

1. **Containerization**: Use Docker with GPU support for consistent environments
2. **Connection Pooling**: Configure appropriate pool sizes based on your workload
3. **Caching**: Enable result caching to improve performance
4. **TensorRT Optimization**: Enable TensorRT for maximum GPU performance
5. **Health Monitoring**: Implement health checks and monitoring
6. **Security**: Enable authentication and use environment secrets

### Memory Optimization

To optimize memory usage:

1. Set appropriate batch sizes based on your GPU memory
2. Use `HALF_VECTOR` type for memory-efficient storage
3. Enable dynamic batch sizing with conservative growth factors
4. Consider using `fp16` precision for a good balance of performance and memory usage
5. Monitor memory usage and adjust parameters as needed

### Performance Tuning

For optimal performance:

1. Test different embedding models to find the right balance of quality and speed
2. Tune HNSW index parameters based on your dataset size and query patterns
3. Enable query result caching for frequently accessed data
4. Configure appropriate connection pool settings
5. Use the batch processing API for large document sets
6. Monitor and profile to identify bottlenecks

### Understanding Vector Column Types

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

### Understanding Distance Strategies

* **COSINE** (`distance_function="COSINE_DISTANCE"`): 
  * Measures similarity based on the angle between vectors
  * Range: 0 to 1 (higher is more similar)
  * Best for: Most text embedding models, semantic search
  * Recommended for: General-purpose applications

* **EUCLIDEAN** (`distance_function="EUCLIDEAN_DISTANCE"`):
  * Measures similarity based on straight-line distance between vectors
  * Range: 0 to infinity (lower is more similar)
  * Best for: Some machine learning applications
  * Recommended for: Specialized use cases

## Error Message Reference

Common error messages and their solutions:

| Error Message | Likely Cause | Solution |
|---------------|--------------|----------|
| "Vector type 'X' is not available on your SAP HANA Cloud instance" | Your HANA instance doesn't support the vector type | Use 'REAL_VECTOR' or upgrade your HANA instance |
| "Internal embedding model ID cannot be None" | Missing model ID for internal embeddings | Specify a valid model ID or use external embeddings |
| "Column X does not exist" | Table structure mismatch | Check table structure and column names |
| "Unsupported distance function" | Invalid distance function value | Use "COSINE_DISTANCE" or "EUCLIDEAN_DISTANCE" |
| "Invalid filter operator" | Incorrect filter syntax | Check filter syntax and use supported operators |
| "CUDA out of memory" | Batch size too large | Reduce batch size or enable dynamic batch sizing |
| "Connection to the server has been lost" | Network connectivity issue | Check network connectivity and retry |
| "Table X not found" | Table doesn't exist | Create the table or check schema name |