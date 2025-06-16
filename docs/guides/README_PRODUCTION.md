# Production-Ready LangChain Integration for SAP HANA Cloud

This guide focuses on using the LangChain SAP HANA Cloud integration in production environments, with a focus on reliability, performance, and security.

## Production Features

The integration includes the following production-ready features:

### Connection Management

- **Connection Pooling**: Efficiently manage database connections for high-throughput applications
- **Automatic Reconnection**: Transparently handle connection failures with configurable retry policies
- **Connection Health Checks**: Regularly validate connections to detect issues before they affect your application
- **Connection Lifecycle Management**: Control connection age and reuse

### Performance Optimization

- **HNSW Indexing**: High-performance vector search with Hierarchical Navigable Small World algorithm
- **Embedding Caching**: Avoid redundant embedding computations with configurable caching
- **Batch Operations**: Efficiently process multiple documents in a single operation
- **GPU Acceleration**: Optional TensorRT support for high-throughput embedding generation

### Reliability

- **Retry Logic**: Automatic retries with exponential backoff for transient errors
- **Transaction Management**: Proper handling of database transactions
- **Error Handling**: Comprehensive error reporting and recovery mechanisms
- **Input Validation**: Extensive validation of inputs to prevent issues

### Security

- **Secure Connections**: Encrypted connections to SAP HANA Cloud
- **SQL Injection Prevention**: Parameterized queries and identifier sanitization
- **Flexible Authentication**: Support for various authentication methods

## Quick Start for Production

```python
# Import required modules
from langchain_core.embeddings import HuggingFaceEmbeddings
from langchain_hana.vectorstore import HanaVectorStore
from langchain_hana.connection import create_connection_pool, close_all_connection_pools
from langchain_hana.embeddings import HanaEmbeddingsCache

# Create embedding model with caching
base_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
cached_embeddings = HanaEmbeddingsCache(
    base_embeddings=base_embeddings,
    ttl_seconds=3600,
    max_size=10000,
    persist_path="/path/to/cache.pkl"
)

# Create connection pool
create_connection_pool(
    pool_name="production_pool",
    min_connections=5,
    max_connections=20,
    host="your-hana-host.hanacloud.ondemand.com",
    port=443,
    user="your-username",
    password="your-password"
)

# Create vector store with production settings
vector_store = HanaVectorStore(
    embedding=cached_embeddings,
    use_connection_pool=True,
    connection_pool_name="production_pool",
    table_name="PRODUCTION_VECTORS",
    create_table=True,
    create_hnsw_index=True,
    retry_attempts=3
)

# Use the vector store
docs = ["Document 1", "Document 2", "Document 3"]
metadatas = [{"source": "wiki"}, {"source": "web"}, {"source": "book"}]
vector_store.add_texts(docs, metadatas)

results = vector_store.similarity_search("your query", k=3)

# Don't forget to clean up when shutting down
close_all_connection_pools()
```

## Production Configuration Recommendations

### Connection Pool Sizing

| Workload | Min Connections | Max Connections |
|----------|----------------|-----------------|
| Low volume (< 10 req/sec) | 2-3 | 5-10 |
| Medium volume (10-50 req/sec) | 5-10 | 20-30 |
| High volume (50+ req/sec) | 10-20 | 30-50+ |

### HNSW Index Parameters

| Dataset Size | M | ef_construction | ef_search |
|--------------|---|-----------------|-----------|
| Small (< 10K vectors) | 16 | 64 | 32 |
| Medium (10K-100K vectors) | 16 | 128 | 64 |
| Large (100K-1M vectors) | 24 | 256 | 128 |
| Very Large (1M+ vectors) | 32 | 512 | 256 |

### Retry Configuration

| Environment | Retry Attempts | Backoff Strategy |
|-------------|---------------|------------------|
| Development | 1-2 | Linear |
| Testing | 2-3 | Exponential |
| Production | 3-5 | Exponential |

## Monitoring

Key metrics to monitor:

1. **Query Latency**: Average time for similarity searches
2. **Connection Pool Utilization**: Number of active connections
3. **Error Rates**: Percentage of failed operations
4. **Cache Performance**: Hit rate for embedding cache
5. **Database Health**: Response time of the SAP HANA Cloud instance

## Error Handling

The integration provides detailed error information. Common errors and solutions:

| Error | Possible Causes | Solution |
|-------|----------------|----------|
| Connection failure | Network issues, invalid credentials | Check connectivity, verify credentials |
| Timeout | Database overload, network latency | Increase timeouts, optimize queries |
| Out of connections | Connection pool exhausted | Increase max_connections, check for leaks |
| SQL error | Invalid query, missing table | Check table permissions, validate schema |

## Security Best Practices

1. **Credential Management**:
   - Use environment variables or a secrets manager
   - Rotate credentials regularly
   - Use the minimum necessary permissions

2. **Network Security**:
   - Enable connection encryption
   - Use certificate validation in production
   - Implement proper network segmentation

3. **Data Protection**:
   - Be aware of what data is stored in the vector database
   - Implement data masking for sensitive information
   - Use role-based access control

## Health Check Implementation

```python
from langchain_hana.connection import test_connection, get_connection_pool

def health_check():
    """Comprehensive health check for the SAP HANA integration."""
    results = {
        "status": "healthy",
        "components": {}
    }
    
    # Check connection pool
    pool = get_connection_pool("production_pool")
    if not pool:
        results["status"] = "degraded"
        results["components"]["connection_pool"] = {
            "status": "unavailable",
            "message": "Connection pool not found"
        }
    else:
        # Get a connection from the pool
        try:
            conn = pool.get_connection()
            is_healthy, info = test_connection(conn)
            
            if is_healthy:
                results["components"]["database"] = {
                    "status": "healthy",
                    "version": info.get("version"),
                    "current_schema": info.get("current_schema")
                }
            else:
                results["status"] = "unhealthy"
                results["components"]["database"] = {
                    "status": "unhealthy",
                    "message": info.get("error")
                }
            
            # Return the connection to the pool
            pool.release_connection(conn)
            
        except Exception as e:
            results["status"] = "unhealthy"
            results["components"]["database"] = {
                "status": "unavailable",
                "message": str(e)
            }
    
    return results
```

## Production Deployment Example

### Docker Compose Setup

```yaml
version: '3'

services:
  vector-service:
    build: .
    environment:
      - HANA_HOST=${HANA_HOST}
      - HANA_PORT=${HANA_PORT}
      - HANA_USER=${HANA_USER}
      - HANA_PASSWORD=${HANA_PASSWORD}
      - POOL_MIN_CONNECTIONS=5
      - POOL_MAX_CONNECTIONS=20
      - TABLE_NAME=PRODUCTION_VECTORS
      - LOG_LEVEL=INFO
    ports:
      - "8080:8080"
    volumes:
      - embedding-cache:/app/cache
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  embedding-cache:
```

### Kubernetes Configuration

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vector-service
  labels:
    app: vector-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: vector-service
  template:
    metadata:
      labels:
        app: vector-service
    spec:
      containers:
      - name: vector-service
        image: vector-service:latest
        ports:
        - containerPort: 8080
        env:
        - name: HANA_HOST
          valueFrom:
            secretKeyRef:
              name: hana-credentials
              key: host
        - name: HANA_PORT
          valueFrom:
            secretKeyRef:
              name: hana-credentials
              key: port
        - name: HANA_USER
          valueFrom:
            secretKeyRef:
              name: hana-credentials
              key: username
        - name: HANA_PASSWORD
          valueFrom:
            secretKeyRef:
              name: hana-credentials
              key: password
        - name: POOL_MIN_CONNECTIONS
          value: "5"
        - name: POOL_MAX_CONNECTIONS
          value: "20"
        - name: TABLE_NAME
          value: "PRODUCTION_VECTORS"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "512Mi"
            cpu: "200m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        volumeMounts:
        - name: embedding-cache
          mountPath: /app/cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: embedding-cache
        persistentVolumeClaim:
          claimName: embedding-cache-pvc
```

## Advanced Configuration

### Query Optimization

For complex search patterns:

```python
# Filter based on metadata
results = vector_store.similarity_search(
    "advanced query",
    filter={
        "category": "database",
        "year": {"$gte": 2022},
        "$or": [
            {"tags": {"$contains": "cloud"}},
            {"tags": {"$contains": "vector"}}
        ]
    }
)

# Use MMR for diverse results
diverse_results = vector_store.max_marginal_relevance_search(
    "query requiring diversity",
    k=5,
    fetch_k=20,
    lambda_mult=0.5  # 0.0 = max diversity, 1.0 = max relevance
)
```

### Load Testing

Use the production example script to benchmark your setup:

```bash
python examples/langchain_hana_production.py \
  --host your-hana-host.hanacloud.ondemand.com \
  --port 443 \
  --user your-username \
  --password your-password \
  --pool-min 5 \
  --pool-max 20 \
  --test-concurrency 8 \
  --benchmark-time 60
```

## Conclusion

By utilizing the production features of the LangChain SAP HANA Cloud integration, you can build robust, high-performance applications that reliably scale to meet your needs. The integration is designed with production use cases in mind, providing the tools necessary for enterprise-grade deployments.

For detailed implementation guidance, refer to the [Production Guide](docs/production/production_guide.md) and explore the example production script at [examples/langchain_hana_production.py](examples/langchain_hana_production.py).