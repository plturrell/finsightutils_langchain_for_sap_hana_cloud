# SAP HANA Cloud LangChain Integration API Documentation

This documentation covers the REST API for the SAP HANA Cloud LangChain integration. The API provides endpoints for vector similarity search, document embedding, and knowledge graph operations.

## Accessing the API Documentation

The API documentation is available at the following endpoints:

- **Swagger UI**: `/docs` - Interactive API documentation that allows you to try out the API endpoints directly from your browser
- **ReDoc**: `/redoc` - Alternative API documentation with a different UI
- **OpenAPI Specification**: `/openapi.json` - Raw OpenAPI specification for generating clients

For example, if your backend is running at `http://localhost:8000`, the Swagger UI would be at `http://localhost:8000/docs`.

## Base URLs

- **Production**: `https://api.example.com/api/v1`
- **Development**: `http://localhost:8000/api/v1`

## Authentication

The API supports two authentication methods:

1. **API Key Authentication**:
   ```
   X-API-Key: your_api_key
   ```

2. **OAuth2 Authentication**:
   First obtain a token from the OAuth server, then include it in your requests:
   ```
   Authorization: Bearer your_access_token
   ```

## Endpoints

### Vector Store Operations

#### Similarity Search

Searches for documents similar to the provided query using vector similarity.

**URL**: `/vector-store/similarity-search`  
**Method**: `POST`  
**Content-Type**: `application/json`  

**Request Body**:
```json
{
  "query": "How does SAP HANA Cloud support vector search?",
  "k": 5,
  "filter": {
    "source": "documentation"
  },
  "use_tensorrt": true
}
```

**Parameters**:
- `query` (string, required): The text query to search for
- `k` (integer, optional): Number of results to return (default: 4)
- `filter` (object, optional): Metadata filter to apply
- `use_tensorrt` (boolean, optional): Whether to use TensorRT acceleration (default: true)

**Response**:
```json
{
  "results": [
    {
      "document": {
        "page_content": "SAP HANA Cloud supports vector search through the REAL_VECTOR data type...",
        "metadata": {
          "source": "documentation",
          "title": "Vector Search Guide",
          "page": 5
        }
      },
      "score": 0.923
    },
    // More results...
  ],
  "execution_stats": {
    "total_time_ms": 125.3,
    "embedding_time_ms": 45.7,
    "search_time_ms": 79.6,
    "batch_size": 1,
    "gpu_utilized": true
  }
}
```

#### MMR Search (Maximum Marginal Relevance)

Search for relevant and diverse documents using MMR, which balances similarity to the query with diversity among results.

**URL**: `/vector-store/mmr-search`  
**Method**: `POST`  
**Content-Type**: `application/json`  

**Request Body**:
```json
{
  "query": "How does SAP HANA Cloud support vector search?",
  "k": 5,
  "fetch_k": 20,
  "lambda_mult": 0.5,
  "filter": {
    "source": "documentation"
  },
  "use_tensorrt": true
}
```

**Parameters**:
- `query` (string, required): The text query to search for
- `k` (integer, optional): Number of results to return (default: 4)
- `fetch_k` (integer, optional): Number of documents to fetch before reranking (default: 20)
- `lambda_mult` (float, optional): Balance between relevance and diversity, 0 = max diversity, 1 = max relevance (default: 0.5)
- `filter` (object, optional): Metadata filter to apply
- `use_tensorrt` (boolean, optional): Whether to use TensorRT acceleration (default: true)

**Response**: Same as Similarity Search

#### Add Documents

Add documents to the vector store by generating embeddings and storing them along with the document content and metadata.

**URL**: `/vector-store/add-documents`  
**Method**: `POST`  
**Content-Type**: `application/json`  

**Request Body**:
```json
{
  "documents": [
    {
      "page_content": "SAP HANA Cloud supports vector search through the REAL_VECTOR data type...",
      "metadata": {
        "source": "documentation",
        "title": "Vector Search Guide",
        "page": 5
      }
    },
    // More documents...
  ],
  "batch_size": 64,
  "use_tensorrt": true
}
```

**Parameters**:
- `documents` (array, required): Array of documents to add
- `batch_size` (integer, optional): Batch size for processing (default: 64)
- `use_tensorrt` (boolean, optional): Whether to use TensorRT acceleration (default: true)

**Response**:
```json
{
  "document_count": 10,
  "execution_stats": {
    "total_time_ms": 1253.7,
    "embedding_time_ms": 982.3,
    "insertion_time_ms": 271.4,
    "batch_size": 64,
    "gpu_utilized": true
  }
}
```

#### Delete Documents

Delete documents from the vector store based on metadata filters.

**URL**: `/vector-store/delete-documents`  
**Method**: `POST`  
**Content-Type**: `application/json`  

**Request Body**:
```json
{
  "filter": {
    "source": "documentation",
    "title": "Vector Search Guide"
  }
}
```

**Parameters**:
- `filter` (object, required): Metadata filter to select documents to delete

**Response**:
```json
{
  "deleted_count": 5
}
```

### Embedding Operations

#### Create Embeddings

Generate embeddings for the provided texts using the specified model.

**URL**: `/embeddings/create`  
**Method**: `POST`  
**Content-Type**: `application/json`  

**Request Body**:
```json
{
  "texts": [
    "SAP HANA Cloud supports vector search",
    "Vector search is useful for semantic similarity"
  ],
  "model": "all-MiniLM-L6-v2",
  "batch_size": 64,
  "use_tensorrt": true
}
```

**Parameters**:
- `texts` (array, required): Array of texts to generate embeddings for
- `model` (string, optional): Name of the embedding model to use (default: "all-MiniLM-L6-v2")
- `batch_size` (integer, optional): Batch size for processing (default: 64)
- `use_tensorrt` (boolean, optional): Whether to use TensorRT acceleration (default: true)

**Response**:
```json
{
  "embeddings": [
    [0.123, 0.456, ...],  // First embedding vector
    [0.789, 0.012, ...]   // Second embedding vector
  ],
  "execution_stats": {
    "total_time_ms": 125.3,
    "embeddings_per_second": 15.96,
    "batch_size": 2,
    "gpu_utilized": true
  }
}
```

### Knowledge Graph Operations

#### Execute SPARQL Query

Execute a SPARQL query against the RDF graph stored in SAP HANA Cloud.

**URL**: `/knowledge-graph/sparql`  
**Method**: `POST`  
**Content-Type**: `application/json`  

**Request Body**:
```json
{
  "query": "PREFIX : <http://example.org/>\nSELECT ?product ?name\nWHERE {\n  ?product a :Product .\n  ?product :name ?name .\n}\nLIMIT 10"
}
```

**Parameters**:
- `query` (string, required): SPARQL query to execute

**Response**:
```json
{
  "results": [
    {
      "product": "http://example.org/product1",
      "name": "Product 1"
    },
    {
      "product": "http://example.org/product2",
      "name": "Product 2"
    }
    // More results...
  ],
  "execution_stats": {
    "total_time_ms": 87.2
  }
}
```

### Health and Status

#### Get Service Health

Check the health status of the service, including database connectivity and GPU availability.

**URL**: `/health`  
**Method**: `GET`  

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "database": {
    "status": "connected",
    "message": "Connected to SAP HANA Cloud (version 4.00.000.00)"
  },
  "gpu": {
    "status": "available",
    "device": "NVIDIA T4",
    "message": "TensorRT engines available"
  },
  "uptime": 3600.5
}
```

## Error Handling

All API endpoints return standard error responses in the following format:

```json
{
  "status": 404,
  "statusText": "Not Found",
  "detail": {
    "message": "The requested data table could not be found.",
    "operation": "similarity_search",
    "suggestions": [
      "Verify table name and schema",
      "Check if the table exists in the database"
    ],
    "common_issues": [
      "Table was deleted or renamed",
      "Table is in a different schema"
    ],
    "original_error": "table 'VECTOR_STORE' not found"
  }
}
```

The API uses standard HTTP status codes:
- `200 OK`: Request succeeded
- `400 Bad Request`: Invalid input
- `401 Unauthorized`: Authentication failed
- `403 Forbidden`: Permission denied
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

## Rate Limiting

The API enforces rate limits to ensure fair usage:

- 100 requests per minute per API key
- 1000 requests per hour per API key

Rate limit headers are included in the response:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1620000000
```

## Configuration Options

The API behavior can be configured through several options:

### GPU Acceleration

GPU acceleration can be enabled/disabled globally or per request:

1. **Global Configuration** (in `config.py`):
   ```python
   GPU_ACCELERATION_ENABLED = True
   DEFAULT_BATCH_SIZE = 64
   DEFAULT_MODEL = "all-MiniLM-L6-v2"
   ```

2. **Per-Request Configuration**:
   ```json
   {
     "use_tensorrt": true,
     "batch_size": 32
   }
   ```

### Connection Management

Database connection pooling can be configured:

```python
DB_POOL_SIZE = 10
DB_POOL_TIMEOUT = 30
DB_POOL_RECYCLE = 3600
```

### Retry Configuration

Automatic retry for transient errors:

```python
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_FACTOR = 1.5
RETRY_STATUS_CODES = [500, 502, 503, 504]
```

### Caching

Query result caching:

```python
CACHE_ENABLED = True
CACHE_TTL = 3600  # seconds
CACHE_MAX_ITEMS = 1000
```

## Code Examples

### Python

```python
import requests

# Authentication
headers = {
    "X-API-Key": "YOUR_API_KEY"
}

# Similarity Search
response = requests.post(
    "https://api.example.com/api/v1/vector-store/similarity-search",
    headers=headers,
    json={
        "query": "How does SAP HANA Cloud support vector search?",
        "k": 5,
        "filter": {"source": "documentation"}
    }
)

results = response.json()["results"]
for result in results:
    print(f"Score: {result['score']}")
    print(f"Content: {result['document']['page_content']}")
    print(f"Metadata: {result['document']['metadata']}")
    print("---")

# Adding documents with batch processing
documents = [
    {
        "page_content": f"Document {i} content",
        "metadata": {"source": "api_example", "id": i}
    }
    for i in range(1, 101)
]

response = requests.post(
    "https://api.example.com/api/v1/vector-store/add-documents",
    headers=headers,
    json={
        "documents": documents,
        "batch_size": 32,
        "use_tensorrt": True
    }
)

print(f"Added {response.json()['document_count']} documents")
print(f"Total time: {response.json()['execution_stats']['total_time_ms']} ms")
```

### JavaScript

```javascript
// Similarity Search
const searchDocuments = async (query, k = 5, filter = {}) => {
  try {
    const response = await fetch('https://api.example.com/api/v1/vector-store/similarity-search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'YOUR_API_KEY'
      },
      body: JSON.stringify({
        query,
        k,
        filter,
        use_tensorrt: true
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail.message);
    }
    
    const data = await response.json();
    return data.results;
  } catch (error) {
    console.error('Search error:', error);
    throw error;
  }
};

// Example usage
searchDocuments('How does SAP HANA Cloud support vector search?', 5, { source: 'documentation' })
  .then(results => {
    results.forEach(result => {
      console.log(`Score: ${result.score}`);
      console.log(`Content: ${result.document.page_content}`);
      console.log(`Metadata:`, result.document.metadata);
      console.log('---');
    });
  })
  .catch(error => {
    console.error('Error:', error.message);
  });

// MMR search with dynamic batching
const mmrSearch = async (query, k = 5, fetchK = 20, lambdaMult = 0.5, filter = {}) => {
  try {
    const response = await fetch('https://api.example.com/api/v1/vector-store/mmr-search', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'YOUR_API_KEY'
      },
      body: JSON.stringify({
        query,
        k,
        fetch_k: fetchK,
        lambda_mult: lambdaMult,
        filter,
        use_tensorrt: true
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail.message);
    }
    
    return await response.json();
  } catch (error) {
    console.error('MMR search error:', error);
    throw error;
  }
};
```

## SDK Libraries

For easier integration, we provide client SDKs for popular programming languages:

- Python: [GitHub - SAP/langchain-integration-for-sap-hana-cloud-python](https://github.com/SAP/langchain-integration-for-sap-hana-cloud)
- JavaScript: [GitHub - SAP/langchain-integration-for-sap-hana-cloud-js](https://github.com/SAP/langchain-integration-for-sap-hana-cloud-js)

## Troubleshooting

### Common Error Codes

| Status Code | Error Type | Description | Solution |
|-------------|------------|-------------|----------|
| 400 | invalid_request | Invalid request format | Check request body against API documentation |
| 401 | unauthorized | Invalid or missing API key | Verify your API key |
| 403 | forbidden | Insufficient permissions | Request access to required resources |
| 404 | not_found | Resource not found | Check resource identifier |
| 413 | payload_too_large | Request payload too large | Reduce batch size or split into multiple requests |
| 429 | too_many_requests | Rate limit exceeded | Implement backoff or reduce request frequency |
| 500 | internal_error | Server error | Retry with exponential backoff |
| 503 | service_unavailable | Service temporarily unavailable | Retry with exponential backoff |

### Debugging Tips

1. **Enable detailed error messages**:
   Add `debug=true` query parameter to get more detailed error information:
   ```
   GET /health?debug=true
   ```

2. **Check response headers**:
   Look for diagnostic headers in the response:
   ```
   X-Request-ID: unique-request-identifier
   X-Process-Time: processing-time-ms
   ```

3. **Use the `/debug` endpoint**:
   This endpoint provides detailed system diagnostics (requires admin API key):
   ```
   GET /debug
   ```

## Support

For issues, questions, or feedback, please [create an issue](https://github.com/SAP/langchain-integration-for-sap-hana-cloud/issues) on our GitHub repository.