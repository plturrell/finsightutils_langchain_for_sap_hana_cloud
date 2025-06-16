# API Documentation for LangChain SAP HANA Integration

This project includes comprehensive API documentation through OpenAPI and Swagger UI. This document explains how to access and use the API documentation.

## Accessing the API Documentation

The API documentation is available at the following endpoints:

- **Swagger UI**: `/docs` - Interactive API documentation that allows you to try out the API endpoints directly from your browser
- **ReDoc**: `/redoc` - Alternative API documentation with a different UI

For example, if your backend is running at `http://localhost:8000`, the Swagger UI would be at `http://localhost:8000/docs`.

## Documentation Features

### Interactive Testing

The Swagger UI allows you to:

1. Browse all available API endpoints
2. View request and response schemas
3. Try out endpoints directly in the browser
4. See detailed descriptions and examples for each endpoint
5. Understand the authentication requirements

### API Categories

The endpoints are organized into the following categories:

- **General**: Basic information about the API
- **Vector Store**: Operations for managing vector embeddings (add, delete)
- **Query**: Vector similarity search operations (standard, MMR, vector-based)
- **Health**: Health check and monitoring endpoints
- **GPU**: GPU information and acceleration settings
- **Benchmarks**: Performance benchmarking endpoints
- **Developer**: Developer-specific operations and debugging

## Using the API Documentation for Testing

1. Navigate to the Swagger UI (`/docs`)
2. Expand an endpoint section (e.g., "Vector Store")
3. Click on an endpoint (e.g., POST `/texts`)
4. Click "Try it out"
5. Fill in the required parameters
6. Click "Execute"
7. View the response

## Examples

Here are some common use cases with example requests:

### Adding Texts to the Vector Store

```json
POST /texts
{
  "texts": [
    "SAP HANA Cloud is an in-memory database platform.",
    "Vector stores enable semantic search capabilities.",
    "GPU acceleration improves embedding generation performance."
  ],
  "metadatas": [
    {"source": "documentation", "category": "database"},
    {"source": "documentation", "category": "vector-search"},
    {"source": "documentation", "category": "performance"}
  ]
}
```

### Performing a Similarity Search

```json
POST /query
{
  "query": "How does GPU acceleration improve performance?",
  "k": 3
}
```

### Using MMR Search for Diverse Results

```json
POST /query/mmr
{
  "query": "HANA Cloud features",
  "k": 5,
  "fetch_k": 20,
  "lambda_mult": 0.5
}
```

## Authentication

Most endpoints require authentication. The API supports:

- API Key Authentication: Pass the API key in the `X-API-Key` header
- JWT Token Authentication: Pass a JWT token in the `Authorization` header with the format `Bearer <token>`

## Response Formats

All endpoints return responses in a consistent format:

### Success Responses

```json
{
  "success": true,
  "message": "Operation succeeded",
  "data": { ... }
}
```

### Error Responses

```json
{
  "detail": {
    "error": "Error type",
    "message": "Error description",
    "context": {
      "operation": "Operation being performed",
      "suggestion": "How to fix the issue",
      "additional_info": { ... }
    }
  }
}
```

## Generating API Clients

You can use the OpenAPI specification to generate client libraries for different programming languages:

1. Download the OpenAPI specification from `/openapi.json`
2. Use tools like [OpenAPI Generator](https://openapi-generator.tech/) to generate client libraries
3. Alternatively, use HTTP client libraries with OpenAPI support like Swagger Client

## API Versioning

The API version is included in:
- The response headers (`X-API-Version`)
- The OpenAPI specification metadata
- The root endpoint response

## Rate Limiting and Quotas

The API implements rate limiting to ensure fair usage:
- Requests are limited to 100 per minute per API key
- Larger batch requests count as multiple requests
- Rate limit headers are included in responses