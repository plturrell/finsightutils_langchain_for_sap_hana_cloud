# API Testing Guide

This document provides instructions for testing the SAP HANA Cloud LangChain integration API locally without requiring a real SAP HANA Cloud connection.

## Overview

The API can be tested in two ways:

1. **Docker-based testing**: Using Docker to build and run the API container
2. **Local testing**: Running the API directly on your local machine

Both approaches use the test mode, which provides mock implementations of SAP HANA Cloud functionality.

## Prerequisites

- Python 3.9 or later (for local testing)
- Docker and Docker Compose (for Docker-based testing)
- curl or Postman (for testing API endpoints)

## Docker-based Testing

### 1. Build and Start the Docker Container

```bash
# Clone the repository if you haven't already
git clone https://github.com/yourusername/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud

# Build and start the container in test mode
docker-compose -f docker-compose.local.yml up -d
```

### 2. Run the Automated Test Script

The `test_api_endpoints.sh` script will test all API endpoints automatically:

```bash
./test_api_endpoints.sh
```

This script:
1. Builds and starts the Docker container
2. Tests all API endpoints
3. Prints the results
4. Stops the container

### 3. Manual Testing

You can also test the endpoints manually with curl:

```bash
# Health check
curl -X GET http://localhost:8000/health/ping

# Version information
curl -X GET http://localhost:8000/api/v1/version

# Generate embeddings
curl -X POST http://localhost:8000/api/v1/embeddings/generate \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This is a test document"], "model": "sentence-transformers/all-MiniLM-L6-v2"}'

# Vector search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "k": 4, "use_mmr": false}'
```

## Local Testing (Without Docker)

### 1. Set Up the Environment

```bash
# Clone the repository if you haven't already
git clone https://github.com/yourusername/langchain-integration-for-sap-hana-cloud.git
cd langchain-integration-for-sap-hana-cloud

# Run the setup script (this creates a virtual environment and installs dependencies)
./run_api_local.sh
```

### 2. Manual Testing

After the API is running, you can test it with curl or Postman as described in the Docker-based testing section.

## API Endpoints

The following endpoints are available for testing:

### Health Endpoints

- `GET /health/ping`: Basic health check
- `GET /health/complete`: Comprehensive health check
- `GET /health/database`: Database health check
- `GET /health/metrics`: API metrics in Prometheus format

### Basic API Endpoints

- `GET /api/v1/version`: Get API version information
- `GET /api/v1/status`: Get API status
- `GET /api/v1/config`: Get API configuration

### GPU Endpoints

- `GET /api/v1/gpu/info`: Get GPU information
- `GET /api/v1/gpu/status`: Get GPU status
- `GET /api/v1/gpu/features`: Get available GPU features

### Embedding Endpoints

- `POST /api/v1/embeddings/generate`: Generate embeddings for documents
- `POST /api/v1/embeddings/query`: Generate an embedding for a query
- `GET /api/v1/embeddings/models`: List available embedding models
- `POST /api/v1/embeddings/benchmark`: Benchmark embedding generation

### Vector Operations

- `POST /api/v1/vectors/similarity`: Calculate similarity between vectors
- `POST /api/v1/vectors/normalize`: Normalize a vector
- `POST /api/v1/vectors/mmr`: Perform Maximal Marginal Relevance calculation

### HANA DB Operations

- `POST /api/v1/hana/test-connection`: Test HANA connection
- `GET /api/v1/hana/vector-types`: List available vector types

### Documentation

- `GET /docs`: Swagger UI
- `GET /redoc`: ReDoc documentation
- `GET /openapi.json`: OpenAPI schema

## Test Mode

The test mode is enabled by setting the `TEST_MODE` environment variable to `true`. When test mode is enabled, the API uses mock implementations instead of connecting to a real SAP HANA Cloud database.

The mock implementations simulate:
- Database connections
- SQL query execution
- Vector embeddings
- Search operations

This allows testing the API functionality without requiring a real SAP HANA Cloud connection.

## Troubleshooting

### Common Issues

#### Docker Container Won't Start

**Issue**: The Docker container fails to start.

**Solution**:
1. Check Docker logs: `docker-compose -f docker-compose.local.yml logs`
2. Verify that the required ports are not in use: `lsof -i :8000`
3. Ensure Docker has sufficient resources

#### API Endpoints Return Errors

**Issue**: API endpoints return errors even in test mode.

**Solution**:
1. Check the logs: `docker-compose -f docker-compose.local.yml logs api`
2. Ensure test mode is enabled: `TEST_MODE=true`
3. Verify that all required packages are installed

#### Embeddings API Returns Errors

**Issue**: Embeddings API returns errors.

**Solution**:
1. Ensure the sentence-transformers package is installed
2. Check if the specified model exists
3. Verify that the request format is correct

## Advanced Testing

### Performance Testing

For performance testing, you can use the benchmarking API:

```bash
# Benchmark embedding generation
curl -X POST http://localhost:8000/api/v1/embeddings/benchmark \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "batch_sizes": [1, 8, 16, 32],
    "iterations": 10
  }'
```

### Custom Test Data

You can modify the `test_mode.py` file to include custom test data for more specific testing scenarios.

## Conclusion

This testing approach allows you to validate the API functionality without requiring a real SAP HANA Cloud connection. It's useful for development, testing, and CI/CD pipelines.