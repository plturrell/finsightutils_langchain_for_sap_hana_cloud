# API Documentation

This directory contains all the API-related documentation for the SAP HANA Cloud LangChain Integration with NVIDIA GPU Acceleration.

## Overview

The SAP HANA Cloud LangChain Integration provides a comprehensive API for interacting with SAP HANA Cloud's vector capabilities, GPU-accelerated embedding generation, and more. This index will help you navigate through the various API documentation resources.

## API Documentation Resources

* [API Documentation](api_documentation.md) - Complete API documentation with examples
* [API Reference](reference.md) - Detailed API reference for all endpoints
* [API Design Guidelines](api_design_guidelines.md) - Guidelines followed for API design
* [API Documentation YAML](api_documentation.yaml) - OpenAPI specification

## API Categories

The API is organized into the following categories:

### 1. Vector Store Operations

Endpoints for managing vector embeddings in SAP HANA Cloud.

* **POST /texts** - Add texts and their embeddings to the vector store
* **DELETE /texts** - Delete texts from the vector store
* **GET /texts/count** - Get the count of texts in the vector store
* **GET /texts/metadata** - Get the metadata schema

### 2. Query Operations

Endpoints for performing similarity searches and retrievals.

* **POST /query** - Perform a similarity search
* **POST /query/mmr** - Perform a Maximum Marginal Relevance search
* **POST /query/by-vector** - Search using a raw vector

### 3. Health and Monitoring

Endpoints for monitoring the health and performance of the API.

* **GET /health/ping** - Basic health check
* **GET /health/ready** - Readiness probe
* **GET /health/startup** - Startup probe
* **GET /metrics** - Prometheus metrics endpoint

### 4. GPU Operations

Endpoints for managing and monitoring GPU acceleration.

* **GET /gpu/info** - Get information about available GPUs
* **GET /gpu/status** - Get GPU utilization status
* **GET /gpu/optimization** - Get current GPU optimization settings
* **POST /gpu/optimization** - Update GPU optimization settings

### 5. Developer Operations

Endpoints for development and debugging.

* **GET /developer/debug** - Get debug information
* **POST /developer/profile** - Profile a specific operation
* **GET /developer/config** - Get current configuration

## Authentication

The API supports the following authentication methods:

* **API Key Authentication**: Pass the API key in the `X-API-Key` header
* **JWT Token Authentication**: Pass a JWT token in the `Authorization` header with the format `Bearer <token>`

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

## Using the API

The API can be accessed through the Swagger UI at `/docs` when the service is running. The Swagger UI provides interactive documentation that allows you to:

1. Browse all available API endpoints
2. View request and response schemas
3. Try out endpoints directly in the browser
4. See detailed descriptions and examples for each endpoint

## Code Examples

For code examples of how to use the API, refer to:

* [Example API Client](../../api/examples/client_example.py)
* [API Client Notebook](../../api/examples/api_client.ipynb)

## API Client Libraries

The OpenAPI specification can be used to generate client libraries for different programming languages:

1. Download the OpenAPI specification from `/openapi.json`
2. Use tools like [OpenAPI Generator](https://openapi-generator.tech/) to generate client libraries

## API Versioning

The API version is included in:
* The response headers (`X-API-Version`)
* The OpenAPI specification metadata
* The root endpoint response

## Rate Limiting and Quotas

The API implements rate limiting to ensure fair usage:
* Requests are limited to 100 per minute per API key
* Larger batch requests count as multiple requests
* Rate limit headers are included in responses