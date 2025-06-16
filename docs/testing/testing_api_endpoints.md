# API Endpoint Testing Guide

This document explains how to run and maintain the API endpoint tests for the SAP HANA Cloud LangChain Integration.

## Overview

The API endpoint tests ensure that all API endpoints are working correctly. These tests use Docker to create an isolated test environment with mock implementations of dependencies like the SAP HANA client library (`hdbcli`) and LangChain components.

## Test Structure

The testing infrastructure consists of:

1. **Mock Implementations**:
   - `api/mocks/hdbcli/` - Mock implementation of the SAP HANA client library
   - `api/mocks/langchain_hana/` - Mock implementation of the LangChain-HANA integration
   - `api/mocks/sentence_transformers/` - Mock implementation of the sentence-transformers library

2. **Test API Implementation**:
   - `api/index_test.py` - A simplified version of the API that uses mock implementations

3. **Test Scripts**:
   - `run_api_tests.sh` - Script to build a test Docker container and test all API endpoints
   - `test_api_endpoints.sh` - Original test script (preserved for reference)

4. **Docker Configuration**:
   - `docker-compose.test.yml` - Docker Compose configuration for testing
   - `api/Dockerfile.test` - Dockerfile for building the test container

## Running the Tests

To run the API endpoint tests:

```bash
./run_api_tests.sh
```

The script will:

1. Create a test Dockerfile and Docker Compose configuration
2. Build and start a Docker container with the test API
3. Run tests against all API endpoints
4. Report success or failure for each endpoint
5. Shut down the container

## Test Environment

The test environment runs with:

- `TEST_MODE=true` - Enables testing without real dependencies
- Port 8001 mapped to container port 8000
- Mock implementations for all external dependencies
- No real connection to SAP HANA Cloud

## Adding New Endpoints

When adding new API endpoints:

1. Add the endpoint implementation to `api/index.py`
2. Add a mock implementation to `api/index_test.py`
3. Add the endpoint test to `run_api_tests.sh`

## Troubleshooting

If tests fail:

1. Check that the container is running (`docker ps`)
2. Check container logs (`docker logs langchain-integration-for-sap-hana-cloud-api-1`)
3. Verify that the endpoint is implemented in `api/index_test.py`
4. Check that all mock implementations are correctly loaded

## Maintaining Mock Implementations

The mock implementations are simplified versions of the real components. When the real components change, the mocks should be updated to maintain compatibility.

Key mock files:
- `/api/mocks/hdbcli/dbapi.py` - Mocks SAP HANA database connections
- `/api/mocks/langchain_hana/vectorstores.py` - Mocks vector search functionality
- `/api/mocks/langchain_hana/embeddings.py` - Mocks embedding generation

## CI/CD Integration

These tests can be integrated into CI/CD pipelines to ensure API compatibility and functionality before deployment.