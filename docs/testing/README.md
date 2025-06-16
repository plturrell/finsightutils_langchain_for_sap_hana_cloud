# Testing Documentation

This directory contains all the testing-related documentation for the SAP HANA Cloud LangChain Integration with NVIDIA GPU Acceleration.

## Overview

The SAP HANA Cloud LangChain Integration includes comprehensive testing across multiple levels to ensure reliability, performance, and correctness. This index will help you navigate through the various testing resources.

## Testing Resources

* [Testing Overview](README_TESTING.md) - General overview of the testing approach
* [Testing Plan](testing_plan.md) - Comprehensive testing plan for the project
* [API Testing](api-testing.md) - Guidelines for testing the API endpoints
* [E2E Testing Guide](e2e_testing_guide.md) - End-to-end testing guidelines
* [T4 GPU Testing Plan](T4_GPU_TESTING_PLAN.md) - Specific plan for testing T4 GPU acceleration

## Testing Levels

The project implements testing at multiple levels:

### 1. Unit Testing

Unit tests validate individual components in isolation.

* Location: `/tests/unit_tests/`
* Key Test Files:
  * `test_batch_processor.py` - Tests for the batch processing component
  * `test_error_utils.py` - Tests for error handling utilities
  * `test_gpu_embeddings.py` - Tests for GPU-accelerated embeddings
  * `test_multi_gpu_manager.py` - Tests for multi-GPU management
  * `test_vectorstores.py` - Tests for vector store operations

### 2. Integration Testing

Integration tests validate the interaction between components.

* Location: `/tests/integration_tests/`
* Key Test Files:
  * `test_vectorstores.py` - Tests for vector store integration
  * `test_cross_platform.py` - Tests for cross-platform compatibility
  * `test_optimization_api.py` - Tests for optimization API

### 3. End-to-End Testing

E2E tests validate the entire system from end to end.

* Location: `/tests/e2e_tests/`
* Key Test Files:
  * `test_basic_functionality.py` - Tests for basic functionality
  * `test_error_handling.py` - Tests for error handling
  * `test_hana_integration.py` - Tests for integration with SAP HANA Cloud

### 4. Performance Testing

Performance tests validate the system's performance under various conditions.

* Location: `/tests/` (various files)
* Key Test Files:
  * `benchmark_embeddings.py` - Benchmarks for embedding generation
  * `load_test.py` - Load testing for API endpoints

## Running Tests

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit_tests/

# Run specific test modules
pytest tests/unit_tests/test_gpu_embeddings.py
pytest tests/unit_tests/test_multi_gpu_manager.py
```

### Integration Tests

```bash
# Run all integration tests
pytest tests/integration_tests/

# Run specific test modules
pytest tests/integration_tests/test_vectorstores.py
```

### End-to-End Tests

```bash
# Run all E2E tests
pytest tests/e2e_tests/

# Run specific test modules
pytest tests/e2e_tests/test_basic_functionality.py
```

### Docker-Based Tests

```bash
# Run tests in Docker
./tests/docker/run_docker_tests.sh
```

## Test Coverage

The project maintains a high level of test coverage to ensure reliability. Test coverage reports are generated automatically as part of the CI/CD pipeline.

## Mocking

The tests use mocking to isolate components and test them independently:

* `/api/mocks/` - Contains mock implementations for external dependencies
* `/tests/integration_tests/fixtures/` - Contains test fixtures for integration tests

## Testing GPU Components

Testing GPU-accelerated components requires additional considerations:

1. **GPU Availability**: Tests that require a GPU will be skipped if no GPU is available
2. **TensorRT Optimization**: Tests for TensorRT-optimized components require TensorRT to be installed
3. **Multi-GPU Testing**: Tests for multi-GPU functionality require multiple GPUs to be available

See the [T4 GPU Testing Plan](T4_GPU_TESTING_PLAN.md) for more details.

## Test Automation

The project includes automated testing as part of the CI/CD pipeline:

* GitHub Actions workflows run tests automatically on pull requests
* Test results are published as part of the PR checks
* Test coverage is tracked over time

## Test Data

The tests use the following test data:

* `/tests/create_test_data.py` - Script for creating test data
* `/tests/integration_tests/fixtures/filtering_test_cases.py` - Test cases for filtering
* `/test_results/sample_documents.json` - Sample documents for testing

## Adding New Tests

When adding new functionality, please ensure that appropriate tests are added:

1. Add unit tests for new components
2. Add integration tests for new component interactions
3. Update E2E tests to cover new functionality
4. Add performance tests for performance-critical components