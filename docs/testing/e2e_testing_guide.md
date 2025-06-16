# End-to-End Testing Guide

This guide explains the comprehensive end-to-end testing approach for the LangChain integration for SAP HANA Cloud project.

## Overview

End-to-end (E2E) tests verify that the entire application stack works together correctly. Our E2E tests cover:

1. **Basic API Functionality**: Verifying core endpoints and features
2. **Error Handling**: Testing how the API handles invalid inputs and error conditions
3. **Frontend Integration**: Ensuring the API provides what the frontend expects
4. **SAP HANA Integration**: Validating the connection to and operations with SAP HANA Cloud

## Test Architecture

Our E2E tests follow a layered architecture:

1. **Base Test Class**: Provides common utilities and setup for all tests
2. **Test Categories**: Organized by functional area
3. **Test Runner**: Coordinates test execution and report generation

## Running the Tests

### Prerequisites

- Python 3.10+
- Required packages: `requests`, `unittest-xml-reporting`
- For local testing: FastAPI backend code with dependencies

### Basic Usage

Run all E2E tests against a local server:

```bash
cd tests/e2e_tests
python run_tests.py --run-local
```

Run against a deployed backend:

```bash
python run_tests.py --backend-url https://your-backend-url --api-key your-api-key --run-local=false
```

### Command Line Options

- `--backend-url`: URL of the API to test (default: http://localhost:8000)
- `--api-key`: API key for authentication (default: test-api-key)
- `--run-local`: Start a local server for testing (default: true)
- `--timeout`: Request timeout in seconds (default: 30)
- `--test-hana`: Run SAP HANA integration tests (default: false)
- `--output-dir`: Directory for test reports (default: test_results)
- `--verbose`: Enable verbose output
- `--pattern`: Pattern for test files to run (default: test_*.py)

### Environment Variables

You can also configure the tests using environment variables:

- `E2E_BACKEND_URL`: Backend API URL
- `E2E_API_KEY`: API key for authentication
- `E2E_RUN_LOCAL`: Whether to start a local server (true/false)
- `E2E_TEST_TIMEOUT`: Request timeout in seconds
- `E2E_TEST_HANA`: Whether to run SAP HANA tests (true/false)
- `E2E_OUTPUT_DIR`: Directory for test reports

For SAP HANA tests, additional variables are required:
- `E2E_HANA_HOST`: SAP HANA Cloud host
- `E2E_HANA_PORT`: SAP HANA Cloud port
- `E2E_HANA_USER`: SAP HANA Cloud username
- `E2E_HANA_PASSWORD`: SAP HANA Cloud password

## Test Categories

### Basic Functionality Tests

These tests verify that core API features work correctly:

- Health check endpoints
- Embedding generation
- Text storage and retrieval
- Query functionality with and without filters
- MMR search for diverse results

### Error Handling Tests

These tests verify that the API properly handles error conditions:

- Invalid endpoints
- Missing required parameters
- Invalid parameter types
- Invalid authentication
- Request validation

### Frontend Integration Tests

These tests simulate how the frontend would interact with the API:

- CORS headers for cross-origin requests
- Response format expected by frontend components
- JSON serialization consistency
- Error response format for frontend error handling

### SAP HANA Integration Tests

These tests verify the connection to SAP HANA Cloud:

- Database connection establishment
- Table creation and management
- Data storage and retrieval
- Metadata filtering
- Data deletion

## Test Reports

The test runner generates several types of reports:

1. **JUnit XML Report**: Compatible with CI/CD systems (Jenkins, GitLab, etc.)
2. **Text Report**: Human-readable summary of test results
3. **JSON Summary**: Structured data about the test run
4. **Console Output**: Real-time test progress

Reports are saved in the specified output directory (default: `test_results`).

## Extending the Tests

### Adding New Test Cases

1. Create a new test file in the `tests/e2e_tests` directory following the naming convention `test_*.py`
2. Extend the `BaseEndToEndTest` class
3. Implement test methods with names starting with `test_`

Example:

```python
from .base import BaseEndToEndTest

class MyNewTest(BaseEndToEndTest):
    def test_my_new_feature(self):
        response, status_code = self.api_request("POST", "/my-endpoint", {"data": "value"})
        self.assertEqual(status_code, 200)
        self.assertTrue(response["success"])
```

### Mocking Dependencies

For tests that need to mock dependencies:

1. Use the `unittest.mock` module to patch dependencies
2. Consider using the `pytest-mock` fixture for pytest-style tests

Example with mocking:

```python
from unittest.mock import patch
from .base import BaseEndToEndTest

class MockTest(BaseEndToEndTest):
    @patch("api.services.expensive_operation")
    def test_with_mock(self, mock_operation):
        mock_operation.return_value = "mocked result"
        response, _ = self.api_request("GET", "/operation")
        self.assertEqual(response["result"], "mocked result")
```

## CI/CD Integration

The E2E tests are integrated with our CI/CD pipeline:

1. Tests run automatically on pull requests to the main branch
2. Tests run before deployment to staging and production
3. Test results are published as artifacts
4. Failed tests block deployment

Configuration can be found in `.github/workflows/ci-cd.yml`.