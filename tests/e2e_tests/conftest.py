"""
Pytest configuration for end-to-end tests.

This file contains pytest fixtures and configuration for the end-to-end tests.
"""

import os
import time
import pytest
import subprocess
import requests
from typing import Iterator, Optional


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--backend-url",
        default=os.environ.get("E2E_BACKEND_URL", "http://localhost:8000"),
        help="Backend API URL (default: http://localhost:8000)"
    )
    parser.addoption(
        "--api-key",
        default=os.environ.get("E2E_API_KEY", "test-api-key"),
        help="API key for authentication"
    )
    parser.addoption(
        "--run-local",
        action="store_true",
        default=os.environ.get("E2E_RUN_LOCAL", "true").lower() == "true",
        help="Start a local server for testing"
    )
    parser.addoption(
        "--test-hana",
        action="store_true",
        default=os.environ.get("E2E_TEST_HANA", "false").lower() == "true",
        help="Run SAP HANA integration tests"
    )


@pytest.fixture(scope="session")
def backend_url(request) -> str:
    """Get the backend URL."""
    return request.config.getoption("--backend-url")


@pytest.fixture(scope="session")
def api_key(request) -> str:
    """Get the API key."""
    return request.config.getoption("--api-key")


@pytest.fixture(scope="session")
def run_local(request) -> bool:
    """Determine if a local server should be started."""
    return request.config.getoption("--run-local")


@pytest.fixture(scope="session")
def test_hana(request) -> bool:
    """Determine if SAP HANA integration tests should be run."""
    return request.config.getoption("--test-hana")


@pytest.fixture(scope="session")
def local_server(run_local, test_hana) -> Iterator[Optional[subprocess.Popen]]:
    """Start a local server for testing if requested."""
    process = None
    
    if run_local:
        # Set environment variables for the test server
        env = os.environ.copy()
        env.update({
            "GPU_ENABLED": "false",  # Disable GPU for tests
            "USE_TENSORRT": "false",
            "LOG_LEVEL": "DEBUG",
            "ENABLE_CORS": "true",
            "CORS_ORIGINS": "*",
            "API_KEY": os.environ.get("E2E_API_KEY", "test-api-key"),
            "TESTING": "true",
        })
        
        # Add HANA config if testing with HANA
        if test_hana:
            required_vars = ["E2E_HANA_HOST", "E2E_HANA_PORT", "E2E_HANA_USER", "E2E_HANA_PASSWORD"]
            missing_vars = [var for var in required_vars if not os.environ.get(var)]
            
            if missing_vars:
                pytest.skip(f"Missing required environment variables for SAP HANA tests: {', '.join(missing_vars)}")
            
            env.update({
                # HANA configuration
                "DB_HOST": os.environ.get("E2E_HANA_HOST"),
                "DB_PORT": os.environ.get("E2E_HANA_PORT"),
                "DB_USER": os.environ.get("E2E_HANA_USER"),
                "DB_PASSWORD": os.environ.get("E2E_HANA_PASSWORD"),
                # Use a test-specific table name
                "DEFAULT_TABLE_NAME": f"E2E_TEST_{int(time.time())}",
            })
        else:
            # Use mock database for non-HANA tests
            env.update({
                "MOCK_DB": "true",
            })
        
        # Start the server
        process = subprocess.Popen(
            ["uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "8000"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # Allow time for server to start
        time.sleep(5)
    
    yield process
    
    # Clean up
    if process:
        process.terminate()
        process.wait()


@pytest.fixture(scope="session", autouse=True)
def check_server(backend_url, local_server):
    """Check that the server is reachable."""
    max_retries = 5
    retry_delay = 2
    
    for retry in range(max_retries):
        try:
            response = requests.get(f"{backend_url}/health/ping", timeout=10)
            response.raise_for_status()
            return
        except Exception as e:
            if retry < max_retries - 1:
                time.sleep(retry_delay)
            else:
                pytest.fail(f"Server at {backend_url} is not reachable: {str(e)}")


@pytest.fixture
def api_client(backend_url, api_key):
    """Create an API client for making requests."""
    session = requests.Session()
    session.headers.update({
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "Accept": "application/json",
    })
    
    class APIClient:
        """API client for making requests."""
        
        def __init__(self, session, base_url):
            self.session = session
            self.base_url = base_url
        
        def request(self, method, path, data=None, expected_status=200, timeout=30):
            """Make an API request."""
            url = f"{self.base_url}/{path.lstrip('/')}"
            
            if method.upper() == "GET":
                response = self.session.get(url, timeout=timeout)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=timeout)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, json=data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            if response.status_code != expected_status:
                raise AssertionError(
                    f"Expected status {expected_status}, got {response.status_code}: {response.text}"
                )
            
            try:
                return response.json()
            except ValueError:
                return {"text": response.text}
    
    client = APIClient(session, backend_url)
    yield client
    session.close()


@pytest.fixture
def test_data():
    """Create test data for use in tests."""
    timestamp = int(time.time())
    return {
        "timestamp": timestamp,
        "texts": [
            f"This is a test document for SAP HANA Cloud ({timestamp}).",
            f"Vector embeddings enable semantic search ({timestamp}).",
            f"GPU acceleration improves performance ({timestamp}).",
        ],
        "metadatas": [
            {"source": f"pytest-{timestamp}", "category": "database"},
            {"source": f"pytest-{timestamp}", "category": "vector-search"},
            {"source": f"pytest-{timestamp}", "category": "performance"},
        ],
        "table_name": f"PYTEST_TABLE_{timestamp}"
    }