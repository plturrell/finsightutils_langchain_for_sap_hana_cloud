"""
Base classes and utilities for end-to-end tests.

This module provides the foundation for running end-to-end tests against
the LangChain SAP HANA Cloud integration API.
"""

import os
import time
import json
import logging
import requests
import subprocess
from typing import Dict, List, Any, Optional, Tuple
import unittest
from unittest.mock import patch

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("e2e_tests")


class BaseEndToEndTest(unittest.TestCase):
    """
    Base class for end-to-end tests of the LangChain SAP HANA integration.
    
    This class provides common setup and teardown methods for running tests against
    the API, as well as utilities for starting local services, making API requests,
    and validating responses.
    """
    
    # Default test environment variables
    BACKEND_URL = os.environ.get("E2E_BACKEND_URL", "http://localhost:8000")
    API_KEY = os.environ.get("E2E_API_KEY", "test-api-key")
    TEST_TIMEOUT = int(os.environ.get("E2E_TEST_TIMEOUT", "30"))
    RUN_LOCAL = os.environ.get("E2E_RUN_LOCAL", "true").lower() == "true"
    
    @classmethod
    def setUpClass(cls):
        """Set up test class, starting a local server if needed."""
        logger.info(f"Setting up end-to-end tests against {cls.BACKEND_URL}")
        
        cls.local_server_process = None
        if cls.RUN_LOCAL:
            cls._start_local_server()
            # Allow server time to start
            logger.info("Waiting for local server to start...")
            time.sleep(5)
            
        # Test that the server is reachable
        cls._check_server_reachable()

    @classmethod
    def tearDownClass(cls):
        """Tear down test class, stopping the local server if it was started."""
        if cls.local_server_process:
            logger.info("Stopping local server...")
            cls.local_server_process.terminate()
            cls.local_server_process.wait()
            logger.info("Local server stopped")
    
    @classmethod
    def _start_local_server(cls):
        """Start a local server for testing."""
        logger.info("Starting local server for testing...")
        
        # Set environment variables for the test server
        env = os.environ.copy()
        env.update({
            "GPU_ENABLED": "false",  # Disable GPU for tests
            "USE_TENSORRT": "false",
            "LOG_LEVEL": "DEBUG",
            "ENABLE_CORS": "true",
            "CORS_ORIGINS": "*",
            "API_KEY": cls.API_KEY,
            "TESTING": "true",
            # Use mock database for tests
            "MOCK_DB": "true",
        })
        
        # Start the server
        cls.local_server_process = subprocess.Popen(
            ["uvicorn", "api.app:app", "--host", "127.0.0.1", "--port", "8000"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        logger.info(f"Local server started with PID {cls.local_server_process.pid}")
    
    @classmethod
    def _check_server_reachable(cls):
        """Check that the server is reachable."""
        max_retries = 5
        retry_delay = 2
        
        for retry in range(max_retries):
            try:
                response = requests.get(f"{cls.BACKEND_URL}/health/ping", timeout=cls.TEST_TIMEOUT)
                response.raise_for_status()
                logger.info(f"Server is reachable: {response.json()}")
                return
            except Exception as e:
                logger.warning(f"Server not yet reachable (attempt {retry+1}/{max_retries}): {str(e)}")
                if retry < max_retries - 1:
                    time.sleep(retry_delay)
        
        # If we get here, the server is not reachable
        raise RuntimeError(f"Server at {cls.BACKEND_URL} is not reachable after {max_retries} attempts")
    
    def setUp(self):
        """Set up test case."""
        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
    
    def tearDown(self):
        """Tear down test case."""
        self.session.close()
    
    def get_api_url(self, path: str) -> str:
        """Get the full URL for an API endpoint."""
        return f"{self.BACKEND_URL}/{path.lstrip('/')}"
    
    def api_request(
        self, 
        method: str, 
        path: str, 
        data: Optional[Dict[str, Any]] = None,
        expected_status: int = 200,
        timeout: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Make an API request and return the response.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            data: Request data (for POST, PUT, etc.)
            expected_status: Expected HTTP status code
            timeout: Request timeout in seconds
            
        Returns:
            Tuple of (response data, status code)
            
        Raises:
            AssertionError: If the response status code doesn't match the expected status
        """
        url = self.get_api_url(path)
        timeout = timeout or self.TEST_TIMEOUT
        
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
        
        # Check status code
        self.assertEqual(
            response.status_code, 
            expected_status, 
            f"Expected status {expected_status}, got {response.status_code}: {response.text}"
        )
        
        # Try to parse as JSON, fall back to text if it's not JSON
        try:
            response_data = response.json()
        except ValueError:
            response_data = {"text": response.text}
        
        return response_data, response.status_code
    
    def assert_response_contains_keys(self, response: Dict[str, Any], keys: List[str]) -> None:
        """Assert that a response contains all the specified keys."""
        for key in keys:
            self.assertIn(key, response, f"Response missing key: {key}")
    
    def assert_embedding_dimensions(self, embedding: List[float], expected_dim: int = 384) -> None:
        """
        Assert that an embedding has the expected dimensions.
        
        Args:
            embedding: The embedding to check
            expected_dim: Expected dimensionality (default: 384 for all-MiniLM-L6-v2)
        """
        self.assertIsInstance(embedding, list, "Embedding should be a list")
        self.assertEqual(len(embedding), expected_dim, f"Expected {expected_dim} dimensions, got {len(embedding)}")
        self.assertTrue(all(isinstance(v, float) for v in embedding), "Embedding should contain only floats")