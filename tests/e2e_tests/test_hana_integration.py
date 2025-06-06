"""
End-to-end tests for SAP HANA Cloud integration.

These tests verify the integration with SAP HANA Cloud,
including database connections, table creation, data storage, and retrieval.
"""

import os
import unittest
import time
from typing import Dict, List, Any, Optional

from .base import BaseEndToEndTest, logger


@unittest.skipIf(
    os.environ.get("E2E_TEST_HANA", "false").lower() != "true",
    "SAP HANA tests are disabled. Set E2E_TEST_HANA=true to enable."
)
class SAP_HANA_IntegrationTest(BaseEndToEndTest):
    """
    Test the integration with SAP HANA Cloud.
    
    These tests require a real SAP HANA Cloud instance and will be skipped
    unless the E2E_TEST_HANA environment variable is set to "true".
    
    Required environment variables:
    - E2E_HANA_HOST: SAP HANA Cloud host
    - E2E_HANA_PORT: SAP HANA Cloud port
    - E2E_HANA_USER: SAP HANA Cloud username
    - E2E_HANA_PASSWORD: SAP HANA Cloud password
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test class with HANA-specific configuration."""
        # Check for required environment variables
        required_vars = ["E2E_HANA_HOST", "E2E_HANA_PORT", "E2E_HANA_USER", "E2E_HANA_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables for SAP HANA tests: {', '.join(missing_vars)}")
        
        # Override RUN_LOCAL to use a proper configuration for HANA tests
        cls.RUN_LOCAL = os.environ.get("E2E_RUN_LOCAL", "true").lower() == "true"
        
        if cls.RUN_LOCAL:
            # Set up a test server with HANA configuration
            logger.info("Starting test server with SAP HANA configuration...")
            cls._start_local_server_with_hana()
            # Allow server time to start
            logger.info("Waiting for local server to start...")
            time.sleep(5)
        
        # Verify server is reachable
        cls._check_server_reachable()
        
        # Verify HANA connection through the API
        cls._verify_hana_connection()
    
    @classmethod
    def _start_local_server_with_hana(cls):
        """Start a local server with SAP HANA configuration."""
        # Set environment variables for HANA connection
        env = os.environ.copy()
        env.update({
            "GPU_ENABLED": "false",  # Disable GPU for tests
            "USE_TENSORRT": "false",
            "LOG_LEVEL": "DEBUG",
            "ENABLE_CORS": "true",
            "CORS_ORIGINS": "*",
            "API_KEY": cls.API_KEY,
            "TESTING": "true",
            # HANA configuration
            "DB_HOST": os.environ.get("E2E_HANA_HOST"),
            "DB_PORT": os.environ.get("E2E_HANA_PORT"),
            "DB_USER": os.environ.get("E2E_HANA_USER"),
            "DB_PASSWORD": os.environ.get("E2E_HANA_PASSWORD"),
            # Use a test-specific table name
            "DEFAULT_TABLE_NAME": f"E2E_TEST_{int(time.time())}",
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
    def _verify_hana_connection(cls):
        """Verify that the API can connect to SAP HANA Cloud."""
        # Use a session for this check
        session = requests.Session()
        session.headers.update({
            "X-API-Key": cls.API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        
        try:
            # Check database health endpoint
            response = session.get(f"{cls.BACKEND_URL}/health/database")
            response.raise_for_status()
            data = response.json()
            
            # Verify connection is successful
            if data.get("status") != "ok" or not data.get("details", {}).get("connected", False):
                raise ValueError(f"Failed to connect to SAP HANA: {data}")
            
            logger.info("SAP HANA connection verified")
        finally:
            session.close()
    
    def setUp(self):
        """Set up test case."""
        super().setUp()
        
        # Get a unique table name for this test
        self.test_table = f"E2E_TEST_{int(time.time())}"
        self.test_metadata_source = f"e2e-test-{int(time.time())}"
        
        # Create test data
        self.test_texts = [
            "SAP HANA Cloud is an in-memory database platform optimized for business applications.",
            "Vector stores enable semantic search capabilities for unstructured data.",
            "GPU acceleration significantly improves embedding generation performance.",
            "TensorRT optimization reduces inference time for embedding models.",
            "Connection pooling improves database access efficiency in high-throughput applications.",
        ]
        
        self.test_metadatas = [
            {"source": self.test_metadata_source, "category": "database", "id": 1},
            {"source": self.test_metadata_source, "category": "vector-search", "id": 2},
            {"source": self.test_metadata_source, "category": "performance", "id": 3},
            {"source": self.test_metadata_source, "category": "optimization", "id": 4},
            {"source": self.test_metadata_source, "category": "database", "id": 5},
        ]
    
    def tearDown(self):
        """Clean up after the test."""
        # Delete test data
        try:
            self.api_request("POST", "/delete", {
                "filter": {"source": self.test_metadata_source}
            })
        except Exception as e:
            logger.warning(f"Failed to clean up test data: {e}")
        
        super().tearDown()
    
    def test_database_connection(self):
        """Test that the API can connect to the database."""
        response, _ = self.api_request("GET", "/health/database")
        self.assertEqual(response["status"], "ok")
        self.assertTrue(response["details"]["connected"])
    
    def test_add_texts_to_hana(self):
        """Test adding texts to SAP HANA Cloud."""
        response, _ = self.api_request("POST", "/texts", {
            "texts": self.test_texts,
            "metadatas": self.test_metadatas,
            "table_name": self.test_table
        })
        
        self.assertTrue(response["success"])
        self.assertIn("message", response)
    
    def test_query_with_filtering(self):
        """Test querying with metadata filtering."""
        # First, add the texts
        self.api_request("POST", "/texts", {
            "texts": self.test_texts,
            "metadatas": self.test_metadatas,
            "table_name": self.test_table
        })
        
        # Query with category filter
        response, _ = self.api_request("POST", "/query", {
            "query": "database",
            "k": 2,
            "filter": {"category": "database", "source": self.test_metadata_source},
            "table_name": self.test_table
        })
        
        self.assertGreaterEqual(len(response["results"]), 1)
        for result in response["results"]:
            self.assertEqual(result["metadata"]["category"], "database")
    
    def test_delete_by_filter(self):
        """Test deleting documents by metadata filter."""
        # First, add the texts
        self.api_request("POST", "/texts", {
            "texts": self.test_texts,
            "metadatas": self.test_metadatas,
            "table_name": self.test_table
        })
        
        # Delete documents with category "database"
        response, _ = self.api_request("POST", "/delete", {
            "filter": {"category": "database", "source": self.test_metadata_source},
            "table_name": self.test_table
        })
        
        self.assertTrue(response["success"])
        
        # Verify they're gone by querying
        response, _ = self.api_request("POST", "/query", {
            "query": "database",
            "k": 5,
            "filter": {"category": "database", "source": self.test_metadata_source},
            "table_name": self.test_table
        })
        
        self.assertEqual(len(response["results"]), 0)
        
        # But other documents should still be there
        response, _ = self.api_request("POST", "/query", {
            "query": "vector",
            "k": 5,
            "filter": {"category": "vector-search", "source": self.test_metadata_source},
            "table_name": self.test_table
        })
        
        self.assertGreaterEqual(len(response["results"]), 1)


if __name__ == "__main__":
    import subprocess
    import requests
    unittest.main()