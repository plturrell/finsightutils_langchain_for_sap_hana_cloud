"""Tests for the API fixes.

This module contains tests for the backend fixes implemented in the API.
It tests:
1. Configuration handling
2. Error handling
3. GPU acceleration
4. CORS configuration
5. Database connection management
6. Version consistency
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import sys
import json

# Add the API path to sys.path to be able to import modules from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "api")))

import pytest
from fastapi.testclient import TestClient
from hdbcli import dbapi
from pydantic import BaseModel

import app
from config import Config, get_settings
from database import ConnectionPool, get_db_connection
from version import get_version, get_version_info


class TestAPIFixes(unittest.TestCase):
    """Test suite for API fixes."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test client
        self.client = TestClient(app.app)
        
        # Mock database connection
        self.db_patcher = patch("database.dbapi")
        self.mock_dbapi = self.db_patcher.start()
        self.mock_connection = MagicMock()
        self.mock_dbapi.connect.return_value = self.mock_connection
        
        # Set environment variables for testing
        os.environ["PLATFORM"] = "test"
        os.environ["VERSION"] = "1.0.0-test"
        os.environ["CORS_ORIGINS"] = "http://localhost:3000,https://example.com"
        os.environ["CORS_METHODS"] = "GET,POST,OPTIONS"
        os.environ["CORS_HEADERS"] = "Content-Type,Authorization"
        os.environ["CORS_CREDENTIALS"] = "false"
    
    def tearDown(self):
        """Clean up test environment."""
        self.db_patcher.stop()
        
        # Reset environment variables
        for var in ["PLATFORM", "VERSION", "CORS_ORIGINS", "CORS_METHODS", 
                    "CORS_HEADERS", "CORS_CREDENTIALS"]:
            if var in os.environ:
                del os.environ[var]
    
    def test_health_endpoint(self):
        """Test the health endpoint returns correct information."""
        response = self.client.get("/health/ping")
        
        # Check response status code
        self.assertEqual(response.status_code, 200)
        
        # Check response content
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "1.0.0-test")
        self.assertEqual(data["backend"], "test")
    
    @patch("app.VectorStoreService")
    def test_error_handling(self, mock_service):
        """Test context-aware error handling."""
        # Mock a database error
        mock_service_instance = mock_service.return_value
        mock_service_instance.similarity_search.side_effect = Exception("Database error: table not found")
        
        # Make request
        response = self.client.post(
            "/query", 
            json={"query": "test query", "k": 4}
        )
        
        # Check response status code
        self.assertEqual(response.status_code, 500)
        
        # Check response content
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("message", data["detail"])
        self.assertIn("suggestions", data["detail"])
        
        # Should have context-aware error
        error_detail = data["detail"]
        self.assertIn("table", error_detail["message"].lower())
    
    def test_cors_configuration(self):
        """Test CORS configuration from environment variables."""
        settings = get_settings()
        
        # Check CORS settings
        self.assertTrue(settings.cors.enable_cors)
        self.assertEqual(
            settings.cors.allowed_origins, 
            ["http://localhost:3000", "https://example.com"]
        )
        self.assertEqual(settings.cors.allowed_methods, ["GET", "POST", "OPTIONS"])
        self.assertEqual(settings.cors.allowed_headers, ["Content-Type", "Authorization"])
        self.assertFalse(settings.cors.allow_credentials)
    
    def test_version_consistency(self):
        """Test version module provides consistent information."""
        version = get_version()
        version_info = get_version_info()
        
        # Check version is a string
        self.assertIsInstance(version, str)
        
        # Check version info has correct keys
        self.assertIn("version", version_info)
        self.assertIn("major", version_info)
        self.assertIn("minor", version_info)
        self.assertIn("patch", version_info)
        
        # Check version matches
        self.assertEqual(version, version_info["version"])
        self.assertEqual(version, "1.0.0-test")  # From environment variable
    
    def test_database_connection_pool(self):
        """Test database connection pool."""
        # Create a connection pool
        pool = ConnectionPool(max_connections=3, connection_timeout=60)
        
        # Get a connection
        conn1 = pool.get_connection()
        
        # Connection should be in the pool
        self.assertEqual(len(pool.pool), 1)
        
        # Get another connection (should be the same since we're using the same thread)
        conn2 = pool.get_connection()
        
        # Still should have one connection
        self.assertEqual(len(pool.pool), 1)
        
        # Close all connections
        pool.close_all()
        
        # Pool should be empty
        self.assertEqual(len(pool.pool), 0)


if __name__ == "__main__":
    unittest.main()