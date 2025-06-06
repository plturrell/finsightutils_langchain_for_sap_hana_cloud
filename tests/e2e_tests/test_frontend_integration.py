"""
End-to-end tests for frontend integration.

These tests verify the integration between the frontend and backend,
ensuring that the API provides the functionality needed by the frontend components.
"""

import os
import unittest
import json
from typing import Dict, List, Any, Optional
from unittest.mock import patch

from .base import BaseEndToEndTest, logger


class FrontendIntegrationTest(BaseEndToEndTest):
    """
    Test the API from a frontend perspective.
    
    These tests simulate the API calls that would be made by the frontend
    and verify that the responses meet the frontend's expectations.
    """
    
    def test_cors_headers(self):
        """Test that CORS headers are properly set."""
        # Send OPTIONS request to check CORS headers
        url = self.get_api_url("/")
        response = requests.options(
            url, 
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "X-API-Key",
            }
        )
        
        # Check CORS headers
        self.assertIn("Access-Control-Allow-Origin", response.headers)
        self.assertIn("Access-Control-Allow-Methods", response.headers)
        self.assertIn("Access-Control-Allow-Headers", response.headers)
    
    def test_query_with_frontend_format(self):
        """
        Test the query endpoint with the format used by the frontend.
        
        This test simulates how the frontend would call the query endpoint
        and verifies that the response format matches what the frontend expects.
        """
        # Create test data with unique identifier
        timestamp = int(time.time())
        test_data = [
            {"text": f"This is test document 1 ({timestamp})", "metadata": {"source": "frontend-test", "id": 1, "timestamp": timestamp}},
            {"text": f"This is test document 2 about SAP HANA Cloud ({timestamp})", "metadata": {"source": "frontend-test", "id": 2, "timestamp": timestamp}},
            {"text": f"This document discusses vector embeddings and semantic search ({timestamp})", "metadata": {"source": "frontend-test", "id": 3, "timestamp": timestamp}},
        ]
        
        # Add test documents
        texts = [item["text"] for item in test_data]
        metadatas = [item["metadata"] for item in test_data]
        
        add_response, _ = self.api_request("POST", "/texts", {
            "texts": texts,
            "metadatas": metadatas
        })
        
        self.assertTrue(add_response["success"])
        
        # Search query that would come from the frontend
        frontend_query = {
            "query": "SAP HANA Cloud",
            "k": 2,
            "filter": {"source": "frontend-test", "timestamp": timestamp}
        }
        
        query_response, _ = self.api_request("POST", "/query", frontend_query)
        
        # Verify the response format meets frontend expectations
        self.assertIn("results", query_response)
        self.assertIsInstance(query_response["results"], list)
        self.assertGreaterEqual(len(query_response["results"]), 1)
        
        # Verify each result has the expected structure
        for result in query_response["results"]:
            self.assertIn("text", result)
            self.assertIn("metadata", result)
            self.assertIn("score", result)  # Relevance score
            
            # Verify metadata format
            self.assertIn("source", result["metadata"])
            self.assertIn("id", result["metadata"])
        
        # Clean up
        delete_response, _ = self.api_request("POST", "/delete", {
            "filter": {"source": "frontend-test", "timestamp": timestamp}
        })
        self.assertTrue(delete_response["success"])
    
    def test_version_info_for_frontend(self):
        """Test version information endpoint used by the frontend."""
        # Version info might be used by the frontend to check compatibility
        response, _ = self.api_request("GET", "/")
        self.assertIn("version", response)
    
    def test_json_serialization(self):
        """
        Test that all responses can be properly JSON serialized.
        
        This is important for frontend integration as the frontend will
        need to parse all responses as JSON.
        """
        # Test several endpoints and verify all responses are JSON serializable
        endpoints = [
            ("GET", "/"),
            ("GET", "/health/ping"),
            ("GET", "/gpu/info"),
            ("POST", "/embeddings", {"texts": ["Test text"]}),
        ]
        
        for method, path, *args in endpoints:
            data = args[0] if args else None
            response, _ = self.api_request(method, path, data)
            
            # Verify we can serialize to JSON and back
            try:
                json_string = json.dumps(response)
                parsed_back = json.loads(json_string)
                self.assertEqual(response, parsed_back)
            except Exception as e:
                self.fail(f"Failed to JSON serialize response from {method} {path}: {str(e)}")
    
    def test_error_response_format(self):
        """
        Test that error responses have a consistent format that the frontend can handle.
        """
        # Test with an invalid request to various endpoints
        test_cases = [
            # Endpoint, data, expected status
            ("/query", {}, 422),  # Missing required parameter
            ("/texts", {"texts": []}, 422),  # Empty texts array
            ("/embeddings", {"texts": []}, 422),  # Empty texts array
            ("/query/vector", {"embedding": [0.1]}, 422),  # Invalid embedding
        ]
        
        for path, data, expected_status in test_cases:
            response, status = self.api_request(
                "POST", path, data, 
                expected_status=expected_status
            )
            
            # Verify error response format is consistent
            self.assertIn("detail", response)
            
            # Check if it's our enhanced error format
            detail = response["detail"]
            if isinstance(detail, dict):
                # This should be our enhanced format
                self.assertIn("error", detail)
                self.assertIn("message", detail)
                
                # Context may be optional but is preferred
                if "context" in detail:
                    self.assertIsInstance(detail["context"], dict)


if __name__ == "__main__":
    import requests  # Import here to avoid issues with mock patching
    unittest.main()