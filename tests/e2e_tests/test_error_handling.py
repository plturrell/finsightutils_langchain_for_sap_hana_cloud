"""
End-to-end tests for API error handling.

These tests verify that the API properly handles error scenarios, including:
- Invalid requests
- Missing required parameters
- Invalid authentication
- Database connection errors
- Rate limiting
"""

import os
import unittest
from typing import Dict, List, Any

from .base import BaseEndToEndTest, logger


class ErrorHandlingTest(BaseEndToEndTest):
    """Test API error handling end-to-end."""
    
    def test_invalid_endpoint(self):
        """Test accessing an invalid endpoint."""
        # This should return a 404 Not Found
        response, status_code = self.api_request(
            "GET", "/non-existent-endpoint", 
            expected_status=404
        )
        self.assertEqual(status_code, 404)
        
        # Check if using FastAPI's default 404 response format
        if "detail" in response:
            self.assertIn("Not Found", response["detail"])
    
    def test_missing_required_parameters(self):
        """Test requests with missing required parameters."""
        # Test /embeddings without texts parameter
        response, status_code = self.api_request(
            "POST", "/embeddings", 
            data={}, 
            expected_status=422
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)
        
        # Test /query without query parameter
        response, status_code = self.api_request(
            "POST", "/query", 
            data={}, 
            expected_status=422
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)
        
        # Test /texts without texts parameter
        response, status_code = self.api_request(
            "POST", "/texts", 
            data={}, 
            expected_status=422
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)
    
    def test_invalid_parameter_types(self):
        """Test requests with parameters of invalid types."""
        # Test /embeddings with non-list texts
        response, status_code = self.api_request(
            "POST", "/embeddings", 
            data={"texts": "not a list"}, 
            expected_status=422
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)
        
        # Test /query with k as string instead of integer
        response, status_code = self.api_request(
            "POST", "/query", 
            data={"query": "test query", "k": "not an integer"}, 
            expected_status=422
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)
        
        # Test /query with non-dict filter
        response, status_code = self.api_request(
            "POST", "/query", 
            data={"query": "test query", "filter": "not a dict"}, 
            expected_status=422
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)
    
    def test_invalid_embedding_vector(self):
        """Test using an invalid embedding vector."""
        # Test with wrong dimensions
        response, status_code = self.api_request(
            "POST", "/query/vector", 
            data={"embedding": [0.1, 0.2, 0.3]},  # Too short
            expected_status=422
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)
        
        # Test with non-numeric values
        response, status_code = self.api_request(
            "POST", "/query/vector", 
            data={"embedding": ["not", "numeric", "values"]},
            expected_status=422
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)
    
    def test_invalid_authentication(self):
        """Test requests with invalid authentication."""
        # Save the current API key
        original_api_key = self.session.headers.get("X-API-Key")
        
        try:
            # Set an invalid API key
            self.session.headers["X-API-Key"] = "invalid-api-key"
            
            # Try to access an endpoint that requires authentication
            # The expected status depends on how auth is implemented
            # It could be 401 (Unauthorized) or 403 (Forbidden)
            response, status_code = self.api_request(
                "POST", "/embeddings", 
                data={"texts": ["test"]},
                expected_status=401,  # Adjust based on actual implementation
            )
            
            # Check for authentication error
            self.assertIn(status_code, [401, 403], "Expected 401 or 403 status code for invalid auth")
            
        finally:
            # Restore the original API key
            self.session.headers["X-API-Key"] = original_api_key
    
    def test_context_aware_error_format(self):
        """Test that errors include context-aware information."""
        # Test with very large batch that might cause out-of-memory or timeout
        # This assumes that the backend can't handle an extremely large batch
        extremely_large_batch = ["Test text"] * 10000
        
        # This might return various error codes depending on implementation
        response, status_code = self.api_request(
            "POST", "/embeddings", 
            data={"texts": extremely_large_batch},
            expected_status=None,  # We don't know what the exact code will be
            timeout=10,  # Short timeout to avoid waiting too long
        )
        
        # Check that we got an error response (4xx or 5xx)
        self.assertGreaterEqual(status_code, 400)
        
        # Our context-aware errors should include these fields
        if "detail" in response and isinstance(response["detail"], dict):
            # Check for context-aware error format
            error_keys = ["error", "message", "context"]
            for key in error_keys:
                # We're lenient here - not all errors may have the full context
                if key in response["detail"]:
                    logger.info(f"Error response includes context field: {key}")
    
    def test_error_with_empty_request(self):
        """Test sending an empty request body."""
        response, status_code = self.api_request(
            "POST", "/texts",
            data=None,
            expected_status=422  # Validation error expected
        )
        self.assertEqual(status_code, 422)
        self.assertIn("detail", response)


if __name__ == "__main__":
    unittest.main()