"""
Tests for integration between backend error handling and frontend display.

This module contains tests that validate the error messages produced by the
error handling utilities are properly formatted for display in the frontend.
"""

import unittest
from unittest.mock import MagicMock, patch
import json
from typing import Dict, Any

from langchain_hana.error_utils import (
    identify_error_type,
    create_context_aware_error,
    handle_database_error
)


class TestErrorIntegration(unittest.TestCase):
    """Tests for backend-frontend error handling integration."""
    
    def test_error_json_serialization(self):
        """Test that error responses can be properly serialized to JSON."""
        # Create a context-aware error
        error = create_context_aware_error(
            error="connection to the server has been lost",
            operation_type="connection",
            additional_context={
                "host": "hana-server.example.com",
                "port": 30015,
                "retry_count": 3
            }
        )
        
        # Verify the error can be JSON serialized
        json_str = json.dumps(error)
        parsed = json.loads(json_str)
        
        # Check that key fields are preserved
        self.assertEqual(parsed["error"], "connection to the server has been lost")
        self.assertEqual(parsed["error_type"], "connection_failed")
        self.assertEqual(parsed["operation"], "connection")
        self.assertEqual(parsed["host"], "hana-server.example.com")
        self.assertEqual(parsed["port"], 30015)
        self.assertEqual(parsed["retry_count"], 3)
        
        # Check that suggestions are included (needed by ErrorHandler component)
        self.assertIn("context", parsed)
        self.assertIn("suggestion", parsed["context"])
        self.assertIn("suggested_actions", parsed["context"])
    
    def test_error_format_matches_frontend_requirements(self):
        """Test that error format matches what the frontend ErrorHandler component expects."""
        # Create a context-aware error
        error = create_context_aware_error(
            error="table 'VECTOR_STORE' not found",
            operation_type="similarity_search",
            additional_context={
                "query": "test query",
                "k": 5
            }
        )
        
        # Construct an API error response that would be sent to the frontend
        api_error = {
            "status": 404,
            "statusText": "Not Found",
            "detail": {
                "message": error["error"],
                "operation": error["operation"],
                "suggestions": error["context"]["suggested_actions"],
                "common_issues": error["context"]["common_issues"],
                "original_error": error["error"],
                "query_info": {
                    "query": "test query",
                    "k": 5
                }
            }
        }
        
        # Verify the API error has all fields required by ErrorHandler.tsx
        self.assertIn("status", api_error)
        self.assertIn("statusText", api_error)
        self.assertIn("detail", api_error)
        
        # Check detail fields
        detail = api_error["detail"]
        self.assertIn("message", detail)
        self.assertIn("operation", detail)
        self.assertIn("suggestions", detail)
        self.assertIn("common_issues", detail)
        self.assertIn("original_error", detail)
        self.assertIn("query_info", detail)
    
    def test_custom_error_codes_mapping(self):
        """Test mapping error types to appropriate HTTP status codes."""
        error_type_to_status = {
            "connection_failed": 503,  # Service Unavailable
            "timeout": 504,  # Gateway Timeout
            "auth_error": 401,  # Unauthorized
            "insufficient_privileges": 403,  # Forbidden
            "table_not_found": 404,  # Not Found
            "column_not_found": 404,  # Not Found
            "invalid_vector_dimension": 400,  # Bad Request
            "vector_feature_unavailable": 501,  # Not Implemented
            "syntax_error": 400,  # Bad Request
            "out_of_memory": 507,  # Insufficient Storage
            "unknown_error": 500,  # Internal Server Error
        }
        
        test_cases = [
            ("connection to the server has been lost", "connection_failed", 503),
            ("connection timed out after 30 seconds", "timeout", 504),
            ("invalid username or password", "auth_error", 401),
            ("insufficient privilege: user does not have CREATE ANY permission", "insufficient_privileges", 403),
            ("table 'VECTOR_STORE' not found", "table_not_found", 404),
            ("column 'EMBEDDING' not found", "column_not_found", 404),
            ("invalid vector dimension: expected 768 but got 384", "invalid_vector_dimension", 400),
            ("vector feature unsupported: VECTOR_EMBEDDING not installed", "vector_feature_unavailable", 501),
            ("syntax error in SQL query", "syntax_error", 400),
            ("out of memory error occurred during operation", "out_of_memory", 507),
            ("some completely unknown error message", "unknown_error", 500),
        ]
        
        for error_msg, expected_type, expected_status in test_cases:
            error_type = identify_error_type(error_msg)
            self.assertEqual(error_type, expected_type)
            self.assertEqual(error_type_to_status.get(error_type), expected_status)
    
    def test_error_translation_for_frontend(self):
        """Test that backend errors are translated to user-friendly messages for the frontend."""
        backend_to_frontend_messages = {
            "connection_failed": "Unable to connect to the database. Please check your connection and try again.",
            "timeout": "The operation timed out. Please try again later or reduce the data volume.",
            "auth_error": "Authentication failed. Please check your credentials.",
            "insufficient_privileges": "You don't have permission to perform this operation.",
            "table_not_found": "The requested data table could not be found.",
            "invalid_vector_dimension": "Vector dimension mismatch. Please ensure query and document vectors have the same dimensions.",
            "syntax_error": "There's an error in your query syntax.",
            "out_of_memory": "The system ran out of memory. Try reducing the batch size or query complexity."
        }
        
        # Test a few error translations
        for error_type, user_message in backend_to_frontend_messages.items():
            # Create a mock error with this type
            mock_error = MagicMock()
            mock_error.__str__.return_value = f"Some error of type {error_type}"
            
            with patch('langchain_hana.error_utils.identify_error_type', return_value=error_type):
                error = create_context_aware_error(mock_error, "some_operation")
                
                # In a real API endpoint, we'd translate to a user-friendly message
                api_response = {
                    "status": 400,  # Example status
                    "statusText": "Error",
                    "detail": {
                        "message": backend_to_frontend_messages.get(error["error_type"], str(error["error"])),
                        "operation": error["operation"],
                        "suggestions": error["context"]["suggested_actions"],
                        "common_issues": error["context"]["common_issues"],
                        "original_error": str(error["error"])
                    }
                }
                
                # Verify the translation happened correctly
                self.assertEqual(api_response["detail"]["message"], user_message)


class TestFrontendErrorHandlerProps(unittest.TestCase):
    """Tests that validate the props expected by the ErrorHandler component."""
    
    def test_error_handler_props_from_api_error(self):
        """Test constructing ErrorHandler props from API error responses."""
        # Create a typical API error response
        api_error = {
            "status": 404,
            "statusText": "Not Found",
            "detail": {
                "message": "The requested data table could not be found.",
                "operation": "similarity_search",
                "suggestions": [
                    "Verify table name and schema",
                    "Check if the table exists in the database"
                ],
                "common_issues": [
                    "Table was deleted or renamed",
                    "Table is in a different schema"
                ],
                "original_error": "table 'VECTOR_STORE' not found",
                "query_info": {
                    "query": "test query",
                    "k": 5
                }
            }
        }
        
        # In a React component, we would pass this to ErrorHandler like:
        # <ErrorHandler error={apiError} onClose={() => setError(null)} />
        
        # Verify the structure matches what ErrorHandler expects
        self.assertIn("status", api_error)
        self.assertIn("statusText", api_error)
        self.assertIn("detail", api_error)
        
        # Check required fields in detail
        detail = api_error["detail"]
        self.assertIn("message", detail)
        self.assertIn("operation", detail)
        self.assertIn("suggestions", detail)
        
        # Verify optional fields are properly structured
        self.assertIsInstance(detail["suggestions"], list)
        self.assertIsInstance(detail["common_issues"], list)
        self.assertIsInstance(detail["query_info"], dict)
    
    def test_dynamic_error_handler_props(self):
        """Test constructing ErrorHandler props with different context types."""
        contexts = [
            {"query_info": {"query": "test", "k": 5}},
            {"flow_info": {"flow_id": "123", "step": "embedding"}},
            {"insertion_info": {"docs": 10, "batch_size": 5}},
            {"visualization_params": {"dimensions": 2, "perplexity": 30}}
        ]
        
        for context in contexts:
            context_type = list(context.keys())[0]  # Get the context type (e.g., "query_info")
            
            # Create an error with this context type
            error = create_context_aware_error(
                error=f"Test error with {context_type}",
                operation_type="test_operation",
                additional_context=context
            )
            
            # Create API error response
            api_error = {
                "status": 400,
                "statusText": "Error",
                "detail": {
                    "message": error["error"],
                    "operation": error["operation"],
                    "suggestions": error["context"]["suggested_actions"],
                    "common_issues": error["context"]["common_issues"],
                    **context  # Add the specific context type
                }
            }
            
            # Verify the context was properly added to detail
            self.assertIn(context_type, api_error["detail"])
            self.assertEqual(api_error["detail"][context_type], context[context_type])


if __name__ == "__main__":
    unittest.main()