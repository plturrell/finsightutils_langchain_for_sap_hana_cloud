"""
Tests for error handling utilities.

This module contains unit tests for the error handling utilities provided
by the langchain_hana.error_utils module.
"""

import unittest
from unittest.mock import Mock, patch

from hdbcli import dbapi

from langchain_hana.error_utils import (
    identify_error_type,
    get_error_suggestions,
    create_context_aware_error,
    handle_database_error
)

class TestErrorTypeIdentification(unittest.TestCase):
    """Tests for error type identification."""
    
    def test_connection_error_identification(self):
        """Test identification of connection-related errors."""
        error_msg = "connection to the server has been lost during query execution"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "connection_failed")
        
        error_msg = "connection timed out after 30 seconds"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "timeout")
        
        error_msg = "authentication failed: invalid username or password"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "auth_error")
    
    def test_permission_error_identification(self):
        """Test identification of permission-related errors."""
        error_msg = "insufficient privilege: user does not have CREATE ANY permission"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "insufficient_privileges")
    
    def test_resource_error_identification(self):
        """Test identification of resource-related errors."""
        error_msg = "out of memory error occurred during operation"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "out_of_memory")
        
        error_msg = "resource limit exceeded: too many concurrent connections"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "resource_limit")
    
    def test_table_column_error_identification(self):
        """Test identification of table and column related errors."""
        error_msg = "table 'VECTOR_STORE' not found"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "table_not_found")
        
        error_msg = "column 'EMBEDDING' not found"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "column_not_found")
        
        error_msg = "data type mismatch: cannot convert VARCHAR to REAL_VECTOR"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "datatype_mismatch")
    
    def test_vector_specific_error_identification(self):
        """Test identification of vector-specific errors."""
        error_msg = "invalid vector dimension: expected 768 but got 384"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "invalid_vector_dimension")
        
        error_msg = "vector feature unsupported: VECTOR_EMBEDDING not installed"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "vector_feature_unavailable")
        
        error_msg = "embedding model 'unknown_model' not found"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "embedding_model_error")
    
    def test_unknown_error_identification(self):
        """Test identification of unknown errors."""
        error_msg = "some completely unknown error message"
        error_type = identify_error_type(error_msg)
        self.assertEqual(error_type, "unknown_error")


class TestErrorSuggestions(unittest.TestCase):
    """Tests for error suggestion generation."""
    
    def test_connection_error_suggestions(self):
        """Test suggestions for connection errors."""
        suggestions = get_error_suggestions("connection_failed", "connection")
        self.assertEqual(suggestions["error_type"], "connection_failed")
        self.assertEqual(suggestions["suggestion"], "Check network connectivity and connection parameters")
        self.assertIn("Network connectivity problems", suggestions["common_issues"])
    
    def test_table_creation_error_suggestions(self):
        """Test suggestions for table creation errors."""
        suggestions = get_error_suggestions("insufficient_privileges", "table_creation")
        self.assertEqual(suggestions["error_type"], "insufficient_privileges")
        self.assertEqual(suggestions["suggestion"], "Request necessary permissions from database administrator")
        self.assertIn("Verify that you have CREATE TABLE privileges in the schema", suggestions["suggested_actions"])
    
    def test_vector_operation_error_suggestions(self):
        """Test suggestions for vector operation errors."""
        suggestions = get_error_suggestions("invalid_vector_dimension", "similarity_search")
        self.assertEqual(suggestions["error_type"], "invalid_vector_dimension")
        self.assertEqual(suggestions["suggestion"], "Ensure vector dimensions match between query and documents")
        
        suggestions = get_error_suggestions("embedding_model_error", "embedding_generation")
        self.assertEqual(suggestions["error_type"], "embedding_model_error")
        self.assertEqual(suggestions["suggestion"], "Verify embedding model ID and availability")
    
    def test_unknown_operation_suggestions(self):
        """Test suggestions for unknown operations."""
        suggestions = get_error_suggestions("syntax_error", "unknown_operation")
        self.assertEqual(suggestions["error_type"], "syntax_error")
        self.assertEqual(suggestions["suggestion"], "Check SQL syntax for errors")
        self.assertEqual(suggestions["operation"], "Performing unknown_operation operation")


class TestContextAwareError(unittest.TestCase):
    """Tests for context-aware error creation."""
    
    def test_create_context_aware_error_with_string(self):
        """Test creating context-aware error from a string."""
        error = "table 'VECTOR_STORE' not found"
        result = create_context_aware_error(error, "similarity_search")
        
        self.assertEqual(result["error"], error)
        self.assertEqual(result["error_type"], "table_not_found")
        self.assertEqual(result["operation"], "similarity_search")
        self.assertIn("context", result)
        self.assertIn("suggestion", result["context"])
    
    def test_create_context_aware_error_with_exception(self):
        """Test creating context-aware error from an exception."""
        # Create a mock dbapi.Error
        db_error = Mock(spec=dbapi.Error)
        db_error.__str__.return_value = "insufficient privilege: user cannot create table"
        
        result = create_context_aware_error(db_error, "table_creation")
        
        self.assertEqual(result["error"], "insufficient privilege: user cannot create table")
        self.assertEqual(result["error_type"], "insufficient_privileges")
        self.assertEqual(result["operation"], "table_creation")
    
    def test_create_context_aware_error_with_additional_context(self):
        """Test creating context-aware error with additional context."""
        error = "connection timed out after 30 seconds"
        additional_context = {
            "host": "hana-server.example.com",
            "port": 30015,
            "retry_count": 3
        }
        
        result = create_context_aware_error(error, "connection", additional_context)
        
        self.assertEqual(result["error"], error)
        self.assertEqual(result["error_type"], "timeout")
        self.assertEqual(result["host"], "hana-server.example.com")
        self.assertEqual(result["port"], 30015)
        self.assertEqual(result["retry_count"], 3)


class TestHandleDatabaseError(unittest.TestCase):
    """Tests for database error handling."""
    
    def test_handle_database_error_without_raising(self):
        """Test handling database error without raising."""
        error = Exception("column 'EMBEDDING' not found")
        result = handle_database_error(error, "similarity_search", raise_exception=False)
        
        self.assertEqual(result["error"], "column 'EMBEDDING' not found")
        self.assertEqual(result["error_type"], "column_not_found")
        self.assertEqual(result["operation"], "similarity_search")
    
    @patch('langchain_hana.error_utils.create_context_aware_error')
    def test_handle_database_error_with_raising(self, mock_create_error):
        """Test handling database error with raising."""
        # Set up the mock
        mock_create_error.return_value = {
            "error": "data type mismatch",
            "error_type": "datatype_mismatch",
            "operation": "add_texts",
            "context": {
                "operation": "Adding documents to vector store",
                "suggestion": "Check data types and conversion compatibility",
                "suggested_actions": ["Verify metadata types", "Check schema"]
            }
        }
        
        # Create a test exception
        error = ValueError("data type mismatch")
        
        # Test that the exception is raised with enhanced message
        with self.assertRaises(ValueError) as context:
            handle_database_error(error, "add_texts", raise_exception=True)
        
        # Check that the exception message is enhanced
        self.assertIn("data type mismatch", str(context.exception))
        self.assertIn("Adding documents to vector store", str(context.exception))
        self.assertIn("Check data types and conversion compatibility", str(context.exception))
    
    @patch('langchain_hana.error_utils.create_context_aware_error')
    def test_handle_database_error_with_dbapi_error(self, mock_create_error):
        """Test handling dbapi.Error with raising."""
        # Set up the mock
        mock_create_error.return_value = {
            "error": "connection to the server has been lost",
            "error_type": "connection_failed",
            "operation": "connection",
            "context": {
                "operation": "Connecting to SAP HANA Cloud database",
                "suggestion": "Check network connectivity and connection parameters",
                "suggested_actions": ["Verify connection parameters", "Check network connectivity"]
            }
        }
        
        # Create a test dbapi exception
        db_error = Mock(spec=dbapi.Error)
        db_error.__str__.return_value = "connection to the server has been lost"
        
        # Test that the exception is raised with enhanced message
        with self.assertRaises(Exception) as context:
            handle_database_error(db_error, "connection", raise_exception=True)
        
        # Check that the exception message is enhanced
        self.assertIn("connection to the server has been lost", str(context.exception))
        self.assertIn("Connecting to SAP HANA Cloud database", str(context.exception))
        self.assertIn("Check network connectivity and connection parameters", str(context.exception))


if __name__ == "__main__":
    unittest.main()