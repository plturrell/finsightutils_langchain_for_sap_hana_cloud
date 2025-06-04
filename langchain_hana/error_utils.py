"""
Error handling utilities for the SAP HANA Cloud integration.

This module provides context-aware error handling for operations against SAP HANA Cloud,
including intelligent error interpretation and suggested actions for various error scenarios.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Pattern, Tuple, Union

from hdbcli import dbapi

# Define patterns for common SQL error messages
SQL_ERROR_PATTERNS = {
    # Connection errors
    "connection_failed": re.compile(r"(connection to the server has been lost|cannot connect to .+|network error|connection refused)", re.IGNORECASE),
    "timeout": re.compile(r"(connection timed out|timeout expired|operation timed out)", re.IGNORECASE),
    "auth_error": re.compile(r"(invalid username or password|authentication failed|not authorized)", re.IGNORECASE),
    
    # Permission errors
    "insufficient_privileges": re.compile(r"(insufficient privilege|not authorized to|permission denied)", re.IGNORECASE),
    
    # Resource errors
    "out_of_memory": re.compile(r"(out of memory|memory limit exceeded|not enough memory)", re.IGNORECASE),
    "resource_limit": re.compile(r"(resource exhausted|resource limit exceeded|too many connections)", re.IGNORECASE),
    
    # Table and column errors
    "table_not_found": re.compile(r"(table .+ not found|relation .+ does not exist|invalid table name)", re.IGNORECASE),
    "column_not_found": re.compile(r"(column .+ not found|invalid column name|no such column)", re.IGNORECASE),
    "datatype_mismatch": re.compile(r"(data type mismatch|incompatible data types|cannot convert|invalid data format)", re.IGNORECASE),
    
    # Vector-specific errors
    "invalid_vector_dimension": re.compile(r"(invalid vector dimension|vector length mismatch|vector dimension .+ not matching|vector size mismatch)", re.IGNORECASE),
    "vector_feature_unavailable": re.compile(r"(vector .+ not available|unsupported vector type|VECTOR_EMBEDDING not installed)", re.IGNORECASE),
    "embedding_model_error": re.compile(r"(embedding model .+ not found|invalid model|model not available|invalid embedding configuration)", re.IGNORECASE),
    
    # Syntax errors
    "syntax_error": re.compile(r"(syntax error|parse error|unexpected token|syntax not supported)", re.IGNORECASE),
    
    # Index errors
    "index_error": re.compile(r"(index .+ not found|invalid index|failed to create index)", re.IGNORECASE),
    "hnsw_error": re.compile(r"(HNSW .+ failed|invalid HNSW parameter|HNSW .+ error)", re.IGNORECASE),
    
    # Transaction errors
    "transaction_error": re.compile(r"(transaction aborted|deadlock detected|lock timeout|serialization failure)", re.IGNORECASE),
    
    # Constraint violations
    "constraint_violation": re.compile(r"(constraint violation|unique constraint|duplicate key|check constraint)", re.IGNORECASE),
    
    # Database limit errors
    "limit_exceeded": re.compile(r"(limit exceeded|too many|maximum .+ exceeded|value too large)", re.IGNORECASE),
}

# Define operation-specific context for errors
OPERATION_CONTEXT = {
    "connection": {
        "description": "Connecting to SAP HANA Cloud database",
        "common_issues": [
            "Network connectivity problems",
            "Incorrect connection parameters",
            "Authentication issues",
            "Firewall restrictions"
        ],
        "suggested_actions": [
            "Verify connection parameters (address, port, username, password)",
            "Check network connectivity to the database server",
            "Ensure your IP address is allowed by the database firewall",
            "Verify that the SAP HANA Cloud instance is running"
        ]
    },
    "table_creation": {
        "description": "Creating vector table in SAP HANA Cloud",
        "common_issues": [
            "Insufficient privileges",
            "Invalid table name or column specifications",
            "Resource limitations",
            "Schema conflicts"
        ],
        "suggested_actions": [
            "Verify that you have CREATE TABLE privileges in the schema",
            "Check table and column naming for invalid characters",
            "Ensure vector column types (REAL_VECTOR, HALF_VECTOR) are supported in your HANA version",
            "Check for existing tables with the same name"
        ]
    },
    "embedding_generation": {
        "description": "Generating embeddings using SAP HANA's internal embedding function",
        "common_issues": [
            "Invalid embedding model ID",
            "VECTOR_EMBEDDING function not available",
            "Text input format issues",
            "Resource limitations"
        ],
        "suggested_actions": [
            "Verify that the embedding model ID is valid (e.g., 'SAP_NEB.20240715')",
            "Check that VECTOR_EMBEDDING function is available in your SAP HANA version",
            "Ensure input text is properly formatted and encoded",
            "Reduce batch size for large document collections"
        ]
    },
    "similarity_search": {
        "description": "Performing vector similarity search",
        "common_issues": [
            "Empty vector table or no matching results",
            "Filter criteria issues",
            "Performance problems with large datasets",
            "Vector dimension mismatch"
        ],
        "suggested_actions": [
            "Verify that documents have been added to the vector store",
            "Check filter criteria syntax and values",
            "Create an HNSW index to improve search performance",
            "Ensure query vector dimensions match document vector dimensions"
        ]
    },
    "add_texts": {
        "description": "Adding documents to vector store",
        "common_issues": [
            "Large batch sizes causing timeouts or memory issues",
            "Invalid metadata format",
            "Schema mismatches",
            "Embedding generation failures"
        ],
        "suggested_actions": [
            "Reduce batch size when adding documents",
            "Ensure metadata keys contain only alphanumeric characters and underscores",
            "Verify that the table schema matches expected columns",
            "Check embedding model configuration"
        ]
    },
    "mmr_search": {
        "description": "Performing maximal marginal relevance search",
        "common_issues": [
            "Fetch_k parameter too large",
            "Resource limitations during processing",
            "Empty result set",
            "Algorithm optimization issues"
        ],
        "suggested_actions": [
            "Reduce the fetch_k parameter value",
            "Create an HNSW index to improve initial retrieval performance",
            "Verify that documents exist in the vector store",
            "Adjust lambda_mult parameter for better diversity/relevance balance"
        ]
    },
    "delete": {
        "description": "Deleting documents from vector store",
        "common_issues": [
            "Invalid filter criteria",
            "No matching documents to delete",
            "Transaction or lock conflicts",
            "Insufficient privileges"
        ],
        "suggested_actions": [
            "Check filter criteria syntax and values",
            "Ensure your user has DELETE privileges on the table",
            "Use more specific filter criteria to target the correct documents",
            "Handle concurrent access with appropriate transaction management"
        ]
    },
    "index_creation": {
        "description": "Creating vector search index",
        "common_issues": [
            "Invalid index parameters",
            "Resource limitations",
            "Insufficient privileges",
            "Existing index with the same name"
        ],
        "suggested_actions": [
            "Check that index parameters are within valid ranges",
            "Ensure sufficient database resources for index creation",
            "Verify you have CREATE INDEX privileges",
            "Check for existing indexes with the same name"
        ]
    }
}

def identify_error_type(error_message: str) -> str:
    """
    Identify the type of error based on the error message pattern.
    
    Args:
        error_message: The error message string to analyze
        
    Returns:
        str: The identified error type or "unknown_error" if no pattern matches
    """
    for error_type, pattern in SQL_ERROR_PATTERNS.items():
        if pattern.search(error_message):
            return error_type
    
    return "unknown_error"

def get_error_suggestions(error_type: str, operation_type: str) -> Dict[str, Any]:
    """
    Get operation-specific error context and suggestions based on error type.
    
    Args:
        error_type: The type of error identified
        operation_type: The type of operation being performed
    
    Returns:
        Dict containing context, common issues, and suggested actions
    """
    # Get operation context
    context = OPERATION_CONTEXT.get(operation_type, {
        "description": f"Performing {operation_type} operation",
        "common_issues": ["Unknown issue"],
        "suggested_actions": ["Check SAP HANA Cloud documentation"]
    })
    
    # Add error-specific information
    error_info = {
        "error_type": error_type,
        "operation": context["description"],
    }
    
    # Customize suggestions based on error type
    if error_type == "connection_failed":
        error_info["suggestion"] = "Check network connectivity and connection parameters"
    elif error_type == "auth_error":
        error_info["suggestion"] = "Verify username and password"
    elif error_type == "insufficient_privileges":
        error_info["suggestion"] = "Request necessary permissions from database administrator"
    elif error_type == "table_not_found":
        error_info["suggestion"] = "Verify table name and schema"
    elif error_type == "column_not_found":
        error_info["suggestion"] = "Verify column names and table structure"
    elif error_type == "datatype_mismatch":
        error_info["suggestion"] = "Check data types and conversion compatibility"
    elif error_type == "invalid_vector_dimension":
        error_info["suggestion"] = "Ensure vector dimensions match between query and documents"
    elif error_type == "vector_feature_unavailable":
        error_info["suggestion"] = "Upgrade to a SAP HANA Cloud version that supports vector features"
    elif error_type == "embedding_model_error":
        error_info["suggestion"] = "Verify embedding model ID and availability"
    elif error_type == "syntax_error":
        error_info["suggestion"] = "Check SQL syntax for errors"
    elif error_type == "out_of_memory":
        error_info["suggestion"] = "Reduce batch size or query complexity"
    elif error_type == "resource_limit":
        error_info["suggestion"] = "Optimize operations or request more resources"
    elif error_type == "transaction_error":
        error_info["suggestion"] = "Retry the operation or optimize transaction management"
    elif error_type == "index_error":
        error_info["suggestion"] = "Check index name and creation parameters"
    elif error_type == "hnsw_error":
        error_info["suggestion"] = "Verify HNSW parameters are within valid ranges"
    else:
        error_info["suggestion"] = "Consult SAP HANA Cloud documentation for more information"
    
    # Include all context information
    error_info["common_issues"] = context["common_issues"]
    error_info["suggested_actions"] = context["suggested_actions"]
    
    return error_info

def create_context_aware_error(
    error: Union[str, Exception], 
    operation_type: str, 
    additional_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create a context-aware error with useful suggestions based on the error and operation type.
    
    Args:
        error: The original error message or exception
        operation_type: The type of operation being performed
        additional_context: Optional additional context information
        
    Returns:
        Dict containing error details with context-aware suggestions
    """
    # Extract error message from different types of inputs
    if isinstance(error, dbapi.Error):
        error_message = str(error)
    elif isinstance(error, Exception):
        error_message = str(error)
    else:
        error_message = str(error)
    
    # Identify error type
    error_type = identify_error_type(error_message)
    
    # Get suggestions
    error_context = get_error_suggestions(error_type, operation_type)
    
    # Build complete error response
    error_response = {
        "error": error_message,
        "error_type": error_type,
        "operation": operation_type,
        "context": error_context,
    }
    
    # Add any additional context
    if additional_context:
        error_response.update(additional_context)
    
    return error_response

def handle_database_error(
    exception: Exception, 
    operation_type: str, 
    additional_context: Optional[Dict[str, Any]] = None,
    raise_exception: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Handle a database error with context-aware information.
    
    Args:
        exception: The exception that occurred
        operation_type: The type of operation being performed
        additional_context: Optional additional context information
        raise_exception: Whether to raise the exception after handling
        
    Returns:
        Dict containing error details if raise_exception is False, otherwise None
        
    Raises:
        The original exception with enhanced context information if raise_exception is True
    """
    # Create context-aware error
    error_response = create_context_aware_error(
        exception, 
        operation_type, 
        additional_context
    )
    
    if raise_exception:
        # Enhance the exception message with context
        enhanced_message = (
            f"{str(exception)}\n\n"
            f"Operation: {error_response['context']['operation']}\n"
            f"Suggestion: {error_response['context']['suggestion']}\n"
            f"Actions: {', '.join(error_response['context']['suggested_actions'][:2])}"
        )
        
        # Raise the same type of exception with enhanced message
        if isinstance(exception, dbapi.Error):
            raise type(exception)(enhanced_message) from exception
        else:
            raise type(exception)(enhanced_message) from exception
    
    return error_response