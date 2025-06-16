"""
Unified error handling system for SAP HANA Cloud integration.

This module provides a consolidated error handling system that combines functionality 
from both the API and core library error handlers, eliminating duplication.
"""

import re
import logging
import json
from typing import Dict, Any, Optional, List, Union, Pattern, Tuple
from fastapi import HTTPException

from hdbcli import dbapi

logger = logging.getLogger(__name__)

# Common SQL error patterns shared across the application
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

# Unified operation context information
OPERATION_CONTEXTS = {
    "connection": {
        "name": "Database Connection",
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
        "name": "Table Creation",
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
        "name": "Embedding Generation",
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
    "vector_search": {
        "name": "Vector Search",
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
    "data_insertion": {
        "name": "Data Insertion",
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
        "name": "MMR Search",
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
        "name": "Document Deletion",
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
        "name": "Index Creation", 
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
    },
    "flow_execution": {
        "name": "Flow Execution",
        "description": "Executing workflow operations",
        "common_issues": [
            "Missing required nodes in the flow",
            "Invalid node connections",
            "Database connection issues",
            "Vector embedding errors"
        ],
        "suggested_actions": [
            "Check that your flow contains all required nodes (connection, embedding, vector store, query)",
            "Verify that nodes are properly connected in the flow",
            "Ensure your database credentials are correct",
            "Check embedding model configuration"
        ]
    },
    "vector_visualization": {
        "name": "Vector Visualization",
        "description": "Visualizing vector embeddings",
        "common_issues": [
            "Too many vectors requested for visualization",
            "Dimensionality reduction errors",
            "Invalid filter parameters",
            "Timeout during vector retrieval"
        ],
        "suggested_actions": [
            "Reduce the number of vectors to visualize",
            "Apply more specific filters to reduce the dataset size",
            "Check for invalid filter syntax",
            "Verify that the table contains valid vector data"
        ]
    }
}

# Mapping of error types to suggestions
ERROR_TYPE_SUGGESTIONS = {
    "connection_failed": "Check network connectivity and connection parameters",
    "auth_error": "Verify username and password",
    "insufficient_privileges": "Request necessary permissions from database administrator",
    "table_not_found": "Verify table name and schema",
    "column_not_found": "Verify column names and table structure",
    "datatype_mismatch": "Check data types and conversion compatibility",
    "invalid_vector_dimension": "Ensure vector dimensions match between query and documents",
    "vector_feature_unavailable": "Upgrade to a SAP HANA Cloud version that supports vector features",
    "embedding_model_error": "Verify embedding model ID and availability",
    "syntax_error": "Check SQL syntax for errors",
    "out_of_memory": "Reduce batch size or query complexity",
    "resource_limit": "Optimize operations or request more resources",
    "transaction_error": "Retry the operation or optimize transaction management",
    "index_error": "Check index name and creation parameters",
    "hnsw_error": "Verify HNSW parameters are within valid ranges",
    "unknown_error": "Consult SAP HANA Cloud documentation for more information"
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

def create_error_context(
    error_type: str, 
    operation_type: str,
    error_message: str
) -> Dict[str, Any]:
    """
    Create context-aware error information based on error type and operation.
    
    Args:
        error_type: The identified error type
        operation_type: The type of operation being performed
        error_message: The original error message
        
    Returns:
        Dict containing context-aware error information
    """
    # Get operation context
    context = OPERATION_CONTEXTS.get(operation_type, {
        "name": operation_type.replace("_", " ").title(),
        "description": f"Performing {operation_type} operation",
        "common_issues": ["Unknown issue"],
        "suggested_actions": ["Check SAP HANA Cloud documentation"]
    })
    
    # Get suggestion for this error type
    suggestion = ERROR_TYPE_SUGGESTIONS.get(error_type, "Check documentation for more information")
    
    # Build error context
    error_context = {
        "error_type": error_type,
        "operation": context["description"],
        "operation_name": context["name"],
        "message": f"An error occurred during {context['name']}: {error_message}",
        "suggestion": suggestion,
        "common_issues": context.get("common_issues", []),
        "suggested_actions": context.get("suggested_actions", [])
    }
    
    return error_context

def handle_error(
    error: Union[str, Exception],
    operation_type: str,
    additional_context: Optional[Dict[str, Any]] = None,
    status_code: int = 500,
    raise_exception: bool = True
) -> Union[Dict[str, Any], HTTPException]:
    """
    Unified error handler that works for both API and core library contexts.
    
    Args:
        error: The original error message or exception
        operation_type: The type of operation being performed
        additional_context: Optional additional context information
        status_code: HTTP status code for API errors
        raise_exception: Whether to raise an exception (for core library) or 
                        return HTTPException (for API)
        
    Returns:
        Dict or HTTPException depending on the context
        
    Raises:
        Exception with enhanced context if raise_exception is True
    """
    # Extract error message
    if isinstance(error, dbapi.Error):
        error_message = str(error)
    elif isinstance(error, Exception):
        error_message = str(error)
    else:
        error_message = str(error)
    
    # Identify error type
    error_type = identify_error_type(error_message)
    
    # Create error context
    error_context = create_error_context(error_type, operation_type, error_message)
    
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
    
    # Log the error with full context
    log_message = f"Error in {error_context['operation_name']}: {error_message}"
    logger.error(log_message)
    
    if raise_exception:
        if isinstance(error, Exception):
            # Create enhanced error message
            enhanced_message = (
                f"{error_message}\n\n"
                f"Operation: {error_context['operation']}\n"
                f"Suggestion: {error_context['suggestion']}\n"
                f"Actions: {', '.join(error_context['suggested_actions'][:2])}"
            )
            
            # Raise the same type of exception with enhanced message
            if isinstance(error, dbapi.Error):
                raise type(error)(enhanced_message) from error
            else:
                raise type(error)(enhanced_message) from error
        else:
            # If error is just a string, raise a generic exception
            raise Exception(error_message)
    else:
        # For API use, return HTTPException
        return HTTPException(
            status_code=status_code,
            detail=error_response
        )

# Specialized handlers for specific operations
def handle_vector_search_error(
    error: Exception, 
    query_info: Optional[Dict[str, Any]] = None,
    status_code: int = 500
) -> HTTPException:
    """
    Specialized handler for vector search errors.
    
    Args:
        error: The exception that occurred
        query_info: Optional information about the query
        status_code: HTTP status code
        
    Returns:
        HTTPException with detailed error information
    """
    error_message = str(error)
    additional_context = {"query_info": query_info} if query_info else None
    
    # Check for specific vector search errors
    if "dimension mismatch" in error_message.lower():
        return handle_error(
            error,
            "vector_search",
            additional_context={
                "message": "The query vector dimension doesn't match the stored vectors.",
                "suggestions": [
                    "Ensure your query vector has the same dimension as the vectors in the database.",
                    "Check the embedding model configuration.",
                    "Verify that you're using the correct embedding function."
                ],
                **(additional_context or {})
            },
            status_code=400,
            raise_exception=False
        )
    
    return handle_error(
        error, 
        "vector_search", 
        additional_context=additional_context,
        status_code=status_code,
        raise_exception=False
    )

def handle_data_operation_error(
    error: Exception,
    operation_type: str,
    data_info: Optional[Dict[str, Any]] = None,
    status_code: int = 500
) -> HTTPException:
    """
    Handler for data operations (insertion, update, delete).
    
    Args:
        error: The exception that occurred
        operation_type: Type of data operation
        data_info: Optional information about the data
        status_code: HTTP status code
        
    Returns:
        HTTPException with detailed error information
    """
    error_message = str(error)
    additional_context = {"data_info": data_info} if data_info else None
    
    return handle_error(
        error, 
        operation_type, 
        additional_context=additional_context,
        status_code=status_code,
        raise_exception=False
    )

# Core library error handling function
def handle_database_error(
    exception: Exception, 
    operation_type: str, 
    additional_context: Optional[Dict[str, Any]] = None,
    raise_exception: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Handle a database error with context-aware information.
    Compatible with the existing langchain_hana.error_utils.handle_database_error.
    
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
    if raise_exception:
        handle_error(exception, operation_type, additional_context, raise_exception=True)
        return None  # This line is never reached because an exception is raised
    else:
        result = handle_error(exception, operation_type, additional_context, raise_exception=False)
        # Convert HTTPException to dict format expected by original API
        if isinstance(result, HTTPException):
            return result.detail
        return result