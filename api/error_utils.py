"""Utilities for handling errors in a user-friendly way."""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from fastapi import HTTPException

logger = logging.getLogger(__name__)

# SQL error patterns and their user-friendly interpretations
SQL_ERROR_PATTERNS = {
    r"table .* does not exist": {
        "message": "The requested table doesn't exist in the database.",
        "suggestions": [
            "Check the table name for typos.",
            "Verify that the table has been created.",
            "Ensure you have the correct schema name if using schema.table notation."
        ]
    },
    r"column .* does not exist": {
        "message": "A column referenced in your query doesn't exist.",
        "suggestions": [
            "Check the column name for typos.",
            "Verify that the column exists in the table.",
            "Check if you need to create this column first."
        ]
    },
    r"permission denied": {
        "message": "You don't have permission to perform this operation.",
        "suggestions": [
            "Contact your database administrator for access.",
            "Check if you're using the correct user credentials.",
            "Verify that your user has the required privileges."
        ]
    },
    r"syntax error": {
        "message": "There's a syntax error in your SQL query.",
        "suggestions": [
            "Check for missing quotes, parentheses, or keywords.",
            "Verify that your SQL follows SAP HANA SQL syntax.",
            "Review the query structure for logical errors."
        ]
    },
    r"timeout expired": {
        "message": "The operation took too long and timed out.",
        "suggestions": [
            "Try again with a smaller dataset.",
            "Optimize your query to be more efficient.",
            "Consider increasing the timeout value if available."
        ]
    },
    r"connection .* closed": {
        "message": "The database connection was closed unexpectedly.",
        "suggestions": [
            "Check your network connection.",
            "Verify that the database server is running.",
            "Try reconnecting to the database."
        ]
    },
    r"deadlock detected": {
        "message": "A deadlock was detected in the database.",
        "suggestions": [
            "Try the operation again after a brief delay.",
            "Consider modifying your transaction to access resources in a consistent order.",
            "Review your application logic to avoid concurrent modification of the same resources."
        ]
    },
    r"out of memory": {
        "message": "The operation ran out of memory.",
        "suggestions": [
            "Reduce the size of your data set.",
            "Break your operation into smaller batches.",
            "Consider using server-side processing techniques."
        ]
    },
    r"invalid identifier": {
        "message": "The query contains an invalid identifier.",
        "suggestions": [
            "Check table and column names for typos.",
            "Ensure identifiers are properly quoted if they contain special characters.",
            "Verify that all referenced objects exist in the database."
        ]
    },
    r"unique constraint .* violated": {
        "message": "A unique constraint violation occurred.",
        "suggestions": [
            "Check for duplicate values in unique columns.",
            "Verify that you're not trying to insert a record that already exists.",
            "Use UPDATE instead of INSERT if the record already exists."
        ]
    },
    r"value too long for column": {
        "message": "The data is too large for the column.",
        "suggestions": [
            "Reduce the size of the data being inserted.",
            "Alter the column to accommodate larger data if appropriate.",
            "Consider using a different data type for the column."
        ]
    },
    r"not authorized .* on .* (table|column)": {
        "message": "You don't have permission to access this table or column.",
        "suggestions": [
            "Contact your database administrator for the required permissions.",
            "Check if you're using the correct user credentials.",
            "Verify that you're trying to access the correct table."
        ]
    },
    r"vector index .* does not exist": {
        "message": "The vector index doesn't exist.",
        "suggestions": [
            "Create the vector index before using it.",
            "Check the index name for typos.",
            "Verify that the index has been created correctly."
        ]
    },
    r"table .* has no .* vector column": {
        "message": "The table doesn't have a vector column as required.",
        "suggestions": [
            "Add a vector column to the table using ALTER TABLE.",
            "Verify that you're using the correct table for vector operations.",
            "Check if the vector column has a different name than expected."
        ]
    },
    r"sql compilation error": {
        "message": "There was an error compiling the SQL query.",
        "suggestions": [
            "Check your SQL syntax for errors.",
            "Verify that all referenced objects exist and are accessible.",
            "Ensure you're using supported SQL features for SAP HANA."
        ]
    },
    r"could not connect to server": {
        "message": "Could not connect to the database server.",
        "suggestions": [
            "Check that the server is running.",
            "Verify your network connection.",
            "Ensure the server address and port are correct.",
            "Check if any firewall is blocking the connection."
        ]
    }
}

# Operation-specific context information
OPERATION_CONTEXTS = {
    "vector_search": {
        "name": "Vector Search",
        "common_issues": [
            "Missing or invalid vector index",
            "Incorrect vector dimensionality",
            "Table doesn't contain the expected vector column",
            "Insufficient permissions for vector operations"
        ],
        "suggested_actions": [
            "Verify that the vector table exists and has an index",
            "Check that your query vector has the same dimension as the stored vectors",
            "Ensure you have the necessary permissions to perform vector operations"
        ]
    },
    "table_creation": {
        "name": "Table Creation",
        "common_issues": [
            "Table already exists",
            "Invalid column definitions",
            "Insufficient permissions to create tables",
            "Schema doesn't exist"
        ],
        "suggested_actions": [
            "Use CREATE TABLE IF NOT EXISTS to avoid errors if the table already exists",
            "Check column definitions for correct data types and constraints",
            "Verify you have CREATE privileges in the target schema"
        ]
    },
    "data_insertion": {
        "name": "Data Insertion",
        "common_issues": [
            "Unique constraint violations",
            "Data type mismatches",
            "Column count doesn't match value count",
            "Value too large for column definition"
        ],
        "suggested_actions": [
            "Check for duplicate keys when inserting with unique constraints",
            "Ensure data types match the column definitions",
            "Verify that the number of values matches the number of columns"
        ]
    },
    "flow_execution": {
        "name": "Flow Execution",
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


def interpret_sql_error(error_message: str) -> Dict[str, Any]:
    """
    Interprets SQL errors and provides user-friendly messages with suggestions.
    
    Args:
        error_message: The original error message
        
    Returns:
        Dict containing user-friendly message and suggestions
    """
    # Default error interpretation
    interpretation = {
        "message": "An error occurred while executing the database operation.",
        "suggestions": [
            "Try the operation again.",
            "Check your inputs for any errors.",
            "Contact support if the problem persists."
        ],
        "original_error": error_message
    }
    
    # Check for known error patterns
    for pattern, info in SQL_ERROR_PATTERNS.items():
        if re.search(pattern, error_message, re.IGNORECASE):
            interpretation["message"] = info["message"]
            interpretation["suggestions"] = info["suggestions"]
            break
    
    return interpretation


def create_context_aware_error(
    error_message: str, 
    operation_type: str,
    status_code: int = 500,
    additional_context: Optional[Dict[str, Any]] = None
) -> HTTPException:
    """
    Creates a context-aware error with useful suggestions based on the operation type.
    
    Args:
        error_message: The original error message
        operation_type: The type of operation being performed (e.g., "vector_search")
        status_code: HTTP status code for the error
        additional_context: Any additional context information
        
    Returns:
        HTTPException with detailed error information
    """
    # Get operation context if available
    context = OPERATION_CONTEXTS.get(operation_type, {
        "name": "Operation",
        "common_issues": [],
        "suggested_actions": []
    })
    
    # Interpret SQL errors if present
    if any(re.search(pattern, error_message, re.IGNORECASE) for pattern in SQL_ERROR_PATTERNS.keys()):
        interpretation = interpret_sql_error(error_message)
    else:
        interpretation = {
            "message": f"An error occurred during {context['name']}.",
            "suggestions": context.get("suggested_actions", []),
            "original_error": error_message
        }
    
    # Combine all information
    error_detail = {
        "message": interpretation["message"],
        "operation": context["name"],
        "suggestions": interpretation["suggestions"],
        "common_issues": context.get("common_issues", []),
        "original_error": error_message
    }
    
    # Add any additional context
    if additional_context:
        error_detail.update(additional_context)
    
    # Log the error with full context
    logger.error(f"Error in {context['name']}: {error_message}")
    
    return HTTPException(status_code=status_code, detail=error_detail)


def handle_vector_search_error(error: Exception, query_info: Optional[Dict[str, Any]] = None) -> HTTPException:
    """
    Specialized handler for vector search errors.
    
    Args:
        error: The exception that occurred
        query_info: Optional information about the query
        
    Returns:
        HTTPException with detailed error information
    """
    error_message = str(error)
    additional_context = {"query_info": query_info} if query_info else None
    
    # Check for specific vector search errors
    if "dimension mismatch" in error_message.lower():
        return create_context_aware_error(
            error_message,
            "vector_search",
            status_code=400,
            additional_context={
                "message": "The query vector dimension doesn't match the stored vectors.",
                "suggestions": [
                    "Ensure your query vector has the same dimension as the vectors in the database.",
                    "Check the embedding model configuration.",
                    "Verify that you're using the correct embedding function."
                ],
                **(additional_context or {})
            }
        )
    
    return create_context_aware_error(error_message, "vector_search", additional_context=additional_context)


def handle_flow_execution_error(error: Exception, flow_info: Optional[Dict[str, Any]] = None) -> HTTPException:
    """
    Specialized handler for flow execution errors.
    
    Args:
        error: The exception that occurred
        flow_info: Optional information about the flow
        
    Returns:
        HTTPException with detailed error information
    """
    error_message = str(error)
    additional_context = {"flow_info": flow_info} if flow_info else None
    
    # Check for specific flow errors
    if "no query node" in error_message.lower():
        return create_context_aware_error(
            error_message,
            "flow_execution",
            status_code=400,
            additional_context={
                "message": "The flow is missing a required query node.",
                "suggestions": [
                    "Add a query node to your flow.",
                    "Ensure all required nodes are properly connected.",
                    "Check the flow structure for any missing components."
                ],
                **(additional_context or {})
            }
        )
    
    return create_context_aware_error(error_message, "flow_execution", additional_context=additional_context)


def handle_data_insertion_error(error: Exception, insertion_info: Optional[Dict[str, Any]] = None) -> HTTPException:
    """
    Specialized handler for data insertion errors.
    
    Args:
        error: The exception that occurred
        insertion_info: Optional information about the data being inserted
        
    Returns:
        HTTPException with detailed error information
    """
    error_message = str(error)
    additional_context = {"insertion_info": insertion_info} if insertion_info else None
    
    return create_context_aware_error(error_message, "data_insertion", additional_context=additional_context)


def handle_vector_visualization_error(error: Exception, viz_params: Optional[Dict[str, Any]] = None) -> HTTPException:
    """
    Specialized handler for vector visualization errors.
    
    Args:
        error: The exception that occurred
        viz_params: Optional information about the visualization parameters
        
    Returns:
        HTTPException with detailed error information
    """
    error_message = str(error)
    additional_context = {"visualization_params": viz_params} if viz_params else None
    
    return create_context_aware_error(error_message, "vector_visualization", additional_context=additional_context)