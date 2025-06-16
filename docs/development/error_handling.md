# Error Handling in LangChain SAP HANA Cloud Integration

This document explains the intelligent error handling system implemented in the LangChain SAP HANA Cloud integration.

## Overview

Working with databases in LLM applications can lead to various types of errors, including:
- Connection issues
- Permission problems
- Invalid SQL syntax
- Data format mismatches
- Resource limitations
- Vector-specific issues (dimensions, models, etc.)

Our context-aware error handling system automatically identifies the error type and provides specific suggestions to fix the problem, making troubleshooting faster and more efficient.

## Key Features

1. **Error Type Detection**: Automatically identifies the type of error based on pattern matching
2. **Operation Context**: Adds context about what operation was being performed when the error occurred
3. **Suggested Actions**: Provides specific actions to resolve the issue
4. **Common Issues List**: Includes a list of common issues related to the specific operation
5. **Additional Context**: Captures relevant parameters and configuration that might be related to the error

## How It Works

The error handling system works by:

1. Intercepting exceptions from database operations
2. Analyzing the error message to identify the type of error
3. Adding operation-specific context and suggestions
4. Returning an enhanced error message with actionable information

## Example Error Messages

Here are some examples of the context-aware error messages:

### Connection Error

```
connection to the server has been lost

Operation: Connecting to SAP HANA Cloud database
Suggestion: Check network connectivity and connection parameters
Actions: Verify connection parameters (address, port, username, password), Check network connectivity to the database server
```

### Table Creation Error

```
invalid vector column type: INTEGER

Operation: Creating vector table in SAP HANA Cloud
Suggestion: Check data types and conversion compatibility
Actions: Verify that you have CREATE TABLE privileges in the schema, Check table and column naming for invalid characters
```

### Vector Dimension Error

```
vector dimension 768 not matching column dimension 384

Operation: Adding documents to vector store
Suggestion: Ensure vector dimensions match between query and documents
Actions: Reduce batch size when adding documents, Ensure metadata keys contain only alphanumeric characters and underscores
```

### Embedding Model Error

```
embedding model 'SAP_NEB.20240715' not found

Operation: Generating embeddings using SAP HANA's internal embedding function
Suggestion: Verify embedding model ID and availability
Actions: Verify that the embedding model ID is valid (e.g., 'SAP_NEB.20240715'), Check that VECTOR_EMBEDDING function is available in your SAP HANA version
```

## Supported Error Types

The system recognizes and provides specific guidance for these error categories:

### Connection Errors
- Connection failures
- Timeout issues
- Authentication problems

### Permission Errors
- Insufficient privileges
- Unauthorized access

### Resource Errors
- Out of memory
- Resource limitations
- Too many connections

### Table and Column Errors
- Table not found
- Column not found
- Data type mismatches

### Vector-specific Errors
- Invalid vector dimensions
- Vector feature unavailability
- Embedding model errors

### Syntax Errors
- SQL syntax errors
- Parse errors

### Index Errors
- HNSW index creation failures
- Invalid index parameters

### Transaction Errors
- Transaction aborts
- Deadlocks
- Lock timeouts

## Operation-specific Context

The system provides different suggestions based on the type of operation being performed:

- **Connection**: Connecting to SAP HANA Cloud database
- **Table Creation**: Creating vector table in SAP HANA Cloud
- **Embedding Generation**: Generating embeddings using SAP HANA's internal embedding function
- **Similarity Search**: Performing vector similarity search
- **Add Texts**: Adding documents to vector store
- **MMR Search**: Performing maximal marginal relevance search
- **Delete**: Deleting documents from vector store
- **Index Creation**: Creating vector search index

## Using Error Handling in Custom Code

If you want to integrate our error handling into your own code that works with SAP HANA Cloud, you can use the `handle_database_error` function:

```python
from langchain_hana.error_utils import handle_database_error

try:
    # Your database operation code here
    cur = connection.cursor()
    cur.execute("YOUR SQL QUERY")
except Exception as e:
    # Handle the error with context
    error_info = handle_database_error(
        e,
        operation_type="your_operation",  # Type of operation being performed
        additional_context={"your_param": value},  # Additional context
        raise_exception=False  # Set to False to return error info instead of raising
    )
    
    # Now you can use the structured error information
    print(f"Error: {error_info['error']}")
    print(f"Suggestion: {error_info['context']['suggestion']}")
    print(f"Actions: {', '.join(error_info['context']['suggested_actions'])}")
```

## Benefits

The context-aware error handling provides several benefits:

1. **Faster Troubleshooting**: Immediately understand what went wrong and how to fix it
2. **Reduced Support Needs**: More users can self-resolve issues with the detailed guidance
3. **Better User Experience**: Clear, actionable error messages instead of cryptic database errors
4. **Education**: Helps users understand the system better through specific guidance
5. **Consistency**: Provides a consistent approach to error handling across the integration

## Best Practices

When working with the SAP HANA Cloud integration:

1. Always handle exceptions with try/except blocks
2. Log the full error message for debugging
3. Present users with the suggested actions from the error
4. For security-sensitive applications, consider filtering out sensitive information from error messages before displaying them to end-users