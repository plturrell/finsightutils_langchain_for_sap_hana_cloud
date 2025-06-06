"""
Example: Robust error handling with the SAP HANA Cloud LangChain integration.

This example demonstrates how to:
1. Implement robust error handling with context-aware error messages
2. Use automatic retry with exponential backoff for transient errors
3. Log and respond to different error types appropriately
4. Create a resilient application that can gracefully handle failures

Requirements:
- langchain-hana package
- requests
- tenacity (for retry logic)
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

# Import the SAP HANA Cloud vectorstore and error utilities
from langchain_hana.vectorstores import HanaVectorStore
from langchain_hana.embeddings import HanaEmbeddings
from langchain_hana.error_utils import (
    identify_error_type,
    create_context_aware_error,
    handle_database_error
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Connection details (from environment variables)
HANA_HOST = os.environ.get("HANA_HOST")
HANA_PORT = int(os.environ.get("HANA_PORT", "443"))
HANA_USER = os.environ.get("HANA_USER")
HANA_PASSWORD = os.environ.get("HANA_PASSWORD")

# Constants
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5


class ConnectionManager:
    """Manages database connections with retry logic."""
    
    def __init__(
        self, 
        host: str, 
        port: int, 
        user: str, 
        password: str,
        schema: str = "ML_DATA",
        max_retries: int = MAX_RETRIES
    ):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.schema = schema
        self.max_retries = max_retries
        self.vectorstore = None
        
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.exceptions.ConnectionError, TimeoutError)),
        reraise=True
    )
    def connect(self) -> HanaVectorStore:
        """Establish connection to SAP HANA Cloud with retry logic."""
        try:
            logger.info(f"Connecting to SAP HANA Cloud at {self.host}:{self.port}")
            
            # Create connection to the vector store
            self.vectorstore = HanaVectorStore.create_connection(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                schema=self.schema,
            )
            
            # Test connection with a simple query
            self.vectorstore.connection.execute("SELECT 1 FROM DUMMY")
            
            logger.info("Successfully connected to SAP HANA Cloud")
            return self.vectorstore
            
        except Exception as e:
            # Create context-aware error for the connection operation
            error_info = create_context_aware_error(e, "connection")
            error_type = error_info["error_type"]
            
            logger.error(f"Connection error: {error_info['error']}")
            logger.error(f"Error type: {error_type}")
            logger.error(f"Suggestion: {error_info['context']['suggestion']}")
            
            # Different handling based on error type
            if error_type == "connection_failed":
                logger.warning("Network connectivity issue detected. Retrying...")
                raise requests.exceptions.ConnectionError(str(e))
            elif error_type == "auth_error":
                logger.critical("Authentication failed. Please check credentials.")
                raise ValueError("Authentication failed") from e
            elif error_type == "timeout":
                logger.warning("Connection timed out. Retrying...")
                raise TimeoutError(str(e))
            else:
                # For other errors, re-raise the original exception
                raise


class VectorStoreOperations:
    """Handles vector store operations with robust error handling."""
    
    def __init__(self, vectorstore: HanaVectorStore):
        self.vectorstore = vectorstore
        
    def safe_similarity_search(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Perform similarity search with error handling.
        
        Returns:
            Tuple containing (results, error_info)
            If successful, error_info will be None
        """
        try:
            results = self.vectorstore.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            
            # Convert to dictionaries for easier handling
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
                
            return formatted_results, None
            
        except Exception as e:
            # Handle the error and create context-aware error information
            error_info = create_context_aware_error(e, "similarity_search", {
                "query": query,
                "k": k,
                "filter": filter
            })
            
            logger.error(f"Search error: {error_info['error']}")
            logger.error(f"Error type: {error_info['error_type']}")
            logger.error(f"Suggestion: {error_info['context']['suggestion']}")
            
            # Return empty results and the error information
            return [], error_info
    
    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=BACKOFF_FACTOR, min=1, max=10),
        reraise=True
    )
    def add_documents_with_retry(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        table: Optional[str] = None
    ) -> int:
        """
        Add documents to the vector store with retry logic for transient errors.
        
        Returns:
            Number of documents added
        """
        try:
            # Use internal embedding model in SAP HANA
            result = self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                table=table
            )
            
            return len(texts)
            
        except Exception as e:
            # Create context-aware error
            error_info = create_context_aware_error(e, "add_texts")
            error_type = error_info["error_type"]
            
            # Log the error
            logger.error(f"Error adding documents: {error_info['error']}")
            logger.error(f"Error type: {error_type}")
            logger.error(f"Suggestion: {error_info['context']['suggestion']}")
            
            # Determine if we should retry based on error type
            retriable_error_types = [
                "connection_failed", 
                "timeout", 
                "resource_limit",
                "transaction_error"
            ]
            
            if error_type in retriable_error_types:
                logger.info(f"Retriable error detected. Will retry...")
                raise  # Let the retry decorator handle it
            else:
                # Non-retriable error
                logger.critical(f"Non-retriable error. Aborting operation.")
                raise ValueError(f"Cannot add documents: {error_info['error']}") from e
    
    def delete_documents_with_fallback(
        self,
        filter: Dict[str, Any],
        table: Optional[str] = None
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """
        Delete documents with error handling and fallback strategies.
        
        Returns:
            Tuple of (deleted_count, error_info)
        """
        try:
            # Try optimistic deletion first
            deleted_count = self.vectorstore.delete(
                filter=filter,
                table=table
            )
            
            return deleted_count, None
            
        except Exception as e:
            # Create context-aware error
            error_info = create_context_aware_error(e, "delete")
            error_type = error_info["error_type"]
            
            logger.error(f"Error deleting documents: {error_info['error']}")
            logger.error(f"Error type: {error_type}")
            
            # For permission errors, return immediately
            if error_type == "insufficient_privileges":
                logger.critical("Insufficient privileges to delete documents")
                return 0, error_info
                
            # For invalid filter, try to provide helpful information
            if error_type == "syntax_error":
                logger.warning("Filter syntax error. Checking for valid fields...")
                try:
                    # Try to get metadata fields to help with filter construction
                    sample = self.vectorstore.similarity_search("sample", k=1)
                    if sample:
                        metadata_fields = list(sample[0].metadata.keys())
                        error_info["context"]["valid_fields"] = metadata_fields
                except:
                    pass
                
                return 0, error_info
            
            # For other errors, attempt a fallback approach for deletion
            if table:
                try:
                    logger.info("Attempting alternative deletion approach...")
                    
                    # Build deletion query based on filter
                    conditions = []
                    params = {}
                    
                    for key, value in filter.items():
                        param_name = f"p_{key}"
                        conditions.append(f"JSON_VALUE(metadata, '$.{key}') = :{param_name}")
                        params[param_name] = value
                    
                    where_clause = " AND ".join(conditions)
                    delete_query = f"DELETE FROM {table} WHERE {where_clause}"
                    
                    # Execute direct deletion
                    result = self.vectorstore.connection.execute(delete_query, params)
                    deleted_count = result.rowcount
                    
                    logger.info(f"Fallback deletion successful: {deleted_count} documents deleted")
                    return deleted_count, None
                    
                except Exception as fallback_error:
                    logger.error(f"Fallback deletion also failed: {str(fallback_error)}")
            
            # If all strategies fail, return 0 with the original error
            return 0, error_info


def format_error_for_api_response(error_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format error information for API response.
    This creates a user-friendly error response suitable for frontend display.
    """
    # Determine HTTP status code based on error type
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
    
    status_code = error_type_to_status.get(error_info["error_type"], 500)
    
    # Create API response
    response = {
        "status": status_code,
        "statusText": get_status_text(status_code),
        "detail": {
            "message": translate_error_message(error_info["error_type"], error_info["error"]),
            "operation": error_info["operation"],
            "suggestions": error_info["context"]["suggested_actions"],
            "common_issues": error_info["context"]["common_issues"],
            "original_error": error_info["error"]
        }
    }
    
    # Add any additional context from the error
    for key, value in error_info.items():
        if key not in ["error", "error_type", "operation", "context"]:
            response["detail"][key] = value
    
    return response


def get_status_text(status_code: int) -> str:
    """Get HTTP status text for a status code."""
    status_texts = {
        200: "OK",
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        500: "Internal Server Error",
        501: "Not Implemented",
        503: "Service Unavailable",
        504: "Gateway Timeout",
        507: "Insufficient Storage",
    }
    return status_texts.get(status_code, "Unknown")


def translate_error_message(error_type: str, original_error: str) -> str:
    """Translate technical error messages to user-friendly messages."""
    translations = {
        "connection_failed": "Unable to connect to the database. Please check your connection and try again.",
        "timeout": "The operation timed out. Please try again later or reduce the data volume.",
        "auth_error": "Authentication failed. Please check your credentials.",
        "insufficient_privileges": "You don't have permission to perform this operation.",
        "table_not_found": "The requested data table could not be found.",
        "column_not_found": "A required column is missing from the data table.",
        "invalid_vector_dimension": "Vector dimension mismatch. Please ensure query and document vectors have the same dimensions.",
        "vector_feature_unavailable": "Vector search features are not available in your SAP HANA Cloud instance.",
        "syntax_error": "There's an error in your query syntax.",
        "out_of_memory": "The system ran out of memory. Try reducing the batch size or query complexity.",
        "resource_limit": "Resource limit exceeded. Please optimize your query or request more resources.",
        "transaction_error": "A database transaction error occurred. Please try again.",
    }
    
    return translations.get(error_type, original_error)


def main():
    # Demonstrate connection with retry and error handling
    try:
        # Establish connection
        connection_manager = ConnectionManager(
            host=HANA_HOST,
            port=HANA_PORT,
            user=HANA_USER,
            password=HANA_PASSWORD
        )
        
        vectorstore = connection_manager.connect()
        logger.info("Connection successful")
        
        # Create operations handler
        ops = VectorStoreOperations(vectorstore)
        
        # Demonstrate similarity search with error handling
        results, error_info = ops.safe_similarity_search(
            query="What are the benefits of SAP HANA Cloud?",
            k=3,
            filter={"category": "product_benefits"}
        )
        
        if error_info:
            # Format error for API response
            api_error = format_error_for_api_response(error_info)
            logger.error(f"Search failed with API error: {api_error}")
        else:
            # Process results
            logger.info(f"Found {len(results)} results")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: {result['content'][:50]}...")
        
        # Demonstrate document addition with retry
        try:
            docs_added = ops.add_documents_with_retry(
                texts=["This is a test document", "Another test document"],
                metadatas=[
                    {"source": "example", "category": "test"},
                    {"source": "example", "category": "test"}
                ]
            )
            logger.info(f"Added {docs_added} documents")
        except RetryError:
            logger.error("Failed to add documents after multiple retries")
        except Exception as e:
            logger.error(f"Non-retriable error adding documents: {str(e)}")
        
        # Demonstrate document deletion with fallback
        deleted_count, error_info = ops.delete_documents_with_fallback(
            filter={"category": "test"}
        )
        
        if error_info:
            # Format error for API response
            api_error = format_error_for_api_response(error_info)
            logger.error(f"Deletion failed with API error: {api_error}")
        else:
            logger.info(f"Deleted {deleted_count} documents")
        
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")


if __name__ == "__main__":
    main()