"""
Custom exceptions for the LangChain SAP HANA Cloud integration.

This module defines a comprehensive exception hierarchy for detailed
error handling and reporting in production environments.
"""

from typing import Dict, Any, Optional


class HanaIntegrationError(Exception):
    """Base exception for all SAP HANA integration errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional details.
        
        Args:
            message: Error message
            details: Additional error details and context
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """String representation of the error."""
        if self.details:
            return f"{self.message} - Details: {self.details}"
        return self.message


class ConnectionError(HanaIntegrationError):
    """Exception raised for connection-related errors."""
    pass


class DatabaseError(HanaIntegrationError):
    """Exception raised for database operation errors."""
    pass


class ConfigurationError(HanaIntegrationError):
    """Exception raised for configuration-related errors."""
    pass


class VectorOperationError(HanaIntegrationError):
    """Exception raised for vector operation errors."""
    pass


class EmbeddingError(HanaIntegrationError):
    """Exception raised for embedding generation errors."""
    pass


class InvalidSchemaError(HanaIntegrationError):
    """Exception raised for schema validation errors."""
    pass


class ResourceExhaustedError(HanaIntegrationError):
    """Exception raised when resources are exhausted (e.g., memory, connections)."""
    pass


class TimeoutError(HanaIntegrationError):
    """Exception raised when an operation times out."""
    pass


class AuthenticationError(HanaIntegrationError):
    """Exception raised for authentication failures."""
    pass


class NotSupportedError(HanaIntegrationError):
    """Exception raised when an operation is not supported."""
    pass


def convert_db_error(error: Exception, operation: str) -> HanaIntegrationError:
    """
    Convert a database error to an appropriate custom exception.
    
    Args:
        error: Original database error
        operation: Operation that was being performed
        
    Returns:
        Appropriate custom exception
    """
    error_message = str(error)
    error_code = getattr(error, "errorcode", None)
    
    # Authentication errors
    if error_code in (10, 13001) or "authentication failed" in error_message.lower():
        return AuthenticationError(
            f"Authentication failed during {operation}", 
            {"error_code": error_code, "original_error": error_message}
        )
    
    # Connection errors
    if error_code in (10061, 10060) or "connection" in error_message.lower():
        return ConnectionError(
            f"Connection error during {operation}", 
            {"error_code": error_code, "original_error": error_message}
        )
    
    # Timeout errors
    if "timeout" in error_message.lower():
        return TimeoutError(
            f"Operation timed out during {operation}", 
            {"error_code": error_code, "original_error": error_message}
        )
    
    # Schema errors
    if "schema" in error_message.lower() or "column" in error_message.lower():
        return InvalidSchemaError(
            f"Schema error during {operation}", 
            {"error_code": error_code, "original_error": error_message}
        )
    
    # Default to database error
    return DatabaseError(
        f"Database error during {operation}: {error_message}", 
        {"error_code": error_code, "original_error": error_message}
    )