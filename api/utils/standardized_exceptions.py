"""
Standardized exceptions for the SAP HANA Cloud LangChain Integration API.

This module provides a consistent exception hierarchy and error response format
for the API, ensuring uniform error handling across all endpoints.
"""

from typing import Any, Dict, Optional, List, Type, Union, Callable
import re

from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Base API exception with consistent error format."""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        error_code: str = "error",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        docs_url: Optional[str] = None
    ):
        """Initialize the base API exception."""
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.details = details or {}
        self.suggestion = suggestion
        self.docs_url = docs_url


class BadRequestException(BaseAPIException):
    """Exception for invalid request data (400)."""
    
    def __init__(
        self,
        detail: str = "Bad request",
        error_code: str = "bad_request",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        docs_url: Optional[str] = None
    ):
        """Initialize the bad request exception."""
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class UnauthorizedException(BaseAPIException):
    """Exception for authentication failures (401)."""
    
    def __init__(
        self,
        detail: str = "Authentication required",
        error_code: str = "unauthorized",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please provide valid authentication credentials",
        docs_url: Optional[str] = None
    ):
        """Initialize the unauthorized exception."""
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code=error_code,
            headers=headers or {"WWW-Authenticate": "Bearer"},
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class ForbiddenException(BaseAPIException):
    """Exception for permission failures (403)."""
    
    def __init__(
        self,
        detail: str = "Access forbidden",
        error_code: str = "forbidden",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please ensure you have the necessary permissions",
        docs_url: Optional[str] = None
    ):
        """Initialize the forbidden exception."""
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class NotFoundException(BaseAPIException):
    """Exception for resource not found (404)."""
    
    def __init__(
        self,
        detail: str = "Resource not found",
        error_code: str = "not_found",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please check the resource identifier",
        docs_url: Optional[str] = None
    ):
        """Initialize the not found exception."""
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class ConflictException(BaseAPIException):
    """Exception for resource conflicts (409)."""
    
    def __init__(
        self,
        detail: str = "Resource conflict",
        error_code: str = "conflict",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None,
        docs_url: Optional[str] = None
    ):
        """Initialize the conflict exception."""
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class UnprocessableEntityException(BaseAPIException):
    """Exception for validation failures (422)."""
    
    def __init__(
        self,
        detail: str = "Validation error",
        error_code: str = "validation_error",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please check the request data",
        docs_url: Optional[str] = None
    ):
        """Initialize the unprocessable entity exception."""
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class RateLimitExceededException(BaseAPIException):
    """Exception for rate limit exceeded (429)."""
    
    def __init__(
        self,
        detail: str = "Rate limit exceeded",
        error_code: str = "rate_limit_exceeded",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please reduce request frequency",
        docs_url: Optional[str] = None
    ):
        """Initialize the rate limit exceeded exception."""
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class InternalServerErrorException(BaseAPIException):
    """Exception for server errors (500)."""
    
    def __init__(
        self,
        detail: str = "Internal server error",
        error_code: str = "internal_server_error",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please contact support if the problem persists",
        docs_url: Optional[str] = None
    ):
        """Initialize the internal server error exception."""
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class ServiceUnavailableException(BaseAPIException):
    """Exception for service unavailable (503)."""
    
    def __init__(
        self,
        detail: str = "Service unavailable",
        error_code: str = "service_unavailable",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please try again later",
        docs_url: Optional[str] = None
    ):
        """Initialize the service unavailable exception."""
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


# Database specific exceptions
class DatabaseConnectionException(ServiceUnavailableException):
    """Exception for database connection issues."""
    
    def __init__(
        self,
        detail: str = "Cannot connect to database",
        error_code: str = "database_connection_error",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please check database connection settings",
        docs_url: Optional[str] = None
    ):
        """Initialize the database connection exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class DatabaseQueryException(InternalServerErrorException):
    """Exception for database query failures."""
    
    def __init__(
        self,
        detail: str = "Database query failed",
        error_code: str = "database_query_error",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Please check your query syntax and parameters",
        docs_url: Optional[str] = None
    ):
        """Initialize the database query exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


# Vector specific exceptions
class VectorDimensionMismatchException(BadRequestException):
    """Exception for vector dimension mismatch."""
    
    def __init__(
        self,
        detail: str = "Vector dimension mismatch",
        error_code: str = "vector_dimension_mismatch",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Ensure query vector has the same dimension as stored vectors",
        docs_url: Optional[str] = None
    ):
        """Initialize the vector dimension mismatch exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class VectorTableNotFoundException(NotFoundException):
    """Exception for vector table not found."""
    
    def __init__(
        self,
        detail: str = "Vector table not found",
        error_code: str = "vector_table_not_found",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Ensure the vector table exists in the database",
        docs_url: Optional[str] = None
    ):
        """Initialize the vector table not found exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class EmbeddingGenerationException(InternalServerErrorException):
    """Exception for embedding generation failures."""
    
    def __init__(
        self,
        detail: str = "Failed to generate embeddings",
        error_code: str = "embedding_generation_error",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Check embedding model configuration",
        docs_url: Optional[str] = None
    ):
        """Initialize the embedding generation exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


# GPU specific exceptions
class GPUNotAvailableException(ServiceUnavailableException):
    """Exception for GPU not available."""
    
    def __init__(
        self,
        detail: str = "GPU acceleration is not available",
        error_code: str = "gpu_not_available",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Try again later or use CPU-only mode",
        docs_url: Optional[str] = None
    ):
        """Initialize the GPU not available exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class TensorRTNotAvailableException(ServiceUnavailableException):
    """Exception for TensorRT not available."""
    
    def __init__(
        self,
        detail: str = "TensorRT acceleration is not available",
        error_code: str = "tensorrt_not_available",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Try again later or use standard mode",
        docs_url: Optional[str] = None
    ):
        """Initialize the TensorRT not available exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


# Arrow Flight specific exceptions
class ArrowFlightUnavailableException(ServiceUnavailableException):
    """Exception for Arrow Flight service unavailable."""
    
    def __init__(
        self,
        detail: str = "Arrow Flight service is not available",
        error_code: str = "arrow_flight_unavailable",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Try again later or use REST API",
        docs_url: Optional[str] = None
    ):
        """Initialize the Arrow Flight unavailable exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


class ArrowFlightAuthException(UnauthorizedException):
    """Exception for Arrow Flight authentication failures."""
    
    def __init__(
        self,
        detail: str = "Arrow Flight authentication failed",
        error_code: str = "arrow_flight_auth_error",
        headers: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = "Provide valid Arrow Flight credentials",
        docs_url: Optional[str] = None
    ):
        """Initialize the Arrow Flight authentication exception."""
        super().__init__(
            detail=detail,
            error_code=error_code,
            headers=headers,
            details=details,
            suggestion=suggestion,
            docs_url=docs_url
        )


# Helper functions to convert from existing error handling
def convert_error_to_exception(
    error_type: str,
    operation_type: str,
    error_message: str,
    status_code: int = 500
) -> BaseAPIException:
    """Convert error information from unified_error_handler to a standardized exception."""
    # Extract suggestion from context
    from api.utils.unified_error_handler import ERROR_TYPE_SUGGESTIONS, OPERATION_CONTEXTS
    
    suggestion = ERROR_TYPE_SUGGESTIONS.get(error_type, "Check documentation for more information")
    
    # Get operation context
    context = OPERATION_CONTEXTS.get(operation_type, {
        "name": operation_type.replace("_", " ").title(),
        "description": f"Performing {operation_type} operation",
        "common_issues": ["Unknown issue"],
        "suggested_actions": ["Check SAP HANA Cloud documentation"]
    })
    
    # Map error types to exception classes
    exception_map = {
        # Connection errors
        "connection_failed": DatabaseConnectionException,
        "timeout": DatabaseConnectionException,
        "auth_error": UnauthorizedException,
        
        # Permission errors
        "insufficient_privileges": ForbiddenException,
        
        # Resource errors
        "out_of_memory": ServiceUnavailableException,
        "resource_limit": ServiceUnavailableException,
        
        # Table and column errors
        "table_not_found": VectorTableNotFoundException,
        "column_not_found": BadRequestException,
        "datatype_mismatch": BadRequestException,
        
        # Vector-specific errors
        "invalid_vector_dimension": VectorDimensionMismatchException,
        "vector_feature_unavailable": ServiceUnavailableException,
        "embedding_model_error": EmbeddingGenerationException,
        
        # Syntax errors
        "syntax_error": BadRequestException,
        
        # Index errors
        "index_error": BadRequestException,
        "hnsw_error": BadRequestException,
        
        # Transaction errors
        "transaction_error": DatabaseQueryException,
        
        # Constraint violations
        "constraint_violation": ConflictException,
        
        # Database limit errors
        "limit_exceeded": BadRequestException,
        
        # Default
        "unknown_error": InternalServerErrorException
    }
    
    # Get the appropriate exception class
    exception_class = exception_map.get(error_type, InternalServerErrorException)
    
    # Return an instance of the exception
    return exception_class(
        detail=error_message,
        error_code=f"{operation_type}_{error_type}",
        suggestion=suggestion,
        details={
            "operation": context["description"],
            "operation_name": context["name"],
            "common_issues": context.get("common_issues", []),
            "suggested_actions": context.get("suggested_actions", [])
        }
    )


def extract_error_code_from_message(error_message: str) -> str:
    """Extract error code from an exception message."""
    # Common SQL error codes from SAP HANA
    sql_error_code_pattern = re.compile(r"(SQL Error|Error:)\s*(\d+)")
    match = sql_error_code_pattern.search(error_message)
    
    if match:
        return f"sql_{match.group(2)}"
    return "unknown_error"


def convert_db_exception(exception: Exception) -> BaseAPIException:
    """Convert a database exception to a standardized exception."""
    error_message = str(exception)
    
    # Detect connection errors
    if any(pattern in error_message.lower() for pattern in [
        "connection", "network error", "timeout", "unable to connect"
    ]):
        return DatabaseConnectionException(detail=error_message)
    
    # Detect authentication errors
    if any(pattern in error_message.lower() for pattern in [
        "authentication failed", "access denied", "invalid username", "password"
    ]):
        return UnauthorizedException(detail=error_message)
    
    # Detect table not found errors
    if any(pattern in error_message.lower() for pattern in [
        "table not found", "relation does not exist"
    ]):
        return VectorTableNotFoundException(detail=error_message)
    
    # Detect vector dimension errors
    if any(pattern in error_message.lower() for pattern in [
        "dimension mismatch", "vector length", "expected dimension"
    ]):
        return VectorDimensionMismatchException(detail=error_message)
    
    # Default to database query exception
    return DatabaseQueryException(detail=error_message)