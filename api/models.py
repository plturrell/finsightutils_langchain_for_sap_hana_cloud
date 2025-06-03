"""Data models for the API."""

from enum import Enum
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field, validator

from langchain_hana.utils import DistanceStrategy


class VectorColumnType(str, Enum):
    """Vector column type enum."""
    REAL_VECTOR = "REAL_VECTOR"
    HALF_VECTOR = "HALF_VECTOR"


class DistanceStrategyEnum(str, Enum):
    """Distance strategy enum."""
    COSINE = "COSINE"
    EUCLIDEAN = "EUCLIDEAN_DISTANCE"


class DocumentModel(BaseModel):
    """Document model."""
    page_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AddTextsRequest(BaseModel):
    """Request model for adding texts."""
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    table_name: Optional[str] = None


class QueryRequest(BaseModel):
    """Request model for querying."""
    query: str
    k: int = 4
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = None


class VectorQueryRequest(BaseModel):
    """Request model for vector querying."""
    embedding: List[float]
    k: int = 4
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = None
    
    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding vector."""
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        return v


class MMRQueryRequest(BaseModel):
    """Request model for MMR querying."""
    query: str
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = None
    
    @validator('lambda_mult')
    def validate_lambda_mult(cls, v):
        """Validate lambda multiplier."""
        if not 0 <= v <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        return v


class MMRVectorQueryRequest(BaseModel):
    """Request model for MMR vector querying."""
    embedding: List[float]
    k: int = 4
    fetch_k: int = 20
    lambda_mult: float = 0.5
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = None
    
    @validator('lambda_mult')
    def validate_lambda_mult(cls, v):
        """Validate lambda multiplier."""
        if not 0 <= v <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        return v
    
    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding vector."""
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        return v


class DeleteRequest(BaseModel):
    """Request model for deleting documents."""
    filter: Dict[str, Any]
    table_name: Optional[str] = None
    
    @validator('filter')
    def validate_filter(cls, v):
        """Validate filter."""
        if not v:
            raise ValueError("Filter cannot be empty")
        return v


class DocumentResponse(BaseModel):
    """Response model for document."""
    document: DocumentModel
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for query."""
    results: List[DocumentResponse]


class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool
    message: str
    data: Optional[Any] = None