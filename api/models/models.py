"""Data models for the API."""

from enum import Enum
import uuid
from typing import Dict, List, Optional, Union, Any, Literal

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
    texts: List[str] = Field(
        ...,  # Required field
        min_items=1,
        max_items=1000,  # Limit batch size to prevent DoS
        description="List of text documents to add (1-1000 items)"
    )
    metadatas: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of metadata dictionaries, one per document"
    )
    table_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Name of the table to add documents to"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate text documents."""
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Document at index {i} must be a string")
            if len(text) > 100000:  # 100KB limit per document
                raise ValueError(f"Document at index {i} exceeds maximum length (100KB)")
        return v
    
    @validator('metadatas')
    def validate_metadatas(cls, v, values):
        """Validate metadata structure and length."""
        if v is not None:
            texts = values.get('texts', [])
            if len(v) != len(texts):
                raise ValueError(f"Number of metadata items ({len(v)}) must match number of texts ({len(texts)})")
            
            for i, metadata in enumerate(v):
                if not isinstance(metadata, dict):
                    raise ValueError(f"Metadata at index {i} must be a dictionary")
                
                # Check metadata keys are valid
                for key in metadata:
                    if not isinstance(key, str):
                        raise ValueError(f"Metadata key {key} at index {i} must be a string")
                    if not key.isalnum() and not key.startswith("_"):
                        raise ValueError(f"Metadata key {key} at index {i} contains invalid characters")
                    if len(key) > 64:
                        raise ValueError(f"Metadata key {key} at index {i} is too long (max 64 chars)")
                
                # Limit total metadata size to prevent abuse
                import json
                metadata_size = len(json.dumps(metadata))
                if metadata_size > 10000:  # 10KB limit per metadata object
                    raise ValueError(f"Metadata at index {i} exceeds maximum size (10KB)")
        return v


class UpdateTextsRequest(BaseModel):
    """Request model for updating texts."""
    texts: List[str]
    filter: Dict[str, Any]
    metadatas: Optional[List[Dict[str, Any]]] = None
    update_embeddings: bool = True
    table_name: Optional[str] = None


class UpsertTextsRequest(BaseModel):
    """Request model for upserting texts."""
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = None


class DeleteRequest(BaseModel):
    """Request model for deleting documents."""
    filter: Dict[str, Any]
    table_name: Optional[str] = None


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    success: bool
    operation: str
    count: int = 0
    message: str
    processing_time: float


class QueryRequest(BaseModel):
    """Request model for querying."""
    query: str
    k: int = Field(
        default=4, 
        ge=1, 
        le=100, 
        description="Number of documents to return (1-100)"
    )
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Name of the table to query"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query string."""
        if not v.strip():
            raise ValueError("Query string cannot be empty")
        if len(v) > 8192:
            raise ValueError("Query string is too long (max 8192 characters)")
        return v
    
    @validator('filter')
    def validate_filter(cls, v):
        """Validate filter structure."""
        if v is not None:
            # Check for potentially dangerous patterns in filter keys
            for key in v.keys():
                if not isinstance(key, str):
                    raise ValueError(f"Filter key {key} must be a string")
                if "$" in key and not key.startswith("$"):
                    raise ValueError(f"Invalid filter key format: {key}")
            
            # Check nested depth to prevent filter bombs
            def check_depth(obj, current_depth=0, max_depth=5):
                if current_depth > max_depth:
                    raise ValueError(f"Filter too deeply nested (max {max_depth} levels)")
                if isinstance(obj, dict):
                    for _, v in obj.items():
                        check_depth(v, current_depth + 1, max_depth)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, current_depth + 1, max_depth)
            
            check_depth(v)
        return v


class VectorQueryRequest(BaseModel):
    """Request model for vector querying."""
    embedding: List[float]
    k: int = Field(
        default=4, 
        ge=1, 
        le=100, 
        description="Number of documents to return (1-100)"
    )
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Name of the table to query"
    )
    
    @validator('embedding')
    def validate_embedding(cls, v):
        """Validate embedding vector."""
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        if len(v) > 4096:
            raise ValueError("Embedding vector too large (max 4096 dimensions)")
        for value in v:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid embedding value: {value}. Must be numeric")
        return v
    
    @validator('filter')
    def validate_filter(cls, v):
        """Validate filter structure."""
        if v is not None:
            # Check for potentially dangerous patterns in filter keys
            for key in v.keys():
                if not isinstance(key, str):
                    raise ValueError(f"Filter key {key} must be a string")
                if "$" in key and not key.startswith("$"):
                    raise ValueError(f"Invalid filter key format: {key}")
            
            # Check nested depth to prevent filter bombs
            def check_depth(obj, current_depth=0, max_depth=5):
                if current_depth > max_depth:
                    raise ValueError(f"Filter too deeply nested (max {max_depth} levels)")
                if isinstance(obj, dict):
                    for _, v in obj.items():
                        check_depth(v, current_depth + 1, max_depth)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, current_depth + 1, max_depth)
            
            check_depth(v)
        return v


class MMRQueryRequest(BaseModel):
    """Request model for MMR querying."""
    query: str
    k: int = Field(
        default=4, 
        ge=1, 
        le=100, 
        description="Number of documents to return (1-100)"
    )
    fetch_k: int = Field(
        default=20, 
        ge=1, 
        le=1000, 
        description="Number of documents to fetch before applying MMR (1-1000)"
    )
    lambda_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between relevance and diversity (0.0-1.0)"
    )
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Name of the table to query"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query string."""
        if not v.strip():
            raise ValueError("Query string cannot be empty")
        if len(v) > 8192:
            raise ValueError("Query string is too long (max 8192 characters)")
        return v
    
    @validator('lambda_mult')
    def validate_lambda_mult(cls, v):
        """Validate lambda multiplier."""
        if not 0 <= v <= 1:
            raise ValueError("lambda_mult must be between 0 and 1")
        return v
    
    @validator('fetch_k')
    def validate_fetch_k_relation(cls, v, values):
        """Validate fetch_k in relation to k."""
        if 'k' in values and v < values['k']:
            raise ValueError("fetch_k must be greater than or equal to k")
        return v
    
    @validator('filter')
    def validate_filter(cls, v):
        """Validate filter structure."""
        if v is not None:
            # Check for potentially dangerous patterns in filter keys
            for key in v.keys():
                if not isinstance(key, str):
                    raise ValueError(f"Filter key {key} must be a string")
                if "$" in key and not key.startswith("$"):
                    raise ValueError(f"Invalid filter key format: {key}")
            
            # Check nested depth to prevent filter bombs
            def check_depth(obj, current_depth=0, max_depth=5):
                if current_depth > max_depth:
                    raise ValueError(f"Filter too deeply nested (max {max_depth} levels)")
                if isinstance(obj, dict):
                    for _, v in obj.items():
                        check_depth(v, current_depth + 1, max_depth)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, current_depth + 1, max_depth)
            
            check_depth(v)
        return v


class MMRVectorQueryRequest(BaseModel):
    """Request model for MMR vector querying."""
    embedding: List[float]
    k: int = Field(
        default=4, 
        ge=1, 
        le=100, 
        description="Number of documents to return (1-100)"
    )
    fetch_k: int = Field(
        default=20, 
        ge=1, 
        le=1000, 
        description="Number of documents to fetch before applying MMR (1-1000)"
    )
    lambda_mult: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Balance between relevance and diversity (0.0-1.0)"
    )
    filter: Optional[Dict[str, Any]] = None
    table_name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Name of the table to query"
    )
    
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
        if len(v) > 4096:
            raise ValueError("Embedding vector too large (max 4096 dimensions)")
        for value in v:
            if not isinstance(value, (int, float)):
                raise ValueError(f"Invalid embedding value: {value}. Must be numeric")
        return v
    
    @validator('fetch_k')
    def validate_fetch_k_relation(cls, v, values):
        """Validate fetch_k in relation to k."""
        if 'k' in values and v < values['k']:
            raise ValueError("fetch_k must be greater than or equal to k")
        return v
    
    @validator('filter')
    def validate_filter(cls, v):
        """Validate filter structure."""
        if v is not None:
            # Check for potentially dangerous patterns in filter keys
            for key in v.keys():
                if not isinstance(key, str):
                    raise ValueError(f"Filter key {key} must be a string")
                if "$" in key and not key.startswith("$"):
                    raise ValueError(f"Invalid filter key format: {key}")
            
            # Check nested depth to prevent filter bombs
            def check_depth(obj, current_depth=0, max_depth=5):
                if current_depth > max_depth:
                    raise ValueError(f"Filter too deeply nested (max {max_depth} levels)")
                if isinstance(obj, dict):
                    for _, v in obj.items():
                        check_depth(v, current_depth + 1, max_depth)
                elif isinstance(obj, list):
                    for item in obj:
                        check_depth(item, current_depth + 1, max_depth)
            
            check_depth(v)
        return v


class QueryResultDocumentResponse(BaseModel):
    """Response model for document query results."""
    document: DocumentModel
    score: Optional[float] = None


class QueryResponse(BaseModel):
    """Response model for query."""
    results: List[QueryResultDocumentResponse]


class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool
    message: str
    data: Optional[Any] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: Optional[str] = None
    environment: Optional[str] = None
    gpu_info: Optional[Dict[str, Any]] = None


class ConfigurationResponse(BaseModel):
    """Configuration response model."""
    config: Dict[str, Any]
    environment: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    status_code: int = 500
    error_type: Optional[str] = None


# Flow models for the visual development environment
class FlowNodeData(BaseModel):
    """Data for a node in a flow."""
    label: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    results: Optional[List[Any]] = None
    
    class Config:
        extra = "allow"

class Position(BaseModel):
    """Position of a node in a flow."""
    x: float
    y: float

class FlowNode(BaseModel):
    """Node in a flow."""
    id: str
    type: str
    data: FlowNodeData
    position: Position

class FlowEdge(BaseModel):
    """Edge in a flow."""
    id: str
    source: str
    target: str
    type: Optional[str] = None
    animated: Optional[bool] = None
    markerEnd: Optional[Dict[str, Any]] = None

class Flow(BaseModel):
    """Flow definition."""
    id: Optional[str] = None
    name: str
    description: str
    nodes: List[FlowNode]
    edges: List[FlowEdge]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

# API request/response models
class RunFlowRequest(BaseModel):
    """Request to run a flow."""
    flow: Flow

class RunFlowResponse(BaseModel):
    """Response from running a flow."""
    success: bool
    results: List[Dict[str, Any]]
    execution_time: float
    generated_code: str

class GenerateCodeRequest(BaseModel):
    """Request to generate code from a flow."""
    name: str
    description: str
    nodes: List[FlowNode]
    edges: List[FlowEdge]
    
    class Config:
        extra = "allow"

class GenerateCodeResponse(BaseModel):
    """Response with generated code."""
    code: str

class SaveFlowRequest(BaseModel):
    """Request to save a flow."""
    flow: Flow

class SaveFlowResponse(BaseModel):
    """Response from saving a flow."""
    success: bool
    flow_id: str
    message: str

class ListFlowsResponse(BaseModel):
    """Response with list of flows."""
    flows: List[Flow]

class GetFlowResponse(Flow):
    """Response with a flow."""
    pass

class DeleteFlowResponse(BaseModel):
    """Response from deleting a flow."""
    success: bool

# Embedding models
class EmbeddingRequest(BaseModel):
    """Request model for generating embeddings."""
    text: str
    model_name: Optional[str] = None
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text."""
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    embedding: List[float]
    text: str
    model_name: str
    processing_time: float

# VectorStore models
class VectorStoreRequest(BaseModel):
    """Request model for vector store operations."""
    operation: str  # 'add', 'update', 'delete', 'query'
    texts: Optional[List[str]] = None
    vectors: Optional[List[List[float]]] = None
    metadatas: Optional[List[Dict[str, Any]]] = None
    filters: Optional[Dict[str, Any]] = None

class VectorStoreResponse(BaseModel):
    """Response model for vector store operations."""
    success: bool
    message: str
    operation: str
    count: int = 0
    results: Optional[List[Any]] = None
    processing_time: float

# Vector visualization models
class VectorDataPoint(BaseModel):
    """Vector data point for visualization."""
    id: str
    content: str
    metadata: Dict[str, Any]
    vector: List[float]
    reduced_vector: List[float]

class GetVectorsRequest(BaseModel):
    """Request to get vectors for visualization."""
    tableName: str
    filter: Optional[Dict[str, Any]] = None
    maxPoints: Optional[int] = 500
    page: Optional[int] = 1
    pageSize: Optional[int] = 100
    clusteringAlgorithm: Optional[str] = "kmeans"  # Options: kmeans, dbscan, hdbscan
    dimensionalityReduction: Optional[str] = "tsne"  # Options: tsne, umap, pca

class GetVectorsResponse(BaseModel):
    """Response with vectors for visualization."""
    vectors: List[VectorDataPoint]
    total_count: Optional[int] = 0
    page: Optional[int] = 1
    page_size: Optional[int] = 100
    total_pages: Optional[int] = 1

# Debugging models
class DebugBreakpoint(BaseModel):
    """A breakpoint in a flow."""
    node_id: str
    enabled: bool = True
    condition: Optional[str] = None

class DebugNodeData(BaseModel):
    """Debug data for a node."""
    node_id: str
    input_data: Optional[Any] = None
    output_data: Optional[Any] = None
    execution_time: Optional[float] = None
    status: Literal["not_executed", "executing", "completed", "error"] = "not_executed"
    error: Optional[str] = None

class DebugSession(BaseModel):
    """Debug session for a flow."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    flow_id: Optional[str] = None
    breakpoints: List[DebugBreakpoint] = Field(default_factory=list)
    current_node_id: Optional[str] = None
    status: Literal["ready", "running", "paused", "completed", "error"] = "ready"
    node_data: Dict[str, DebugNodeData] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class CreateDebugSessionRequest(BaseModel):
    """Request to create a debug session."""
    flow: Flow
    breakpoints: Optional[List[DebugBreakpoint]] = None

class CreateDebugSessionResponse(BaseModel):
    """Response from creating a debug session."""
    session_id: str
    status: str
    message: str

class DebugStepRequest(BaseModel):
    """Request to step in a debug session."""
    session_id: str
    step_type: Literal["step", "step_over", "continue", "reset"] = "step"

class DebugStepResponse(BaseModel):
    """Response from stepping in a debug session."""
    session: DebugSession
    node_output: Optional[Any] = None
    execution_time: float = 0.0
    message: str

class SetBreakpointRequest(BaseModel):
    """Request to set a breakpoint."""
    session_id: str
    breakpoint: DebugBreakpoint

class SetBreakpointResponse(BaseModel):
    """Response from setting a breakpoint."""
    success: bool
    message: str

class GetVariablesRequest(BaseModel):
    """Request to get variables from a debug session."""
    session_id: str
    variable_names: Optional[List[str]] = None

class GetVariablesResponse(BaseModel):
    """Response with variables from a debug session."""
    variables: Dict[str, Any]