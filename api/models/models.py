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
    texts: List[str]
    metadatas: Optional[List[Dict[str, Any]]] = None
    table_name: Optional[str] = None


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