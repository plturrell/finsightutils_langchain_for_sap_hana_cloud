"""
Models for the Arrow Flight API endpoints.

This module defines the data models for the Arrow Flight API endpoints,
including support for multi-GPU operations and high-performance data transfer.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

class FlightQueryRequest(BaseModel):
    """
    Request model for creating a Flight query.
    
    This model represents the parameters for retrieving vectors using
    Arrow Flight protocol.
    """
    
    table_name: str = Field(..., description="The name of the vector table")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional filter criteria")
    limit: int = Field(1000, description="Maximum number of vectors to retrieve")
    offset: int = Field(0, description="Offset for pagination")

class FlightQueryResponse(BaseModel):
    """
    Response model for a Flight query.
    
    This model contains the ticket and location information needed to
    retrieve vectors using Arrow Flight protocol.
    """
    
    ticket: str = Field(..., description="The Flight ticket as a JSON string")
    location: str = Field(..., description="The Flight server location")
    schema: Optional[Dict[str, Any]] = Field(None, description="Optional schema information")

class FlightUploadRequest(BaseModel):
    """
    Request model for uploading vectors using Flight.
    
    This model represents the parameters for uploading vectors using
    Arrow Flight protocol.
    """
    
    table_name: str = Field(..., description="The name of the vector table")
    create_if_not_exists: bool = Field(True, description="Create the table if it doesn't exist")
    
class FlightUploadResponse(BaseModel):
    """
    Response model for a Flight upload.
    
    This model contains the descriptor and location information needed to
    upload vectors using Arrow Flight protocol.
    """
    
    descriptor: str = Field(..., description="The Flight descriptor as a JSON string")
    location: str = Field(..., description="The Flight server location")

class FlightCollection(BaseModel):
    """
    Model for a vector collection available through Flight.
    
    This model represents a vector collection that can be accessed using
    Arrow Flight protocol.
    """
    
    name: str = Field(..., description="The name of the collection")
    total_records: int = Field(0, description="The total number of vectors in the collection")

class FlightListResponse(BaseModel):
    """
    Response model for listing available vector collections.
    
    This model contains the list of available vector collections that can be
    accessed using Arrow Flight protocol.
    """
    
    collections: List[Dict[str, Any]] = Field(..., description="List of available collections")
    location: str = Field(..., description="The Flight server location")

class FlightInfoResponse(BaseModel):
    """
    Response model for Flight server information.
    
    This model contains information about the Flight server configuration.
    """
    
    host: str = Field(..., description="The Flight server host")
    port: int = Field(..., description="The Flight server port")
    location: str = Field(..., description="The Flight server location URI")
    status: str = Field(..., description="The Flight server status (running or stopped)")


class FlightMultiGPUConfig(BaseModel):
    """
    Configuration model for multi-GPU Flight operations.
    
    This model represents the configuration for multi-GPU operations using
    Arrow Flight protocol.
    """
    
    enabled: bool = Field(True, description="Whether multi-GPU mode is enabled")
    gpu_ids: Optional[List[int]] = Field(None, description="List of GPU device IDs to use")
    distribution_strategy: str = Field("round_robin", description="Strategy for distributing work across GPUs")
    batch_size: int = Field(1000, description="Default batch size for operations")
    memory_fraction: float = Field(0.8, description="Maximum fraction of GPU memory to use")


class FlightMultiGPURequest(BaseModel):
    """
    Request model for multi-GPU Flight operations.
    
    This model represents the parameters for operations using multiple GPUs
    with Arrow Flight protocol.
    """
    
    table_name: str = Field(..., description="The name of the vector table")
    config: Optional[FlightMultiGPUConfig] = Field(None, description="Multi-GPU configuration")


class FlightMultiGPUResponse(BaseModel):
    """
    Response model for multi-GPU Flight operations.
    
    This model contains information about the multi-GPU configuration and
    performance metrics.
    """
    
    gpu_ids: List[int] = Field(..., description="List of GPU device IDs used")
    num_gpus: int = Field(..., description="Number of GPUs used")
    batch_sizes: List[int] = Field(..., description="Optimal batch sizes for each GPU")
    location: str = Field(..., description="The Flight server location")
    config: FlightMultiGPUConfig = Field(..., description="Multi-GPU configuration")