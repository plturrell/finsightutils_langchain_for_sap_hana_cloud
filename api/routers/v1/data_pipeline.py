"""
Data pipeline routes for version 1 of the API.

This module provides endpoints for tracking data transformations from
tables to vectors in SAP HANA Cloud.
"""

import logging
import uuid
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import Depends, Request, HTTPException, Path, Body
from pydantic import BaseModel, Field

from ...config_standardized import get_standardized_settings
from ...utils.standardized_exceptions import DatabaseException
from ...utils.error_utils import create_context_aware_error
from ..base import BaseRouter
from ..dependencies import get_current_user
from ...db import get_connection

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Import data pipeline components
try:
    from langchain_hana.reasoning.data_pipeline import (
        DataPipelineManager,
        DataPipeline,
        DataSourceMetadata,
        IntermediateRepresentation,
        VectorRepresentation,
        TransformationRule
    )
    DATA_PIPELINE_AVAILABLE = True
except ImportError:
    DATA_PIPELINE_AVAILABLE = False
    logger.warning("Data Pipeline module is not available. Install the full package with data pipeline components.")

# Models
class CreatePipelineRequest(BaseModel):
    """Request model for creating a new data pipeline."""
    connection_id: Optional[str] = Field(None, description="Optional connection ID for the database")


class CreatePipelineResponse(BaseModel):
    """Response model for pipeline creation."""
    pipeline_id: str = Field(..., description="Unique ID of the created pipeline")
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Informational message")


class RegisterDataSourceRequest(BaseModel):
    """Request model for registering a data source."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    schema_name: str = Field(..., description="Schema name for the table")
    table_name: str = Field(..., description="Name of the table")
    include_sample: bool = Field(True, description="Whether to include a data sample")
    sample_size: int = Field(5, description="Number of rows to include in the sample")
    custom_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the source")


class RegisterDataSourceResponse(BaseModel):
    """Response model for data source registration."""
    source_id: str = Field(..., description="Unique ID of the registered data source")
    pipeline_id: str = Field(..., description="ID of the pipeline")
    schema_name: str = Field(..., description="Schema name of the source table")
    table_name: str = Field(..., description="Name of the source table")
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Informational message")


class RegisterIntermediateStageRequest(BaseModel):
    """Request model for registering an intermediate transformation stage."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    stage_name: str = Field(..., description="Name of the transformation stage")
    stage_description: str = Field(..., description="Description of the transformation stage")
    source_id: str = Field(..., description="ID of the source data")
    column_mapping: Dict[str, List[str]] = Field(..., description="Mapping of input to output columns")
    data_sample: Optional[List[Dict[str, Any]]] = Field(None, description="Sample data after transformation")
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for processing")


class RegisterIntermediateStageResponse(BaseModel):
    """Response model for intermediate stage registration."""
    stage_id: str = Field(..., description="Unique ID of the registered stage")
    pipeline_id: str = Field(..., description="ID of the pipeline")
    stage_name: str = Field(..., description="Name of the transformation stage")
    source_id: str = Field(..., description="ID of the source data")
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Informational message")


class RegisterVectorRequest(BaseModel):
    """Request model for registering a vector representation."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    source_id: str = Field(..., description="ID of the source data")
    model_name: str = Field(..., description="Name of the embedding model")
    vector_dimensions: int = Field(..., description="Number of dimensions in the vector")
    vector_sample: Optional[List[float]] = Field(None, description="Sample vector embedding")
    original_text: Optional[str] = Field(None, description="Original text used to generate the vector")
    processing_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for processing")


class RegisterVectorResponse(BaseModel):
    """Response model for vector registration."""
    vector_id: str = Field(..., description="Unique ID of the registered vector")
    pipeline_id: str = Field(..., description="ID of the pipeline")
    source_id: str = Field(..., description="ID of the source data")
    model_name: str = Field(..., description="Name of the embedding model")
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Informational message")


class RegisterTransformationRuleRequest(BaseModel):
    """Request model for registering a transformation rule."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    rule_name: str = Field(..., description="Name of the transformation rule")
    rule_description: str = Field(..., description="Description of the rule")
    input_columns: List[str] = Field(..., description="Columns used as input")
    output_columns: List[str] = Field(..., description="Columns produced as output")
    transformation_type: str = Field(..., description="Type of transformation")
    transformation_params: Dict[str, Any] = Field(..., description="Parameters for the transformation")


class RegisterTransformationRuleResponse(BaseModel):
    """Response model for transformation rule registration."""
    rule_id: str = Field(..., description="Unique ID of the registered rule")
    pipeline_id: str = Field(..., description="ID of the pipeline")
    rule_name: str = Field(..., description="Name of the transformation rule")
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Informational message")


class GetPipelineRequest(BaseModel):
    """Request model for getting pipeline data."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    source_id: Optional[str] = Field(None, description="Optional ID of the specific source to filter by")


class GetPipelineResponse(BaseModel):
    """Response model for pipeline data."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    data_sources: Dict[str, Any] = Field(..., description="Data sources in the pipeline")
    intermediate_stages: Dict[str, Any] = Field(..., description="Intermediate transformation stages")
    vector_representations: Dict[str, Any] = Field(..., description="Vector representations")
    transformation_rules: Dict[str, Any] = Field(..., description="Transformation rules")
    created_at: float = Field(..., description="Creation timestamp")


class GetDataLineageRequest(BaseModel):
    """Request model for data lineage retrieval."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    vector_id: str = Field(..., description="ID of the vector")


class GetDataLineageResponse(BaseModel):
    """Response model for data lineage."""
    vector_id: str = Field(..., description="ID of the vector")
    vector_data: Dict[str, Any] = Field(..., description="Vector data")
    source_data: Dict[str, Any] = Field(..., description="Source data")
    transformation_stages: List[Dict[str, Any]] = Field(..., description="Transformation stages")
    created_at: float = Field(..., description="Creation timestamp")


class GetReverseMapRequest(BaseModel):
    """Request model for reverse mapping retrieval."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    vector_id: str = Field(..., description="ID of the vector")
    similarity_threshold: float = Field(0.8, description="Similarity threshold for matching")


class GetReverseMapResponse(BaseModel):
    """Response model for reverse mapping."""
    vector_id: str = Field(..., description="ID of the vector")
    source_data: Dict[str, Any] = Field(..., description="Source data")
    similar_vectors: List[Dict[str, Any]] = Field(..., description="Similar vectors")
    threshold: float = Field(..., description="Similarity threshold used")
    created_at: float = Field(..., description="Creation timestamp")


class ListPipelinesResponse(BaseModel):
    """Response model for pipeline listing."""
    pipelines: List[Dict[str, Any]] = Field(..., description="List of pipelines")
    count: int = Field(..., description="Number of pipelines")


class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Informational message")
    data: Dict[str, Any] = Field(..., description="Response data")


# Create router
router = BaseRouter(tags=["Data Pipeline"])


@router.post("/create", response_model=CreatePipelineResponse)
async def create_pipeline(
    request: Request,
    input_data: CreatePipelineRequest
):
    """
    Create a new data pipeline for tracking the transformation from tables to vectors.
    
    Initializes a new data pipeline for tracking the full lineage of data
    transformations from source tables to vector embeddings.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get database connection if needed
        connection = None
        if input_data.connection_id:
            connection = get_connection()
        
        # Create a new pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline_id = pipeline_manager.create_pipeline(connection)
        
        return CreatePipelineResponse(
            pipeline_id=pipeline_id,
            status="success",
            message="Data pipeline created successfully"
        )
    except Exception as e:
        logger.error(f"Error creating data pipeline: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "data_pipeline_creation",
            status_code=500,
            additional_context={
                "message": "Error creating data pipeline.",
                "suggestions": [
                    "Check that the database connection is valid.",
                    "Ensure that the data pipeline module is properly installed."
                ]
            }
        )


@router.post("/register-source", response_model=RegisterDataSourceResponse)
async def register_data_source(
    request: Request,
    input_data: RegisterDataSourceRequest
):
    """
    Register a HANA table as a data source in the pipeline.
    
    Adds a table from SAP HANA Cloud as a data source in the pipeline,
    capturing metadata and optionally a sample of the data.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(input_data.pipeline_id)
        
        # Get database connection
        connection = get_connection()
        pipeline.connection = connection
        
        # Register the data source
        source_id = pipeline.register_data_source(
            schema_name=input_data.schema_name,
            table_name=input_data.table_name,
            include_sample=input_data.include_sample,
            sample_size=input_data.sample_size,
            custom_metadata=input_data.custom_metadata
        )
        
        return RegisterDataSourceResponse(
            source_id=source_id,
            pipeline_id=input_data.pipeline_id,
            schema_name=input_data.schema_name,
            table_name=input_data.table_name,
            status="success",
            message="Data source registered successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering data source: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "data_source_registration",
            status_code=500,
            additional_context={
                "message": "Error registering data source.",
                "suggestions": [
                    "Check that the schema and table names are correct.",
                    "Verify that the table exists and is accessible.",
                    "Ensure that the pipeline ID is valid."
                ]
            }
        )


@router.post("/register-intermediate", response_model=RegisterIntermediateStageResponse)
async def register_intermediate_stage(
    request: Request,
    input_data: RegisterIntermediateStageRequest
):
    """
    Register an intermediate transformation stage in the pipeline.
    
    Records information about a transformation stage between source data
    and final vector representations.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(input_data.pipeline_id)
        
        # Register the intermediate stage
        stage_id = pipeline.register_intermediate_stage(
            stage_name=input_data.stage_name,
            stage_description=input_data.stage_description,
            source_id=input_data.source_id,
            column_mapping=input_data.column_mapping,
            data_sample=input_data.data_sample,
            processing_metadata=input_data.processing_metadata
        )
        
        return RegisterIntermediateStageResponse(
            stage_id=stage_id,
            pipeline_id=input_data.pipeline_id,
            stage_name=input_data.stage_name,
            source_id=input_data.source_id,
            status="success",
            message="Intermediate stage registered successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering intermediate stage: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "intermediate_stage_registration",
            status_code=500,
            additional_context={
                "message": "Error registering intermediate transformation stage.",
                "suggestions": [
                    "Check that the source ID is valid.",
                    "Verify that the column mapping is correct.",
                    "Ensure that the pipeline ID is valid."
                ]
            }
        )


@router.post("/register-vector", response_model=RegisterVectorResponse)
async def register_vector(
    request: Request,
    input_data: RegisterVectorRequest
):
    """
    Register a vector representation in the pipeline.
    
    Records information about a vector embedding created from source data,
    including the embedding model used and dimensionality.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(input_data.pipeline_id)
        
        # Register the vector representation
        vector_id = pipeline.register_vector_representation(
            source_id=input_data.source_id,
            model_name=input_data.model_name,
            vector_dimensions=input_data.vector_dimensions,
            vector_sample=input_data.vector_sample,
            original_text=input_data.original_text,
            processing_metadata=input_data.processing_metadata
        )
        
        return RegisterVectorResponse(
            vector_id=vector_id,
            pipeline_id=input_data.pipeline_id,
            source_id=input_data.source_id,
            model_name=input_data.model_name,
            status="success",
            message="Vector representation registered successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering vector representation: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "vector_registration",
            status_code=500,
            additional_context={
                "message": "Error registering vector representation.",
                "suggestions": [
                    "Check that the source ID is valid.",
                    "Verify that the vector dimensions are correct.",
                    "Ensure that the pipeline ID is valid."
                ]
            }
        )


@router.post("/register-rule", response_model=RegisterTransformationRuleResponse)
async def register_transformation_rule(
    request: Request,
    input_data: RegisterTransformationRuleRequest
):
    """
    Register a transformation rule in the pipeline.
    
    Records information about a transformation rule used to process data
    between source tables and vector embeddings.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(input_data.pipeline_id)
        
        # Register the transformation rule
        rule_id = pipeline.register_transformation_rule(
            rule_name=input_data.rule_name,
            rule_description=input_data.rule_description,
            input_columns=input_data.input_columns,
            output_columns=input_data.output_columns,
            transformation_type=input_data.transformation_type,
            transformation_params=input_data.transformation_params
        )
        
        return RegisterTransformationRuleResponse(
            rule_id=rule_id,
            pipeline_id=input_data.pipeline_id,
            rule_name=input_data.rule_name,
            status="success",
            message="Transformation rule registered successfully"
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error registering transformation rule: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "rule_registration",
            status_code=500,
            additional_context={
                "message": "Error registering transformation rule.",
                "suggestions": [
                    "Check that the input and output columns are valid.",
                    "Verify that the transformation type is supported.",
                    "Ensure that the pipeline ID is valid."
                ]
            }
        )


@router.post("/get", response_model=GetPipelineResponse)
async def get_pipeline(
    request: Request,
    input_data: GetPipelineRequest
):
    """
    Get the complete data pipeline visualization data.
    
    Retrieves the full pipeline data, including all sources, stages,
    vectors, and transformation rules.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(input_data.pipeline_id)
        
        # Get the complete pipeline data
        pipeline_data = pipeline.get_complete_pipeline(input_data.source_id)
        
        return GetPipelineResponse(**pipeline_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting pipeline data: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "pipeline_retrieval",
            status_code=500,
            additional_context={
                "message": "Error retrieving pipeline data.",
                "suggestions": [
                    "Check that the pipeline ID is valid.",
                    "Verify that the source ID is valid if specified."
                ]
            }
        )


@router.post("/lineage", response_model=GetDataLineageResponse)
async def get_data_lineage(
    request: Request,
    input_data: GetDataLineageRequest
):
    """
    Get data lineage for a specific vector.
    
    Retrieves the complete lineage for a vector, showing all transformation
    steps from source data to the final vector representation.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(input_data.pipeline_id)
        
        # Get the data lineage
        lineage_data = pipeline.get_data_lineage(input_data.vector_id)
        
        return GetDataLineageResponse(**lineage_data)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting data lineage: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "lineage_retrieval",
            status_code=500,
            additional_context={
                "message": "Error retrieving data lineage.",
                "suggestions": [
                    "Check that the pipeline ID is valid.",
                    "Verify that the vector ID is valid.",
                    "Ensure that the vector exists in the pipeline."
                ]
            }
        )


@router.post("/reverse-map", response_model=GetReverseMapResponse)
async def get_reverse_map(
    request: Request,
    input_data: GetReverseMapRequest
):
    """
    Get the reverse mapping from vector back to source data.
    
    Maps a vector back to its source data, and identifies other vectors
    that have similar source data based on the similarity threshold.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(input_data.pipeline_id)
        
        # Get the reverse mapping
        reverse_map = pipeline.get_reverse_mapping(
            input_data.vector_id,
            input_data.similarity_threshold
        )
        
        return GetReverseMapResponse(**reverse_map)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting reverse mapping: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "reverse_map_retrieval",
            status_code=500,
            additional_context={
                "message": "Error retrieving reverse mapping.",
                "suggestions": [
                    "Check that the pipeline ID is valid.",
                    "Verify that the vector ID is valid.",
                    "Ensure that the vector exists in the pipeline."
                ]
            }
        )


@router.get("/list", response_model=ListPipelinesResponse)
async def list_pipelines(
    request: Request
):
    """
    List all data pipelines.
    
    Returns a list of all data pipelines in the system.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline manager
        pipeline_manager = DataPipelineManager.get_instance()
        
        # List all pipelines
        pipelines = pipeline_manager.list_pipelines()
        
        return ListPipelinesResponse(
            pipelines=pipelines,
            count=len(pipelines)
        )
    except Exception as e:
        logger.error(f"Error listing pipelines: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "pipeline_listing",
            status_code=500,
            additional_context={
                "message": "Error listing data pipelines.",
                "suggestions": [
                    "Check that the data pipeline module is properly installed."
                ]
            }
        )


@router.get("/status", response_model=APIResponse)
async def get_data_pipeline_status(
    request: Request
):
    """
    Check if the data pipeline module is available and return its status.
    
    Returns information about the availability and version of the
    data pipeline module.
    """
    try:
        return APIResponse(
            success=True,
            message="Data Pipeline module status retrieved successfully",
            data={
                "available": DATA_PIPELINE_AVAILABLE,
                "version": "1.0.0" if DATA_PIPELINE_AVAILABLE else None
            }
        )
    except Exception as e:
        logger.error(f"Error getting data pipeline status: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "data_pipeline_status",
            status_code=500,
            additional_context={
                "message": "Error retrieving data pipeline module status.",
                "suggestions": [
                    "Check the server logs for more information.",
                    "Verify that the data pipeline module is properly installed."
                ]
            }
        )