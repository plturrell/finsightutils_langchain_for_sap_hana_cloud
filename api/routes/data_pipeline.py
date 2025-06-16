from fastapi import APIRouter, Depends, HTTPException, Path, Body
from typing import Dict, List, Optional, Any
import uuid
import json
import os
import time
from datetime import datetime
from pydantic import BaseModel, Field

# Import database helpers and models
from ..db import get_db_connection
from ..models import APIResponse

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

# Import logging
import logging
from ..error_utils import create_context_aware_error
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/data-pipeline",
    tags=["data-pipeline"],
)

# Models for API requests and responses
class CreatePipelineRequest(BaseModel):
    connection_id: Optional[str] = None

class CreatePipelineResponse(BaseModel):
    pipeline_id: str
    status: str
    message: str

class RegisterDataSourceRequest(BaseModel):
    pipeline_id: str
    schema_name: str
    table_name: str
    include_sample: bool = True
    sample_size: int = 5
    custom_metadata: Optional[Dict[str, Any]] = None

class RegisterDataSourceResponse(BaseModel):
    source_id: str
    pipeline_id: str
    schema_name: str
    table_name: str
    status: str
    message: str

class RegisterIntermediateStageRequest(BaseModel):
    pipeline_id: str
    stage_name: str
    stage_description: str
    source_id: str
    column_mapping: Dict[str, List[str]]
    data_sample: Optional[List[Dict[str, Any]]] = None
    processing_metadata: Optional[Dict[str, Any]] = None

class RegisterIntermediateStageResponse(BaseModel):
    stage_id: str
    pipeline_id: str
    stage_name: str
    source_id: str
    status: str
    message: str

class RegisterVectorRequest(BaseModel):
    pipeline_id: str
    source_id: str
    model_name: str
    vector_dimensions: int
    vector_sample: Optional[List[float]] = None
    original_text: Optional[str] = None
    processing_metadata: Optional[Dict[str, Any]] = None

class RegisterVectorResponse(BaseModel):
    vector_id: str
    pipeline_id: str
    source_id: str
    model_name: str
    status: str
    message: str

class RegisterTransformationRuleRequest(BaseModel):
    pipeline_id: str
    rule_name: str
    rule_description: str
    input_columns: List[str]
    output_columns: List[str]
    transformation_type: str
    transformation_params: Dict[str, Any]

class RegisterTransformationRuleResponse(BaseModel):
    rule_id: str
    pipeline_id: str
    rule_name: str
    status: str
    message: str

class GetPipelineRequest(BaseModel):
    pipeline_id: str
    source_id: Optional[str] = None

class GetPipelineResponse(BaseModel):
    pipeline_id: str
    data_sources: Dict[str, Any]
    intermediate_stages: Dict[str, Any]
    vector_representations: Dict[str, Any]
    transformation_rules: Dict[str, Any]
    created_at: float

class GetDataLineageRequest(BaseModel):
    pipeline_id: str
    vector_id: str

class GetDataLineageResponse(BaseModel):
    vector_id: str
    vector_data: Dict[str, Any]
    source_data: Dict[str, Any]
    transformation_stages: List[Dict[str, Any]]
    created_at: float

class GetReverseMapRequest(BaseModel):
    pipeline_id: str
    vector_id: str
    similarity_threshold: float = 0.8

class GetReverseMapResponse(BaseModel):
    vector_id: str
    source_data: Dict[str, Any]
    similar_vectors: List[Dict[str, Any]]
    threshold: float
    created_at: float

class ListPipelinesResponse(BaseModel):
    pipelines: List[Dict[str, Any]]
    count: int

@router.post("/create", response_model=CreatePipelineResponse)
async def create_pipeline(request: CreatePipelineRequest):
    """
    Create a new data pipeline for tracking the transformation from tables to vectors.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get database connection if needed
        connection = None
        if request.connection_id:
            connection = get_db_connection()
        
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
async def register_data_source(request: RegisterDataSourceRequest):
    """
    Register a HANA table as a data source in the pipeline.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(request.pipeline_id)
        
        # Get database connection
        connection = get_db_connection()
        pipeline.connection = connection
        
        # Register the data source
        source_id = pipeline.register_data_source(
            schema_name=request.schema_name,
            table_name=request.table_name,
            include_sample=request.include_sample,
            sample_size=request.sample_size,
            custom_metadata=request.custom_metadata
        )
        
        return RegisterDataSourceResponse(
            source_id=source_id,
            pipeline_id=request.pipeline_id,
            schema_name=request.schema_name,
            table_name=request.table_name,
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
async def register_intermediate_stage(request: RegisterIntermediateStageRequest):
    """
    Register an intermediate transformation stage in the pipeline.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(request.pipeline_id)
        
        # Register the intermediate stage
        stage_id = pipeline.register_intermediate_stage(
            stage_name=request.stage_name,
            stage_description=request.stage_description,
            source_id=request.source_id,
            column_mapping=request.column_mapping,
            data_sample=request.data_sample,
            processing_metadata=request.processing_metadata
        )
        
        return RegisterIntermediateStageResponse(
            stage_id=stage_id,
            pipeline_id=request.pipeline_id,
            stage_name=request.stage_name,
            source_id=request.source_id,
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
async def register_vector(request: RegisterVectorRequest):
    """
    Register a vector representation in the pipeline.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(request.pipeline_id)
        
        # Register the vector representation
        vector_id = pipeline.register_vector_representation(
            source_id=request.source_id,
            model_name=request.model_name,
            vector_dimensions=request.vector_dimensions,
            vector_sample=request.vector_sample,
            original_text=request.original_text,
            processing_metadata=request.processing_metadata
        )
        
        return RegisterVectorResponse(
            vector_id=vector_id,
            pipeline_id=request.pipeline_id,
            source_id=request.source_id,
            model_name=request.model_name,
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
async def register_transformation_rule(request: RegisterTransformationRuleRequest):
    """
    Register a transformation rule in the pipeline.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(request.pipeline_id)
        
        # Register the transformation rule
        rule_id = pipeline.register_transformation_rule(
            rule_name=request.rule_name,
            rule_description=request.rule_description,
            input_columns=request.input_columns,
            output_columns=request.output_columns,
            transformation_type=request.transformation_type,
            transformation_params=request.transformation_params
        )
        
        return RegisterTransformationRuleResponse(
            rule_id=rule_id,
            pipeline_id=request.pipeline_id,
            rule_name=request.rule_name,
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
async def get_pipeline(request: GetPipelineRequest):
    """
    Get the complete data pipeline visualization data.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(request.pipeline_id)
        
        # Get the complete pipeline data
        pipeline_data = pipeline.get_complete_pipeline(request.source_id)
        
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
async def get_data_lineage(request: GetDataLineageRequest):
    """
    Get data lineage for a specific vector.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(request.pipeline_id)
        
        # Get the data lineage
        lineage_data = pipeline.get_data_lineage(request.vector_id)
        
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
async def get_reverse_map(request: GetReverseMapRequest):
    """
    Get the reverse mapping from vector back to source data.
    """
    if not DATA_PIPELINE_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Data Pipeline module is not available. Install the full package with data pipeline components."
        )
    
    try:
        # Get the pipeline
        pipeline_manager = DataPipelineManager.get_instance()
        pipeline = pipeline_manager.get_pipeline(request.pipeline_id)
        
        # Get the reverse mapping
        reverse_map = pipeline.get_reverse_mapping(
            request.vector_id,
            request.similarity_threshold
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
async def list_pipelines():
    """
    List all data pipelines.
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
async def get_data_pipeline_status():
    """
    Check if the data pipeline module is available and return its status.
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