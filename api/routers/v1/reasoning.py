"""
Reasoning routes for version 1 of the API.

This module provides endpoints for reasoning, explanation, validation,
and tracking of vector operations in SAP HANA Cloud.
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

# Import langchain components
try:
    from langchain_hana import HanaVectorStore
    from langchain_hana.embeddings import HanaInternalEmbeddings
    VECTOR_STORE_AVAILABLE = True
except ImportError:
    VECTOR_STORE_AVAILABLE = False
    logger.warning("HANA Vector Store is not available. Install the langchain-hana package.")

# Import reasoning framework components
try:
    from langchain_hana.reasoning import (
        ReasoningPathTracker,
        TransformationTracker,
        ReasoningValidator,
        InformationPreservationMetrics,
        UserFeedbackCollector,
        InformationFingerprint
    )
    REASONING_AVAILABLE = True
except ImportError:
    REASONING_AVAILABLE = False
    logger.warning("Reasoning framework is not available. Install the full package with reasoning components.")

# Models for API requests and responses
class ReasoningPathRequest(BaseModel):
    """Request model for tracking reasoning paths."""
    query: str = Field(..., description="The query to reason about")
    table_name: str = Field("EMBEDDINGS", description="Name of the vector table")
    document_id: Optional[str] = Field(None, description="Optional specific document ID to focus on")
    include_content: bool = Field(True, description="Whether to include document content in the response")
    max_steps: int = Field(5, description="Maximum number of reasoning steps to track")


class ReasoningPathResponse(BaseModel):
    """Response model for reasoning path tracking."""
    path_id: str = Field(..., description="Unique ID for the reasoning path")
    query: str = Field(..., description="The original query")
    steps: List[Dict[str, Any]] = Field(..., description="Reasoning steps with references")
    final_result: Optional[str] = Field(None, description="Final conclusion from reasoning")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata about the reasoning path")
    execution_time: float = Field(..., description="Time taken to process in seconds")


class TransformationRequest(BaseModel):
    """Request model for transformation tracking."""
    document_id: str = Field(..., description="ID of the document to track transformations for")
    table_name: str = Field("EMBEDDINGS", description="Name of the vector table")
    include_intermediate: bool = Field(True, description="Whether to include intermediate transformation data")
    track_only: bool = Field(False, description="Whether to only track metadata without re-running transformations")


class TransformationResponse(BaseModel):
    """Response model for transformation tracking."""
    transformation_id: str = Field(..., description="Unique ID for the transformation")
    document_id: str = Field(..., description="ID of the document")
    stages: List[Dict[str, Any]] = Field(..., description="Transformation stages")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata about the transformation")
    execution_time: float = Field(..., description="Time taken to process in seconds")


class ValidationRequest(BaseModel):
    """Request model for reasoning validation."""
    reasoning_path_id: str = Field(..., description="ID of the reasoning path to validate")
    validation_types: List[str] = Field(["consistency", "calculations", "citations", "hallucination"], 
                                     description="Types of validation to perform")


class ValidationResponse(BaseModel):
    """Response model for reasoning validation."""
    validation_id: str = Field(..., description="Unique ID for the validation")
    reasoning_path_id: str = Field(..., description="ID of the validated reasoning path")
    results: Dict[str, Any] = Field(..., description="Validation results by type")
    score: float = Field(..., description="Overall validation score (0-1)")
    suggestions: List[str] = Field([], description="Suggestions for improvement")
    execution_time: float = Field(..., description="Time taken to process in seconds")


class MetricsRequest(BaseModel):
    """Request model for metrics calculation."""
    document_id: Optional[str] = Field(None, description="Optional document ID to calculate metrics for")
    table_name: str = Field("EMBEDDINGS", description="Name of the vector table")
    metric_types: List[str] = Field(["cosine_similarity", "information_retention", "structural_integrity"], 
                                  description="Types of metrics to calculate")


class MetricsResponse(BaseModel):
    """Response model for metrics calculation."""
    metrics_id: str = Field(..., description="Unique ID for the metrics calculation")
    document_id: Optional[str] = Field(None, description="ID of the document if specified")
    metrics: Dict[str, Any] = Field(..., description="Calculated metrics by type")
    execution_time: float = Field(..., description="Time taken to process in seconds")


class FeedbackRequest(BaseModel):
    """Request model for user feedback."""
    query: str = Field(..., description="The query that feedback relates to")
    document_id: Optional[str] = Field(None, description="Optional document ID that feedback relates to")
    reasoning_path_id: Optional[str] = Field(None, description="Optional reasoning path ID that feedback relates to")
    feedback_type: str = Field(..., description="Type of feedback being provided")
    feedback_content: str = Field(..., description="Content of the feedback")
    rating: Optional[int] = Field(None, description="Optional numerical rating")
    user_id: Optional[str] = Field(None, description="Optional ID of the user providing feedback")


class FeedbackResponse(BaseModel):
    """Response model for user feedback."""
    feedback_id: str = Field(..., description="Unique ID for the feedback")
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Informational message")


class FingerprintRequest(BaseModel):
    """Request model for information fingerprinting."""
    document_id: str = Field(..., description="ID of the document to fingerprint")
    table_name: str = Field("EMBEDDINGS", description="Name of the vector table")
    include_lineage: bool = Field(True, description="Whether to include information lineage")


class FingerprintResponse(BaseModel):
    """Response model for information fingerprinting."""
    fingerprint_id: str = Field(..., description="Unique ID for the fingerprint")
    document_id: str = Field(..., description="ID of the document")
    signatures: Dict[str, Any] = Field(..., description="Information signatures")
    lineage: Optional[Dict[str, Any]] = Field(None, description="Information lineage if requested")
    execution_time: float = Field(..., description="Time taken to process in seconds")


class APIResponse(BaseModel):
    """Generic API response model."""
    success: bool = Field(..., description="Whether the operation was successful")
    message: str = Field(..., description="Informational message")
    data: Dict[str, Any] = Field(..., description="Response data")


# Create router
router = BaseRouter(tags=["Reasoning"])


@router.post("/track", response_model=ReasoningPathResponse)
async def track_reasoning_path(
    request: Request,
    input_data: ReasoningPathRequest
):
    """
    Track the reasoning path through vector data for a given query.
    
    Shows how the system navigates from the query to the final answer
    by tracking reasoning steps and document references.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize vector store
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=input_data.table_name
        )
        
        # Initialize reasoning path tracker
        path_tracker = ReasoningPathTracker(
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        
        # Track reasoning path
        path_id = str(uuid.uuid4())
        
        if input_data.document_id:
            # Get specific document by ID
            result = vector_store.get_by_id(input_data.document_id)
            if not result:
                raise HTTPException(status_code=404, detail=f"Document with ID {input_data.document_id} not found")
                
            # Track reasoning path for this specific document
            reasoning_path = path_tracker.track_for_document(
                query=input_data.query,
                document=result,
                max_steps=input_data.max_steps
            )
        else:
            # Track reasoning path across all documents
            reasoning_path = path_tracker.track(
                query=input_data.query,
                max_steps=input_data.max_steps
            )
        
        # Format the response
        steps = []
        for i, step in enumerate(reasoning_path.steps):
            step_data = {
                "step_number": i + 1,
                "description": step.description,
                "reasoning": step.reasoning,
                "confidence": step.confidence,
                "references": [
                    {
                        "id": ref.document_id,
                        "relevance": ref.relevance,
                        "content": ref.content if input_data.include_content else None
                    }
                    for ref in step.references
                ]
            }
            steps.append(step_data)
        
        execution_time = time.time() - start_time
        
        return ReasoningPathResponse(
            path_id=path_id,
            query=input_data.query,
            steps=steps,
            final_result=reasoning_path.conclusion,
            metadata={
                "step_count": len(steps),
                "average_confidence": sum(step.confidence for step in reasoning_path.steps) / len(reasoning_path.steps) if reasoning_path.steps else 0,
                "created_at": datetime.now().isoformat()
            },
            execution_time=execution_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking reasoning path: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "reasoning_tracking",
            status_code=500,
            additional_context={
                "message": "Error tracking reasoning path.",
                "suggestions": [
                    "Verify that the query is valid.",
                    "Check that the table exists and contains vector embeddings.",
                    "Ensure the document ID is correct if specified."
                ]
            }
        )


@router.post("/transformation", response_model=TransformationResponse)
async def track_transformation(
    request: Request,
    input_data: TransformationRequest
):
    """
    Track the transformation of data into vector embeddings.
    
    Shows how source data is processed and transformed into vectors
    by tracking the transformation pipeline stages.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize vector store
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=input_data.table_name
        )
        
        # Get document by ID
        document = vector_store.get_by_id(input_data.document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {input_data.document_id} not found")
        
        # Initialize transformation tracker
        transformation_tracker = TransformationTracker(
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        
        # Track transformation
        transformation_id = str(uuid.uuid4())
        
        if input_data.track_only:
            # Just track the transformation metadata without re-running the transformation
            transformation = transformation_tracker.get_transformation_metadata(document)
        else:
            # Track full transformation pipeline
            transformation = transformation_tracker.track_transformation(
                document=document,
                include_intermediate=input_data.include_intermediate
            )
        
        # Format the response
        stages = []
        for i, stage in enumerate(transformation.stages):
            stage_data = {
                "stage_number": i + 1,
                "name": stage.name,
                "description": stage.description,
                "input_type": stage.input_type,
                "output_type": stage.output_type,
                "duration_ms": stage.duration_ms,
                "metadata": stage.metadata
            }
            
            if input_data.include_intermediate and hasattr(stage, 'sample_output'):
                stage_data["sample_output"] = stage.sample_output
                
            stages.append(stage_data)
        
        execution_time = time.time() - start_time
        
        return TransformationResponse(
            transformation_id=transformation_id,
            document_id=input_data.document_id,
            stages=stages,
            metadata={
                "stage_count": len(stages),
                "total_transformation_time_ms": sum(stage.duration_ms for stage in transformation.stages),
                "created_at": datetime.now().isoformat()
            },
            execution_time=execution_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error tracking transformation: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "transformation_tracking",
            status_code=500,
            additional_context={
                "message": "Error tracking transformation process.",
                "suggestions": [
                    "Verify that the document ID is valid.",
                    "Check that the table exists and contains vector embeddings.",
                    "Ensure the document has valid transformation metadata."
                ]
            }
        )


@router.post("/validate", response_model=ValidationResponse)
async def validate_reasoning(
    request: Request,
    input_data: ValidationRequest
):
    """
    Validate the quality of a reasoning path using various techniques.
    
    Assesses the quality of a reasoning path using validation techniques
    such as consistency checking, calculation verification, citation
    validation, and hallucination detection.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize reasoning validator
        validator = ReasoningValidator()
        
        # Validate reasoning path
        validation_id = str(uuid.uuid4())
        
        # Validate reasoning
        validation_results = validator.validate(
            reasoning_path_id=input_data.reasoning_path_id,
            validation_types=input_data.validation_types
        )
        
        # Calculate overall score (0-1)
        score = validator.calculate_score(validation_results)
        
        # Generate suggestions
        suggestions = validator.generate_suggestions(validation_results)
        
        execution_time = time.time() - start_time
        
        return ValidationResponse(
            validation_id=validation_id,
            reasoning_path_id=input_data.reasoning_path_id,
            results=validation_results,
            score=score,
            suggestions=suggestions,
            execution_time=execution_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating reasoning: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "reasoning_validation",
            status_code=500,
            additional_context={
                "message": "Error validating reasoning path.",
                "suggestions": [
                    "Verify that the reasoning path ID is valid.",
                    "Check that the validation types are supported.",
                    "Ensure the reasoning path has the necessary information for validation."
                ]
            }
        )


@router.post("/metrics", response_model=MetricsResponse)
async def calculate_metrics(
    request: Request,
    input_data: MetricsRequest
):
    """
    Calculate metrics for information preservation in vector embeddings.
    
    Computes metrics to assess how well information is preserved
    during the transformation from text to vector embeddings.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize vector store
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=input_data.table_name
        )
        
        # Initialize metrics calculator
        metrics_calculator = InformationPreservationMetrics(
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        
        # Calculate metrics
        metrics_id = str(uuid.uuid4())
        
        if input_data.document_id:
            # Get document by ID
            document = vector_store.get_by_id(input_data.document_id)
            if not document:
                raise HTTPException(status_code=404, detail=f"Document with ID {input_data.document_id} not found")
                
            # Calculate metrics for specific document
            metrics = metrics_calculator.calculate_for_document(
                document=document,
                metric_types=input_data.metric_types
            )
        else:
            # Calculate metrics across the entire table
            metrics = metrics_calculator.calculate_global(
                metric_types=input_data.metric_types
            )
        
        execution_time = time.time() - start_time
        
        return MetricsResponse(
            metrics_id=metrics_id,
            document_id=input_data.document_id,
            metrics=metrics,
            execution_time=execution_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "metrics_calculation",
            status_code=500,
            additional_context={
                "message": "Error calculating information preservation metrics.",
                "suggestions": [
                    "Verify that the document ID is valid if specified.",
                    "Check that the table exists and contains vector embeddings.",
                    "Ensure the metric types are supported."
                ]
            }
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: Request,
    input_data: FeedbackRequest
):
    """
    Submit user feedback on the reasoning process.
    
    Collects and processes user feedback about the reasoning results,
    which can be used to improve the system over time.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        # Get database connection
        conn = get_connection()
        
        # Initialize feedback collector
        feedback_collector = UserFeedbackCollector(
            connection=conn
        )
        
        # Submit feedback
        feedback_id = str(uuid.uuid4())
        
        feedback_collector.collect(
            feedback_id=feedback_id,
            query=input_data.query,
            document_id=input_data.document_id,
            reasoning_path_id=input_data.reasoning_path_id,
            feedback_type=input_data.feedback_type,
            feedback_content=input_data.feedback_content,
            rating=input_data.rating,
            user_id=input_data.user_id or "anonymous"
        )
        
        # Process feedback (this would typically be async in production)
        feedback_collector.process(feedback_id)
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="success",
            message="Feedback submitted and processed successfully"
        )
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "feedback_submission",
            status_code=500,
            additional_context={
                "message": "Error submitting user feedback.",
                "suggestions": [
                    "Verify that all required fields are provided.",
                    "Check that the feedback type is supported.",
                    "Ensure the document ID and reasoning path ID are valid if specified."
                ]
            }
        )


@router.post("/fingerprint", response_model=FingerprintResponse)
async def get_information_fingerprint(
    request: Request,
    input_data: FingerprintRequest
):
    """
    Get the information fingerprint for a document.
    
    Generates a unique fingerprint that tracks the lineage of information
    through transformations, useful for provenance tracking.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize vector store
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=input_data.table_name
        )
        
        # Get document by ID
        document = vector_store.get_by_id(input_data.document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {input_data.document_id} not found")
        
        # Initialize fingerprint manager
        fingerprint_manager = InformationFingerprint.get_manager()
        
        # Get information fingerprint
        fingerprint_id = str(uuid.uuid4())
        
        fingerprint = fingerprint_manager.get_fingerprint(document)
        
        signatures = fingerprint.to_dict()
        lineage = None
        
        if input_data.include_lineage:
            lineage = fingerprint_manager.get_lineage(document)
        
        execution_time = time.time() - start_time
        
        return FingerprintResponse(
            fingerprint_id=fingerprint_id,
            document_id=input_data.document_id,
            signatures=signatures,
            lineage=lineage,
            execution_time=execution_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting information fingerprint: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "fingerprint_retrieval",
            status_code=500,
            additional_context={
                "message": "Error retrieving information fingerprint.",
                "suggestions": [
                    "Verify that the document ID is valid.",
                    "Check that the table exists and contains vector embeddings.",
                    "Ensure the document has valid fingerprint metadata."
                ]
            }
        )


@router.get("/status", response_model=APIResponse)
async def get_reasoning_status(
    request: Request
):
    """
    Check if the reasoning framework is available and return its status.
    
    Returns information about the availability and capabilities of
    the reasoning framework components.
    """
    try:
        features = {
            "tracking": REASONING_AVAILABLE,
            "transformation": REASONING_AVAILABLE,
            "validation": REASONING_AVAILABLE,
            "metrics": REASONING_AVAILABLE,
            "feedback": REASONING_AVAILABLE,
            "fingerprinting": REASONING_AVAILABLE
        }
        
        return APIResponse(
            success=True,
            message="Reasoning framework status retrieved successfully",
            data={
                "available": REASONING_AVAILABLE,
                "features": features,
                "version": "1.0.0" if REASONING_AVAILABLE else None
            }
        )
    except Exception as e:
        logger.error(f"Error getting reasoning status: {str(e)}")
        raise create_context_aware_error(
            str(e),
            "reasoning_status",
            status_code=500,
            additional_context={
                "message": "Error retrieving reasoning framework status.",
                "suggestions": [
                    "Check the server logs for more information.",
                    "Verify that the reasoning framework is properly installed."
                ]
            }
        )