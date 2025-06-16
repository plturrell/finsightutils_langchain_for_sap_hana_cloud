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

# Import langchain components
from langchain_hana import HanaVectorStore
from langchain_hana.embeddings import HanaInternalEmbeddings
try:
    # Import the reasoning framework components if available
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

# Import logging
import logging
from ..error_utils import create_context_aware_error
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/reasoning",
    tags=["reasoning"],
)

# Models for API requests and responses
class ReasoningPathRequest(BaseModel):
    query: str
    table_name: str = "EMBEDDINGS"
    document_id: Optional[str] = None
    include_content: bool = True
    max_steps: int = 5

class ReasoningPathResponse(BaseModel):
    path_id: str
    query: str
    steps: List[Dict[str, Any]]
    final_result: Optional[str] = None
    metadata: Dict[str, Any] = {}
    execution_time: float

class TransformationRequest(BaseModel):
    document_id: str
    table_name: str = "EMBEDDINGS"
    include_intermediate: bool = True
    track_only: bool = False

class TransformationResponse(BaseModel):
    transformation_id: str
    document_id: str
    stages: List[Dict[str, Any]]
    metadata: Dict[str, Any] = {}
    execution_time: float

class ValidationRequest(BaseModel):
    reasoning_path_id: str
    validation_types: List[str] = ["consistency", "calculations", "citations", "hallucination"]

class ValidationResponse(BaseModel):
    validation_id: str
    reasoning_path_id: str
    results: Dict[str, Any]
    score: float
    suggestions: List[str] = []
    execution_time: float

class MetricsRequest(BaseModel):
    document_id: Optional[str] = None
    table_name: str = "EMBEDDINGS"
    metric_types: List[str] = ["cosine_similarity", "information_retention", "structural_integrity"]

class MetricsResponse(BaseModel):
    metrics_id: str
    document_id: Optional[str]
    metrics: Dict[str, Any]
    execution_time: float

class FeedbackRequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    reasoning_path_id: Optional[str] = None
    feedback_type: str
    feedback_content: str
    rating: Optional[int] = None
    user_id: Optional[str] = None

class FeedbackResponse(BaseModel):
    feedback_id: str
    status: str
    message: str

class FingerprintRequest(BaseModel):
    document_id: str
    table_name: str = "EMBEDDINGS"
    include_lineage: bool = True

class FingerprintResponse(BaseModel):
    fingerprint_id: str
    document_id: str
    signatures: Dict[str, Any]
    lineage: Optional[Dict[str, Any]] = None
    execution_time: float

@router.post("/track", response_model=ReasoningPathResponse)
async def track_reasoning_path(request: ReasoningPathRequest):
    """
    Track the reasoning path through vector data for a given query.
    This shows how the system navigates from the query to the final answer.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_db_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize vector store
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=request.table_name
        )
        
        # Initialize reasoning path tracker
        path_tracker = ReasoningPathTracker(
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        
        # Track reasoning path
        path_id = str(uuid.uuid4())
        
        if request.document_id:
            # Get specific document by ID
            result = vector_store.get_by_id(request.document_id)
            if not result:
                raise HTTPException(status_code=404, detail=f"Document with ID {request.document_id} not found")
                
            # Track reasoning path for this specific document
            reasoning_path = path_tracker.track_for_document(
                query=request.query,
                document=result,
                max_steps=request.max_steps
            )
        else:
            # Track reasoning path across all documents
            reasoning_path = path_tracker.track(
                query=request.query,
                max_steps=request.max_steps
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
                        "content": ref.content if request.include_content else None
                    }
                    for ref in step.references
                ]
            }
            steps.append(step_data)
        
        execution_time = time.time() - start_time
        
        return ReasoningPathResponse(
            path_id=path_id,
            query=request.query,
            steps=steps,
            final_result=reasoning_path.conclusion,
            metadata={
                "step_count": len(steps),
                "average_confidence": sum(step.confidence for step in reasoning_path.steps) / len(reasoning_path.steps) if reasoning_path.steps else 0,
                "created_at": datetime.now().isoformat()
            },
            execution_time=execution_time
        )
    except HTTPException as e:
        # Pass through already formatted HTTP exceptions
        raise e
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
async def track_transformation(request: TransformationRequest):
    """
    Track the transformation of data into vector embeddings.
    This shows how source data is processed and transformed into vectors.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_db_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize vector store
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=request.table_name
        )
        
        # Get document by ID
        document = vector_store.get_by_id(request.document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {request.document_id} not found")
        
        # Initialize transformation tracker
        transformation_tracker = TransformationTracker(
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        
        # Track transformation
        transformation_id = str(uuid.uuid4())
        
        if request.track_only:
            # Just track the transformation metadata without re-running the transformation
            transformation = transformation_tracker.get_transformation_metadata(document)
        else:
            # Track full transformation pipeline
            transformation = transformation_tracker.track_transformation(
                document=document,
                include_intermediate=request.include_intermediate
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
            
            if request.include_intermediate and hasattr(stage, 'sample_output'):
                stage_data["sample_output"] = stage.sample_output
                
            stages.append(stage_data)
        
        execution_time = time.time() - start_time
        
        return TransformationResponse(
            transformation_id=transformation_id,
            document_id=request.document_id,
            stages=stages,
            metadata={
                "stage_count": len(stages),
                "total_transformation_time_ms": sum(stage.duration_ms for stage in transformation.stages),
                "created_at": datetime.now().isoformat()
            },
            execution_time=execution_time
        )
    except HTTPException as e:
        # Pass through already formatted HTTP exceptions
        raise e
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
async def validate_reasoning(request: ValidationRequest):
    """
    Validate the quality of a reasoning path using various validation techniques.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_db_connection()
        
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
            reasoning_path_id=request.reasoning_path_id,
            validation_types=request.validation_types
        )
        
        # Calculate overall score (0-1)
        score = validator.calculate_score(validation_results)
        
        # Generate suggestions
        suggestions = validator.generate_suggestions(validation_results)
        
        execution_time = time.time() - start_time
        
        return ValidationResponse(
            validation_id=validation_id,
            reasoning_path_id=request.reasoning_path_id,
            results=validation_results,
            score=score,
            suggestions=suggestions,
            execution_time=execution_time
        )
    except HTTPException as e:
        # Pass through already formatted HTTP exceptions
        raise e
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
async def calculate_metrics(request: MetricsRequest):
    """
    Calculate metrics for information preservation in vector embeddings.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_db_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize vector store
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=request.table_name
        )
        
        # Initialize metrics calculator
        metrics_calculator = InformationPreservationMetrics(
            vector_store=vector_store,
            embedding_model=embedding_model
        )
        
        # Calculate metrics
        metrics_id = str(uuid.uuid4())
        
        if request.document_id:
            # Get document by ID
            document = vector_store.get_by_id(request.document_id)
            if not document:
                raise HTTPException(status_code=404, detail=f"Document with ID {request.document_id} not found")
                
            # Calculate metrics for specific document
            metrics = metrics_calculator.calculate_for_document(
                document=document,
                metric_types=request.metric_types
            )
        else:
            # Calculate metrics across the entire table
            metrics = metrics_calculator.calculate_global(
                metric_types=request.metric_types
            )
        
        execution_time = time.time() - start_time
        
        return MetricsResponse(
            metrics_id=metrics_id,
            document_id=request.document_id,
            metrics=metrics,
            execution_time=execution_time
        )
    except HTTPException as e:
        # Pass through already formatted HTTP exceptions
        raise e
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
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on the reasoning process.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        # Get database connection
        conn = get_db_connection()
        
        # Initialize feedback collector
        feedback_collector = UserFeedbackCollector(
            connection=conn
        )
        
        # Submit feedback
        feedback_id = str(uuid.uuid4())
        
        feedback_collector.collect(
            feedback_id=feedback_id,
            query=request.query,
            document_id=request.document_id,
            reasoning_path_id=request.reasoning_path_id,
            feedback_type=request.feedback_type,
            feedback_content=request.feedback_content,
            rating=request.rating,
            user_id=request.user_id or "anonymous"
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
async def get_information_fingerprint(request: FingerprintRequest):
    """
    Get the information fingerprint for a document.
    This tracks the lineage of information through transformations.
    """
    if not REASONING_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Reasoning framework is not available. Install the full package with reasoning components."
        )
    
    try:
        start_time = time.time()
        
        # Get database connection
        conn = get_db_connection()
        
        # Initialize embedding model
        embedding_model = HanaInternalEmbeddings(
            internal_embedding_model_id="SAP_NEB.20240715"
        )
        
        # Initialize vector store
        vector_store = HanaVectorStore(
            connection=conn,
            embedding=embedding_model,
            table_name=request.table_name
        )
        
        # Get document by ID
        document = vector_store.get_by_id(request.document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {request.document_id} not found")
        
        # Initialize fingerprint manager
        fingerprint_manager = InformationFingerprint.get_manager()
        
        # Get information fingerprint
        fingerprint_id = str(uuid.uuid4())
        
        fingerprint = fingerprint_manager.get_fingerprint(document)
        
        signatures = fingerprint.to_dict()
        lineage = None
        
        if request.include_lineage:
            lineage = fingerprint_manager.get_lineage(document)
        
        execution_time = time.time() - start_time
        
        return FingerprintResponse(
            fingerprint_id=fingerprint_id,
            document_id=request.document_id,
            signatures=signatures,
            lineage=lineage,
            execution_time=execution_time
        )
    except HTTPException as e:
        # Pass through already formatted HTTP exceptions
        raise e
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
async def get_reasoning_status():
    """
    Check if the reasoning framework is available and return its status.
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