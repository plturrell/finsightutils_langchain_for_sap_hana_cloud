"""
API routes for advanced optimization features.

This module provides endpoints for:
1. Data valuation with DVRL
2. Interpretable embeddings with Neural Additive Models
3. Optimized hyperparameters with opt_list
4. Model compression with state_of_sparsity
"""

import logging
import os
import json
import tempfile
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, HTTPException, Body, Query, Depends
from pydantic import BaseModel, Field

# Import optimization components
from langchain_hana.optimization.data_valuation import DVRLDataValuation
from langchain_hana.optimization.interpretable_embeddings import NAMEmbeddings
from langchain_hana.optimization.hyperparameters import OptimizedHyperparameters
from langchain_hana.optimization.model_compression import SparseEmbeddingModel

# Import from core API components
from api.models.models import Document, EmbeddingResponse
from api.db import get_db_connection
from api.services.services import get_embedding_model, get_vectorstore

# Set up router
router = APIRouter(prefix="/optimization", tags=["optimization"])

# Set up logging
logger = logging.getLogger(__name__)


# Data valuation models
class DataValueRequest(BaseModel):
    documents: List[Document] = Field(..., description="List of documents to evaluate")
    threshold: Optional[float] = Field(0.5, description="Value threshold (0.0-1.0)")
    top_k: Optional[int] = Field(None, description="Number of top documents to return")
    
    class Config:
        schema_extra = {
            "example": {
                "documents": [
                    {"page_content": "This is a high quality document about finance.", 
                     "metadata": {"id": "doc_1", "category": "finance"}},
                    {"page_content": "Short text.", 
                     "metadata": {"id": "doc_2", "category": "other"}},
                ],
                "threshold": 0.7,
                "top_k": 10
            }
        }


class DataValueResponse(BaseModel):
    document_values: List[float] = Field(..., description="Document importance scores")
    valuable_documents: List[Document] = Field(..., description="Filtered valuable documents")
    total_documents: int = Field(..., description="Total number of documents")
    valuable_count: int = Field(..., description="Number of valuable documents")
    
    class Config:
        schema_extra = {
            "example": {
                "document_values": [0.82, 0.35],
                "valuable_documents": [
                    {"page_content": "This is a high quality document about finance.", 
                     "metadata": {"id": "doc_1", "category": "finance", "dvrl_value": 0.82}}
                ],
                "total_documents": 2,
                "valuable_count": 1
            }
        }


# Interpretable embeddings models
class SimilarityExplanationRequest(BaseModel):
    query: str = Field(..., description="Query text")
    document: str = Field(..., description="Document text")
    top_k: Optional[int] = Field(5, description="Number of top features to include")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Tell me about finance",
                "document": "This document discusses financial markets and investment strategies.",
                "top_k": 5
            }
        }


class SimilarityExplanationResponse(BaseModel):
    similarity_score: float = Field(..., description="Overall similarity score")
    top_matching_features: List[Dict[str, Any]] = Field(..., description="Top matching features")
    least_matching_features: List[Dict[str, Any]] = Field(..., description="Least matching features")
    query: str = Field(..., description="Original query")
    document: str = Field(..., description="Original document")
    
    class Config:
        schema_extra = {
            "example": {
                "similarity_score": 0.78,
                "top_matching_features": [
                    {"feature": "feature_12", "importance": 0.23},
                    {"feature": "feature_45", "importance": 0.18}
                ],
                "least_matching_features": [
                    {"feature": "feature_3", "importance": -0.05},
                    {"feature": "feature_27", "importance": -0.03}
                ],
                "query": "Tell me about finance",
                "document": "This document discusses financial markets and investment strategies."
            }
        }


# Hyperparameter optimization models
class HyperparametersRequest(BaseModel):
    model_size: int = Field(..., description="Number of parameters in the model")
    batch_size: int = Field(..., description="Batch size for training")
    dataset_size: Optional[int] = Field(None, description="Size of the dataset")
    embedding_dimension: Optional[int] = Field(None, description="Dimension of embeddings")
    vocabulary_size: Optional[int] = Field(None, description="Size of vocabulary")
    max_sequence_length: Optional[int] = Field(None, description="Maximum sequence length")
    
    class Config:
        schema_extra = {
            "example": {
                "model_size": 10000000,
                "batch_size": 32,
                "dataset_size": 50000,
                "embedding_dimension": 768,
                "vocabulary_size": 30000,
                "max_sequence_length": 512
            }
        }


class HyperparametersResponse(BaseModel):
    learning_rate: float = Field(..., description="Optimized learning rate")
    batch_size: int = Field(..., description="Optimized batch size")
    embedding_parameters: Optional[Dict[str, Any]] = Field(None, description="Embedding model parameters")
    training_schedule: Optional[Dict[str, Any]] = Field(None, description="Training schedule")
    
    class Config:
        schema_extra = {
            "example": {
                "learning_rate": 0.0003,
                "batch_size": 64,
                "embedding_parameters": {
                    "dropout_rate": 0.2,
                    "weight_decay": 0.01,
                    "hidden_dimension": 3072
                },
                "training_schedule": {
                    "warmup_steps": 100,
                    "total_steps": 15000,
                    "base_learning_rate": 0.0003
                }
            }
        }


# Model compression models
class CompressionRequest(BaseModel):
    texts: List[str] = Field(..., description="Texts to generate embeddings for")
    compression_ratio: float = Field(0.5, description="Target compression ratio (0.0-1.0)")
    compression_strategy: str = Field("magnitude", description="Compression strategy")
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "This is the first document to embed.",
                    "This is the second document to embed."
                ],
                "compression_ratio": 0.7,
                "compression_strategy": "magnitude"
            }
        }


class CompressionResponse(BaseModel):
    embeddings: List[List[float]] = Field(..., description="Compressed embeddings")
    compression_stats: Dict[str, Any] = Field(..., description="Compression statistics")
    
    class Config:
        schema_extra = {
            "example": {
                "embeddings": [
                    [0.1, 0.0, 0.3, 0.0, 0.5],
                    [0.0, 0.2, 0.0, 0.4, 0.0]
                ],
                "compression_stats": {
                    "compression_ratio": 0.7,
                    "compression_strategy": "magnitude",
                    "total_sparsity": 0.65,
                    "compressed_shapes": {
                        "shape_768": {
                            "elements": 1536,
                            "nonzeros": 538,
                            "sparsity": 0.65
                        }
                    }
                }
            }
        }


# Data valuation endpoints
@router.post("/data-valuation", response_model=DataValueResponse, 
             summary="Evaluate document importance")
async def evaluate_documents(
    request: DataValueRequest = Body(...),
):
    """
    Evaluate the importance of documents using Data Valuation with Reinforcement Learning (DVRL).
    
    This endpoint:
    1. Computes importance scores for each document
    2. Filters documents based on a value threshold
    3. Returns both scores and filtered documents
    
    The importance score indicates how valuable a document is for retrieval quality.
    Higher scores indicate more valuable documents.
    """
    try:
        # Convert to LangChain documents
        from langchain_core.documents import Document as LCDocument
        lc_documents = [
            LCDocument(page_content=doc.page_content, metadata=doc.metadata)
            for doc in request.documents
        ]
        
        # Create data valuation component
        data_valuation = DVRLDataValuation(
            embedding_dimension=768,  # Default dimension
            value_threshold=request.threshold,
        )
        
        # Compute document values
        doc_values = data_valuation.compute_document_values(lc_documents)
        
        # Filter valuable documents
        valuable_docs = data_valuation.filter_valuable_documents(
            lc_documents,
            threshold=request.threshold,
            top_k=request.top_k,
        )
        
        # Convert back to API documents
        api_valuable_docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in valuable_docs
        ]
        
        # Create response
        response = DataValueResponse(
            document_values=doc_values,
            valuable_documents=api_valuable_docs,
            total_documents=len(lc_documents),
            valuable_count=len(valuable_docs),
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in data valuation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data valuation error: {str(e)}")


# Interpretable embeddings endpoints
@router.post("/explain-similarity", response_model=SimilarityExplanationResponse,
             summary="Explain text similarity")
async def explain_similarity(
    request: SimilarityExplanationRequest = Body(...),
):
    """
    Explain why a document is similar to a query using Neural Additive Models.
    
    This endpoint:
    1. Analyzes which features contribute to similarity
    2. Returns top matching and least matching features
    3. Provides interpretable explanation of similarity
    
    The explanation helps understand which aspects of the text make it relevant to the query.
    """
    try:
        # Get base embedding model
        base_embeddings = get_embedding_model()
        
        # Create interpretable embeddings model
        interpretable_embeddings = NAMEmbeddings(
            base_embeddings=base_embeddings,
            dimension=768,  # Default dimension
            num_features=64,  # Number of interpretable features
        )
        
        # Get similarity explanation
        explanation = interpretable_embeddings.explain_similarity(
            query=request.query,
            document=request.document,
            top_k=request.top_k,
        )
        
        # Format feature importance for response
        top_features = [
            {"feature": feature, "importance": float(score)}
            for feature, score in explanation["top_matching_features"]
        ]
        
        least_features = [
            {"feature": feature, "importance": float(score)}
            for feature, score in explanation["least_matching_features"]
        ]
        
        # Create response
        response = SimilarityExplanationResponse(
            similarity_score=float(explanation["similarity_score"]),
            top_matching_features=top_features,
            least_matching_features=least_features,
            query=request.query,
            document=request.document,
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in similarity explanation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Similarity explanation error: {str(e)}")


# Hyperparameter optimization endpoints
@router.post("/optimized-hyperparameters", response_model=HyperparametersResponse,
             summary="Get optimized hyperparameters")
async def get_optimized_hyperparameters(
    request: HyperparametersRequest = Body(...),
):
    """
    Get optimized hyperparameters for embedding models using opt_list.
    
    This endpoint:
    1. Provides optimized learning rates based on model size and batch size
    2. Recommends optimal batch sizes for training
    3. Returns optimized parameters for embedding models
    4. Generates training schedules with warmup and decay
    
    These optimized parameters help improve training stability and performance.
    """
    try:
        # Create hyperparameter optimizer
        optimizer = OptimizedHyperparameters()
        
        # Get optimized learning rate
        learning_rate = optimizer.get_learning_rate(
            model_size=request.model_size,
            batch_size=request.batch_size,
            dataset_size=request.dataset_size,
        )
        
        # Get optimized batch size
        batch_size = optimizer.get_batch_size(
            model_size=request.model_size,
            dataset_size=request.dataset_size,
        )
        
        # Get embedding parameters if relevant fields are provided
        embedding_parameters = None
        if (request.embedding_dimension is not None and 
            request.vocabulary_size is not None and 
            request.max_sequence_length is not None):
            
            embedding_parameters = optimizer.get_embedding_parameters(
                embedding_dimension=request.embedding_dimension,
                vocabulary_size=request.vocabulary_size,
                max_sequence_length=request.max_sequence_length,
            )
        
        # Get training schedule if dataset size is provided
        training_schedule = None
        if request.dataset_size is not None:
            training_schedule = optimizer.get_training_schedule(
                model_size=request.model_size,
                dataset_size=request.dataset_size,
                batch_size=request.batch_size,
            )
        
        # Create response
        response = HyperparametersResponse(
            learning_rate=learning_rate,
            batch_size=batch_size,
            embedding_parameters=embedding_parameters,
            training_schedule=training_schedule,
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in hyperparameter optimization: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Hyperparameter optimization error: {str(e)}")


# Model compression endpoints
@router.post("/compressed-embeddings", response_model=CompressionResponse,
             summary="Generate compressed embeddings")
async def generate_compressed_embeddings(
    request: CompressionRequest = Body(...),
):
    """
    Generate compressed embeddings with reduced memory footprint using state_of_sparsity.
    
    This endpoint:
    1. Applies model compression to reduce embedding size
    2. Creates sparse embeddings with minimal accuracy loss
    3. Returns both compressed embeddings and compression statistics
    
    Compressed embeddings require less memory and can be faster to process.
    """
    try:
        # Get base embedding model
        base_embeddings = get_embedding_model()
        
        # Create compressed embedding model
        compressed_embeddings = SparseEmbeddingModel(
            base_embeddings=base_embeddings,
            compression_ratio=request.compression_ratio,
            compression_strategy=request.compression_strategy,
        )
        
        # Generate compressed embeddings
        embeddings = compressed_embeddings.embed_documents(request.texts)
        
        # Get compression statistics
        stats = compressed_embeddings.get_compression_stats()
        
        # Create response
        response = CompressionResponse(
            embeddings=embeddings,
            compression_stats=stats,
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error in embedding compression: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Embedding compression error: {str(e)}")