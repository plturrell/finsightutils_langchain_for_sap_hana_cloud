"""
Embedding routes for version 1 of the API.

This module provides routes for generating embeddings for text documents
using various embedding models.
"""

import logging
from typing import Dict, List, Any, Optional, Union

from fastapi import Depends, Request, Query, Body
from pydantic import BaseModel, Field

from ...config_standardized import get_standardized_settings
from ...utils.standardized_exceptions import GPUNotAvailableException
from ..base import BaseRouter
from ..dependencies import get_current_user, get_gpu_info

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Define embedding models
class TextInput(BaseModel):
    """Input model for text to be embedded."""
    
    text: str = Field(..., description="Text to generate embeddings for")


class BatchTextInput(BaseModel):
    """Input model for batch text embedding."""
    
    texts: List[str] = Field(..., description="List of texts to generate embeddings for")
    model_name: str = Field("sentence-transformers/all-MiniLM-L6-v2", 
                            description="Name of the embedding model to use")
    normalize: bool = Field(True, description="Whether to normalize embeddings")


class EmbeddingResponse(BaseModel):
    """Response model for embeddings."""
    
    embedding: List[float] = Field(..., description="Vector embedding")
    model_name: str = Field(..., description="Name of the model used")
    dimensions: int = Field(..., description="Dimensionality of the embedding")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embeddings."""
    
    embeddings: List[List[float]] = Field(..., description="List of vector embeddings")
    model_name: str = Field(..., description="Name of the model used")
    dimensions: int = Field(..., description="Dimensionality of the embeddings")
    count: int = Field(..., description="Number of embeddings generated")
    gpu_accelerated: bool = Field(False, description="Whether GPU acceleration was used")
    tensorrt_accelerated: bool = Field(False, description="Whether TensorRT acceleration was used")


# Create router
router = BaseRouter(tags=["Embeddings"])


@router.post("/embeddings", response_model=BatchEmbeddingResponse)
async def create_embeddings(
    request: Request,
    input_data: BatchTextInput,
    gpu_info: Dict[str, Any] = Depends(get_gpu_info),
):
    """
    Generate embeddings for a batch of texts.
    
    Creates vector embeddings for the provided texts using the specified model.
    Supports GPU acceleration if available.
    """
    try:
        # Get model name
        model_name = input_data.model_name
        
        # Initialize embedding provider based on GPU availability
        if gpu_info.get("requested", False) and gpu_info.get("available", False):
            # Use GPU embeddings
            try:
                from langchain_hana.embeddings import HanaGPUEmbeddings
                
                # Check if TensorRT is requested and available
                use_tensorrt = gpu_info.get("tensorrt_requested", False) and gpu_info.get("tensorrt_available", False)
                
                provider = HanaGPUEmbeddings(
                    model_name=model_name,
                    device="cuda:0",  # Use first GPU
                    use_tensorrt=use_tensorrt,
                    normalize_embeddings=input_data.normalize
                )
                
                # Generate embeddings
                embeddings = provider.embed_documents(input_data.texts)
                
                # Get dimensions
                dimensions = len(embeddings[0]) if embeddings else 0
                
                return BatchEmbeddingResponse(
                    embeddings=embeddings,
                    model_name=model_name,
                    dimensions=dimensions,
                    count=len(embeddings),
                    gpu_accelerated=True,
                    tensorrt_accelerated=use_tensorrt
                )
            except ImportError:
                logger.warning("GPU embeddings requested but HanaGPUEmbeddings not available")
                # Fall back to CPU embeddings
                pass
        
        # Use CPU embeddings
        try:
            from langchain_hana.embeddings import HanaEmbeddings
            
            provider = HanaEmbeddings(
                model_name=model_name,
                normalize_embeddings=input_data.normalize
            )
            
            # Generate embeddings
            embeddings = provider.embed_documents(input_data.texts)
            
            # Get dimensions
            dimensions = len(embeddings[0]) if embeddings else 0
            
            return BatchEmbeddingResponse(
                embeddings=embeddings,
                model_name=model_name,
                dimensions=dimensions,
                count=len(embeddings),
                gpu_accelerated=False,
                tensorrt_accelerated=False
            )
        except ImportError:
            logger.error("HanaEmbeddings not available")
            raise RuntimeError("Embedding provider not available")
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise


@router.post("/embeddings/single", response_model=EmbeddingResponse)
async def create_single_embedding(
    request: Request,
    input_data: TextInput,
    model_name: str = Query("sentence-transformers/all-MiniLM-L6-v2", description="Name of the embedding model to use"),
    normalize: bool = Query(True, description="Whether to normalize embeddings"),
    gpu_info: Dict[str, Any] = Depends(get_gpu_info),
):
    """
    Generate embedding for a single text.
    
    Creates a vector embedding for the provided text using the specified model.
    Supports GPU acceleration if available.
    """
    try:
        # Initialize embedding provider based on GPU availability
        if gpu_info.get("requested", False) and gpu_info.get("available", False):
            # Use GPU embeddings
            try:
                from langchain_hana.embeddings import HanaGPUEmbeddings
                
                # Check if TensorRT is requested and available
                use_tensorrt = gpu_info.get("tensorrt_requested", False) and gpu_info.get("tensorrt_available", False)
                
                provider = HanaGPUEmbeddings(
                    model_name=model_name,
                    device="cuda:0",  # Use first GPU
                    use_tensorrt=use_tensorrt,
                    normalize_embeddings=normalize
                )
                
                # Generate embeddings
                embedding = provider.embed_query(input_data.text)
                
                return EmbeddingResponse(
                    embedding=embedding,
                    model_name=model_name,
                    dimensions=len(embedding)
                )
            except ImportError:
                logger.warning("GPU embeddings requested but HanaGPUEmbeddings not available")
                # Fall back to CPU embeddings
                pass
        
        # Use CPU embeddings
        try:
            from langchain_hana.embeddings import HanaEmbeddings
            
            provider = HanaEmbeddings(
                model_name=model_name,
                normalize_embeddings=normalize
            )
            
            # Generate embeddings
            embedding = provider.embed_query(input_data.text)
            
            return EmbeddingResponse(
                embedding=embedding,
                model_name=model_name,
                dimensions=len(embedding)
            )
        except ImportError:
            logger.error("HanaEmbeddings not available")
            raise RuntimeError("Embedding provider not available")
        
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise


@router.get("/embeddings/models")
async def list_embedding_models():
    """
    List available embedding models.
    
    Returns a list of supported embedding models for generating embeddings.
    """
    # Return a list of supported models
    return {
        "models": [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "dimensions": 384,
                "description": "General purpose embedding model",
                "default": True
            },
            {
                "name": "BAAI/bge-small-en",
                "dimensions": 384,
                "description": "BGE small English model"
            },
            {
                "name": "BAAI/bge-base-en",
                "dimensions": 768,
                "description": "BGE base English model"
            },
            {
                "name": "intfloat/e5-small",
                "dimensions": 384,
                "description": "E5 small model"
            },
            {
                "name": "intfloat/e5-base",
                "dimensions": 768,
                "description": "E5 base model"
            },
            {
                "name": "jinaai/jina-embeddings-v2-small-en",
                "dimensions": 512,
                "description": "Jina embeddings small model"
            },
            {
                "name": "jinaai/jina-embeddings-v2-base-en",
                "dimensions": 768,
                "description": "Jina embeddings base model"
            },
            {
                "name": "finance-e5-small",
                "dimensions": 384,
                "description": "Financial domain-specific embedding model"
            }
        ]
    }