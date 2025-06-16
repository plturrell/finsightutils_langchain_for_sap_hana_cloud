"""
Vector operations routes for version 1 of the API.

This module provides routes for vector operations such as creating, querying,
and managing vector embeddings in SAP HANA Cloud.
"""

import logging
import uuid
import time
from typing import Dict, List, Optional, Any, Union

from fastapi import Depends, Request, HTTPException, Query
from pydantic import BaseModel, Field

from ...config_standardized import get_standardized_settings
from ...utils.standardized_exceptions import GPUNotAvailableException
from ..base import BaseRouter
from ..dependencies import get_current_user, get_gpu_info
from ...db import get_connection

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Models
class CreateVectorRequest(BaseModel):
    """Request model for creating vector embeddings."""
    pipeline_id: str = Field(..., description="ID of the pipeline")
    source_id: str = Field(..., description="ID of the source data")
    table_name: str = Field(..., description="Name of the table containing the data")
    schema_name: Optional[str] = Field(None, description="Schema name for the table")
    model_name: str = Field("SAP_NEB.20240715", description="HANA's native embedding model")
    vector_dimensions: int = Field(768, description="Dimensions of the vector embeddings")
    normalize_vectors: bool = Field(True, description="Whether to normalize vectors")
    chunking_strategy: str = Field("none", description="Strategy for chunking text: none, fixed, sentence, paragraph")
    chunk_size: int = Field(256, description="Size of chunks when chunking is enabled")
    chunk_overlap: int = Field(50, description="Overlap between chunks when chunking is enabled")
    max_records: Optional[int] = Field(None, description="Maximum number of records to process")
    filter_condition: Optional[str] = Field(None, description="SQL filter condition for source data")
    embedding_type: str = Field("DOCUMENT", description="Type of embedding: DOCUMENT, QUERY, or CODE")
    pal_batch_size: int = Field(64, description="Batch size for PAL processing")
    use_pal_service: bool = Field(False, description="Whether to use PAL Text Embedding Service")


class CreateVectorResponse(BaseModel):
    """Response model for vector creation."""
    vector_id: str = Field(..., description="ID of the created vector collection")
    table_name: str = Field(..., description="Name of the vector table")
    vector_count: int = Field(..., description="Number of vectors created")
    model_name: str = Field(..., description="Name of the embedding model used")
    dimensions: int = Field(..., description="Dimensions of the vectors")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Informational message")


class VectorInfoRequest(BaseModel):
    """Request model for vector information."""
    vector_id: str = Field(..., description="ID of the vector collection")


class VectorInfoResponse(BaseModel):
    """Response model for vector information."""
    vector_id: str = Field(..., description="ID of the vector collection")
    table_name: str = Field(..., description="Name of the source table")
    vector_count: int = Field(..., description="Number of vectors")
    model_name: str = Field(..., description="Name of the embedding model used")
    dimensions: int = Field(..., description="Dimensions of the vectors")
    sample_vector: Optional[List[float]] = Field(None, description="Sample vector (first 10 dimensions)")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata about the vectors")
    created_at: Optional[str] = Field(None, description="Creation timestamp")


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding."""
    texts: List[str] = Field(..., description="List of texts to embed")
    model_name: str = Field("SAP_NEB.20240715", description="Name of the embedding model to use")
    embedding_type: str = Field("DOCUMENT", description="Type of embedding: DOCUMENT, QUERY, or CODE")
    use_pal_service: bool = Field(False, description="Whether to use PAL Text Embedding Service")


class BatchEmbeddingResponse(BaseModel):
    """Response model for batch embedding."""
    embeddings: List[List[float]] = Field(..., description="List of vector embeddings")
    dimensions: int = Field(..., description="Dimensionality of the embeddings")
    processing_time: float = Field(..., description="Time taken to process in seconds")
    tokens_processed: int = Field(..., description="Estimated number of tokens processed")


# Create router
router = BaseRouter(tags=["Vector Operations"])


# Helper Functions
def _get_embedding_model(model_name: str, embedding_type: str = "DOCUMENT", use_pal_service: bool = False):
    """Get the appropriate embedding model based on settings"""
    try:
        # Use HANA's native embedding capabilities via LangChain integration
        from langchain_hana.embeddings import HanaInternalEmbeddings
        
        return HanaInternalEmbeddings(
            model_name=model_name,
            embedding_type=embedding_type,
            use_pal_service=use_pal_service
        )
    except ImportError as e:
        logger.error(f"Error importing HanaInternalEmbeddings: {str(e)}")
        raise RuntimeError(f"Embedding provider not available: {str(e)}")


def _process_text_chunks(texts: List[str], chunking_strategy: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Process texts using the specified chunking strategy"""
    if chunking_strategy == "none":
        return texts
    
    chunked_texts = []
    
    for text in texts:
        if not text or len(text.strip()) == 0:
            continue
            
        if chunking_strategy == "fixed":
            # Simple fixed-size chunking
            if len(text) <= chunk_size:
                chunked_texts.append(text)
            else:
                for i in range(0, len(text), chunk_size - chunk_overlap):
                    chunk = text[i:i + chunk_size]
                    if len(chunk) >= chunk_size / 2:  # Only add chunks that are at least half the target size
                        chunked_texts.append(chunk)
        
        elif chunking_strategy == "sentence":
            # Chunk by sentences, combining into chunks of approximately chunk_size
            import re
            sentences = re.split(r'(?<=[.!?])\s+', text)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk:
                        chunked_texts.append(current_chunk)
                    current_chunk = sentence
            
            if current_chunk:
                chunked_texts.append(current_chunk)
        
        elif chunking_strategy == "paragraph":
            # Chunk by paragraphs
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                if para.strip():
                    if len(para) <= chunk_size:
                        chunked_texts.append(para)
                    else:
                        # If paragraph is too large, use fixed-size chunking
                        for i in range(0, len(para), chunk_size - chunk_overlap):
                            chunk = para[i:i + chunk_size]
                            if len(chunk) >= chunk_size / 2:
                                chunked_texts.append(chunk)
    
    return chunked_texts


# Endpoints
@router.post("/create", response_model=CreateVectorResponse)
async def create_vectors(
    request: Request,
    input_data: CreateVectorRequest
):
    """
    Create vector embeddings for data in the specified table.
    
    Generates vector embeddings for text data stored in the specified table
    using HANA's native embedding capabilities.
    """
    try:
        start_time = time.time()
        vector_id = str(uuid.uuid4())
        
        # Get database connection
        conn = get_connection()
        
        # Fetch data from the source table
        query = f"""
            SELECT * FROM {input_data.schema_name + "." if input_data.schema_name else ""}{input_data.table_name}
            {f"WHERE {input_data.filter_condition}" if input_data.filter_condition else ""}
            {f"LIMIT {input_data.max_records}" if input_data.max_records else ""}
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No data found in table {input_data.table_name}")
        
        # Extract column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Prepare data
        documents = []
        metadatas = []
        
        for row in results:
            # Convert row to dict
            row_dict = {column_names[i]: value for i, value in enumerate(row)}
            
            # Extract text content (assuming 'content' or 'text' column)
            content_col = next((col for col in ['content', 'text', 'description'] 
                              if col in row_dict and row_dict[col]), None)
            
            if not content_col:
                # If no specific text column found, concatenate all string values
                text_content = " ".join([str(v) for k, v in row_dict.items() 
                                       if isinstance(v, str) and len(str(v)) > 3])
            else:
                text_content = row_dict[content_col]
            
            # Skip empty content
            if not text_content or len(text_content.strip()) == 0:
                continue
                
            # Extract metadata (all other columns)
            metadata = {k: v for k, v in row_dict.items() if k != content_col}
            
            documents.append(text_content)
            metadatas.append(metadata)
        
        # Apply text chunking if specified
        if input_data.chunking_strategy != "none":
            chunked_documents = _process_text_chunks(
                documents, 
                input_data.chunking_strategy, 
                input_data.chunk_size, 
                input_data.chunk_overlap
            )
            
            # Also need to duplicate metadata for each chunk
            chunked_metadatas = []
            chunk_index = 0
            for i, doc in enumerate(documents):
                # Count how many chunks were created from this document
                orig_text = doc
                chunk_count = 0
                while (chunk_index + chunk_count < len(chunked_documents) and 
                      chunked_documents[chunk_index + chunk_count] in orig_text):
                    chunk_count += 1
                
                # Duplicate metadata for each chunk
                for j in range(chunk_count):
                    chunk_metadata = metadatas[i].copy()
                    chunk_metadata["chunk_index"] = j
                    chunk_metadata["original_document_index"] = i
                    chunked_metadatas.append(chunk_metadata)
                
                chunk_index += chunk_count
            
            documents = chunked_documents
            metadatas = chunked_metadatas
        
        # Get embedding model - using HANA's native VECTOR_EMBEDDING function
        embedding_model = _get_embedding_model(
            model_name=input_data.model_name,
            embedding_type=input_data.embedding_type,
            use_pal_service=input_data.use_pal_service
        )
        
        # Create vector store - HanaDB uses REAL_VECTOR data type for storage
        vector_table_name = f"VECTOR_{vector_id.replace('-', '_')}"
        
        # Use HANA's native vector store capabilities
        try:
            from langchain_hana.vectorstores import HanaDB
            
            vector_store = HanaDB.from_texts(
                texts=documents,
                embedding=embedding_model,
                metadatas=metadatas,
                connection=conn,
                table_name=vector_table_name,
                normalize_embeddings=input_data.normalize_vectors,
                batch_size=input_data.pal_batch_size if input_data.use_pal_service else None
            )
        except ImportError as e:
            logger.error(f"Error importing HanaDB: {str(e)}")
            raise RuntimeError(f"Vector store not available: {str(e)}")
        
        # Record the vector creation in the pipeline tracking table
        # This would typically be part of a more comprehensive pipeline tracking system
        tracking_query = """
        INSERT INTO PIPELINE_VECTORS (
            VECTOR_ID, PIPELINE_ID, SOURCE_ID, TABLE_NAME, VECTOR_TABLE, 
            MODEL_NAME, DIMENSIONS, VECTOR_COUNT, CREATED_AT, METADATA
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        """
        
        metadata_json = {
            "model_name": input_data.model_name,
            "embedding_type": input_data.embedding_type,
            "use_pal_service": input_data.use_pal_service,
            "pal_batch_size": input_data.pal_batch_size if input_data.use_pal_service else None,
            "normalize_vectors": input_data.normalize_vectors,
            "chunking_strategy": input_data.chunking_strategy,
            "chunk_size": input_data.chunk_size if input_data.chunking_strategy != "none" else None,
            "chunk_overlap": input_data.chunk_overlap if input_data.chunking_strategy != "none" else None,
            "processing_time": time.time() - start_time
        }
        
        try:
            cursor.execute(tracking_query, (
                vector_id, 
                input_data.pipeline_id,
                input_data.source_id,
                f"{input_data.schema_name + '.' if input_data.schema_name else ''}{input_data.table_name}",
                vector_table_name,
                input_data.model_name,
                input_data.vector_dimensions,
                len(documents),
                str(metadata_json)
            ))
            conn.commit()
        except Exception as e:
            # If tracking fails, still return success for the vector creation
            logger.warning(f"Failed to record vector creation in tracking table: {str(e)}")
        
        processing_time = time.time() - start_time
        
        return CreateVectorResponse(
            vector_id=vector_id,
            table_name=vector_table_name,
            vector_count=len(documents),
            model_name=input_data.model_name,
            dimensions=input_data.vector_dimensions,
            processing_time=processing_time,
            status="success",
            message=f"Successfully created {len(documents)} vectors in {processing_time:.2f} seconds"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating vectors: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create vectors: {str(e)}")


@router.post("/info", response_model=VectorInfoResponse)
async def get_vector_info(
    request: Request,
    input_data: VectorInfoRequest
):
    """
    Get information about a vector embedding table.
    
    Retrieves detailed information about a vector collection,
    including metadata and a sample vector.
    """
    try:
        # Get database connection
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query the tracking table for vector information
        query = """
        SELECT VECTOR_ID, TABLE_NAME, VECTOR_TABLE, MODEL_NAME, DIMENSIONS, 
               VECTOR_COUNT, CREATED_AT, METADATA
        FROM PIPELINE_VECTORS
        WHERE VECTOR_ID = ?
        """
        
        cursor.execute(query, (input_data.vector_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Vector ID {input_data.vector_id} not found")
        
        vector_id, table_name, vector_table, model_name, dimensions, vector_count, created_at, metadata_str = result
        
        # Get a sample vector
        sample_query = f"""
        SELECT TOP 1 VECTOR FROM {vector_table}
        """
        
        try:
            cursor.execute(sample_query)
            sample_result = cursor.fetchone()
            if sample_result:
                # Vector is typically stored in binary format, need to decode
                import numpy as np
                sample_vector = np.frombuffer(sample_result[0], dtype=np.float32).tolist()
                # Limit to first 10 dimensions for response
                sample_vector = sample_vector[:10]
            else:
                sample_vector = None
        except Exception as e:
            sample_vector = None
            logger.warning(f"Could not retrieve sample vector: {str(e)}")
        
        # Parse metadata
        try:
            import json
            metadata = json.loads(metadata_str)
        except:
            metadata = {}
        
        return VectorInfoResponse(
            vector_id=vector_id,
            table_name=table_name,
            vector_count=vector_count,
            model_name=model_name,
            dimensions=dimensions,
            sample_vector=sample_vector,
            metadata=metadata,
            created_at=created_at.isoformat() if created_at else None
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving vector information: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get vector information: {str(e)}")


@router.post("/batch-embed", response_model=BatchEmbeddingResponse)
async def batch_embed_texts(
    request: Request,
    input_data: BatchEmbeddingRequest
):
    """
    Create embeddings for a batch of texts.
    
    Generates vector embeddings for a batch of texts using HANA's
    native embedding capabilities.
    """
    try:
        start_time = time.time()
        
        if not input_data.texts:
            raise HTTPException(status_code=400, detail="No texts provided for embedding")
        
        # Get embedding model using HANA's native capabilities
        embedding_model = _get_embedding_model(
            model_name=input_data.model_name,
            embedding_type=input_data.embedding_type,
            use_pal_service=input_data.use_pal_service
        )
        
        # Generate embeddings using HANA's VECTOR_EMBEDDING function via LangChain
        embeddings = embedding_model.embed_documents(input_data.texts)
        
        # Calculate token estimate (rough approximation)
        token_estimate = sum(len(text.split()) * 1.3 for text in input_data.texts)
        
        return BatchEmbeddingResponse(
            embeddings=embeddings,
            dimensions=len(embeddings[0]) if embeddings else 0,
            processing_time=time.time() - start_time,
            tokens_processed=int(token_estimate)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")


@router.get("/models")
async def list_available_models():
    """
    List available embedding models and their properties.
    
    Returns information about the embedding models supported by HANA Cloud.
    """
    models = [
        {
            "id": "SAP_NEB.20240715",
            "name": "SAP NEB Standard",
            "dimensions": 768,
            "description": "HANA Cloud's native embedding model for general text",
            "performance": "fast",
            "quality": "good",
            "embedding_types": ["DOCUMENT", "QUERY", "CODE"],
        },
        {
            "id": "SAP_NEB.PAL.20250115",
            "name": "SAP NEB PAL Service",
            "dimensions": 768,
            "description": "Optimized for batch processing through PAL Text Embedding Service",
            "performance": "fast",
            "quality": "good",
            "embedding_types": ["DOCUMENT", "QUERY"],
            "pal_service": True
        }
    ]
    
    # Get database connection to check for PAL service availability
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if PAL procedures are available
        pal_available = True
        try:
            cursor.execute("SELECT * FROM SYS.AFL_FUNCTIONS WHERE AREA_NAME = 'PAL' AND FUNCTION_NAME = 'TEXTEMBEDDING'")
            pal_results = cursor.fetchone()
            pal_available = pal_results is not None
        except:
            pal_available = False
            
        return {
            "models": models,
            "pal_available": pal_available,
            "vector_engine_available": True,  # HANA Cloud Vector Engine is available
            "recommended_model": "SAP_NEB.20240715"
        }
    except Exception as e:
        # If we can't connect, return basic info
        logger.error(f"Error connecting to database: {str(e)}")
        return {
            "models": models,
            "pal_available": False,
            "vector_engine_available": True,
            "recommended_model": "SAP_NEB.20240715",
            "error": str(e)
        }