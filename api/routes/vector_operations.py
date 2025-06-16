from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, List, Optional, Any, Union
import uuid
import time
from langchain_hana.embeddings import HanaInternalEmbeddings
from langchain_hana.vectorstores import HanaDB
from api.db import get_connection
from api.models.models import get_embeddings_model

router = APIRouter(
    prefix="/vector-operations",
    tags=["vector-operations"],
)

# Models
class CreateVectorRequest(BaseModel):
    pipeline_id: str
    source_id: str
    table_name: str
    schema_name: Optional[str] = None
    model_name: str = "SAP_NEB.20240715"  # HANA's native embedding model
    vector_dimensions: int = 768          # Native model dimension
    normalize_vectors: bool = True
    chunking_strategy: str = "none"
    chunk_size: int = 256
    chunk_overlap: int = 50
    max_records: Optional[int] = None
    filter_condition: Optional[str] = None
    embedding_type: str = "DOCUMENT"      # DOCUMENT, QUERY, or CODE
    pal_batch_size: int = 64             # For PAL batch processing
    use_pal_service: bool = False         # Whether to use PAL Text Embedding Service

class CreateVectorResponse(BaseModel):
    vector_id: str
    table_name: str
    vector_count: int
    model_name: str
    dimensions: int
    processing_time: float
    status: str
    message: str

class VectorInfoRequest(BaseModel):
    vector_id: str

class VectorInfoResponse(BaseModel):
    vector_id: str
    table_name: str
    vector_count: int
    model_name: str
    dimensions: int
    sample_vector: Optional[List[float]] = None
    metadata: Dict[str, Any]
    created_at: Optional[str] = None

class BatchEmbeddingRequest(BaseModel):
    texts: List[str]
    model_name: str = "SAP_NEB.20240715"
    embedding_type: str = "DOCUMENT"      # DOCUMENT, QUERY, or CODE
    use_pal_service: bool = False         # Whether to use PAL Text Embedding Service

class BatchEmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    processing_time: float
    tokens_processed: int


# Helper Functions
def _get_embedding_model(model_name: str, embedding_type: str = "DOCUMENT", use_pal_service: bool = False):
    """Get the appropriate embedding model based on settings"""
    # Use HANA's native embedding capabilities via LangChain integration
    return HanaInternalEmbeddings(
        model_name=model_name,
        embedding_type=embedding_type,
        use_pal_service=use_pal_service
    )

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
async def create_vectors(request: CreateVectorRequest):
    """Create vector embeddings for data in the specified table"""
    try:
        start_time = time.time()
        vector_id = str(uuid.uuid4())
        
        # Get database connection
        conn = get_connection()
        
        # Fetch data from the source table
        query = f"""
            SELECT * FROM {request.schema_name + "." if request.schema_name else ""}{request.table_name}
            {f"WHERE {request.filter_condition}" if request.filter_condition else ""}
            {f"LIMIT {request.max_records}" if request.max_records else ""}
        """
        
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        
        if not results:
            raise HTTPException(status_code=404, detail=f"No data found in table {request.table_name}")
        
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
        if request.chunking_strategy != "none":
            chunked_documents = _process_text_chunks(
                documents, 
                request.chunking_strategy, 
                request.chunk_size, 
                request.chunk_overlap
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
            model_name=request.model_name,
            embedding_type=request.embedding_type,
            use_pal_service=request.use_pal_service
        )
        
        # Create vector store - HanaDB uses REAL_VECTOR data type for storage
        vector_table_name = f"VECTOR_{vector_id.replace('-', '_')}"
        
        # Use HANA's native vector store capabilities
        vector_store = HanaDB.from_texts(
            texts=documents,
            embedding=embedding_model,
            metadatas=metadatas,
            connection=conn,
            table_name=vector_table_name,
            normalize_embeddings=request.normalize_vectors,
            batch_size=request.pal_batch_size if request.use_pal_service else None
        )
        
        # Record the vector creation in the pipeline tracking table
        # This would typically be part of a more comprehensive pipeline tracking system
        tracking_query = """
        INSERT INTO PIPELINE_VECTORS (
            VECTOR_ID, PIPELINE_ID, SOURCE_ID, TABLE_NAME, VECTOR_TABLE, 
            MODEL_NAME, DIMENSIONS, VECTOR_COUNT, CREATED_AT, METADATA
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
        """
        
        metadata_json = {
            "model_name": request.model_name,
            "embedding_type": request.embedding_type,
            "use_pal_service": request.use_pal_service,
            "pal_batch_size": request.pal_batch_size if request.use_pal_service else None,
            "normalize_vectors": request.normalize_vectors,
            "chunking_strategy": request.chunking_strategy,
            "chunk_size": request.chunk_size if request.chunking_strategy != "none" else None,
            "chunk_overlap": request.chunk_overlap if request.chunking_strategy != "none" else None,
            "processing_time": time.time() - start_time
        }
        
        try:
            cursor.execute(tracking_query, (
                vector_id, 
                request.pipeline_id,
                request.source_id,
                f"{request.schema_name + '.' if request.schema_name else ''}{request.table_name}",
                vector_table_name,
                request.model_name,
                request.vector_dimensions,
                len(documents),
                str(metadata_json)
            ))
            conn.commit()
        except Exception as e:
            # If tracking fails, still return success for the vector creation
            print(f"Warning: Failed to record vector creation in tracking table: {e}")
        
        processing_time = time.time() - start_time
        
        return CreateVectorResponse(
            vector_id=vector_id,
            table_name=vector_table_name,
            vector_count=len(documents),
            model_name=request.model_name,
            dimensions=request.vector_dimensions,
            processing_time=processing_time,
            status="success",
            message=f"Successfully created {len(documents)} vectors in {processing_time:.2f} seconds"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vectors: {str(e)}")


@router.post("/info", response_model=VectorInfoResponse)
async def get_vector_info(request: VectorInfoRequest):
    """Get information about a vector embedding table"""
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
        
        cursor.execute(query, (request.vector_id,))
        result = cursor.fetchone()
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Vector ID {request.vector_id} not found")
        
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
            print(f"Warning: Could not retrieve sample vector: {e}")
        
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
        raise HTTPException(status_code=500, detail=f"Failed to get vector information: {str(e)}")


@router.post("/batch-embed", response_model=BatchEmbeddingResponse)
async def batch_embed_texts(request: BatchEmbeddingRequest):
    """Create embeddings for a batch of texts using HANA's native VECTOR_EMBEDDING function"""
    try:
        start_time = time.time()
        
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided for embedding")
        
        # Get embedding model using HANA's native capabilities
        embedding_model = _get_embedding_model(
            model_name=request.model_name,
            embedding_type=request.embedding_type,
            use_pal_service=request.use_pal_service
        )
        
        # Generate embeddings using HANA's VECTOR_EMBEDDING function via LangChain
        embeddings = embedding_model.embed_documents(request.texts)
        
        # Calculate token estimate (rough approximation)
        token_estimate = sum(len(text.split()) * 1.3 for text in request.texts)
        
        return BatchEmbeddingResponse(
            embeddings=embeddings,
            dimensions=len(embeddings[0]) if embeddings else 0,
            processing_time=time.time() - start_time,
            tokens_processed=int(token_estimate)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")


@router.get("/models")
async def list_available_models():
    """List available embedding models and their properties"""
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
        return {
            "models": models,
            "pal_available": False,
            "vector_engine_available": True,
            "recommended_model": "SAP_NEB.20240715",
            "error": str(e)
        }