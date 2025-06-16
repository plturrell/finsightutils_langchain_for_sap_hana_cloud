"""
Document update routes for version 1 of the API.

This module provides API endpoints for updating documents in the SAP HANA vector store.
"""

from fastapi import Depends, HTTPException, Body
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
import logging
import time

from ...database import get_vectorstore
from ...models import ErrorResponse, DocumentResponse
from ..base import BaseRouter

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = BaseRouter(
    prefix="/documents",
    tags=["Document Management"],
    responses={
        404: {"model": ErrorResponse, "description": "Not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    }
)


class UpdateDocumentRequest(BaseModel):
    """Model for document update requests."""
    text: str = Field(..., description="The new document text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="The document metadata")
    filter: Dict[str, Any] = Field(..., description="Filter to identify documents to update")
    update_embeddings: bool = Field(True, description="Whether to update the embeddings")

    class Config:
        schema_extra = {
            "example": {
                "text": "Updated document about financial markets",
                "metadata": {
                    "category": "finance",
                    "source": "manual",
                    "updated_at": "2023-05-15"
                },
                "filter": {
                    "id": "doc123"
                },
                "update_embeddings": True
            }
        }


class UpsertDocumentRequest(BaseModel):
    """Model for document upsert requests."""
    text: str = Field(..., description="The document text")
    metadata: Optional[Dict[str, Any]] = Field(None, description="The document metadata")
    filter: Optional[Dict[str, Any]] = Field(None, description="Filter to identify documents to update")

    class Config:
        schema_extra = {
            "example": {
                "text": "Document about investment strategies",
                "metadata": {
                    "category": "finance",
                    "source": "manual",
                    "created_at": "2023-05-15"
                },
                "filter": {
                    "id": "doc123"
                }
            }
        }


class UpdateResponse(BaseModel):
    """Model for update operation response."""
    success: bool = Field(..., description="Whether the operation was successful")
    documents_affected: int = Field(0, description="Number of documents affected")
    operation: str = Field(..., description="Type of operation performed")
    processing_time: float = Field(..., description="Processing time in seconds")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "documents_affected": 1,
                "operation": "update",
                "processing_time": 0.254
            }
        }


@router.put("/update", response_model=UpdateResponse)
async def update_document(
    request: UpdateDocumentRequest,
    vectorstore = Depends(get_vectorstore)
):
    """
    Update documents that match the filter criteria.
    
    This endpoint updates the content and/or metadata of documents in the vector store
    that match the specified filter criteria. It can optionally regenerate embeddings.
    
    Args:
        request: The update request containing text, metadata, filter, and update_embeddings flag
        vectorstore: The vector store to update documents in
        
    Returns:
        UpdateResponse: The result of the update operation
        
    Raises:
        HTTPException: If there is an error updating the documents
    """
    start_time = time.time()
    
    try:
        # Count matching documents before update
        where_clause, parameters = vectorstore._where_clause_builder(request.filter)
        count_sql = f'SELECT COUNT(*) FROM "{vectorstore.table_name}" {where_clause}'
        
        cursor = vectorstore.connection.cursor()
        try:
            cursor.execute(count_sql, parameters)
            count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        finally:
            cursor.close()
            
        if count == 0:
            return UpdateResponse(
                success=True,
                documents_affected=0,
                operation="update",
                processing_time=time.time() - start_time
            )
            
        # Perform the update
        result = vectorstore.update_texts(
            texts=[request.text],
            filter=request.filter,
            metadatas=[request.metadata] if request.metadata else None,
            update_embeddings=request.update_embeddings
        )
        
        return UpdateResponse(
            success=result,
            documents_affected=count,
            operation="update",
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error updating document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "update_error",
                "message": str(e),
                "context": {
                    "operation": "document_update",
                    "suggestion": "Check your filter criteria and database connection"
                }
            }
        )


@router.post("/upsert", response_model=UpdateResponse)
async def upsert_document(
    request: UpsertDocumentRequest,
    vectorstore = Depends(get_vectorstore)
):
    """
    Add or update a document in the vector store.
    
    This endpoint checks if documents matching the filter exist:
    - If they exist, it updates them with the new content and metadata
    - If they don't exist, it adds the document as a new entry
    
    If no filter is provided, the document is always added as new.
    
    Args:
        request: The upsert request containing text, metadata, and optional filter
        vectorstore: The vector store to upsert documents in
        
    Returns:
        UpdateResponse: The result of the upsert operation
        
    Raises:
        HTTPException: If there is an error upserting the document
    """
    start_time = time.time()
    
    try:
        # If filter is provided, check if documents exist
        operation = "insert"
        count = 0
        
        if request.filter:
            where_clause, parameters = vectorstore._where_clause_builder(request.filter)
            count_sql = f'SELECT COUNT(*) FROM "{vectorstore.table_name}" {where_clause}'
            
            cursor = vectorstore.connection.cursor()
            try:
                cursor.execute(count_sql, parameters)
                count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
            finally:
                cursor.close()
                
            operation = "update" if count > 0 else "insert"
        
        # Perform the upsert
        vectorstore.upsert_texts(
            texts=[request.text],
            metadatas=[request.metadata] if request.metadata else None,
            filter=request.filter
        )
        
        return UpdateResponse(
            success=True,
            documents_affected=count if operation == "update" else 1,
            operation=operation,
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error upserting document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "upsert_error",
                "message": str(e),
                "context": {
                    "operation": "document_upsert",
                    "suggestion": "Check your input data and database connection"
                }
            }
        )


@router.delete("/delete", response_model=UpdateResponse)
async def delete_documents(
    filter: Dict[str, Any] = Body(..., description="Filter criteria to identify documents to delete")
):
    """
    Delete documents that match the filter criteria.
    
    This endpoint deletes documents from the vector store that match the specified filter criteria.
    
    Args:
        filter: Filter criteria to identify documents to delete
        
    Returns:
        UpdateResponse: The result of the delete operation
        
    Raises:
        HTTPException: If there is an error deleting the documents
    """
    start_time = time.time()
    
    try:
        vectorstore = await get_vectorstore()
        
        # Count matching documents before deletion
        where_clause, parameters = vectorstore._where_clause_builder(filter)
        count_sql = f'SELECT COUNT(*) FROM "{vectorstore.table_name}" {where_clause}'
        
        cursor = vectorstore.connection.cursor()
        try:
            cursor.execute(count_sql, parameters)
            count = cursor.fetchone()[0] if cursor.rowcount > 0 else 0
        finally:
            cursor.close()
            
        # Perform the deletion
        result = vectorstore.delete(filter=filter)
        
        return UpdateResponse(
            success=result,
            documents_affected=count,
            operation="delete",
            processing_time=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "delete_error",
                "message": str(e),
                "context": {
                    "operation": "document_delete",
                    "suggestion": "Check your filter criteria and database connection"
                }
            }
        )