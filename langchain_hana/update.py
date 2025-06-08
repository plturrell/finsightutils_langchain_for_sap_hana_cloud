"""Update operations for SAP HANA Cloud Vector Engine.

This module provides update operations for the HanaDB vectorstore.
It enables updating document content and metadata in place without needing 
to delete and re-add documents.
"""

from typing import List, Dict, Any, Optional, Iterable, Tuple, Union
import json
import logging
from hdbcli import dbapi

from langchain_core.embeddings import Embeddings
from langchain_hana.query_constructors import CreateWhereClause
from langchain_hana.error_utils import handle_database_error

logger = logging.getLogger(__name__)

def update_document_content(
    connection: dbapi.Connection,
    table_name: str,
    content_column: str,
    vector_column: str,
    filter: Dict[str, Any],
    new_content: str,
    embedding_model: Embeddings = None,
    embedding_function_name: Optional[str] = None,
) -> bool:
    """
    Update the text content of documents matching the filter criteria.
    
    This function can update the document content and regenerate the embedding vector
    either using an external embedding model or HANA's internal VECTOR_EMBEDDING function.
    
    Args:
        connection: SAP HANA database connection
        table_name: Name of the vector table
        content_column: Name of the column containing document text
        vector_column: Name of the column containing embedding vectors
        filter: Filter criteria to identify documents to update
        new_content: New text content for the documents
        embedding_model: Optional external embedding model to generate new embeddings
        embedding_function_name: Optional HANA internal embedding function name
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        # Sanitize names to prevent SQL injection
        table_name = table_name.replace('"', '""')
        content_column = content_column.replace('"', '""')
        vector_column = vector_column.replace('"', '""')
        
        # Create where clause from filter
        where_generator = CreateWhereClause(None)  # Pass None as we're manually handling this
        where_clause, parameters = where_generator(filter)
        
        # Determine how to update the embedding vector
        if embedding_model:
            # Generate new embedding using external model
            new_embedding = embedding_model.embed_query(new_content)
            
            # Convert embedding to binary format
            # Note: This might need adjustment based on your serialization approach
            import struct
            vector_binary = struct.pack(f"<I{len(new_embedding)}f", len(new_embedding), *new_embedding)
            
            # Update SQL with external embedding
            sql = f'''UPDATE "{table_name}" 
                    SET "{content_column}" = ?, "{vector_column}" = ?
                    {where_clause}'''
            
            # Add parameters
            all_params = [new_content, vector_binary] + parameters
            
        elif embedding_function_name:
            # Use HANA's internal VECTOR_EMBEDDING function
            sql = f'''UPDATE "{table_name}" 
                    SET "{content_column}" = ?, 
                        "{vector_column}" = VECTOR_EMBEDDING(?, 'QUERY', ?)
                    {where_clause}'''
            
            # Add parameters
            all_params = [new_content, new_content, embedding_function_name] + parameters
            
        else:
            # Only update the content, not the vector
            sql = f'''UPDATE "{table_name}" 
                    SET "{content_column}" = ?
                    {where_clause}'''
                    
            # Add parameters
            all_params = [new_content] + parameters
        
        # Execute the update
        cursor = connection.cursor()
        try:
            cursor.execute(sql, all_params)
            row_count = cursor.rowcount
            connection.commit()
            logger.info(f"Updated {row_count} document(s) in table '{table_name}'")
            return True
        except dbapi.Error as e:
            connection.rollback()
            # Provide context-aware error handling
            additional_context = {
                "table_name": table_name,
                "filter": filter,
                "operation": "update_document_content"
            }
            handle_database_error(e, "update_document", additional_context)
            return False
        finally:
            cursor.close()
            
    except Exception as e:
        logger.error(f"Error updating document content: {str(e)}")
        return False

def update_document_metadata(
    connection: dbapi.Connection,
    table_name: str,
    metadata_column: str,
    filter: Dict[str, Any],
    new_metadata: Dict[str, Any],
    merge_strategy: str = "replace",
) -> bool:
    """
    Update the metadata of documents matching the filter criteria.
    
    This function can either replace the entire metadata or merge new metadata
    with existing metadata based on the merge_strategy parameter.
    
    Args:
        connection: SAP HANA database connection
        table_name: Name of the vector table
        metadata_column: Name of the column containing document metadata
        filter: Filter criteria to identify documents to update
        new_metadata: New metadata for the documents
        merge_strategy: Strategy for updating metadata ('replace' or 'merge')
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    try:
        # Sanitize names to prevent SQL injection
        table_name = table_name.replace('"', '""')
        metadata_column = metadata_column.replace('"', '""')
        
        # Create where clause from filter
        where_generator = CreateWhereClause(None)  # Pass None as we're manually handling this
        where_clause, parameters = where_generator(filter)
        
        if merge_strategy.lower() == "merge":
            # For merging, we need to fetch existing metadata first
            fetch_sql = f'''SELECT ID, "{metadata_column}" FROM "{table_name}" {where_clause}'''
            
            cursor = connection.cursor()
            try:
                cursor.execute(fetch_sql, parameters)
                rows = cursor.fetchall()
                
                # Process each row to merge metadata
                for row_id, metadata_json in rows:
                    try:
                        # Parse existing metadata
                        existing_metadata = json.loads(metadata_json) if metadata_json else {}
                        
                        # Merge with new metadata (new values override existing ones)
                        merged_metadata = {**existing_metadata, **new_metadata}
                        
                        # Update the specific row
                        update_sql = f'''UPDATE "{table_name}" 
                                      SET "{metadata_column}" = ?
                                      WHERE ID = ?'''
                        
                        cursor.execute(update_sql, [json.dumps(merged_metadata), row_id])
                    except Exception as e:
                        logger.error(f"Error merging metadata for row {row_id}: {str(e)}")
                
                connection.commit()
                logger.info(f"Updated metadata for {len(rows)} document(s) in table '{table_name}'")
                return True
                
            except dbapi.Error as e:
                connection.rollback()
                additional_context = {
                    "table_name": table_name,
                    "filter": filter,
                    "operation": "update_document_metadata",
                    "strategy": "merge"
                }
                handle_database_error(e, "update_metadata", additional_context)
                return False
            finally:
                cursor.close()
        else:
            # Direct replacement of metadata
            update_sql = f'''UPDATE "{table_name}" 
                          SET "{metadata_column}" = ?
                          {where_clause}'''
            
            all_params = [json.dumps(new_metadata)] + parameters
            
            cursor = connection.cursor()
            try:
                cursor.execute(update_sql, all_params)
                row_count = cursor.rowcount
                connection.commit()
                logger.info(f"Updated metadata for {row_count} document(s) in table '{table_name}'")
                return True
            except dbapi.Error as e:
                connection.rollback()
                additional_context = {
                    "table_name": table_name,
                    "filter": filter,
                    "operation": "update_document_metadata",
                    "strategy": "replace"
                }
                handle_database_error(e, "update_metadata", additional_context)
                return False
            finally:
                cursor.close()
                
    except Exception as e:
        logger.error(f"Error updating document metadata: {str(e)}")
        return False