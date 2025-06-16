"""
Utility functions for SAP HANA vectorstore operations.

This module provides helper functions that extract logic from the main vectorstores.py
file to reduce method complexity and improve maintainability.
"""

import json
import logging
import re
import struct
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from hdbcli import dbapi
from langchain_core.documents import Document

from langchain_hana.error_utils import handle_database_error
from langchain_hana.query_constructors import CONTAINS_OPERATOR, LOGICAL_OPERATORS_TO_SQL

logger = logging.getLogger(__name__)

# Intermediate table name for query operations
INTERMEDIATE_TABLE_NAME = "intermediate_result"

def extract_keyword_search_columns(
    filter_obj: Optional[Dict[str, Any]] = None,
    content_column: str = "VEC_TEXT",
    specific_metadata_columns: Optional[List[str]] = None
) -> List[str]:
    """
    Extract metadata columns used with `$contains` in the filter.
    
    Scans the filter to find unspecific metadata columns used
    with the `$contains` operator.
    
    Args:
        filter_obj: A dictionary of filter criteria.
        content_column: Name of the content column to exclude from results.
        specific_metadata_columns: List of specific metadata columns to exclude.
        
    Returns:
        list of metadata column names for keyword searches.
        
    Example:
        filter = {"$or": [
            {"title": {"$contains": "barbie"}},
            {"VEC_TEXT": {"$contains": "fred"}}]}
        Result: ["title"]
    """
    keyword_columns: Set[str] = set()
    
    if not filter_obj:
        return []
    
    # Set default value for specific_metadata_columns
    if specific_metadata_columns is None:
        specific_metadata_columns = []
    
    # Use recursive function to extract columns
    _recurse_filters(
        keyword_columns, 
        filter_obj, 
        parent_key=None, 
        content_column=content_column,
        specific_metadata_columns=specific_metadata_columns
    )
    
    return list(keyword_columns)

def _recurse_filters(
    keyword_columns: Set[str],
    filter_obj: Optional[Dict[Any, Any]],
    parent_key: Optional[str] = None,
    content_column: str = "VEC_TEXT",
    specific_metadata_columns: Optional[List[str]] = None
) -> None:
    """
    Recursively process the filter dictionary to find metadata columns used with `$contains`.
    
    Args:
        keyword_columns: Set to populate with found column names.
        filter_obj: The filter dictionary to process.
        parent_key: The parent key in the current recursion level.
        content_column: Name of the content column to exclude.
        specific_metadata_columns: List of specific metadata columns to exclude.
    """
    if specific_metadata_columns is None:
        specific_metadata_columns = []
        
    if isinstance(filter_obj, dict):
        for key, value in filter_obj.items():
            if key == CONTAINS_OPERATOR:
                # Add the parent key as it's the metadata column being filtered
                if parent_key and not (
                    parent_key == content_column
                    or parent_key in specific_metadata_columns
                ):
                    keyword_columns.add(parent_key)
            elif key in LOGICAL_OPERATORS_TO_SQL:  # Handle logical operators
                for subfilter in value:
                    _recurse_filters(
                        keyword_columns, 
                        subfilter, 
                        content_column=content_column,
                        specific_metadata_columns=specific_metadata_columns
                    )
            else:
                _recurse_filters(
                    keyword_columns, 
                    value, 
                    parent_key=key,
                    content_column=content_column,
                    specific_metadata_columns=specific_metadata_columns
                )

def create_metadata_projection(
    projected_metadata_columns: List[str],
    table_name: str,
    metadata_column: str
) -> str:
    """
    Generate a SQL `WITH` clause to project metadata columns for keyword search.
    
    Args:
        projected_metadata_columns: List of metadata column names for projection.
        table_name: Name of the table.
        metadata_column: Name of the metadata column.
        
    Returns:
        A SQL `WITH` clause string.
        
    Example:
        Input: ["title", "author"], "my_table", "metadata"
        Output:
        WITH intermediate_result AS (
            SELECT *,
            JSON_VALUE(metadata, '$.title') AS "title",
            JSON_VALUE(metadata, '$.author') AS "author"
            FROM "my_table"
        )
    """
    # Return empty string if no columns to project
    if not projected_metadata_columns:
        return ""
    
    # Sanitize column names
    sanitized_cols = [_sanitize_name(col) for col in projected_metadata_columns]
    
    # Create metadata column projections
    metadata_columns = [
        f'JSON_VALUE({metadata_column}, \'$.{col}\') AS "{col}"'
        for col in sanitized_cols
    ]
    
    # Build and return the WITH clause
    return (
        f"WITH {INTERMEDIATE_TABLE_NAME} AS ("
        f"SELECT *, {', '.join(metadata_columns)} "
        f"FROM \"{table_name}\")"
    )

def _sanitize_name(input_str: str) -> str:
    """
    Sanitize a name by removing non-alphanumeric characters.
    
    Args:
        input_str: String to sanitize.
        
    Returns:
        Sanitized string containing only alphanumeric characters and underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "", input_str)

def serialize_binary_vector(
    values: List[float], 
    vector_column_type: str
) -> bytes:
    """
    Converts a list of floats into binary format suitable for HANA vector types.
    
    Args:
        values: List of floating point values representing a vector.
        vector_column_type: Type of vector column ("REAL_VECTOR" or "HALF_VECTOR").
        
    Returns:
        Binary representation of the vector.
        
    Raises:
        ValueError: If the vector column type is not supported.
    """
    if vector_column_type == "HALF_VECTOR":
        # 2-byte half-precision float serialization
        return struct.pack(f"<I{len(values)}e", len(values), *values)
    elif vector_column_type == "REAL_VECTOR":
        # 4-byte float serialization (standard FVECS format)
        return struct.pack(f"<I{len(values)}f", len(values), *values)
    else:
        raise ValueError(f"Unsupported vector column type: {vector_column_type}")

def deserialize_binary_vector(
    fvecs: bytes, 
    vector_column_type: str
) -> List[float]:
    """
    Extracts a list of floats from binary format.
    
    Args:
        fvecs: Binary data from HANA vector column.
        vector_column_type: Type of vector column ("REAL_VECTOR" or "HALF_VECTOR").
        
    Returns:
        List of floating point values representing the vector.
        
    Raises:
        ValueError: If the vector column type is not supported.
    """
    # Extract dimension from first 4 bytes
    dim = struct.unpack_from("<I", fvecs, 0)[0]
    
    if vector_column_type == "HALF_VECTOR":
        # 2-byte half-precision float deserialization
        return list(struct.unpack_from(f"<{dim}e", fvecs, 4))
    elif vector_column_type == "REAL_VECTOR":
        # 4-byte float deserialization (standard FVECS format)
        return list(struct.unpack_from(f"<{dim}f", fvecs, 4))
    else:
        raise ValueError(f"Unsupported vector column type: {vector_column_type}")

def process_similarity_search_results(
    rows: List[Tuple],
    distance_strategy: str,
    vector_column_type: str
) -> List[Tuple[Document, float, List[float]]]:
    """
    Process the results of a similarity search query.
    
    Args:
        rows: Result rows from database query, containing document content, 
              metadata, vector, and similarity score.
        distance_strategy: Distance strategy used for search.
        vector_column_type: Type of vector column for deserialization.
        
    Returns:
        List of tuples, each containing:
        - Document: The matched document with content and metadata
        - float: The similarity score
        - list[float]: The document's embedding vector
    """
    result = []
    
    for row in rows:
        # Extract and parse document content and metadata
        js = json.loads(row[1])
        doc = Document(page_content=row[0], metadata=js)
        
        # Deserialize vector
        result_vector = deserialize_binary_vector(row[2], vector_column_type)
        
        # Get similarity score
        similarity_score = row[3]
        
        # For Euclidean distance, normalize to [0,1] range for consistency
        if distance_strategy == "EUCLIDEAN_DISTANCE":
            # Lower values are better for Euclidean distance, so invert the scale
            if similarity_score > 0.0001:
                # Convert to a similarity score in [0,1] range (approximately)
                # Using 1/(1+distance) to get higher values for lower distances
                similarity_score = 1.0 / (1.0 + similarity_score)
            else:
                # Perfect match or very close
                similarity_score = 1.0
        
        result.append((doc, similarity_score, result_vector))
    
    return result