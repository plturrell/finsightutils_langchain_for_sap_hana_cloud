"""
Utility functions for SAP HANA Cloud LangChain integration.

This module provides utilities for working with SAP HANA Cloud's vector capabilities,
including distance strategy enums, vector serialization helpers, and other common functions.
"""

import json
import logging
import struct
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

class DistanceStrategy(str, Enum):
    """Enumerator of the Distance strategies for calculating distances
    between vectors."""

    EUCLIDEAN_DISTANCE = "EUCLIDEAN_DISTANCE"
    MAX_INNER_PRODUCT = "MAX_INNER_PRODUCT"
    DOT_PRODUCT = "DOT_PRODUCT"
    JACCARD = "JACCARD"
    COSINE = "COSINE"

def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length (L2 norm).
    
    Args:
        vector: The vector to normalize
        
    Returns:
        The normalized vector
    """
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm > 0:
        return (arr / norm).tolist()
    return arr.tolist()

def serialize_vector(vector: List[float], vector_type: str = "REAL_VECTOR") -> bytes:
    """
    Serialize a vector to the binary format expected by SAP HANA Cloud.
    
    Args:
        vector: The vector to serialize
        vector_type: The HANA vector type (REAL_VECTOR or HALF_VECTOR)
        
    Returns:
        Binary representation of the vector
        
    Raises:
        ValueError: If an unsupported vector type is provided
    """
    if vector_type == "HALF_VECTOR":
        # 2-byte half-precision float serialization
        return struct.pack(f"<I{len(vector)}e", len(vector), *vector)
    elif vector_type == "REAL_VECTOR":
        # 4-byte float serialization (standard FVECS format)
        return struct.pack(f"<I{len(vector)}f", len(vector), *vector)
    else:
        raise ValueError(f"Unsupported vector type: {vector_type}")

def deserialize_vector(binary_data: bytes, vector_type: str = "REAL_VECTOR") -> List[float]:
    """
    Deserialize a vector from the binary format used by SAP HANA Cloud.
    
    Args:
        binary_data: The binary vector data
        vector_type: The HANA vector type (REAL_VECTOR or HALF_VECTOR)
        
    Returns:
        The deserialized vector as a list of floats
        
    Raises:
        ValueError: If an unsupported vector type is provided
    """
    # Extract dimension from the first 4 bytes
    dim = struct.unpack_from("<I", binary_data, 0)[0]
    
    if vector_type == "HALF_VECTOR":
        # 2-byte half-precision float deserialization
        return list(struct.unpack_from(f"<{dim}e", binary_data, 4))
    elif vector_type == "REAL_VECTOR":
        # 4-byte float deserialization (standard FVECS format)
        return list(struct.unpack_from(f"<{dim}f", binary_data, 4))
    else:
        raise ValueError(f"Unsupported vector type: {vector_type}")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (1.0 means identical, -1.0 means opposite, 0.0 means orthogonal)
    """
    a = np.array(vec1)
    b = np.array(vec2)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance (0.0 means identical, larger values mean more distant)
    """
    a = np.array(vec1)
    b = np.array(vec2)
    
    return np.linalg.norm(a - b)

def convert_to_distance_score(
    raw_score: float, 
    distance_strategy: DistanceStrategy
) -> float:
    """
    Convert a raw distance/similarity score to a normalized score.
    
    This function normalizes scores from different distance metrics to a common scale
    where higher values always indicate more similar vectors.
    
    Args:
        raw_score: The raw score from the distance function
        distance_strategy: The distance strategy used
        
    Returns:
        Normalized score (higher values mean more similar)
    """
    if distance_strategy == DistanceStrategy.COSINE:
        # Cosine similarity is already normalized between -1 and 1, with 1 being identical
        # Scale to [0, 1] range
        return (raw_score + 1) / 2
    elif distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
        # Euclidean distance is unbounded, with 0 being identical
        # Convert to similarity score using exponential decay
        return np.exp(-raw_score)
    else:
        # For other distance strategies, implement appropriate normalization
        return raw_score

def create_vector_table(
    cursor,
    table_name: str,
    content_column: str = "VEC_TEXT",
    metadata_column: str = "VEC_META",
    vector_column: str = "VEC_VECTOR",
    vector_column_type: str = "REAL_VECTOR",
    vector_column_length: Optional[int] = None,
    specific_metadata_columns: Optional[List[str]] = None,
    if_not_exists: bool = True
) -> None:
    """
    Create a vector table in SAP HANA Cloud.
    
    Args:
        cursor: Database cursor
        table_name: Name of the table to create
        content_column: Name of the column for document content
        metadata_column: Name of the column for document metadata
        vector_column: Name of the column for vector embeddings
        vector_column_type: Type of the vector column (REAL_VECTOR or HALF_VECTOR)
        vector_column_length: Length of vectors (or None for dynamic length)
        specific_metadata_columns: Additional columns for extracting specific metadata fields
        if_not_exists: Whether to use IF NOT EXISTS in the CREATE TABLE statement
    """
    # Start building the SQL statement
    sql = f'CREATE TABLE {"IF NOT EXISTS " if if_not_exists else ""}{table_name} ('
    
    # Add standard columns
    sql += f'"{content_column}" NCLOB, '
    sql += f'"{metadata_column}" NCLOB, '
    
    # Add vector column with appropriate type
    if vector_column_length is not None and vector_column_length > 0:
        sql += f'"{vector_column}" {vector_column_type}({vector_column_length})'
    else:
        sql += f'"{vector_column}" {vector_column_type}'
    
    # Add specific metadata columns if provided
    if specific_metadata_columns:
        for col in specific_metadata_columns:
            sql += f', "{col}" NVARCHAR(5000)'
    
    # Close the statement
    sql += ')'
    
    # Execute the SQL
    cursor.execute(sql)

def create_hnsw_index(
    cursor,
    table_name: str,
    vector_column: str,
    distance_function: str,
    index_name: Optional[str] = None,
    m: Optional[int] = None,
    ef_construction: Optional[int] = None,
    ef_search: Optional[int] = None
) -> None:
    """
    Create an HNSW vector index in SAP HANA Cloud.
    
    Args:
        cursor: Database cursor
        table_name: Name of the table
        vector_column: Name of the vector column
        distance_function: Distance function (COSINE_SIMILARITY or L2DISTANCE)
        index_name: Optional custom index name
        m: Optional M parameter (valid range: [4, 1000])
        ef_construction: Optional efConstruction parameter (valid range: [1, 100000])
        ef_search: Optional efSearch parameter (valid range: [1, 100000])
    """
    # Set default index name if not provided
    if index_name is None:
        index_name = f"{table_name}_{distance_function.lower()}_idx"
    
    # Initialize configurations
    build_config = {}
    search_config = {}
    
    # Add M parameter if provided
    if m is not None:
        if not (4 <= m <= 1000):
            raise ValueError("M must be in the range [4, 1000]")
        build_config["M"] = m
    
    # Add ef_construction parameter if provided
    if ef_construction is not None:
        if not (1 <= ef_construction <= 100000):
            raise ValueError("efConstruction must be in the range [1, 100000]")
        build_config["efConstruction"] = ef_construction
    
    # Add ef_search parameter if provided
    if ef_search is not None:
        if not (1 <= ef_search <= 100000):
            raise ValueError("efSearch must be in the range [1, 100000]")
        search_config["efSearch"] = ef_search
    
    # Convert configs to JSON strings if they contain values
    build_config_str = json.dumps(build_config) if build_config else ""
    search_config_str = json.dumps(search_config) if search_config else ""
    
    # Create the SQL string
    sql = (
        f"CREATE HNSW VECTOR INDEX {index_name} ON {table_name} "
        f"({vector_column}) "
        f"SIMILARITY FUNCTION {distance_function} "
    )
    
    # Add build config if provided
    if build_config_str:
        sql += f"BUILD CONFIGURATION '{build_config_str}' "
    
    # Add search config if provided
    if search_config_str:
        sql += f"SEARCH CONFIGURATION '{search_config_str}' "
    
    # Always add ONLINE option
    sql += "ONLINE"
    
    # Execute the SQL
    cursor.execute(sql)

def validate_vector_type_support(
    cursor,
    vector_type: str
) -> bool:
    """
    Validate that a vector type is supported by the SAP HANA Cloud instance.
    
    Args:
        cursor: Database cursor
        vector_type: Vector type to validate (REAL_VECTOR or HALF_VECTOR)
        
    Returns:
        True if the vector type is supported, False otherwise
    """
    try:
        cursor.execute("SELECT COUNT(*) FROM SYS.DATA_TYPES WHERE TYPE_NAME = ?", (vector_type,))
        count = cursor.fetchone()[0]
        return count > 0
    except Exception as e:
        logger.warning(f"Error checking vector type support: {str(e)}")
        return False

def get_min_supported_version(vector_type: str) -> str:
    """
    Get the minimum SAP HANA Cloud version that supports a vector type.
    
    Args:
        vector_type: Vector type (REAL_VECTOR or HALF_VECTOR)
        
    Returns:
        String describing the minimum version
        
    Raises:
        ValueError: If an unsupported vector type is provided
    """
    if vector_type == "REAL_VECTOR":
        return "2024.2 (QRC 1/2024)"
    elif vector_type == "HALF_VECTOR":
        return "2025.15 (QRC 2/2025)"
    else:
        raise ValueError(f"Unknown vector type: '{vector_type}'")

def convert_distance_strategy_to_sql(
    distance_strategy: DistanceStrategy
) -> Tuple[str, str]:
    """
    Convert a DistanceStrategy to the corresponding SQL function and sort order.
    
    Args:
        distance_strategy: The distance strategy
        
    Returns:
        Tuple containing:
            - String with the SQL function name
            - String with the SQL sort order (ASC or DESC)
    """
    if distance_strategy == DistanceStrategy.COSINE:
        return "COSINE_SIMILARITY", "DESC"  # Higher values are better
    elif distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
        return "L2DISTANCE", "ASC"  # Lower values are better
    elif distance_strategy == DistanceStrategy.DOT_PRODUCT:
        return "DOT_PRODUCT", "DESC"  # Higher values are better
    elif distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
        return "MAX_INNER_PRODUCT", "DESC"  # Higher values are better
    elif distance_strategy == DistanceStrategy.JACCARD:
        return "JACCARD", "DESC"  # Higher values are better
    else:
        raise ValueError(f"Unsupported distance strategy: {distance_strategy}")