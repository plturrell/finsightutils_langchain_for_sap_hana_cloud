"""
Vector serialization utilities for SAP HANA Cloud.

This module provides functions for efficient serialization and deserialization
of vector data for SAP HANA Cloud storage.
"""

import struct
from typing import List, Dict, Any, Union, Optional
import json


def serialize_vector(vector: List[float], vector_type: str = "REAL_VECTOR") -> bytes:
    """
    Serialize a vector to the binary format expected by SAP HANA Cloud.
    
    Args:
        vector: List of floating point values
        vector_type: Type of vector to serialize ('REAL_VECTOR' or 'HALF_VECTOR')
        
    Returns:
        Binary serialized vector
        
    Raises:
        ValueError: If vector_type is not supported
    """
    if not vector:
        raise ValueError("Cannot serialize empty vector")
    
    # Validate vector elements
    for value in vector:
        if not isinstance(value, (int, float)):
            raise ValueError(f"Vector contains non-numeric value: {value}")
    
    # Serialize based on vector type
    if vector_type == "REAL_VECTOR":
        # Standard 32-bit float serialization (4 bytes per value)
        # First 4 bytes contain the vector dimension as uint32
        return struct.pack(f"<I{len(vector)}f", len(vector), *vector)
    
    elif vector_type == "HALF_VECTOR":
        # 16-bit float serialization (2 bytes per value)
        # First 4 bytes contain the vector dimension as uint32
        # Note: This uses half-precision floats which may reduce accuracy
        try:
            import numpy as np
            float16_values = np.array(vector, dtype=np.float16)
            dimension_bytes = struct.pack("<I", len(vector))
            
            # Convert float16 values to bytes
            vector_bytes = float16_values.tobytes()
            
            return dimension_bytes + vector_bytes
        except ImportError:
            raise ImportError("NumPy is required for HALF_VECTOR serialization")
    
    else:
        raise ValueError(f"Unsupported vector type: {vector_type}")


def deserialize_vector(binary_data: bytes, vector_type: str = "REAL_VECTOR") -> List[float]:
    """
    Deserialize a binary vector from SAP HANA Cloud to a list of floats.
    
    Args:
        binary_data: Binary vector data
        vector_type: Type of vector to deserialize ('REAL_VECTOR' or 'HALF_VECTOR')
        
    Returns:
        List of floating point values
        
    Raises:
        ValueError: If vector_type is not supported or data is invalid
    """
    if not binary_data:
        raise ValueError("Cannot deserialize empty binary data")
    
    # Extract dimension from first 4 bytes
    if len(binary_data) < 4:
        raise ValueError("Binary data too short")
    
    dimension = struct.unpack_from("<I", binary_data, 0)[0]
    
    # Deserialize based on vector type
    if vector_type == "REAL_VECTOR":
        # Expected size: 4 bytes for dimension + 4 bytes per float
        expected_size = 4 + (4 * dimension)
        if len(binary_data) < expected_size:
            raise ValueError(f"Binary data too short: {len(binary_data)} bytes, expected {expected_size}")
        
        # Unpack 32-bit floats
        return list(struct.unpack_from(f"<{dimension}f", binary_data, 4))
    
    elif vector_type == "HALF_VECTOR":
        # Expected size: 4 bytes for dimension + 2 bytes per half-float
        expected_size = 4 + (2 * dimension)
        if len(binary_data) < expected_size:
            raise ValueError(f"Binary data too short: {len(binary_data)} bytes, expected {expected_size}")
        
        try:
            import numpy as np
            # Extract the float16 values (skipping first 4 bytes)
            float16_array = np.frombuffer(binary_data[4:expected_size], dtype=np.float16)
            
            # Convert to regular Python floats
            return float16_array.astype(np.float32).tolist()
        except ImportError:
            raise ImportError("NumPy is required for HALF_VECTOR deserialization")
    
    else:
        raise ValueError(f"Unsupported vector type: {vector_type}")


def serialize_metadata(metadata: Dict[str, Any]) -> str:
    """
    Serialize metadata to JSON for storage in SAP HANA Cloud.
    
    This function includes validation and sanitization for better
    database compatibility.
    
    Args:
        metadata: Dictionary of metadata
        
    Returns:
        JSON string representation
        
    Raises:
        ValueError: If metadata contains invalid keys
    """
    if not metadata:
        return "{}"
    
    # Validate metadata keys (alphanumeric and underscores only)
    for key in metadata:
        if not key.replace("_", "").isalnum():
            raise ValueError(f"Invalid metadata key: {key}. Keys must contain only alphanumeric characters and underscores.")
    
    # Sanitize values to ensure they're JSON-serializable
    sanitized_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
            # Convert tuples to lists for JSON serialization
            if isinstance(value, tuple):
                sanitized_metadata[key] = list(value)
            else:
                sanitized_metadata[key] = value
        else:
            # Convert non-serializable values to strings
            sanitized_metadata[key] = str(value)
    
    # Serialize to JSON with sorting for consistent output
    return json.dumps(sanitized_metadata, sort_keys=True)


def deserialize_metadata(json_string: str) -> Dict[str, Any]:
    """
    Deserialize metadata from JSON storage in SAP HANA Cloud.
    
    Args:
        json_string: JSON string representation
        
    Returns:
        Dictionary of metadata
        
    Raises:
        ValueError: If JSON string is invalid
    """
    if not json_string or json_string == "{}":
        return {}
    
    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid metadata JSON: {e}")


def estimate_vector_size(dimension: int, vector_type: str = "REAL_VECTOR") -> int:
    """
    Estimate the size in bytes of a serialized vector.
    
    Args:
        dimension: Vector dimension
        vector_type: Type of vector ('REAL_VECTOR' or 'HALF_VECTOR')
        
    Returns:
        Estimated size in bytes
        
    Raises:
        ValueError: If vector_type is not supported
    """
    if vector_type == "REAL_VECTOR":
        # 4 bytes for dimension + 4 bytes per float
        return 4 + (4 * dimension)
    elif vector_type == "HALF_VECTOR":
        # 4 bytes for dimension + 2 bytes per half-float
        return 4 + (2 * dimension)
    else:
        raise ValueError(f"Unsupported vector type: {vector_type}")


def validate_vector_dimension(vector: List[float], expected_dimension: Optional[int] = None) -> int:
    """
    Validate vector dimension and return it.
    
    Args:
        vector: Vector to validate
        expected_dimension: Expected dimension or None
        
    Returns:
        Vector dimension
        
    Raises:
        ValueError: If vector dimension doesn't match expected_dimension
    """
    dimension = len(vector)
    
    if expected_dimension is not None and dimension != expected_dimension:
        raise ValueError(
            f"Vector dimension mismatch: got {dimension}, expected {expected_dimension}"
        )
    
    return dimension