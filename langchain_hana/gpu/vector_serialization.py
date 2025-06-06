"""
Memory-optimized vector serialization for efficient transfer between GPU and SAP HANA Cloud.

This module provides specialized functions for efficient vector serialization and deserialization,
optimized for transferring embedding vectors between GPU memory and SAP HANA Cloud.
"""

import struct
import logging
import zlib
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


def serialize_vector(
    vector: Union[List[float], np.ndarray, "torch.Tensor"],
    precision: str = "float32",
    compression: bool = False,
    compression_level: int = 6,
) -> bytes:
    """
    Serialize a vector into an efficient binary format.
    
    This function converts a vector (list, numpy array, or PyTorch tensor) into
    a compact binary representation for efficient storage and transfer. It supports
    different precision modes for optimized memory usage.
    
    Supported precision modes:
    - float32: Standard 32-bit floating point (4 bytes per value)
    - float16: Half-precision 16-bit floating point (2 bytes per value)
    - int8: Quantized 8-bit integer (1 byte per value)
    
    Args:
        vector: Vector to serialize
        precision: Precision to use for serialization
        compression: Whether to apply zlib compression
        compression_level: Compression level (1-9, higher = better compression but slower)
        
    Returns:
        Serialized binary data
    
    Technical details:
    - float32: Standard IEEE 754 format
    - float16: IEEE 754 half-precision format
    - int8: Linear quantization in range [-127, 127] with scale factor
    """
    # Convert input to numpy array
    if HAS_TORCH and isinstance(vector, torch.Tensor):
        # Move to CPU if on GPU
        if vector.is_cuda:
            vector = vector.cpu()
        # Convert to numpy
        vector_np = vector.detach().numpy()
    elif isinstance(vector, np.ndarray):
        vector_np = vector
    else:
        vector_np = np.array(vector, dtype=np.float32)
    
    # Ensure vector is flattened
    vector_np = vector_np.reshape(-1)
    
    # Get vector dimension
    dim = len(vector_np)
    
    # Serialize based on precision
    if precision == "float16":
        # Convert to float16
        vector_np = vector_np.astype(np.float16)
        # Serialize dimension and data
        serialized = struct.pack(f"<I{dim}e", dim, *vector_np)
    elif precision == "int8":
        # Normalize and quantize to int8 range
        abs_max = max(abs(np.max(vector_np)), abs(np.min(vector_np)))
        if abs_max > 0:
            scale = 127.0 / abs_max
        else:
            scale = 1.0
        
        # Scale and convert to int8
        vector_i8 = np.clip(vector_np * scale, -127, 127).astype(np.int8)
        
        # Pack dimension, scale factor, and data
        serialized = struct.pack(f"<If{dim}b", dim, float(scale), *vector_i8)
    else:  # float32 or default
        # Convert to float32
        vector_np = vector_np.astype(np.float32)
        # Serialize dimension and data
        serialized = struct.pack(f"<I{dim}f", dim, *vector_np)
    
    # Apply compression if requested
    if compression:
        serialized = zlib.compress(serialized, level=compression_level)
        # Prepend compression flag and original length
        serialized = struct.pack("<BI", 1, len(serialized)) + serialized
    else:
        # Prepend compression flag
        serialized = struct.pack("<B", 0) + serialized
    
    return serialized


def deserialize_vector(serialized: bytes) -> List[float]:
    """
    Deserialize a binary vector format back to a list of floats.
    
    This function converts a binary representation back to a list of floating point
    values, automatically handling different precision modes and compression.
    
    Args:
        serialized: Binary data to deserialize
        
    Returns:
        List of float values
    """
    # Check for compression flag
    compression_flag = struct.unpack_from("<B", serialized, 0)[0]
    
    if compression_flag == 1:
        # Get compressed data length and decompress
        compressed_len = struct.unpack_from("<I", serialized, 1)[0]
        compressed_data = serialized[5:5+compressed_len]
        serialized = zlib.decompress(compressed_data)
        offset = 0
    else:
        # Skip compression flag
        serialized = serialized[1:]
        offset = 0
    
    # Get dimension
    dim = struct.unpack_from("<I", serialized, offset)[0]
    offset += 4
    
    # Determine format based on data size
    remaining_bytes = len(serialized) - offset
    bytes_per_value = remaining_bytes / dim
    
    if bytes_per_value < 1.5:
        # int8 format (including scale factor)
        scale = struct.unpack_from("<f", serialized, offset)[0]
        offset += 4
        values = struct.unpack_from(f"<{dim}b", serialized, offset)
        return [float(v) / scale for v in values]
    elif bytes_per_value < 3:
        # float16 format
        values = struct.unpack_from(f"<{dim}e", serialized, offset)
        return list(values)
    else:
        # float32 format
        values = struct.unpack_from(f"<{dim}f", serialized, offset)
        return list(values)


def serialize_batch(
    vectors: List[Union[List[float], np.ndarray, "torch.Tensor"]],
    precision: str = "float32",
    compression: bool = False,
    compression_level: int = 6,
) -> List[bytes]:
    """
    Serialize a batch of vectors efficiently.
    
    Args:
        vectors: List of vectors to serialize
        precision: Precision to use for serialization
        compression: Whether to apply zlib compression
        compression_level: Compression level (1-9)
        
    Returns:
        List of serialized binary data
    """
    return [
        serialize_vector(vector, precision, compression, compression_level)
        for vector in vectors
    ]


def deserialize_batch(serialized_vectors: List[bytes]) -> List[List[float]]:
    """
    Deserialize a batch of vectors efficiently.
    
    Args:
        serialized_vectors: List of serialized binary data
        
    Returns:
        List of vectors as lists of floats
    """
    return [deserialize_vector(serialized) for serialized in serialized_vectors]


def optimize_vector_for_storage(
    vector: List[float], 
    target_type: str = "REAL_VECTOR",
) -> bytes:
    """
    Optimize a vector for storage in SAP HANA Cloud.
    
    This function converts a vector to the optimal binary format for the
    specified SAP HANA vector column type.
    
    Args:
        vector: Vector to optimize
        target_type: Target SAP HANA vector type ("REAL_VECTOR" or "HALF_VECTOR")
        
    Returns:
        Optimized binary data for storage
    """
    if target_type == "HALF_VECTOR":
        return serialize_vector(vector, precision="float16", compression=False)
    else:  # REAL_VECTOR
        return serialize_vector(vector, precision="float32", compression=False)


def get_vector_memory_usage(
    vector_dimension: int, 
    num_vectors: int, 
    precision: str = "float32",
) -> Dict[str, Any]:
    """
    Calculate memory usage for vectors with different precision options.
    
    Args:
        vector_dimension: Dimension of each vector
        num_vectors: Number of vectors
        precision: Current precision
        
    Returns:
        Dictionary with memory usage information
    """
    bytes_per_float32 = 4
    bytes_per_float16 = 2
    bytes_per_int8 = 1
    
    # Calculate base memory usage (excluding overhead)
    memory_float32 = vector_dimension * num_vectors * bytes_per_float32
    memory_float16 = vector_dimension * num_vectors * bytes_per_float16
    memory_int8 = vector_dimension * num_vectors * bytes_per_int8
    
    # Add overhead for each vector (dimension field, etc.)
    overhead_per_vector = 4  # 4 bytes for dimension
    overhead_int8 = 4  # Additional 4 bytes for scale factor
    
    total_overhead = num_vectors * overhead_per_vector
    total_overhead_int8 = total_overhead + (num_vectors * overhead_int8)
    
    # Calculate total memory with overhead
    total_float32 = memory_float32 + total_overhead
    total_float16 = memory_float16 + total_overhead
    total_int8 = memory_int8 + total_overhead_int8
    
    # Current memory based on precision
    if precision == "float16":
        current_memory = total_float16
    elif precision == "int8":
        current_memory = total_int8
    else:  # float32
        current_memory = total_float32
    
    # Calculate savings percentages
    savings_float16 = (1 - (total_float16 / total_float32)) * 100 if total_float32 > 0 else 0
    savings_int8 = (1 - (total_int8 / total_float32)) * 100 if total_float32 > 0 else 0
    
    return {
        "vector_dimension": vector_dimension,
        "num_vectors": num_vectors,
        "current_precision": precision,
        "current_memory_bytes": current_memory,
        "current_memory_mb": current_memory / (1024 * 1024),
        "memory_float32_mb": total_float32 / (1024 * 1024),
        "memory_float16_mb": total_float16 / (1024 * 1024),
        "memory_int8_mb": total_int8 / (1024 * 1024),
        "savings_float16_vs_float32_percent": savings_float16,
        "savings_int8_vs_float32_percent": savings_int8,
        "bytes_per_vector_float32": vector_dimension * bytes_per_float32 + overhead_per_vector,
        "bytes_per_vector_float16": vector_dimension * bytes_per_float16 + overhead_per_vector,
        "bytes_per_vector_int8": vector_dimension * bytes_per_int8 + overhead_per_vector + overhead_int8,
    }