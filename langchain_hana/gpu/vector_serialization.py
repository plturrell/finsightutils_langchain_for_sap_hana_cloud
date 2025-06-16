"""
Memory-optimized vector serialization for efficient transfer between GPU and SAP HANA Cloud.

This module provides specialized functions for efficient vector serialization and deserialization,
optimized for transferring embedding vectors between GPU memory and SAP HANA Cloud.

It supports both custom binary serialization and Apache Arrow columnar format for
high-performance data transfer.
"""

import json
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

try:
    import pyarrow as pa
    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False

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
    
    # Calculate Arrow format memory usage (estimate)
    arrow_overhead = 64  # Rough estimate for Arrow batch metadata
    arrow_memory_float32 = memory_float32 + arrow_overhead
    arrow_memory_float16 = memory_float16 + arrow_overhead
    
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
    savings_arrow = (1 - (arrow_memory_float32 / total_float32)) * 100 if total_float32 > 0 else 0
    
    return {
        "vector_dimension": vector_dimension,
        "num_vectors": num_vectors,
        "current_precision": precision,
        "current_memory_bytes": current_memory,
        "current_memory_mb": current_memory / (1024 * 1024),
        "memory_float32_mb": total_float32 / (1024 * 1024),
        "memory_float16_mb": total_float16 / (1024 * 1024),
        "memory_int8_mb": total_int8 / (1024 * 1024),
        "memory_arrow_float32_mb": arrow_memory_float32 / (1024 * 1024),
        "memory_arrow_float16_mb": arrow_memory_float16 / (1024 * 1024),
        "savings_float16_vs_float32_percent": savings_float16,
        "savings_int8_vs_float32_percent": savings_int8,
        "savings_arrow_vs_binary_percent": savings_arrow,
        "bytes_per_vector_float32": vector_dimension * bytes_per_float32 + overhead_per_vector,
        "bytes_per_vector_float16": vector_dimension * bytes_per_float16 + overhead_per_vector,
        "bytes_per_vector_int8": vector_dimension * bytes_per_int8 + overhead_per_vector + overhead_int8,
    }


# Arrow serialization functions

def vector_to_arrow_array(
    vector: Union[List[float], np.ndarray, "torch.Tensor"],
    precision: str = "float32",
) -> "pa.Array":
    """
    Convert a vector to an Arrow Array.
    
    Args:
        vector: Vector to convert
        precision: Precision to use ("float32" or "float16")
        
    Returns:
        PyArrow Array
    """
    if not HAS_ARROW:
        raise ImportError(
            "The pyarrow package is required for Arrow serialization. "
            "Install it with 'pip install pyarrow'."
        )
    
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
    
    # Convert to requested precision
    if precision == "float16":
        vector_np = vector_np.astype(np.float16)
        arrow_type = pa.float16()
    else:  # float32
        vector_np = vector_np.astype(np.float32)
        arrow_type = pa.float32()
    
    # Create Arrow array
    return pa.array(vector_np, type=arrow_type)


def vectors_to_arrow_batch(
    vectors: List[Union[List[float], np.ndarray, "torch.Tensor"]],
    precision: str = "float32",
    include_metadata: bool = False,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    texts: Optional[List[str]] = None,
    ids: Optional[List[str]] = None,
) -> "pa.RecordBatch":
    """
    Convert a batch of vectors to an Arrow RecordBatch.
    
    Args:
        vectors: List of vectors to convert
        precision: Precision to use ("float32" or "float16")
        include_metadata: Whether to include metadata columns
        metadatas: Optional list of metadata dictionaries
        texts: Optional list of text strings
        ids: Optional list of IDs
        
    Returns:
        PyArrow RecordBatch
    """
    if not HAS_ARROW:
        raise ImportError(
            "The pyarrow package is required for Arrow serialization. "
            "Install it with 'pip install pyarrow'."
        )
    
    if not vectors:
        raise ValueError("Empty vector list provided")
    
    # Convert all vectors to numpy arrays
    if HAS_TORCH and isinstance(vectors[0], torch.Tensor):
        # Handle PyTorch tensors
        vectors_np = np.vstack([
            v.detach().cpu().numpy().reshape(1, -1) 
            if v.dim() == 1 else v.detach().cpu().numpy() 
            for v in vectors
        ])
    elif isinstance(vectors[0], np.ndarray):
        # Handle numpy arrays
        vectors_np = np.vstack([
            v.reshape(1, -1) if v.ndim == 1 else v 
            for v in vectors
        ])
    else:
        # Handle lists
        vectors_np = np.array(vectors, dtype=np.float32)
    
    # Get dimensions
    num_vectors = len(vectors)
    vector_dim = vectors_np.shape[1]
    
    # Convert to requested precision
    if precision == "float16":
        vectors_np = vectors_np.astype(np.float16)
        arrow_type = pa.float16()
    else:  # float32
        vectors_np = vectors_np.astype(np.float32)
        arrow_type = pa.float32()
    
    # Create a FixedSizeListArray for the vectors
    flattened = pa.array(vectors_np.flatten(), type=arrow_type)
    vectors_array = pa.FixedSizeListArray.from_arrays(flattened, vector_dim)
    
    # Prepare arrays dictionary
    arrays = {"vector": vectors_array}
    
    # Add IDs if provided
    if ids is not None:
        arrays["id"] = pa.array(ids, type=pa.string())
    
    # Add texts if provided
    if texts is not None:
        arrays["text"] = pa.array(texts, type=pa.string())
    
    # Add metadata if requested
    if include_metadata and metadatas is not None:
        # Convert metadata dictionaries to JSON strings
        metadata_jsons = [json.dumps(m) if m else "{}" for m in metadatas]
        arrays["metadata"] = pa.array(metadata_jsons, type=pa.string())
    
    # Create RecordBatch
    return pa.RecordBatch.from_arrays(
        [arrays[name] for name in arrays],
        names=list(arrays.keys())
    )


def arrow_batch_to_vectors(
    batch: "pa.RecordBatch",
    vector_column: str = "vector",
) -> List[List[float]]:
    """
    Convert an Arrow RecordBatch to a list of vectors.
    
    Args:
        batch: PyArrow RecordBatch
        vector_column: Name of the vector column
        
    Returns:
        List of vectors as lists of floats
    """
    if not HAS_ARROW:
        raise ImportError(
            "The pyarrow package is required for Arrow serialization. "
            "Install it with 'pip install pyarrow'."
        )
    
    # Check if vector column exists
    if vector_column not in batch.schema.names:
        raise ValueError(f"Vector column '{vector_column}' not found in RecordBatch")
    
    # Get vector column
    vector_array = batch.column(batch.schema.get_field_index(vector_column))
    
    # Check if it's a FixedSizeListArray
    if not isinstance(vector_array.type, pa.FixedSizeListType):
        raise ValueError(f"Column '{vector_column}' is not a FixedSizeListArray")
    
    # Convert to numpy and then to list of lists
    vector_np = vector_array.to_numpy()
    return vector_np.tolist()


def arrow_batch_to_documents(
    batch: "pa.RecordBatch",
    vector_column: str = "vector",
    text_column: str = "text",
    metadata_column: str = "metadata",
    id_column: str = "id",
) -> Tuple[List[List[float]], List[str], List[Dict[str, Any]], List[str]]:
    """
    Convert an Arrow RecordBatch to vectors, texts, metadata, and IDs.
    
    Args:
        batch: PyArrow RecordBatch
        vector_column: Name of the vector column
        text_column: Name of the text column
        metadata_column: Name of the metadata column
        id_column: Name of the ID column
        
    Returns:
        Tuple of (vectors, texts, metadata, ids)
    """
    if not HAS_ARROW:
        raise ImportError(
            "The pyarrow package is required for Arrow serialization. "
            "Install it with 'pip install pyarrow'."
        )
    
    # Extract vectors
    vectors = arrow_batch_to_vectors(batch, vector_column)
    
    # Extract texts if available
    texts = None
    if text_column in batch.schema.names:
        text_array = batch.column(batch.schema.get_field_index(text_column))
        texts = text_array.to_pylist()
    
    # Extract metadata if available
    metadata = None
    if metadata_column in batch.schema.names:
        metadata_array = batch.column(batch.schema.get_field_index(metadata_column))
        metadata = [json.loads(m) if m else {} for m in metadata_array.to_pylist()]
    
    # Extract IDs if available
    ids = None
    if id_column in batch.schema.names:
        id_array = batch.column(batch.schema.get_field_index(id_column))
        ids = id_array.to_pylist()
    
    return vectors, texts, metadata, ids


def serialize_arrow_batch(batch: "pa.RecordBatch", compression: bool = False) -> bytes:
    """
    Serialize an Arrow RecordBatch to bytes.
    
    Args:
        batch: PyArrow RecordBatch to serialize
        compression: Whether to use compression
        
    Returns:
        Serialized binary data
    """
    if not HAS_ARROW:
        raise ImportError(
            "The pyarrow package is required for Arrow serialization. "
            "Install it with 'pip install pyarrow'."
        )
    
    # Create an IPC output stream
    sink = pa.BufferOutputStream()
    
    if compression:
        # Use compressed IPC format
        options = pa.ipc.IpcWriteOptions(compression="zstd")
        writer = pa.ipc.RecordBatchStreamWriter(sink, batch.schema, options)
    else:
        # Use standard IPC format
        writer = pa.ipc.RecordBatchStreamWriter(sink, batch.schema)
    
    # Write the batch
    writer.write_batch(batch)
    writer.close()
    
    # Get the serialized data
    return sink.getvalue().to_pybytes()


def deserialize_arrow_batch(data: bytes) -> "pa.RecordBatch":
    """
    Deserialize binary data to an Arrow RecordBatch.
    
    Args:
        data: Serialized binary data
        
    Returns:
        PyArrow RecordBatch
    """
    if not HAS_ARROW:
        raise ImportError(
            "The pyarrow package is required for Arrow serialization. "
            "Install it with 'pip install pyarrow'."
        )
    
    # Create an IPC input stream
    source = pa.BufferReader(data)
    reader = pa.ipc.RecordBatchStreamReader(source)
    
    # Read the first batch
    return reader.read_next_batch()