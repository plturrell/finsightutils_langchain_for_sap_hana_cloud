"""
Vector distance strategies and utilities.

This module provides distance strategies and functions for vector similarity
operations in SAP HANA Cloud.
"""

import enum
from typing import Dict, Tuple, List, Any, Optional, Union
import numpy as np


class DistanceStrategy(str, enum.Enum):
    """
    Enum of available distance strategies for vector similarity.
    
    These strategies map to SAP HANA's vector similarity functions.
    """
    
    COSINE = "cosine"
    """Cosine similarity - suitable for semantic similarity."""
    
    EUCLIDEAN_DISTANCE = "euclidean"
    """Euclidean distance - suitable for geospatial and other distance measures."""
    
    DOT_PRODUCT = "dot_product"
    """Dot product - suitable for unnormalized vectors."""
    
    L1_DISTANCE = "l1"
    """L1 distance (Manhattan) - suitable for certain specialized applications."""


# Map distance strategies to SAP HANA function names and sort directions
HANA_DISTANCE_FUNCTIONS: Dict[DistanceStrategy, Tuple[str, str]] = {
    DistanceStrategy.COSINE: ("COSINE_SIMILARITY", "DESC"),
    DistanceStrategy.EUCLIDEAN_DISTANCE: ("L2DISTANCE", "ASC"),
    DistanceStrategy.DOT_PRODUCT: ("DOT_PRODUCT", "DESC"),
    DistanceStrategy.L1_DISTANCE: ("L1DISTANCE", "ASC"),
}


def normalize_vector(vector: List[float]) -> List[float]:
    """
    Normalize a vector to unit length (L2 norm).
    
    Args:
        vector: Input vector
        
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return (np.array(vector) / norm).tolist()


def compute_similarity(
    vector1: List[float],
    vector2: List[float],
    strategy: DistanceStrategy = DistanceStrategy.COSINE
) -> float:
    """
    Compute similarity between two vectors using the specified strategy.
    
    Args:
        vector1: First vector
        vector2: Second vector
        strategy: Distance strategy to use
        
    Returns:
        Similarity score
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(vector1) != len(vector2):
        raise ValueError(f"Vectors have different dimensions: {len(vector1)} vs {len(vector2)}")
    
    v1 = np.array(vector1)
    v2 = np.array(vector2)
    
    if strategy == DistanceStrategy.COSINE:
        # Normalize vectors for cosine similarity
        v1_norm = v1 / np.linalg.norm(v1) if np.linalg.norm(v1) > 0 else v1
        v2_norm = v2 / np.linalg.norm(v2) if np.linalg.norm(v2) > 0 else v2
        return float(np.dot(v1_norm, v2_norm))
    
    elif strategy == DistanceStrategy.DOT_PRODUCT:
        return float(np.dot(v1, v2))
    
    elif strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
        return float(np.linalg.norm(v1 - v2))
    
    elif strategy == DistanceStrategy.L1_DISTANCE:
        return float(np.sum(np.abs(v1 - v2)))
    
    else:
        raise ValueError(f"Unsupported distance strategy: {strategy}")


def convert_distance_to_similarity(
    distance: float,
    strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    max_distance: Optional[float] = None
) -> float:
    """
    Convert a distance measure to a similarity score (higher is more similar).
    
    Args:
        distance: Distance value
        strategy: Distance strategy that was used
        max_distance: Optional maximum distance for normalization
        
    Returns:
        Similarity score
    """
    # For strategies where higher values already mean more similar
    if strategy in (DistanceStrategy.COSINE, DistanceStrategy.DOT_PRODUCT):
        return distance
    
    # For distance measures, convert to similarity (1 is identical, 0 is completely dissimilar)
    if strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
        # If max_distance is not provided, use a reasonable default
        if max_distance is None:
            max_distance = 10.0  # Arbitrary choice for normalization
        
        # Ensure distance is not greater than max_distance
        distance = min(distance, max_distance)
        
        # Convert to similarity: 1 when distance is 0, approaching 0 as distance increases
        return 1.0 - (distance / max_distance)
    
    elif strategy == DistanceStrategy.L1_DISTANCE:
        # If max_distance is not provided, use a reasonable default
        if max_distance is None:
            max_distance = 20.0  # Arbitrary choice for normalization
        
        # Ensure distance is not greater than max_distance
        distance = min(distance, max_distance)
        
        # Convert to similarity: 1 when distance is 0, approaching 0 as distance increases
        return 1.0 - (distance / max_distance)
    
    else:
        raise ValueError(f"Unsupported distance strategy: {strategy}")


def get_sql_order_direction(strategy: DistanceStrategy) -> str:
    """
    Get the SQL ORDER BY direction for a given distance strategy.
    
    Args:
        strategy: Distance strategy
        
    Returns:
        "ASC" or "DESC" depending on whether lower or higher values are better
    """
    if strategy not in HANA_DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance strategy: {strategy}")
    
    return HANA_DISTANCE_FUNCTIONS[strategy][1]


def get_hana_function_name(strategy: DistanceStrategy) -> str:
    """
    Get the SAP HANA function name for a given distance strategy.
    
    Args:
        strategy: Distance strategy
        
    Returns:
        HANA function name for the strategy
    """
    if strategy not in HANA_DISTANCE_FUNCTIONS:
        raise ValueError(f"Unsupported distance strategy: {strategy}")
    
    return HANA_DISTANCE_FUNCTIONS[strategy][0]