"""
Mock utility module for langchain_hana testing.
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Union, Tuple


class DistanceStrategy(Enum):
    """Distance strategies for vector similarity search."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    INNER_PRODUCT = "inner_product"
    L2 = "l2"