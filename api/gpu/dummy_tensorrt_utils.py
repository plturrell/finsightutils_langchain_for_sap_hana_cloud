"""Dummy TensorRT utilities for CPU-only mode."""

import logging
logger = logging.getLogger(__name__)

class TensorRTEngine:
    """Dummy TensorRT engine for CPU-only mode."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the dummy TensorRT engine."""
        logger.warning("TensorRT not available in CPU mode. Using dummy implementation.")

    def run(self, *args, **kwargs):
        """Simulate TensorRT execution in CPU mode."""
        logger.warning("TensorRT execution simulated in CPU mode.")
        return None

def optimize_model(*args, **kwargs):
    """Simulate TensorRT model optimization in CPU mode."""
    logger.warning("TensorRT optimization simulated in CPU mode.")
    return None
