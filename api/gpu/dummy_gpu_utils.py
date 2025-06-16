"""Dummy GPU utilities for CPU-only mode."""

import logging
logger = logging.getLogger(__name__)

def get_gpu_info():
    """Return dummy GPU info in CPU-only mode."""
    return {
        "gpu_count": 0, 
        "gpu_names": [], 
        "cpu_only": True
    }

def get_gpu_utilization():
    """Return dummy GPU utilization in CPU-only mode."""
    return [{
        "id": 0, 
        "name": "CPU", 
        "utilization": 0, 
        "memory": 0
    }]
