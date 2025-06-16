#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script to verify embedding initialization works properly in both CPU and GPU environments.

This script tests:
1. Force CPU-only mode and verify fallback
2. GPU availability detection
3. TensorRT embedding initialization with fallback
4. Standard GPU embedding initialization with fallback
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("embedding_test")

# Add project root to path for imports
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
sys.path.append(os.path.join(project_root, 'api', 'gpu'))
import gpu_utils
from langchain_hana.embeddings import HanaInternalEmbeddings

def test_cpu_only_mode():
    """Test embedding initialization in CPU-only mode."""
    logger.info("TESTING CPU-ONLY MODE")
    
    # Force CPU-only mode
    os.environ["FORCE_CPU"] = "1"
    
    # Reload gpu_utils to apply the environment variable
    import importlib
    importlib.reload(gpu_utils)
    
    # Verify GPU is reported as unavailable
    logger.info(f"GPU reported as available: {gpu_utils.is_gpu_available()}")
    assert not gpu_utils.is_gpu_available(), "GPU should not be available in forced CPU mode"
    
    # Initialize embeddings - should get HanaInternalEmbeddings
    embeddings = initialize_embeddings(model_name="all-MiniLM-L6-v2", use_gpu=True, use_tensorrt=True)
    
    # Verify we got the right type
    logger.info(f"Embedding type: {type(embeddings).__name__}")
    assert isinstance(embeddings, HanaInternalEmbeddings), "Should fall back to HanaInternalEmbeddings in CPU mode"
    
    # Reset environment
    os.environ.pop("FORCE_CPU", None)
    importlib.reload(gpu_utils)
    logger.info("CPU-ONLY TEST COMPLETED")

def test_gpu_mode():
    """Test embedding initialization with GPU when available."""
    logger.info("TESTING GPU MODE (if available)")
    
    # Check if GPU is actually available in this environment
    gpu_available = gpu_utils.is_gpu_available()
    logger.info(f"GPU actually available: {gpu_available}")
    
    # First try with TensorRT
    logger.info("Testing TensorRT embeddings initialization:")
    tensorrt_embeddings = initialize_embeddings(model_name="all-MiniLM-L6-v2", use_gpu=True, use_tensorrt=True)
    logger.info(f"TensorRT test result type: {type(tensorrt_embeddings).__name__}")
    
    # Then try with standard GPU embeddings
    logger.info("Testing standard GPU embeddings initialization:")
    gpu_embeddings = initialize_embeddings(model_name="all-MiniLM-L6-v2", use_gpu=True, use_tensorrt=False)
    logger.info(f"GPU embeddings test result type: {type(gpu_embeddings).__name__}")
    
    # Verify we get expected behavior
    logger.info("Verifying behavior is as expected based on environment")
    if gpu_available:
        try:
            from gpu_embeddings import TensorRTEmbeddings, GPUAcceleratedEmbeddings
            logger.info("GPU embeddings modules are importable")
            # In a real environment with GPU, one of these would work
            # But in most test environments they'll fall back to CPU
        except ImportError:
            logger.info("GPU embeddings modules not importable, expected CPU fallback")
    
    # Always verify embeddings is valid and usable
    assert hasattr(tensorrt_embeddings, "embed_query"), "Embeddings should have embed_query method"
    assert hasattr(gpu_embeddings, "embed_query"), "Embeddings should have embed_query method"
    
    logger.info("GPU MODE TEST COMPLETED")

def initialize_embeddings(model_name="all-MiniLM-L6-v2", use_gpu=True, use_tensorrt=False):
    """
    Initialize embeddings with fallback mechanism (copy of our implementation).
    
    Args:
        model_name: The name of the embedding model to use
        use_gpu: Whether to try using GPU embeddings
        use_tensorrt: Whether to try using TensorRT optimized embeddings
        
    Returns:
        An embedding instance that works in the current environment
    """
    embeddings = None

    # Try TensorRT embeddings first if requested
    if use_gpu and use_tensorrt:
        try:
            if gpu_utils.is_gpu_available():
                try:
                    from gpu_embeddings import TensorRTEmbeddings
                    embeddings = TensorRTEmbeddings(
                        model_name=model_name,
                        device="cuda",
                        precision="fp16"
                    )
                    logger.info("Using TensorRT embeddings")
                except (ImportError, NameError, AttributeError, Exception) as e:
                    logger.warning("TensorRT embeddings unavailable: %s.", str(e))
                    logger.warning("Trying standard GPU embeddings")
                    # Will try next option
        except Exception as e:
            logger.warning("Error checking GPU availability: %s", str(e))
    
    # Try standard GPU embeddings if TensorRT failed or wasn't requested
    if embeddings is None and use_gpu:
        try:
            if gpu_utils.is_gpu_available():
                try:
                    from gpu_embeddings import GPUAcceleratedEmbeddings
                    embeddings = GPUAcceleratedEmbeddings(
                        model_name=model_name,
                        device="cuda",
                        batch_size=32
                    )
                    logger.info("Using GPU-accelerated embeddings")
                except (ImportError, NameError, AttributeError, Exception) as e:
                    logger.warning("GPU embeddings unavailable: %s.", str(e))
                    logger.warning("Falling back to CPU embeddings")
                    # Will fall back to CPU embeddings
        except Exception as e:
            logger.warning("Error checking GPU availability: %s", str(e))
    
    # Fall back to HANA internal embeddings if needed
    if embeddings is None:
        try:
            # Try using the specific model ID if possible
            from config import config
            model_id = config.gpu.internal_embedding_model_id if hasattr(config, 'gpu') else "SAP_NEB.20240715"
            embeddings = HanaInternalEmbeddings(
                internal_embedding_model_id=model_id
            )
        except (ImportError, AttributeError):
            # Simplest fallback when config isn't available
            embeddings = HanaInternalEmbeddings(
                model_name=model_name
            )
        logger.info("Using CPU-compatible HanaInternalEmbeddings")
    
    return embeddings

if __name__ == "__main__":
    logger.info("STARTING EMBEDDING INITIALIZATION TESTS")
    
    # Run the tests
    test_cpu_only_mode()
    test_gpu_mode()
    
    logger.info("ALL TESTS COMPLETED")
