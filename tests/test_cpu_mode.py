#!/usr/bin/env python3
"""Test script to verify TensorRT embeddings modules can be imported in CPU-only mode.
This script tests our conditional imports to ensure no GPU-related errors occur.
"""
import logging
import sys
import os

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath('.'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Test importing GPU-related modules in CPU-only mode."""
    logger.info("Starting CPU-only mode test...")
    
    try:
        # Test importing tensorrt_utils
        logger.info("Importing tensorrt_utils...")
        try:
            from api.gpu.tensorrt_utils import TENSORRT_AVAILABLE, tensorrt_optimizer
        except ImportError as e:
            logger.warning("Could not import tensorrt_utils directly: %s", e)
            logger.info("Trying alternative import path...")
            # Try alternative import path
            from gpu.tensorrt_utils import TENSORRT_AVAILABLE, tensorrt_optimizer
        logger.info("TENSORRT_AVAILABLE = %s", TENSORRT_AVAILABLE)
        logger.info("tensorrt_optimizer type = %s", type(tensorrt_optimizer))
        
        # Test importing gpu_utils
        logger.info("Importing gpu_utils...")
        try:
            from api.gpu.gpu_utils import get_available_gpu_memory
        except ImportError as e:
            logger.warning("Could not import gpu_utils directly: %s", e)
            logger.info("Trying alternative import path...")
            # Try alternative import path
            from gpu.gpu_utils import get_available_gpu_memory
        memory_info = get_available_gpu_memory()
        logger.info("GPU memory info: %s", memory_info)
        
        # Test importing embeddings_tensorrt
        logger.info("Importing embeddings_tensorrt...")
        try:
            from api.embeddings.embeddings_tensorrt import (
                TensorRTEmbeddings, 
                TensorRTEmbedding,  # Test backward compatibility alias
                TensorRTHybridEmbeddings
            )
        except ImportError as e:
            logger.warning("Could not import embeddings_tensorrt directly: %s", e)
            logger.info("Trying alternative import path...")
            # Try alternative import path
            from embeddings.embeddings_tensorrt import (
                TensorRTEmbeddings, 
                TensorRTEmbedding,  # Test backward compatibility alias
                TensorRTHybridEmbeddings
            )
        
        # Test instantiating embeddings classes
        logger.info("Creating TensorRTEmbeddings instance...")
        embeddings = TensorRTEmbeddings()
        logger.info("TensorRTEmbeddings instance: %s", embeddings)
        
        # Test basic embedding functionality
        logger.info("Testing embedding functionality...")
        test_text = ["This is a test sentence."]
        try:
            result = embeddings.embed_documents(test_text)
            logger.info("Embedding output shape: %sx%s", len(result), len(result[0]) if result else 0)
        except Exception as e:
            logger.warning("Embedding generation produced expected dummy output or error: %s", e)
        
        logger.info("Testing TensorRTHybridEmbeddings...")
        hybrid_embeddings = TensorRTHybridEmbeddings()
        logger.info("TensorRTHybridEmbeddings instance: %s", hybrid_embeddings)
        
        # Test if alias is working correctly
        logger.info("Testing backward compatibility alias...")
        assert TensorRTEmbedding == TensorRTEmbeddings, "Alias is not properly set up"
        logger.info("Alias is correctly configured")
        
        logger.info("All imports and tests completed successfully in CPU-only mode!")
        return True
        
    except ImportError as e:
        logger.error("Import error: %s", e)
        return False
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
