#!/usr/bin/env python3
"""Simple test script to verify CPU-only compatibility with proper imports."""

import logging
import sys
import os
import pathlib
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = pathlib.Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
logger.info("Added %s to Python path", project_root)

# Print the current sys.path for debugging
logger.info("Python sys.path:")
for p in sys.path:
    logger.info("  %s", p)

# Verify that the expected module files exist
gpu_utils_path = os.path.join(project_root, "api", "gpu", "gpu_utils.py")
tensorrt_utils_path = os.path.join(project_root, "api", "gpu", "tensorrt_utils.py")
embeddings_path = os.path.join(project_root, "api", "embeddings", "embeddings_tensorrt.py")

logger.info("Checking file existence:")
logger.info("  gpu_utils.py exists: %s", os.path.exists(gpu_utils_path))
logger.info("  tensorrt_utils.py exists: %s", os.path.exists(tensorrt_utils_path))
logger.info("  embeddings_tensorrt.py exists: %s", os.path.exists(embeddings_path))

def import_from_path(module_name, file_path):
    """Import a module from a file path directly."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not spec or not spec.loader:
        return None
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def main():
    """Test basic imports to verify CPU-only compatibility."""
    success = True
    
    # Test direct imports first
    try:
        logger.info("Importing gpu_utils module using standard import...")
        from api.gpu.gpu_utils import get_available_gpu_memory, is_gpu_available
        memory_info = get_available_gpu_memory()
        logger.info("GPU memory info: %s", memory_info)
        logger.info("GPU available: %s", is_gpu_available())
    except Exception as e:
        logger.warning("Standard import of gpu_utils failed: %s", e)
        
        # Try direct file import
        try:
            logger.info("Trying direct file import for gpu_utils...")
            gpu_utils = import_from_path("gpu_utils", gpu_utils_path)
            if gpu_utils:
                memory_info = gpu_utils.get_available_gpu_memory()
                logger.info("GPU memory info: %s", memory_info)
                logger.info("GPU available: %s", gpu_utils.is_gpu_available())
            else:
                logger.error("Failed to import gpu_utils directly from file")
                success = False
        except Exception as e:
            logger.error("Failed to import gpu_utils: %s", e)
            success = False

    try:
        logger.info("Importing tensorrt_utils module...")
        from api.gpu.tensorrt_utils import TENSORRT_AVAILABLE, tensorrt_optimizer
        logger.info("TENSORRT_AVAILABLE = %s", TENSORRT_AVAILABLE)
        logger.info("tensorrt_optimizer type = %s", type(tensorrt_optimizer))
    except Exception as e:
        logger.warning("Standard import of tensorrt_utils failed: %s", e)
        
        # Try direct file import
        try:
            logger.info("Trying direct file import for tensorrt_utils...")
            tensorrt_utils = import_from_path("tensorrt_utils", tensorrt_utils_path)
            if tensorrt_utils:
                logger.info("TENSORRT_AVAILABLE = %s", tensorrt_utils.TENSORRT_AVAILABLE)
                logger.info("tensorrt_optimizer type = %s", type(tensorrt_utils.tensorrt_optimizer))
            else:
                logger.error("Failed to import tensorrt_utils directly from file")
                success = False
        except Exception as e:
            logger.error("Failed to import tensorrt_utils: %s", e)
            success = False
    
    try:
        logger.info("Importing embeddings_tensorrt module...")
        from api.embeddings.embeddings_tensorrt import (
            TensorRTEmbeddings, 
            TensorRTEmbedding
        )
        
        logger.info("Testing TensorRTEmbeddings initialization...")
        # Force CPU mode for testing
        embeddings = TensorRTEmbeddings(
            model_name="all-MiniLM-L6-v2",  # A smaller model for faster testing
            use_tensorrt=False  # Skip TensorRT optimization for this test
        )
        
        # Test basic functionality
        query_embedding = embeddings.embed_query("Hello, world")
        logger.info("Query embedding length: %d", len(query_embedding))
        
        doc_embeddings = embeddings.embed_documents(["Document 1", "Document 2"])
        logger.info("Number of document embeddings: %d", len(doc_embeddings))
        
    except Exception as e:
        logger.warning("Standard import of embeddings_tensorrt failed: %s", e)
        
        # Try direct file import
        try:
            logger.info("Trying direct file import for embeddings_tensorrt...")
            embeddings_module = import_from_path("embeddings_tensorrt", embeddings_path)
            if embeddings_module:
                TensorRTEmbeddings = embeddings_module.TensorRTEmbeddings
                embeddings = TensorRTEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    use_tensorrt=False
                )
                query_embedding = embeddings.embed_query("Hello, world")
                logger.info("Query embedding length: %d", len(query_embedding))
            else:
                logger.error("Failed to import embeddings_tensorrt directly from file")
                success = False
        except Exception as e:
            logger.error("Failed testing embeddings_tensorrt: %s", e)
            success = False
        
    if success:
        logger.info("All tests passed successfully!")
        return 0
    else:
        logger.error("Tests failed. See errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
