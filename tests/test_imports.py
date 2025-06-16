#!/usr/bin/env python3
"""Simple test script to verify CPU-only compatibility."""

import importlib.util
import logging
import os
import sys
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))
logger.info("Added %s to Python path", project_root)
logger.info("Python sys.path:")
for path in sys.path:
    logger.info("  %s", path)


# Check if the module files exist
logger.info("Checking file existence:")
gpu_utils_path = os.path.join(project_root, "api", "gpu", "gpu_utils.py")
tensorrt_utils_path = os.path.join(
    project_root, "api", "gpu", "tensorrt_utils.py")
embeddings_tensorrt_path = os.path.join(
    project_root, "api", "embeddings", "embeddings_tensorrt.py")
logger.info("  gpu_utils.py exists: %s", os.path.exists(gpu_utils_path))
logger.info("  tensorrt_utils.py exists: %s", os.path.exists(tensorrt_utils_path))
logger.info("  embeddings_tensorrt.py exists: %s",
            os.path.exists(embeddings_tensorrt_path))


def main():
    """Test basic imports to verify CPU-only compatibility."""
    success = True

    # Test importing gpu_utils module with fallback
    gpu_utils = None
    logger.info("Importing gpu_utils module using standard import...")
    try:
        import api.gpu.gpu_utils as gpu_utils
    except ImportError as e:
        logger.warning("Standard import of gpu_utils failed: %s", e)

        # Try direct file import
        logger.info("Trying direct file import for gpu_utils...")
        try:
            if os.path.exists(gpu_utils_path):
                spec = importlib.util.spec_from_file_location(
                    "gpu_utils", gpu_utils_path)
                if spec and spec.loader:
                    gpu_utils = importlib.util.module_from_spec(spec)
                    sys.modules["gpu_utils"] = gpu_utils
                    spec.loader.exec_module(gpu_utils)
        except (ImportError, AttributeError) as e:
            logger.error("Direct file import of gpu_utils failed: %s", e)
            success = False

    if gpu_utils:
        memory_info = gpu_utils.get_available_gpu_memory()
        gpu_available = gpu_utils.is_gpu_available()
        logger.info("GPU memory info: %s", memory_info)
        logger.info("GPU available: %s", gpu_available)
    else:
        logger.error("Failed to import gpu_utils through any method")
        success = False

    # Test importing tensorrt_utils module
    tensorrt_utils = None
    logger.info("Importing tensorrt_utils module...")
    try:
        # We need to import the module and assign it to the variable
        import api.gpu.tensorrt_utils
        tensorrt_utils = api.gpu.tensorrt_utils
    except ImportError as e:
        logger.warning("Standard import of tensorrt_utils failed: %s", e)

        # Try direct file import
        logger.info("Trying direct file import for tensorrt_utils...")
        try:
            if os.path.exists(tensorrt_utils_path):
                spec = importlib.util.spec_from_file_location(
                    "tensorrt_utils", tensorrt_utils_path)
                if spec and spec.loader:
                    tensorrt_utils = importlib.util.module_from_spec(spec)
                    sys.modules["tensorrt_utils"] = tensorrt_utils
                    spec.loader.exec_module(tensorrt_utils)
        except (ImportError, AttributeError) as e:
            logger.error("Direct file import of tensorrt_utils failed: %s", e)
            success = False

    if tensorrt_utils:
        logger.info("TENSORRT_AVAILABLE = %s", tensorrt_utils.TENSORRT_AVAILABLE)
        logger.info("tensorrt_optimizer type = %s",
                   type(tensorrt_utils.tensorrt_optimizer))
    else:
        logger.error("Failed to import tensorrt_utils through any method")
        success = False

    # Test importing embeddings_tensorrt module
    embeddings_tensorrt = None
    logger.info("Importing embeddings_tensorrt module...")
    try:
        # Import the module and assign it to the variable
        import api.embeddings.embeddings_tensorrt
        embeddings_tensorrt = api.embeddings.embeddings_tensorrt
    except ImportError as e:
        logger.warning("Standard import of embeddings_tensorrt failed: %s", e)

        # Try direct file import
        logger.info("Trying direct file import for embeddings_tensorrt...")
        try:
            if os.path.exists(embeddings_tensorrt_path):
                spec = importlib.util.spec_from_file_location(
                    "embeddings_tensorrt", embeddings_tensorrt_path)
                if spec and spec.loader:
                    embeddings_tensorrt = importlib.util.module_from_spec(spec)
                    sys.modules["embeddings_tensorrt"] = embeddings_tensorrt
                    spec.loader.exec_module(embeddings_tensorrt)
        except (ImportError, AttributeError) as e:
            logger.error("Direct file import of embeddings_tensorrt failed: %s", e)
            success = False

    if embeddings_tensorrt:
        # Test class initialization (should work with CPU fallback)
        try:
            logger.info("Testing TensorRTEmbeddings initialization...")
            embeddings = embeddings_tensorrt.TensorRTEmbeddings(
                model_name="all-MiniLM-L6-v2", device="cpu")

            # Test embeddings functionality
            query_embedding = embeddings.embed_query("Test query")
            logger.info("Query embedding length: %d", len(query_embedding))

            doc_embeddings = embeddings.embed_documents(
                ["Test document 1", "Test document 2"])
            logger.info("Number of document embeddings: %d",
                        len(doc_embeddings))

            # Also test the backward compatibility class
            logger.info("Testing TensorRTEmbedding (alias) initialization...")
            embeddings_alias = embeddings_tensorrt.TensorRTEmbedding(
                model_name="all-MiniLM-L6-v2", device="cpu")
            logger.info("Created TensorRTEmbedding instance")
        except (ImportError, RuntimeError, AttributeError) as e:
            logger.error("Error testing TensorRTEmbeddings: %s", e)
            success = False
    else:
        logger.error("Failed to import embeddings_tensorrt through any method")
        success = False

    if success:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Tests failed. See errors above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
