"""
This is a reference file showing how to modify the embedding initialization code
in developer_service.py to properly handle both CPU and GPU environments.

INSTRUCTIONS:
1. Look for the embedding initialization blocks in developer_service.py
2. Replace them with the corresponding modified versions below
3. These modifications allow proper fallbacks from GPU to CPU when needed
"""

# For the first embedding initialization block around line 450-472
# Replace with:

# Try TensorRT embeddings if requested
if use_gpu and use_tensorrt and gpu_utils.is_gpu_available():
    try:
        embeddings = TensorRTEmbeddings(
            model_name=model_name,
            device="cuda",
            precision="fp16"
        )
        logger.info("Using TensorRT embeddings")
    except (ImportError, NameError, AttributeError, Exception) as e:
        logger.warning(f"TensorRT embeddings unavailable: {str(e)}. Trying standard GPU embeddings.")
        embeddings = None  # Will try next option

# Try standard GPU embeddings if TensorRT failed or wasn't requested
if (embeddings is None or not 'embeddings' in locals()) and use_gpu and gpu_utils.is_gpu_available():
    try:
        embeddings = GPUAcceleratedEmbeddings(
            model_name=model_name,
            device="cuda",
            batch_size=32
        )
        logger.info("Using GPU-accelerated embeddings")
    except (ImportError, NameError, AttributeError, Exception) as e:
        logger.warning(f"GPU embeddings unavailable: {str(e)}. Falling back to CPU embeddings.")
        embeddings = None  # Will fall back to CPU embeddings

# Fall back to HANA internal embeddings if needed
if embeddings is None or not 'embeddings' in locals():
    embeddings = HanaInternalEmbeddings(
        internal_embedding_model_id=config.gpu.internal_embedding_model_id
        if hasattr(config, 'gpu')
        else "SAP_NEB.20240715"
    )
    logger.info("Using CPU-compatible HanaInternalEmbeddings")


# For the second block (around line 870-890), use similar approach:
# Replace with:

# Try TensorRT embeddings if requested
if use_gpu and use_tensorrt and gpu_utils.is_gpu_available():
    try:
        embeddings = TensorRTEmbeddings(
            model_name=model_name,
            device="cuda",
            precision="fp16"
        )
        logger.info("Using TensorRT embeddings for second instance")
    except (ImportError, NameError, AttributeError, Exception) as e:
        logger.warning(f"TensorRT embeddings unavailable: {str(e)}. Trying standard GPU embeddings.")
        embeddings = None  # Will try next option

# Try standard GPU embeddings if TensorRT failed or wasn't requested
if (embeddings is None or not 'embeddings' in locals()) and use_gpu and gpu_utils.is_gpu_available():
    try:
        embeddings = GPUAcceleratedEmbeddings(
            model_name=model_name,
            device="cuda",
            batch_size=32
        )
        logger.info("Using GPU-accelerated embeddings for second instance")
    except (ImportError, NameError, AttributeError, Exception) as e:
        logger.warning(f"GPU embeddings unavailable: {str(e)}. Falling back to CPU embeddings.")
        embeddings = None  # Will fall back to CPU embeddings

# Fall back to HANA internal embeddings if needed
if embeddings is None or not 'embeddings' in locals():
    embeddings = HanaInternalEmbeddings(
        internal_embedding_model_id=config.gpu.internal_embedding_model_id
        if hasattr(config, 'gpu')
        else "SAP_NEB.20240715"
    )
    logger.info("Using CPU-compatible HanaInternalEmbeddings for second instance")
