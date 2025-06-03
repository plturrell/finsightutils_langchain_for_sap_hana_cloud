"""Configuration module for the FastAPI application."""

import os
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = os.getenv("HANA_HOST", "")
    port: int = int(os.getenv("HANA_PORT", "443"))
    user: str = os.getenv("HANA_USER", "")
    password: str = os.getenv("HANA_PASSWORD", "")


class APIConfig(BaseModel):
    """API configuration."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    table_name: str = os.getenv("DEFAULT_TABLE_NAME", "EMBEDDINGS")
    content_column: str = os.getenv("DEFAULT_CONTENT_COLUMN", "VEC_TEXT")
    metadata_column: str = os.getenv("DEFAULT_METADATA_COLUMN", "VEC_META")
    vector_column: str = os.getenv("DEFAULT_VECTOR_COLUMN", "VEC_VECTOR")
    vector_column_type: str = os.getenv("VECTOR_COLUMN_TYPE", "REAL_VECTOR")
    vector_column_length: int = int(os.getenv("VECTOR_COLUMN_LENGTH", "-1"))


class GPUConfig(BaseModel):
    """GPU configuration."""
    enabled: bool = os.getenv("GPU_ENABLED", "true").lower() == "true"
    device: str = os.getenv("GPU_DEVICE", "auto")
    batch_size: int = int(os.getenv("GPU_BATCH_SIZE", "32"))
    embedding_model: str = os.getenv("GPU_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    use_internal_embeddings: bool = os.getenv("USE_INTERNAL_EMBEDDINGS", "true").lower() == "true"
    internal_embedding_model_id: str = os.getenv("INTERNAL_EMBEDDING_MODEL_ID", "SAP_NEB.20240715")
    
    # TensorRT optimization settings
    use_tensorrt: bool = os.getenv("USE_TENSORRT", "true").lower() == "true"
    tensorrt_precision: str = os.getenv("TENSORRT_PRECISION", "fp16")
    tensorrt_cache_dir: str = os.getenv("TENSORRT_CACHE_DIR", "/tmp/tensorrt_engines")
    tensorrt_dynamic_shapes: bool = os.getenv("TENSORRT_DYNAMIC_SHAPES", "true").lower() == "true"


class Config(BaseModel):
    """Application configuration."""
    db: DatabaseConfig = DatabaseConfig()
    api: APIConfig = APIConfig()
    vectorstore: VectorStoreConfig = VectorStoreConfig()
    gpu: GPUConfig = GPUConfig()


# Application configuration
config = Config()