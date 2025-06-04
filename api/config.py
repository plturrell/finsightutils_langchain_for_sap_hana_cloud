"""
Configuration module for the FastAPI application.

This module provides secure configuration handling, particularly for
database connections and sensitive settings. It prioritizes environment
variables but falls back to a local .env file if available.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger("api_config")

class DatabaseConfig(BaseModel):
    """
    Database configuration with secure credential handling.
    
    These settings can be configured via environment variables:
    - HANA_HOST: SAP HANA Cloud host address
    - HANA_PORT: SAP HANA Cloud port (default: 443)
    - HANA_USER: Database username
    - HANA_PASSWORD: Database password (stored securely)
    - HANA_ENCRYPT: Whether to use SSL/TLS encryption (default: true)
    - HANA_SSL_VALIDATE_CERT: Whether to validate SSL certificates (default: true)
    """
    host: str = Field(default="", env="HANA_HOST")
    port: int = Field(default=443, env="HANA_PORT")
    user: str = Field(default="", env="HANA_USER")
    password: str = Field(default="", env="HANA_PASSWORD")
    encrypt: bool = Field(default=True, env="HANA_ENCRYPT")
    ssl_validate_cert: bool = Field(default=True, env="HANA_SSL_VALIDATE_CERT")
    
    class Config:
        env_prefix = ""
        
    def __init__(self, **data):
        """Initialize with values from environment variables if not provided."""
        if "host" not in data:
            data["host"] = os.getenv("HANA_HOST", "")
        if "port" not in data:
            data["port"] = int(os.getenv("HANA_PORT", "443"))
        if "user" not in data:
            data["user"] = os.getenv("HANA_USER", "")
        if "password" not in data:
            data["password"] = os.getenv("HANA_PASSWORD", "")
        if "encrypt" not in data:
            data["encrypt"] = os.getenv("HANA_ENCRYPT", "true").lower() == "true"
        if "ssl_validate_cert" not in data:
            data["ssl_validate_cert"] = os.getenv("HANA_SSL_VALIDATE_CERT", "true").lower() == "true"
        super().__init__(**data)
        
    def get_connection_params(self) -> Dict[str, Any]:
        """
        Get database connection parameters as a dictionary.
        
        Returns:
            Dict[str, Any]: Connection parameters for hdbcli.
        """
        return {
            "address": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "encrypt": self.encrypt,
            "sslValidateCertificate": self.ssl_validate_cert
        }
    
    def is_configured(self) -> bool:
        """
        Check if database connection is properly configured.
        
        Returns:
            bool: True if all required settings are provided.
        """
        return bool(self.host and self.user and self.password)
        
    def get_connection_string(self, mask_password: bool = True) -> str:
        """
        Get database connection string (with password masked by default).
        
        Args:
            mask_password: Whether to mask the password in the connection string.
            
        Returns:
            str: Connection string format.
        """
        password = "********" if mask_password else self.password
        return f"hana://{self.user}:{password}@{self.host}:{self.port}"


class APIConfig(BaseModel):
    """
    API configuration settings.
    
    These settings can be configured via environment variables:
    - API_HOST: Host to bind the API server (default: 0.0.0.0)
    - API_PORT: Port to bind the API server (default: 8000)
    - LOG_LEVEL: Logging level (default: INFO)
    - ENVIRONMENT: Deployment environment (default: production)
    - DEBUG: Enable debug mode (default: false)
    """
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    environment: str = Field(default="production")
    debug: bool = Field(default=False)
    
    class Config:
        env_prefix = ""
        
    def __init__(self, **data):
        """Initialize with values from environment variables if not provided."""
        if "host" not in data:
            data["host"] = os.getenv("API_HOST", "0.0.0.0")
        if "port" not in data:
            data["port"] = int(os.getenv("API_PORT", "8000"))
        if "log_level" not in data:
            data["log_level"] = os.getenv("LOG_LEVEL", "INFO")
        if "environment" not in data:
            data["environment"] = os.getenv("ENVIRONMENT", "production")
        if "debug" not in data:
            data["debug"] = os.getenv("DEBUG", "false").lower() == "true"
        super().__init__(**data)


class VectorStoreConfig(BaseModel):
    """
    Vector store configuration for SAP HANA Cloud.
    
    These settings configure the vector store table structure:
    - DEFAULT_TABLE_NAME: Name of the vector table (default: EMBEDDINGS)
    - DEFAULT_CONTENT_COLUMN: Column for document content (default: VEC_TEXT)
    - DEFAULT_METADATA_COLUMN: Column for document metadata (default: VEC_META)
    - DEFAULT_VECTOR_COLUMN: Column for vector embeddings (default: VEC_VECTOR)
    - VECTOR_COLUMN_TYPE: Type of vector column (default: REAL_VECTOR)
    - VECTOR_COLUMN_LENGTH: Length of vector embeddings (default: -1 for dynamic)
    """
    table_name: str = Field(default="EMBEDDINGS")
    content_column: str = Field(default="VEC_TEXT")
    metadata_column: str = Field(default="VEC_META")
    vector_column: str = Field(default="VEC_VECTOR")
    vector_column_type: str = Field(default="REAL_VECTOR")
    vector_column_length: int = Field(default=-1)
    
    class Config:
        env_prefix = ""
        
    def __init__(self, **data):
        """Initialize with values from environment variables if not provided."""
        if "table_name" not in data:
            data["table_name"] = os.getenv("DEFAULT_TABLE_NAME", "EMBEDDINGS")
        if "content_column" not in data:
            data["content_column"] = os.getenv("DEFAULT_CONTENT_COLUMN", "VEC_TEXT")
        if "metadata_column" not in data:
            data["metadata_column"] = os.getenv("DEFAULT_METADATA_COLUMN", "VEC_META")
        if "vector_column" not in data:
            data["vector_column"] = os.getenv("DEFAULT_VECTOR_COLUMN", "VEC_VECTOR")
        if "vector_column_type" not in data:
            data["vector_column_type"] = os.getenv("VECTOR_COLUMN_TYPE", "REAL_VECTOR")
        if "vector_column_length" not in data:
            data["vector_column_length"] = int(os.getenv("VECTOR_COLUMN_LENGTH", "-1"))
        super().__init__(**data)
        
    @validator("vector_column_type")
    def validate_vector_column_type(cls, v):
        """Validate that vector column type is supported."""
        valid_types = ["REAL_VECTOR", "HALF_VECTOR"]
        if v not in valid_types:
            raise ValueError(f"vector_column_type must be one of {valid_types}")
        return v


class GPUConfig(BaseModel):
    """
    GPU configuration for optimized embedding generation.
    
    These settings configure GPU usage and TensorRT optimization:
    - GPU_ENABLED: Whether to use GPU acceleration (default: true)
    - GPU_DEVICE: Which GPU device to use (default: auto)
    - GPU_BATCH_SIZE: Batch size for embedding generation (default: 32)
    - GPU_EMBEDDING_MODEL: Embedding model to use (default: all-MiniLM-L6-v2)
    - USE_INTERNAL_EMBEDDINGS: Whether to use SAP HANA's internal embeddings (default: true)
    - INTERNAL_EMBEDDING_MODEL_ID: SAP HANA embedding model ID (default: SAP_NEB.20240715)
    
    TensorRT optimization settings:
    - USE_TENSORRT: Whether to use TensorRT optimization (default: true)
    - TENSORRT_PRECISION: Precision for TensorRT optimization (default: fp16)
    - TENSORRT_CACHE_DIR: Directory for caching TensorRT engines (default: /tmp/tensorrt_engines)
    - TENSORRT_DYNAMIC_SHAPES: Whether to use dynamic shapes (default: true)
    """
    enabled: bool = Field(default=True)
    device: str = Field(default="auto")
    batch_size: int = Field(default=32)
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    use_internal_embeddings: bool = Field(default=True)
    internal_embedding_model_id: str = Field(default="SAP_NEB.20240715")
    
    # TensorRT optimization settings
    use_tensorrt: bool = Field(default=True)
    tensorrt_precision: str = Field(default="fp16")
    tensorrt_cache_dir: str = Field(default="/tmp/tensorrt_engines")
    tensorrt_dynamic_shapes: bool = Field(default=True)
    
    class Config:
        env_prefix = ""
        
    def __init__(self, **data):
        """Initialize with values from environment variables if not provided."""
        if "enabled" not in data:
            data["enabled"] = os.getenv("GPU_ENABLED", "true").lower() == "true"
        if "device" not in data:
            data["device"] = os.getenv("GPU_DEVICE", "auto")
        if "batch_size" not in data:
            data["batch_size"] = int(os.getenv("GPU_BATCH_SIZE", "32"))
        if "embedding_model" not in data:
            data["embedding_model"] = os.getenv("GPU_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        if "use_internal_embeddings" not in data:
            data["use_internal_embeddings"] = os.getenv("USE_INTERNAL_EMBEDDINGS", "true").lower() == "true"
        if "internal_embedding_model_id" not in data:
            data["internal_embedding_model_id"] = os.getenv("INTERNAL_EMBEDDING_MODEL_ID", "SAP_NEB.20240715")
        if "use_tensorrt" not in data:
            data["use_tensorrt"] = os.getenv("USE_TENSORRT", "true").lower() == "true"
        if "tensorrt_precision" not in data:
            data["tensorrt_precision"] = os.getenv("TENSORRT_PRECISION", "fp16")
        if "tensorrt_cache_dir" not in data:
            data["tensorrt_cache_dir"] = os.getenv("TENSORRT_CACHE_DIR", "/tmp/tensorrt_engines")
        if "tensorrt_dynamic_shapes" not in data:
            data["tensorrt_dynamic_shapes"] = os.getenv("TENSORRT_DYNAMIC_SHAPES", "true").lower() == "true"
        super().__init__(**data)
        
    @validator("tensorrt_precision")
    def validate_tensorrt_precision(cls, v):
        """Validate that TensorRT precision is supported."""
        valid_precisions = ["fp32", "fp16", "int8"]
        if v not in valid_precisions:
            raise ValueError(f"tensorrt_precision must be one of {valid_precisions}")
        return v


class FeatureConfig(BaseModel):
    """
    Feature configuration for enabling/disabling functionality.
    
    These settings control which features are enabled:
    - ENABLE_ERROR_CONTEXT: Enable context-aware error handling (default: true)
    - CACHE_VECTOR_REDUCTION: Enable caching for vector reduction operations (default: true)
    - ENABLE_ADVANCED_CLUSTERING: Enable advanced clustering for search results (default: false)
    - ENABLE_KNOWLEDGE_GRAPH: Enable knowledge graph integration (default: true)
    - ENABLE_CORS: Enable CORS support for frontend integration (default: true)
    """
    enable_error_context: bool = Field(default=True)
    cache_vector_reduction: bool = Field(default=True)
    enable_advanced_clustering: bool = Field(default=False)
    enable_knowledge_graph: bool = Field(default=True)
    enable_cors: bool = Field(default=True)
    
    class Config:
        env_prefix = ""
        
    def __init__(self, **data):
        """Initialize with values from environment variables if not provided."""
        if "enable_error_context" not in data:
            data["enable_error_context"] = os.getenv("ENABLE_ERROR_CONTEXT", "true").lower() == "true"
        if "cache_vector_reduction" not in data:
            data["cache_vector_reduction"] = os.getenv("CACHE_VECTOR_REDUCTION", "true").lower() == "true"
        if "enable_advanced_clustering" not in data:
            data["enable_advanced_clustering"] = os.getenv("ENABLE_ADVANCED_CLUSTERING", "false").lower() == "true"
        if "enable_knowledge_graph" not in data:
            data["enable_knowledge_graph"] = os.getenv("ENABLE_KNOWLEDGE_GRAPH", "true").lower() == "true"
        if "enable_cors" not in data:
            data["enable_cors"] = os.getenv("ENABLE_CORS", "true").lower() == "true"
        super().__init__(**data)


class Config(BaseModel):
    """
    Application configuration container.
    
    This class provides access to all configuration settings categories
    and handles validation and secure access to sensitive information.
    """
    db: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    vectorstore: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    gpu: GPUConfig = Field(default_factory=GPUConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.api.environment.lower() == "production"
    
    def is_vercel(self) -> bool:
        """Check if running on Vercel."""
        return os.environ.get("VERCEL", "0") == "1"
    
    def configure_logging(self):
        """Configure logging based on settings."""
        logging_level = getattr(logging, self.api.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=logging_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        
        # Log configuration status but never log sensitive information
        logger.info(f"Environment: {self.api.environment}")
        logger.info(f"API configured on {self.api.host}:{self.api.port}")
        logger.info(f"Database connection configured: {self.db.is_configured()}")
        logger.info(f"GPU enabled: {self.gpu.enabled}")
        
        if self.is_vercel():
            logger.info("Running on Vercel deployment")


# Application configuration
config = Config()

# Configure logging
config.configure_logging()