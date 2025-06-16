"""
Standardized configuration for the SAP HANA Cloud LangChain Integration API.

This module provides a consistent configuration system using Pydantic settings,
supporting environment variables, .env files, and defaults.
"""

import os
import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Union

from dotenv import load_dotenv
from pydantic import BaseSettings, Field, validator

# Load environment variables from .env file
load_dotenv()


class BaseSettingsModel(BaseSettings):
    """Base settings model with common configuration."""
    
    class Config:
        """Pydantic settings configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


class APISettings(BaseSettingsModel):
    """API settings."""
    
    name: str = Field("SAP HANA Cloud LangChain Integration API", description="API name")
    description: str = Field(
        "API for SAP HANA Cloud LangChain Integration with Arrow Flight support",
        description="API description"
    )
    version: str = Field("1.1.0", description="API version")
    host: str = Field("0.0.0.0", description="API host")
    port: int = Field(8000, description="API port")
    docs_url: Optional[str] = Field("/api/docs", description="API docs URL")
    redoc_url: Optional[str] = Field("/api/redoc", description="API redoc URL")
    openapi_url: Optional[str] = Field("/api/openapi.json", description="API OpenAPI URL")
    enable_rate_limiting: bool = Field(True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(100, description="Maximum requests per minute")
    enable_csp: bool = Field(True, description="Enable Content Security Policy")
    include_exception_details: bool = Field(False, description="Include exception details in error responses")
    enforce_https: bool = Field(False, description="Enforce HTTPS")
    restrict_external_calls: bool = Field(True, description="Restrict external calls")
    
    class Config:
        env_prefix = "API_"


class AuthSettings(BaseSettingsModel):
    """Authentication settings."""
    
    enabled: bool = Field(True, description="Enable authentication")
    api_key_header: str = Field("X-API-Key", description="API key header name")
    api_keys: List[str] = Field(
        default_factory=lambda: ["dev-key"],
        description="List of valid API keys"
    )
    secret_key: str = Field(
        "supersecretkey",
        description="Secret key for signing tokens"
    )
    session_secret_key: str = Field(
        "supersecretkey",
        description="Secret key for signing session cookies"
    )
    jwt_algorithm: str = Field("HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(60, description="JWT expiration in minutes")
    
    @validator("api_keys", pre=True)
    def parse_api_keys(cls, v):
        """Parse API keys from string."""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",")]
        return v
    
    class Config:
        env_prefix = "AUTH_"


class CORSSettings(BaseSettingsModel):
    """CORS settings."""
    
    origins: List[str] = Field(["*"], description="Allowed origins")
    allow_credentials: bool = Field(True, description="Allow credentials")
    allow_methods: List[str] = Field(
        ["*"],
        description="Allowed methods"
    )
    allow_headers: List[str] = Field(
        ["*"],
        description="Allowed headers"
    )
    expose_headers: List[str] = Field(
        ["X-Request-ID"],
        description="Exposed headers"
    )
    max_age: int = Field(86400, description="Max age in seconds")
    
    @validator("origins", pre=True)
    def parse_origins(cls, v):
        """Parse origins from string."""
        if isinstance(v, str):
            return [origin.strip() for origin in origin.split(",")] if v != "*" else ["*"]
        return v
    
    @validator("allow_methods", pre=True)
    def parse_methods(cls, v):
        """Parse methods from string."""
        if isinstance(v, str):
            return [method.strip() for method in v.split(",")] if v != "*" else ["*"]
        return v
    
    @validator("allow_headers", pre=True)
    def parse_headers(cls, v):
        """Parse headers from string."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")] if v != "*" else ["*"]
        return v
    
    @validator("expose_headers", pre=True)
    def parse_expose_headers(cls, v):
        """Parse expose headers from string."""
        if isinstance(v, str):
            return [header.strip() for header in v.split(",")]
        return v
    
    class Config:
        env_prefix = "CORS_"


class LoggingSettings(BaseSettingsModel):
    """Logging settings."""
    
    level: str = Field("INFO", description="Logging level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    log_to_file: bool = Field(False, description="Log to file")
    log_file: Optional[str] = Field(None, description="Log file path")
    log_to_console: bool = Field(True, description="Log to console")
    log_headers: bool = Field(False, description="Log headers")
    log_request_body: bool = Field(False, description="Log request body")
    log_response_body: bool = Field(False, description="Log response body")
    
    class Config:
        env_prefix = "LOG_"


class DatabaseSettings(BaseSettingsModel):
    """HANA database settings."""
    
    enabled: bool = Field(True, description="Enable HANA integration")
    host: Optional[str] = Field(None, description="HANA host")
    port: Optional[int] = Field(None, description="HANA port")
    user: Optional[str] = Field(None, description="HANA user")
    password: Optional[str] = Field(None, description="HANA password")
    encrypt: bool = Field(True, description="Whether to use encryption")
    ssl_validate_cert: bool = Field(False, description="Whether to validate SSL certificate")
    connection_timeout: int = Field(30, description="Connection timeout in seconds")
    
    @validator("port")
    def validate_port(cls, v):
        """Validate port number."""
        if v is not None and (v < 1 or v > 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    class Config:
        env_prefix = "HANA_"


class GPUSettings(BaseSettingsModel):
    """GPU settings."""
    
    enabled: bool = Field(True, description="Enable GPU acceleration")
    device: str = Field("auto", description="GPU device to use")
    batch_size: int = Field(32, description="Batch size for embedding generation")
    embedding_model: str = Field("all-MiniLM-L6-v2", description="Embedding model to use")
    cuda_visible_devices: str = Field("all", description="CUDA visible devices")
    cuda_device_order: str = Field("PCI_BUS_ID", description="CUDA device order")
    cuda_memory_fraction: float = Field(0.9, description="CUDA memory fraction")
    use_internal_embeddings: bool = Field(True, description="Use SAP HANA's internal embeddings")
    internal_embedding_model_id: str = Field("SAP_NEB.20240715", description="SAP HANA embedding model ID")
    
    class Config:
        env_prefix = "GPU_"


class TensorRTSettings(BaseSettingsModel):
    """TensorRT settings."""
    
    enabled: bool = Field(True, description="Enable TensorRT")
    cache_dir: str = Field("/tmp/tensorrt_engines", description="TensorRT cache directory")
    precision: str = Field("fp16", description="Precision for TensorRT optimization")
    dynamic_shapes: bool = Field(True, description="Whether to use dynamic shapes")
    max_workspace_size: int = Field(1073741824, description="Max workspace size")
    
    @validator("precision")
    def validate_precision(cls, v):
        """Validate precision value."""
        valid_precisions = ["fp32", "fp16", "int8"]
        if v not in valid_precisions:
            raise ValueError(f"Precision must be one of {valid_precisions}")
        return v
    
    class Config:
        env_prefix = "TENSORRT_"


class ArrowFlightSettings(BaseSettingsModel):
    """Arrow Flight settings."""
    
    enabled: bool = Field(True, description="Enable Arrow Flight")
    host: str = Field("0.0.0.0", description="Arrow Flight host")
    port: int = Field(8815, description="Arrow Flight port")
    auth_enabled: bool = Field(True, description="Enable Arrow Flight authentication")
    tls_enabled: bool = Field(False, description="Enable TLS for Arrow Flight")
    tls_cert_file: Optional[str] = Field(None, description="TLS certificate file")
    tls_key_file: Optional[str] = Field(None, description="TLS key file")
    
    class Config:
        env_prefix = "ARROW_FLIGHT_"


class VectorStoreSettings(BaseSettingsModel):
    """Vector store settings."""
    
    table_name: str = Field("EMBEDDINGS", description="Default vector table name")
    content_column: str = Field("VEC_TEXT", description="Content column name")
    metadata_column: str = Field("VEC_META", description="Metadata column name")
    vector_column: str = Field("VEC_VECTOR", description="Vector column name")
    vector_column_type: str = Field("REAL_VECTOR", description="Vector column type")
    vector_column_length: int = Field(-1, description="Vector column length (-1 for dynamic)")
    
    @validator("vector_column_type")
    def validate_vector_column_type(cls, v):
        """Validate vector column type."""
        valid_types = ["REAL_VECTOR", "HALF_VECTOR"]
        if v not in valid_types:
            raise ValueError(f"Vector column type must be one of {valid_types}")
        return v
    
    class Config:
        env_prefix = "VECTORSTORE_"


class FeatureSettings(BaseSettingsModel):
    """Feature settings."""
    
    enable_error_context: bool = Field(True, description="Enable context-aware error handling")
    cache_vector_reduction: bool = Field(True, description="Enable caching for vector reduction")
    enable_advanced_clustering: bool = Field(False, description="Enable advanced clustering")
    enable_knowledge_graph: bool = Field(True, description="Enable knowledge graph integration")
    
    class Config:
        env_prefix = "FEATURE_"


class EnvironmentSettings(BaseSettingsModel):
    """Environment settings."""
    
    name: str = Field("production", description="Environment name")
    development_mode: bool = Field(False, description="Development mode")
    debug: bool = Field(False, description="Debug mode")
    
    @validator("development_mode", pre=True, always=True)
    def set_development_mode(cls, v, values):
        """Set development mode based on environment name."""
        if v is not None:
            return v
        return values.get("name", "").lower() in ("dev", "development")
    
    class Config:
        env_prefix = "ENVIRONMENT_"


class Settings(BaseSettingsModel):
    """Main settings class that combines all sub-settings."""
    
    # Core settings
    api: APISettings = Field(default_factory=APISettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    cors: CORSSettings = Field(default_factory=CORSSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    gpu: GPUSettings = Field(default_factory=GPUSettings)
    tensorrt: TensorRTSettings = Field(default_factory=TensorRTSettings)
    arrow_flight: ArrowFlightSettings = Field(default_factory=ArrowFlightSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    features: FeatureSettings = Field(default_factory=FeatureSettings)
    environment: EnvironmentSettings = Field(default_factory=EnvironmentSettings)
    
    # Backward compatibility properties for legacy code
    @property
    def API_TITLE(self) -> str:
        return self.api.name
    
    @property
    def API_DESCRIPTION(self) -> str:
        return self.api.description
    
    @property
    def API_VERSION(self) -> str:
        return self.api.version
    
    @property
    def HOST(self) -> str:
        return self.api.host
    
    @property
    def PORT(self) -> int:
        return self.api.port
    
    @property
    def API_HOST(self) -> str:
        return self.api.host
    
    @property
    def API_PORT(self) -> int:
        return self.api.port
    
    @property
    def ENVIRONMENT(self) -> str:
        return self.environment.name
    
    @property
    def DEVELOPMENT_MODE(self) -> bool:
        return self.environment.development_mode
    
    @property
    def DEBUG(self) -> bool:
        return self.environment.debug
    
    @property
    def LOG_LEVEL(self) -> str:
        return self.logging.level
    
    @property
    def API_KEY_HEADER(self) -> str:
        return self.auth.api_key_header
    
    @property
    def API_KEYS(self) -> List[str]:
        return self.auth.api_keys
    
    @property
    def SECRET_KEY(self) -> str:
        return self.auth.secret_key
    
    @property
    def SESSION_SECRET_KEY(self) -> str:
        return self.auth.session_secret_key
    
    @property
    def CORS_ORIGINS(self) -> List[str]:
        return self.cors.origins
    
    @property
    def RATE_LIMIT_PER_MINUTE(self) -> int:
        return self.api.rate_limit_per_minute
    
    @property
    def ENFORCE_HTTPS(self) -> bool:
        return self.api.enforce_https
    
    @property
    def RESTRICT_EXTERNAL_CALLS(self) -> bool:
        return self.api.restrict_external_calls
    
    @property
    def HANA_HOST(self) -> Optional[str]:
        return self.database.host
    
    @property
    def HANA_PORT(self) -> Optional[int]:
        return self.database.port
    
    @property
    def HANA_USER(self) -> Optional[str]:
        return self.database.user
    
    @property
    def HANA_PASSWORD(self) -> Optional[str]:
        return self.database.password
    
    @property
    def HANA_ENCRYPT(self) -> bool:
        return self.database.encrypt
    
    @property
    def HANA_SSL_VALIDATE_CERT(self) -> bool:
        return self.database.ssl_validate_cert
    
    @property
    def GPU_ENABLED(self) -> bool:
        return self.gpu.enabled
    
    @property
    def GPU_DEVICE(self) -> str:
        return self.gpu.device
    
    @property
    def GPU_BATCH_SIZE(self) -> int:
        return self.gpu.batch_size
    
    @property
    def GPU_EMBEDDING_MODEL(self) -> str:
        return self.gpu.embedding_model
    
    @property
    def USE_INTERNAL_EMBEDDINGS(self) -> bool:
        return self.gpu.use_internal_embeddings
    
    @property
    def INTERNAL_EMBEDDING_MODEL_ID(self) -> str:
        return self.gpu.internal_embedding_model_id
    
    @property
    def USE_TENSORRT(self) -> bool:
        return self.tensorrt.enabled
    
    @property
    def TENSORRT_PRECISION(self) -> str:
        return self.tensorrt.precision
    
    @property
    def TENSORRT_CACHE_DIR(self) -> str:
        return self.tensorrt.cache_dir
    
    @property
    def TENSORRT_DYNAMIC_SHAPES(self) -> bool:
        return self.tensorrt.dynamic_shapes
    
    @property
    def DEFAULT_TABLE_NAME(self) -> str:
        return self.vectorstore.table_name
    
    @property
    def DEFAULT_CONTENT_COLUMN(self) -> str:
        return self.vectorstore.content_column
    
    @property
    def DEFAULT_METADATA_COLUMN(self) -> str:
        return self.vectorstore.metadata_column
    
    @property
    def DEFAULT_VECTOR_COLUMN(self) -> str:
        return self.vectorstore.vector_column
    
    @property
    def VECTOR_COLUMN_TYPE(self) -> str:
        return self.vectorstore.vector_column_type
    
    @property
    def VECTOR_COLUMN_LENGTH(self) -> int:
        return self.vectorstore.vector_column_length
    
    @property
    def ARROW_FLIGHT_ENABLED(self) -> bool:
        return self.arrow_flight.enabled
    
    @property
    def ARROW_FLIGHT_HOST(self) -> str:
        return self.arrow_flight.host
    
    @property
    def ARROW_FLIGHT_PORT(self) -> int:
        return self.arrow_flight.port
    
    @property
    def ENABLE_ERROR_CONTEXT(self) -> bool:
        return self.features.enable_error_context
    
    @property
    def CACHE_VECTOR_REDUCTION(self) -> bool:
        return self.features.cache_vector_reduction
    
    @property
    def ENABLE_ADVANCED_CLUSTERING(self) -> bool:
        return self.features.enable_advanced_clustering
    
    @property
    def ENABLE_KNOWLEDGE_GRAPH(self) -> bool:
        return self.features.enable_knowledge_graph


# Cache settings instance
@lru_cache()
def get_standardized_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    
    # Configure logging
    logging_level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    logging.basicConfig(
        level=logging_level,
        format=settings.logging.format,
    )
    
    # Log warnings for insecure defaults
    logger = logging.getLogger(__name__)
    
    if "dev-key" in settings.auth.api_keys:
        logger.warning("Using default API key. For production, set a secure API_KEYS environment variable.")
    
    if settings.auth.secret_key == "supersecretkey":
        logger.warning("Using default SECRET_KEY. For production, set a secure SECRET_KEY environment variable.")
    
    if not all([settings.database.host, settings.database.port, settings.database.user, settings.database.password]):
        logger.warning("HANA database credentials not fully configured. Some features may be unavailable.")
    
    return settings


# Legacy compatibility function for backward compatibility
def get_database_config() -> Dict[str, Any]:
    """Get database configuration from settings."""
    settings = get_standardized_settings()
    return {
        "address": settings.database.host,
        "port": settings.database.port,
        "user": settings.database.user,
        "password": settings.database.password,
        "encrypt": settings.database.encrypt,
        "sslValidateCertificate": settings.database.ssl_validate_cert
    }


# Legacy compatibility function for backward compatibility
def get_vectorstore_config() -> Dict[str, Any]:
    """Get vector store configuration from settings."""
    settings = get_standardized_settings()
    return {
        "table_name": settings.vectorstore.table_name,
        "content_column": settings.vectorstore.content_column,
        "metadata_column": settings.vectorstore.metadata_column,
        "vector_column": settings.vectorstore.vector_column,
        "vector_column_type": settings.vectorstore.vector_column_type,
        "vector_column_length": settings.vectorstore.vector_column_length
    }


# Legacy compatibility function for backward compatibility
def get_gpu_config() -> Dict[str, Any]:
    """Get GPU configuration from settings."""
    settings = get_standardized_settings()
    return {
        "enabled": settings.gpu.enabled,
        "device": settings.gpu.device,
        "batch_size": settings.gpu.batch_size,
        "embedding_model": settings.gpu.embedding_model,
        "use_internal_embeddings": settings.gpu.use_internal_embeddings,
        "internal_embedding_model_id": settings.gpu.internal_embedding_model_id,
        "use_tensorrt": settings.tensorrt.enabled,
        "tensorrt_precision": settings.tensorrt.precision,
        "tensorrt_cache_dir": settings.tensorrt.cache_dir,
        "tensorrt_dynamic_shapes": settings.tensorrt.dynamic_shapes
    }