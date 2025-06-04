"""
Deployment configuration manager for SAP HANA Cloud LangChain integration.

This module provides utilities for managing configuration across different
deployment environments and platforms, including:

1. Together.ai deployment
2. NVIDIA LaunchPad deployment
3. SAP BTP deployment
4. Vercel deployment

The configuration manager handles environment-specific settings, connection
parameters, and platform-specific optimizations.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class DeploymentPlatform(str, Enum):
    """Supported deployment platforms for the backend."""
    TOGETHER_AI = "together_ai"
    NVIDIA_LAUNCHPAD = "nvidia_launchpad"
    SAP_BTP = "sap_btp"
    VERCEL = "vercel"
    LOCAL = "local"


class FrontendPlatform(str, Enum):
    """Supported deployment platforms for the frontend."""
    SAP_BTP = "sap_btp"
    VERCEL = "vercel"
    LOCAL = "local"


class Environment(str, Enum):
    """Deployment environments."""
    DEVELOPMENT = "dev"
    TESTING = "test"
    STAGING = "staging"
    PRODUCTION = "prod"


@dataclass
class HanaConnectionConfig:
    """SAP HANA Cloud database connection configuration."""
    host: str
    port: int
    user: str
    password: str
    encrypt: bool = True
    ssl_validate_certificate: bool = True
    autocommit: bool = True
    max_connections: int = 10
    connect_timeout: int = 30
    connection_pooling: bool = True
    
    @classmethod
    def from_env(cls) -> "HanaConnectionConfig":
        """Create a connection configuration from environment variables."""
        return cls(
            host=os.environ.get("HANA_HOST", ""),
            port=int(os.environ.get("HANA_PORT", "443")),
            user=os.environ.get("HANA_USER", ""),
            password=os.environ.get("HANA_PASSWORD", ""),
            encrypt=os.environ.get("HANA_ENCRYPT", "true").lower() == "true",
            ssl_validate_certificate=os.environ.get("HANA_SSL_VALIDATE", "true").lower() == "true",
            autocommit=os.environ.get("HANA_AUTOCOMMIT", "true").lower() == "true",
            max_connections=int(os.environ.get("HANA_MAX_CONNECTIONS", "10")),
            connect_timeout=int(os.environ.get("HANA_CONNECT_TIMEOUT", "30")),
            connection_pooling=os.environ.get("HANA_CONNECTION_POOLING", "true").lower() == "true",
        )


@dataclass
class GPUConfig:
    """GPU configuration for embedding generation and vector operations."""
    enabled: bool = False
    max_batch_size: int = 32
    memory_threshold: float = 20.0  # Percentage of GPU memory that must be free
    precision: str = "fp16"  # "fp32", "fp16", or "int8"
    enable_tensorrt: bool = False
    tensorrt_cache_dir: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "GPUConfig":
        """Create a GPU configuration from environment variables."""
        return cls(
            enabled=os.environ.get("GPU_ENABLED", "false").lower() == "true",
            max_batch_size=int(os.environ.get("GPU_MAX_BATCH_SIZE", "32")),
            memory_threshold=float(os.environ.get("GPU_MEMORY_THRESHOLD", "20.0")),
            precision=os.environ.get("GPU_PRECISION", "fp16"),
            enable_tensorrt=os.environ.get("GPU_ENABLE_TENSORRT", "false").lower() == "true",
            tensorrt_cache_dir=os.environ.get("GPU_TENSORRT_CACHE_DIR", None),
        )


@dataclass
class EmbeddingConfig:
    """Embedding configuration for vector operations."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_internal_embedding: bool = False
    internal_embedding_model_id: str = "SAP_NEB.20240715"
    cache_embeddings: bool = True
    embedding_cache_dir: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create an embedding configuration from environment variables."""
        return cls(
            model_name=os.environ.get(
                "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            use_internal_embedding=os.environ.get(
                "USE_INTERNAL_EMBEDDING", "false"
            ).lower() == "true",
            internal_embedding_model_id=os.environ.get(
                "INTERNAL_EMBEDDING_MODEL_ID", "SAP_NEB.20240715"
            ),
            cache_embeddings=os.environ.get("CACHE_EMBEDDINGS", "true").lower() == "true",
            embedding_cache_dir=os.environ.get("EMBEDDING_CACHE_DIR", None),
        )


@dataclass
class VectorstoreConfig:
    """Vector store configuration."""
    table_name: str = "EMBEDDINGS"
    content_column: str = "VEC_TEXT"
    metadata_column: str = "VEC_META"
    vector_column: str = "VEC_VECTOR"
    vector_column_type: str = "REAL_VECTOR"
    create_hnsw_index: bool = True
    distance_strategy: str = "COSINE"
    
    @classmethod
    def from_env(cls) -> "VectorstoreConfig":
        """Create a vectorstore configuration from environment variables."""
        return cls(
            table_name=os.environ.get("VECTORSTORE_TABLE_NAME", "EMBEDDINGS"),
            content_column=os.environ.get("VECTORSTORE_CONTENT_COLUMN", "VEC_TEXT"),
            metadata_column=os.environ.get("VECTORSTORE_METADATA_COLUMN", "VEC_META"),
            vector_column=os.environ.get("VECTORSTORE_VECTOR_COLUMN", "VEC_VECTOR"),
            vector_column_type=os.environ.get("VECTORSTORE_VECTOR_COLUMN_TYPE", "REAL_VECTOR"),
            create_hnsw_index=os.environ.get("VECTORSTORE_CREATE_HNSW_INDEX", "true").lower() == "true",
            distance_strategy=os.environ.get("VECTORSTORE_DISTANCE_STRATEGY", "COSINE"),
        )


@dataclass
class DeploymentConfig:
    """Main deployment configuration."""
    # Core settings
    app_name: str = "langchain-hana-integration"
    environment: Environment = Environment.DEVELOPMENT
    backend_platform: DeploymentPlatform = DeploymentPlatform.LOCAL
    frontend_platform: FrontendPlatform = FrontendPlatform.LOCAL
    log_level: str = "INFO"
    
    # Component configurations
    hana_connection: HanaConnectionConfig = field(default_factory=HanaConnectionConfig.from_env)
    gpu_config: GPUConfig = field(default_factory=GPUConfig.from_env)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig.from_env)
    vectorstore_config: VectorstoreConfig = field(default_factory=VectorstoreConfig.from_env)
    
    # Platform-specific configurations
    api_base_url: Optional[str] = None
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=list)
    request_timeout: int = 60
    
    @classmethod
    def from_env(cls) -> "DeploymentConfig":
        """Create a deployment configuration from environment variables."""
        # Load base configuration
        config = cls(
            app_name=os.environ.get("APP_NAME", "langchain-hana-integration"),
            environment=Environment(os.environ.get("ENVIRONMENT", Environment.DEVELOPMENT)),
            backend_platform=DeploymentPlatform(
                os.environ.get("BACKEND_PLATFORM", DeploymentPlatform.LOCAL)
            ),
            frontend_platform=FrontendPlatform(
                os.environ.get("FRONTEND_PLATFORM", FrontendPlatform.LOCAL)
            ),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
            api_base_url=os.environ.get("API_BASE_URL", None),
            enable_cors=os.environ.get("ENABLE_CORS", "true").lower() == "true",
            request_timeout=int(os.environ.get("REQUEST_TIMEOUT", "60")),
        )
        
        # Parse CORS origins
        cors_origins_str = os.environ.get("CORS_ORIGINS", "")
        if cors_origins_str:
            config.cors_origins = [origin.strip() for origin in cors_origins_str.split(",")]
        
        return config
    
    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "DeploymentConfig":
        """Load configuration from a JSON or environment file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load the configuration file
        if file_path.suffix.lower() in [".json", ".jsonc"]:
            with open(file_path, "r") as f:
                config_data = json.load(f)
            return cls._from_dict(config_data)
        elif file_path.suffix.lower() in [".env"]:
            # Load .env file
            from dotenv import load_dotenv
            load_dotenv(file_path)
            return cls.from_env()
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "DeploymentConfig":
        """Create a configuration from a dictionary."""
        # Extract nested configurations
        hana_dict = config_dict.pop("hana_connection", {})
        gpu_dict = config_dict.pop("gpu_config", {})
        embedding_dict = config_dict.pop("embedding_config", {})
        vectorstore_dict = config_dict.pop("vectorstore_config", {})
        
        # Create the configuration object
        config = cls(**config_dict)
        
        # Create nested configurations
        if hana_dict:
            config.hana_connection = HanaConnectionConfig(**hana_dict)
        if gpu_dict:
            config.gpu_config = GPUConfig(**gpu_dict)
        if embedding_dict:
            config.embedding_config = EmbeddingConfig(**embedding_dict)
        if vectorstore_dict:
            config.vectorstore_config = VectorstoreConfig(**vectorstore_dict)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        config_dict = asdict(self)
        # Convert enum values to strings
        config_dict["environment"] = self.environment.value
        config_dict["backend_platform"] = self.backend_platform.value
        config_dict["frontend_platform"] = self.frontend_platform.value
        return config_dict
    
    def save(self, file_path: Union[str, Path]) -> None:
        """Save the configuration to a file."""
        file_path = Path(file_path)
        os.makedirs(file_path.parent, exist_ok=True)
        
        if file_path.suffix.lower() in [".json", ".jsonc"]:
            with open(file_path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
        elif file_path.suffix.lower() in [".env"]:
            with open(file_path, "w") as f:
                for key, value in self._flatten_config().items():
                    f.write(f"{key}={value}\n")
        else:
            raise ValueError(f"Unsupported configuration file format: {file_path.suffix}")
    
    def _flatten_config(self) -> Dict[str, str]:
        """Flatten the configuration to a dictionary of environment variables."""
        flattened = {}
        
        # Add main configuration
        flattened["APP_NAME"] = self.app_name
        flattened["ENVIRONMENT"] = self.environment.value
        flattened["BACKEND_PLATFORM"] = self.backend_platform.value
        flattened["FRONTEND_PLATFORM"] = self.frontend_platform.value
        flattened["LOG_LEVEL"] = self.log_level
        
        if self.api_base_url:
            flattened["API_BASE_URL"] = self.api_base_url
        
        flattened["ENABLE_CORS"] = str(self.enable_cors).lower()
        flattened["CORS_ORIGINS"] = ",".join(self.cors_origins)
        flattened["REQUEST_TIMEOUT"] = str(self.request_timeout)
        
        # Add HANA connection configuration
        flattened["HANA_HOST"] = self.hana_connection.host
        flattened["HANA_PORT"] = str(self.hana_connection.port)
        flattened["HANA_USER"] = self.hana_connection.user
        flattened["HANA_PASSWORD"] = self.hana_connection.password
        flattened["HANA_ENCRYPT"] = str(self.hana_connection.encrypt).lower()
        flattened["HANA_SSL_VALIDATE"] = str(self.hana_connection.ssl_validate_certificate).lower()
        flattened["HANA_AUTOCOMMIT"] = str(self.hana_connection.autocommit).lower()
        flattened["HANA_MAX_CONNECTIONS"] = str(self.hana_connection.max_connections)
        flattened["HANA_CONNECT_TIMEOUT"] = str(self.hana_connection.connect_timeout)
        flattened["HANA_CONNECTION_POOLING"] = str(self.hana_connection.connection_pooling).lower()
        
        # Add GPU configuration
        flattened["GPU_ENABLED"] = str(self.gpu_config.enabled).lower()
        flattened["GPU_MAX_BATCH_SIZE"] = str(self.gpu_config.max_batch_size)
        flattened["GPU_MEMORY_THRESHOLD"] = str(self.gpu_config.memory_threshold)
        flattened["GPU_PRECISION"] = self.gpu_config.precision
        flattened["GPU_ENABLE_TENSORRT"] = str(self.gpu_config.enable_tensorrt).lower()
        
        if self.gpu_config.tensorrt_cache_dir:
            flattened["GPU_TENSORRT_CACHE_DIR"] = self.gpu_config.tensorrt_cache_dir
        
        # Add embedding configuration
        flattened["EMBEDDING_MODEL_NAME"] = self.embedding_config.model_name
        flattened["USE_INTERNAL_EMBEDDING"] = str(self.embedding_config.use_internal_embedding).lower()
        flattened["INTERNAL_EMBEDDING_MODEL_ID"] = self.embedding_config.internal_embedding_model_id
        flattened["CACHE_EMBEDDINGS"] = str(self.embedding_config.cache_embeddings).lower()
        
        if self.embedding_config.embedding_cache_dir:
            flattened["EMBEDDING_CACHE_DIR"] = self.embedding_config.embedding_cache_dir
        
        # Add vectorstore configuration
        flattened["VECTORSTORE_TABLE_NAME"] = self.vectorstore_config.table_name
        flattened["VECTORSTORE_CONTENT_COLUMN"] = self.vectorstore_config.content_column
        flattened["VECTORSTORE_METADATA_COLUMN"] = self.vectorstore_config.metadata_column
        flattened["VECTORSTORE_VECTOR_COLUMN"] = self.vectorstore_config.vector_column
        flattened["VECTORSTORE_VECTOR_COLUMN_TYPE"] = self.vectorstore_config.vector_column_type
        flattened["VECTORSTORE_CREATE_HNSW_INDEX"] = str(self.vectorstore_config.create_hnsw_index).lower()
        flattened["VECTORSTORE_DISTANCE_STRATEGY"] = self.vectorstore_config.distance_strategy
        
        return flattened
    
    def get_platform_specific_settings(self) -> Dict[str, Any]:
        """Get platform-specific settings based on the current deployment platform."""
        if self.backend_platform == DeploymentPlatform.TOGETHER_AI:
            return self._get_together_ai_settings()
        elif self.backend_platform == DeploymentPlatform.NVIDIA_LAUNCHPAD:
            return self._get_nvidia_launchpad_settings()
        elif self.backend_platform == DeploymentPlatform.SAP_BTP:
            return self._get_sap_btp_settings()
        elif self.backend_platform == DeploymentPlatform.VERCEL:
            return self._get_vercel_settings()
        else:
            return {}
    
    def _get_together_ai_settings(self) -> Dict[str, Any]:
        """Get Together.ai-specific settings."""
        return {
            "together_api_key": os.environ.get("TOGETHER_API_KEY", ""),
            "together_model_name": os.environ.get("TOGETHER_MODEL_NAME", "togethercomputer/llama-2-7b"),
            "together_api_base": os.environ.get("TOGETHER_API_BASE", "https://api.together.xyz/v1"),
            "use_together_embeddings": os.environ.get("USE_TOGETHER_EMBEDDINGS", "false").lower() == "true",
        }
    
    def _get_nvidia_launchpad_settings(self) -> Dict[str, Any]:
        """Get NVIDIA LaunchPad-specific settings."""
        return {
            "triton_server_url": os.environ.get("TRITON_SERVER_URL", "localhost:8000"),
            "use_triton_inference": os.environ.get("USE_TRITON_INFERENCE", "true").lower() == "true",
            "triton_model_name": os.environ.get("TRITON_MODEL_NAME", "all-MiniLM-L6-v2"),
            "triton_timeout": int(os.environ.get("TRITON_TIMEOUT", "60")),
            "enable_tensorrt": os.environ.get("ENABLE_TENSORRT", "true").lower() == "true",
        }
    
    def _get_sap_btp_settings(self) -> Dict[str, Any]:
        """Get SAP BTP-specific settings."""
        return {
            "cf_api_url": os.environ.get("CF_API_URL", ""),
            "cf_org": os.environ.get("CF_ORG", ""),
            "cf_space": os.environ.get("CF_SPACE", ""),
            "xsuaa_url": os.environ.get("XSUAA_URL", ""),
            "destination_service": os.environ.get("DESTINATION_SERVICE", ""),
            "use_destination_service": os.environ.get("USE_DESTINATION_SERVICE", "false").lower() == "true",
        }
    
    def _get_vercel_settings(self) -> Dict[str, Any]:
        """Get Vercel-specific settings."""
        return {
            "vercel_region": os.environ.get("VERCEL_REGION", ""),
            "vercel_edge_config": os.environ.get("VERCEL_EDGE_CONFIG", ""),
            "use_vercel_edge": os.environ.get("USE_VERCEL_EDGE", "false").lower() == "true",
            "use_vercel_kv": os.environ.get("USE_VERCEL_KV", "false").lower() == "true",
            "vercel_cron_enabled": os.environ.get("VERCEL_CRON_ENABLED", "false").lower() == "true",
        }


# Global configuration singleton
_global_config: Optional[DeploymentConfig] = None


def get_config() -> DeploymentConfig:
    """Get the global configuration singleton."""
    global _global_config
    if _global_config is None:
        _global_config = DeploymentConfig.from_env()
    return _global_config


def set_config(config: DeploymentConfig) -> None:
    """Set the global configuration singleton."""
    global _global_config
    _global_config = config


def load_config_from_file(file_path: Union[str, Path]) -> DeploymentConfig:
    """Load configuration from a file and set it as the global configuration."""
    config = DeploymentConfig.from_file(file_path)
    set_config(config)
    return config


def create_platform_configs() -> Dict[str, Dict[str, str]]:
    """
    Create platform-specific configuration templates for all supported platforms.
    
    Returns:
        Dictionary with platform configurations where each key is a platform name
        and each value is a dictionary of environment variables for that platform.
    """
    result = {}
    
    # Create base configuration
    base_config = DeploymentConfig()
    
    # Create Together.ai configuration
    together_config = DeploymentConfig(
        backend_platform=DeploymentPlatform.TOGETHER_AI,
        frontend_platform=FrontendPlatform.VERCEL,
        environment=Environment.PRODUCTION,
        gpu_config=GPUConfig(enabled=False),
        embedding_config=EmbeddingConfig(model_name="togethercomputer/m2-bert-80M-8k-retrieval"),
    )
    result["together_ai"] = together_config._flatten_config()
    
    # Create NVIDIA LaunchPad configuration
    nvidia_config = DeploymentConfig(
        backend_platform=DeploymentPlatform.NVIDIA_LAUNCHPAD,
        frontend_platform=FrontendPlatform.VERCEL,
        environment=Environment.PRODUCTION,
        gpu_config=GPUConfig(
            enabled=True,
            max_batch_size=64,
            memory_threshold=10.0,
            precision="fp16",
            enable_tensorrt=True,
        ),
    )
    result["nvidia_launchpad"] = nvidia_config._flatten_config()
    
    # Create SAP BTP configuration
    sap_btp_config = DeploymentConfig(
        backend_platform=DeploymentPlatform.SAP_BTP,
        frontend_platform=FrontendPlatform.SAP_BTP,
        environment=Environment.PRODUCTION,
        gpu_config=GPUConfig(enabled=True),
    )
    result["sap_btp"] = sap_btp_config._flatten_config()
    
    # Create Vercel configuration
    vercel_config = DeploymentConfig(
        backend_platform=DeploymentPlatform.VERCEL,
        frontend_platform=FrontendPlatform.VERCEL,
        environment=Environment.PRODUCTION,
        gpu_config=GPUConfig(enabled=False),
    )
    result["vercel"] = vercel_config._flatten_config()
    
    return result