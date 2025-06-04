"""
Health check utilities for SAP HANA Cloud LangChain integration.

This module provides utilities for checking the health of various components,
including database connections, GPU resources, and platform-specific services.
"""

import logging
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pkg_resources

from langchain_hana.config.deployment import (
    DeploymentConfig,
    DeploymentPlatform,
    get_config,
)
from langchain_hana.gpu.utils import detect_gpu_capabilities

try:
    from hdbcli import dbapi
    HAS_HDBCLI = True
except ImportError:
    HAS_HDBCLI = False

logger = logging.getLogger(__name__)


class ComponentStatus(str, Enum):
    """Status of a component in the health check."""
    OK = "ok"
    WARNING = "warning"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health information for a specific component."""
    name: str
    status: ComponentStatus
    details: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    last_checked: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Initialize timestamp if not provided."""
        if self.last_checked is None:
            self.last_checked = datetime.utcnow().isoformat()


@dataclass
class SystemHealth:
    """Overall system health information."""
    status: ComponentStatus
    components: List[ComponentHealth]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    environment: str = field(default_factory=lambda: get_config().environment.value)
    platform: str = field(default_factory=lambda: get_config().backend_platform.value)
    version: str = field(default_factory=lambda: _get_package_version())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["components"] = [asdict(component) for component in self.components]
        return result


def _get_package_version() -> str:
    """Get the version of the langchain-hana package."""
    try:
        return pkg_resources.get_distribution("langchain-hana").version
    except pkg_resources.DistributionNotFound:
        return "unknown"


def check_database_health(
    connection: Optional[dbapi.Connection] = None
) -> ComponentHealth:
    """
    Check the health of the SAP HANA Cloud database connection.
    
    Args:
        connection: Optional existing database connection to use.
                   If not provided, a new connection will be created
                   using the configuration.
    
    Returns:
        ComponentHealth: Health information for the database component.
    """
    if not HAS_HDBCLI:
        return ComponentHealth(
            name="database",
            status=ComponentStatus.ERROR,
            message="hdbcli not installed",
        )
    
    connection_owned = False
    try:
        # Create a new connection if one wasn't provided
        if connection is None:
            config = get_config().hana_connection
            connection = dbapi.connect(
                address=config.host,
                port=config.port,
                user=config.user,
                password=config.password,
                encrypt=config.encrypt,
                sslValidateCertificate=config.ssl_validate_certificate,
            )
            connection_owned = True
        
        # Check if connection is alive
        cursor = connection.cursor()
        start_time = time.time()
        cursor.execute("SELECT * FROM DUMMY")
        response_time = time.time() - start_time
        
        # Get database version and other info
        cursor.execute("SELECT VERSION, CLOUD_VERSION FROM SYS.M_DATABASE")
        version_info = cursor.fetchone()
        
        # Get schema info
        cursor.execute("SELECT CURRENT_SCHEMA FROM DUMMY")
        schema_info = cursor.fetchone()
        
        # Close cursor
        cursor.close()
        
        # Return health information
        return ComponentHealth(
            name="database",
            status=ComponentStatus.OK,
            details={
                "version": version_info[0] if version_info else "unknown",
                "cloud_version": version_info[1] if version_info else "unknown",
                "schema": schema_info[0] if schema_info else "unknown",
                "response_time_ms": round(response_time * 1000, 2),
                "connected": True,
            },
            message="Database connection successful",
        )
    
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return ComponentHealth(
            name="database",
            status=ComponentStatus.ERROR,
            details={"error": str(e)},
            message="Database connection failed",
        )
    
    finally:
        # Close the connection if we created it
        if connection_owned and connection:
            connection.close()


def check_vectorstore_health(
    connection: Optional[dbapi.Connection] = None
) -> ComponentHealth:
    """
    Check the health of the vector store.
    
    Args:
        connection: Optional existing database connection to use.
    
    Returns:
        ComponentHealth: Health information for the vector store component.
    """
    if not HAS_HDBCLI:
        return ComponentHealth(
            name="vectorstore",
            status=ComponentStatus.ERROR,
            message="hdbcli not installed",
        )
    
    connection_owned = False
    try:
        # Create a new connection if one wasn't provided
        if connection is None:
            config = get_config().hana_connection
            connection = dbapi.connect(
                address=config.host,
                port=config.port,
                user=config.user,
                password=config.password,
                encrypt=config.encrypt,
                sslValidateCertificate=config.ssl_validate_certificate,
            )
            connection_owned = True
        
        # Get vector store configuration
        config = get_config().vectorstore_config
        table_name = config.table_name
        
        # Check if the vector table exists
        cursor = connection.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM SYS.TABLES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
            "AND TABLE_NAME = ?",
            (table_name,)
        )
        table_exists = cursor.fetchone()[0] > 0
        
        details = {"table_exists": table_exists, "table_name": table_name}
        
        if table_exists:
            # Get vector table statistics
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            row_count = cursor.fetchone()[0]
            details["row_count"] = row_count
            
            # Check for vector indexes
            cursor.execute(
                "SELECT COUNT(*) FROM SYS.INDEXES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
                "AND TABLE_NAME = ? AND INDEX_TYPE = 'HNSW'",
                (table_name,)
            )
            has_index = cursor.fetchone()[0] > 0
            details["has_hnsw_index"] = has_index
            
            if has_index:
                # Get index details
                cursor.execute(
                    "SELECT INDEX_NAME FROM SYS.INDEXES WHERE SCHEMA_NAME = CURRENT_SCHEMA "
                    "AND TABLE_NAME = ? AND INDEX_TYPE = 'HNSW'",
                    (table_name,)
                )
                index_names = [row[0] for row in cursor.fetchall()]
                details["index_names"] = index_names
        
        # Close cursor
        cursor.close()
        
        # Determine status
        status = ComponentStatus.OK if table_exists else ComponentStatus.WARNING
        message = "Vector table exists" if table_exists else "Vector table does not exist"
        
        # Return health information
        return ComponentHealth(
            name="vectorstore",
            status=status,
            details=details,
            message=message,
        )
    
    except Exception as e:
        logger.error(f"Vector store health check failed: {str(e)}")
        return ComponentHealth(
            name="vectorstore",
            status=ComponentStatus.ERROR,
            details={"error": str(e)},
            message="Vector store health check failed",
        )
    
    finally:
        # Close the connection if we created it
        if connection_owned and connection:
            connection.close()


def check_gpu_health() -> ComponentHealth:
    """
    Check the health of GPU resources.
    
    Returns:
        ComponentHealth: Health information for the GPU component.
    """
    try:
        # Get GPU configuration
        config = get_config().gpu_config
        
        # Check if GPU is enabled in configuration
        if not config.enabled:
            return ComponentHealth(
                name="gpu",
                status=ComponentStatus.OK,
                details={"enabled": False},
                message="GPU is disabled in configuration",
            )
        
        # Check GPU capabilities
        gpu_info = detect_gpu_capabilities()
        
        if not gpu_info["has_gpu"]:
            return ComponentHealth(
                name="gpu",
                status=ComponentStatus.WARNING,
                details={"enabled": True, "available": False},
                message="GPU is enabled in configuration but not available on this system",
            )
        
        # Return GPU health information
        return ComponentHealth(
            name="gpu",
            status=ComponentStatus.OK,
            details={
                "enabled": True,
                "available": True,
                "count": gpu_info["gpu_count"],
                "names": gpu_info["gpu_names"],
                "cuda_version": gpu_info["cuda_version"],
                "total_memory_mb": gpu_info["total_gpu_memory"],
                "compute_capabilities": gpu_info["compute_capabilities"],
                "tensorrt_enabled": config.enable_tensorrt,
            },
            message=f"GPU is available ({gpu_info['gpu_count']} devices)",
        )
    
    except Exception as e:
        logger.error(f"GPU health check failed: {str(e)}")
        return ComponentHealth(
            name="gpu",
            status=ComponentStatus.ERROR,
            details={"error": str(e)},
            message="GPU health check failed",
        )


def check_embedding_health() -> ComponentHealth:
    """
    Check the health of embedding resources.
    
    Returns:
        ComponentHealth: Health information for the embedding component.
    """
    try:
        # Get embedding configuration
        config = get_config().embedding_config
        
        details = {
            "model_name": config.model_name,
            "use_internal_embedding": config.use_internal_embedding,
        }
        
        if config.use_internal_embedding:
            details["internal_embedding_model_id"] = config.internal_embedding_model_id
        
        # Check if cache is enabled
        details["cache_enabled"] = config.cache_embeddings
        if config.cache_embeddings and config.embedding_cache_dir:
            details["cache_dir"] = config.embedding_cache_dir
        
        # Check if model exists (for external embeddings)
        if not config.use_internal_embedding:
            try:
                # Import conditionally to avoid unnecessary dependencies
                from sentence_transformers import SentenceTransformer
                
                # Try to load the model to check if it exists
                _ = SentenceTransformer(config.model_name)
                details["model_exists"] = True
                status = ComponentStatus.OK
                message = f"Embedding model '{config.model_name}' is available"
            
            except ImportError:
                details["model_exists"] = "unknown"
                details["sentence_transformers_installed"] = False
                status = ComponentStatus.WARNING
                message = "Cannot check embedding model: sentence-transformers not installed"
            
            except Exception as e:
                details["model_exists"] = False
                details["error"] = str(e)
                status = ComponentStatus.ERROR
                message = f"Embedding model '{config.model_name}' is not available"
        
        else:
            # For internal embeddings, we can't check model validity here
            # (would need a database connection)
            status = ComponentStatus.OK
            message = f"Using internal embedding model '{config.internal_embedding_model_id}'"
        
        # Return embedding health information
        return ComponentHealth(
            name="embedding",
            status=status,
            details=details,
            message=message,
        )
    
    except Exception as e:
        logger.error(f"Embedding health check failed: {str(e)}")
        return ComponentHealth(
            name="embedding",
            status=ComponentStatus.ERROR,
            details={"error": str(e)},
            message="Embedding health check failed",
        )


def check_platform_health() -> ComponentHealth:
    """
    Check the health of the deployment platform.
    
    Returns:
        ComponentHealth: Health information for the platform component.
    """
    try:
        # Get platform configuration
        config = get_config()
        platform_type = config.backend_platform
        
        # Common platform details
        details = {
            "platform": platform_type.value,
            "environment": config.environment.value,
            "system_info": {
                "python_version": sys.version,
                "os": platform.system(),
                "os_version": platform.version(),
                "architecture": platform.machine(),
            },
        }
        
        # Platform-specific checks
        if platform_type == DeploymentPlatform.TOGETHER_AI:
            platform_check = _check_together_ai_platform()
        elif platform_type == DeploymentPlatform.NVIDIA_LAUNCHPAD:
            platform_check = _check_nvidia_launchpad_platform()
        elif platform_type == DeploymentPlatform.SAP_BTP:
            platform_check = _check_sap_btp_platform()
        elif platform_type == DeploymentPlatform.VERCEL:
            platform_check = _check_vercel_platform()
        else:
            platform_check = (ComponentStatus.OK, "Local platform", {})
        
        # Combine platform-specific details
        status, message, platform_details = platform_check
        details.update(platform_details)
        
        # Return platform health information
        return ComponentHealth(
            name="platform",
            status=status,
            details=details,
            message=message,
        )
    
    except Exception as e:
        logger.error(f"Platform health check failed: {str(e)}")
        return ComponentHealth(
            name="platform",
            status=ComponentStatus.ERROR,
            details={"error": str(e)},
            message="Platform health check failed",
        )


def _check_together_ai_platform() -> Tuple[ComponentStatus, str, Dict[str, Any]]:
    """Check Together.ai platform-specific health."""
    settings = get_config().get_platform_specific_settings()
    
    # Check if API key is configured
    if not settings.get("together_api_key"):
        return (
            ComponentStatus.WARNING,
            "Together.ai API key not configured",
            {"api_key_configured": False},
        )
    
    # Check API connectivity
    try:
        import requests
        
        api_base = settings.get("together_api_base", "https://api.together.xyz/v1")
        headers = {"Authorization": f"Bearer {settings['together_api_key']}"}
        
        # Test API connectivity with a models request
        response = requests.get(f"{api_base}/models", headers=headers, timeout=5)
        
        if response.status_code == 200:
            models = response.json()
            return (
                ComponentStatus.OK,
                "Together.ai platform is available",
                {
                    "api_key_configured": True,
                    "api_connectivity": True,
                    "model_name": settings.get("together_model_name"),
                    "models_available": len(models.get("data", [])),
                },
            )
        else:
            return (
                ComponentStatus.ERROR,
                f"Together.ai API error: {response.status_code}",
                {
                    "api_key_configured": True,
                    "api_connectivity": False,
                    "error_code": response.status_code,
                    "error_message": response.text,
                },
            )
    
    except ImportError:
        return (
            ComponentStatus.WARNING,
            "Cannot check Together.ai connectivity: requests not installed",
            {"api_key_configured": True, "api_connectivity": "unknown"},
        )
    
    except Exception as e:
        return (
            ComponentStatus.ERROR,
            f"Together.ai connectivity error: {str(e)}",
            {"api_key_configured": True, "api_connectivity": False, "error": str(e)},
        )


def _check_nvidia_launchpad_platform() -> Tuple[ComponentStatus, str, Dict[str, Any]]:
    """Check NVIDIA LaunchPad platform-specific health."""
    settings = get_config().get_platform_specific_settings()
    
    # Check if Triton server is configured
    if not settings.get("triton_server_url"):
        return (
            ComponentStatus.WARNING,
            "Triton server URL not configured",
            {"triton_configured": False},
        )
    
    # Check if TensorRT is enabled
    tensorrt_enabled = settings.get("enable_tensorrt", False)
    
    # Check Triton connectivity
    try:
        import requests
        
        triton_url = settings.get("triton_server_url", "localhost:8000")
        if not triton_url.startswith(("http://", "https://")):
            triton_url = f"http://{triton_url}"
        
        # Test Triton connectivity with a health request
        health_url = f"{triton_url}/v2/health/ready"
        response = requests.get(health_url, timeout=5)
        
        if response.status_code == 200:
            # Get model status
            models_url = f"{triton_url}/v2/models"
            models_response = requests.get(models_url, timeout=5)
            
            if models_response.status_code == 200:
                models = models_response.json()
                model_names = [model.get("name") for model in models.get("models", [])]
                target_model = settings.get("triton_model_name")
                model_available = target_model in model_names
                
                return (
                    ComponentStatus.OK if model_available else ComponentStatus.WARNING,
                    "Triton server is available" if model_available else f"Model '{target_model}' not found on Triton server",
                    {
                        "triton_configured": True,
                        "triton_connectivity": True,
                        "tensorrt_enabled": tensorrt_enabled,
                        "model_name": target_model,
                        "model_available": model_available,
                        "available_models": model_names,
                    },
                )
            else:
                return (
                    ComponentStatus.WARNING,
                    f"Triton models API error: {models_response.status_code}",
                    {
                        "triton_configured": True,
                        "triton_connectivity": True,
                        "tensorrt_enabled": tensorrt_enabled,
                        "models_api_error": models_response.status_code,
                    },
                )
        else:
            return (
                ComponentStatus.ERROR,
                f"Triton server error: {response.status_code}",
                {
                    "triton_configured": True,
                    "triton_connectivity": False,
                    "tensorrt_enabled": tensorrt_enabled,
                    "error_code": response.status_code,
                    "error_message": response.text,
                },
            )
    
    except ImportError:
        return (
            ComponentStatus.WARNING,
            "Cannot check Triton connectivity: requests not installed",
            {
                "triton_configured": True,
                "triton_connectivity": "unknown",
                "tensorrt_enabled": tensorrt_enabled,
            },
        )
    
    except Exception as e:
        return (
            ComponentStatus.ERROR,
            f"Triton connectivity error: {str(e)}",
            {
                "triton_configured": True,
                "triton_connectivity": False,
                "tensorrt_enabled": tensorrt_enabled,
                "error": str(e),
            },
        )


def _check_sap_btp_platform() -> Tuple[ComponentStatus, str, Dict[str, Any]]:
    """Check SAP BTP platform-specific health."""
    settings = get_config().get_platform_specific_settings()
    
    # Check if CF API is configured
    if not settings.get("cf_api_url"):
        return (
            ComponentStatus.WARNING,
            "Cloud Foundry API URL not configured",
            {"cf_configured": False},
        )
    
    # Check environment variables specific to SAP BTP
    vcap_services = os.environ.get("VCAP_SERVICES")
    vcap_application = os.environ.get("VCAP_APPLICATION")
    
    if not vcap_services or not vcap_application:
        return (
            ComponentStatus.WARNING,
            "VCAP environment variables not found",
            {
                "cf_configured": True,
                "vcap_services": bool(vcap_services),
                "vcap_application": bool(vcap_application),
                "environment": "not detected",
            },
        )
    
    # Try to parse VCAP_APPLICATION to get app details
    try:
        import json
        
        app_info = json.loads(vcap_application)
        return (
            ComponentStatus.OK,
            "SAP BTP platform detected",
            {
                "cf_configured": True,
                "vcap_services": True,
                "vcap_application": True,
                "app_name": app_info.get("application_name"),
                "space_name": app_info.get("space_name"),
                "org_name": app_info.get("organization_name"),
                "app_uris": app_info.get("application_uris", []),
                "use_destination_service": settings.get("use_destination_service", False),
            },
        )
    
    except Exception as e:
        return (
            ComponentStatus.WARNING,
            f"Error parsing VCAP environment: {str(e)}",
            {
                "cf_configured": True,
                "vcap_services": True,
                "vcap_application": True,
                "error": str(e),
            },
        )


def _check_vercel_platform() -> Tuple[ComponentStatus, str, Dict[str, Any]]:
    """Check Vercel platform-specific health."""
    settings = get_config().get_platform_specific_settings()
    
    # Check if running on Vercel
    vercel_env = os.environ.get("VERCEL")
    vercel_region = os.environ.get("VERCEL_REGION") or settings.get("vercel_region")
    
    if not vercel_env:
        return (
            ComponentStatus.WARNING,
            "Not running on Vercel",
            {"vercel_detected": False},
        )
    
    # Check Vercel-specific features
    edge_config = settings.get("use_vercel_edge", False)
    kv_storage = settings.get("use_vercel_kv", False)
    cron_enabled = settings.get("vercel_cron_enabled", False)
    
    # Get Vercel environment details
    vercel_env_type = os.environ.get("VERCEL_ENV", "development")
    
    return (
        ComponentStatus.OK,
        "Vercel platform detected",
        {
            "vercel_detected": True,
            "vercel_region": vercel_region,
            "vercel_environment": vercel_env_type,
            "edge_config_enabled": edge_config,
            "kv_storage_enabled": kv_storage,
            "cron_enabled": cron_enabled,
            "build_id": os.environ.get("VERCEL_GIT_COMMIT_SHA"),
        },
    )


def get_system_health(
    connection: Optional[dbapi.Connection] = None
) -> SystemHealth:
    """
    Get comprehensive health information for all system components.
    
    Args:
        connection: Optional existing database connection to use.
    
    Returns:
        SystemHealth: Comprehensive health information for the system.
    """
    # Check all components
    database_health = check_database_health(connection)
    vectorstore_health = check_vectorstore_health(connection)
    gpu_health = check_gpu_health()
    embedding_health = check_embedding_health()
    platform_health = check_platform_health()
    
    # Collect all components
    components = [
        database_health,
        vectorstore_health,
        gpu_health,
        embedding_health,
        platform_health,
    ]
    
    # Determine overall status
    if any(c.status == ComponentStatus.ERROR for c in components):
        status = ComponentStatus.ERROR
    elif any(c.status == ComponentStatus.WARNING for c in components):
        status = ComponentStatus.WARNING
    else:
        status = ComponentStatus.OK
    
    # Return system health
    return SystemHealth(
        status=status,
        components=components,
    )