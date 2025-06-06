"""
Health check API endpoints for monitoring system status.
"""

import os
import platform
import sys
import time
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from config import config, get_settings
from version import VERSION, get_version_info

# Import conditionally to handle missing dependencies
try:
    from langchain_hana.monitoring.health import get_system_health
    HAS_HEALTH_MODULE = True
except ImportError:
    HAS_HEALTH_MODULE = False

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)


class HealthStatus(BaseModel):
    """Basic health status response model."""
    status: str
    timestamp: str
    environment: str
    version: str
    backend: str


class ComponentHealth(BaseModel):
    """Component health information."""
    name: str
    status: str
    details: Dict[str, Any]
    message: Optional[str] = None
    last_checked: Optional[str] = None


class DetailedHealthStatus(HealthStatus):
    """Detailed health status response with component information."""
    components: List[ComponentHealth]


@router.get(
    "/ping",
    summary="Simple health check",
    description="Lightweight health check endpoint that returns 200 OK if the API is running.",
    response_description="Basic health status information including version and environment",
)
async def ping():
    """
    Simple health check that returns a 200 OK response if the API is running.
    
    This is a lightweight endpoint suitable for Kubernetes liveness probes
    or load balancer health checks. It doesn't perform any database or 
    external service validation.
    
    Returns:
        dict: Status information including version and platform details
    """
    settings = get_settings()
    # Get environment variables or config values with appropriate fallbacks
    platform_env = os.environ.get("PLATFORM", "unknown")
    
    # Use version information from version module
    version_info = get_version_info()
    
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.api.environment,
        "version": VERSION,
        "version_details": version_info,
        "backend": platform_env,
    }


@router.get(
    "/complete",
    summary="Complete health check",
    description="Comprehensive health check that validates all system components including database, GPU, and external services.",
    response_model=DetailedHealthStatus,
    responses={
        500: {
            "description": "Server error during health check",
            "content": {
                "application/json": {
                    "example": {
                        "status": "error",
                        "timestamp": "2023-06-05T12:34:56.789012",
                        "message": "Error performing health check",
                        "details": {
                            "error": "Database connection failed"
                        }
                    }
                }
            }
        }
    }
)
async def complete_health():
    """
    Comprehensive health check with all system components.
    
    This endpoint performs a thorough validation of all system components including:
    - API server status
    - Database connection
    - GPU availability and status
    - External services connectivity
    - Memory and resource usage
    
    It's suitable for deep system diagnostics but may be resource-intensive.
    Use the '/ping' endpoint for lightweight monitoring.
    
    Returns:
        DetailedHealthStatus: Detailed health information for all components
    """
    settings = get_settings()
    start_time = time.time()
    
    # If health module is available, use it to get comprehensive health info
    if HAS_HEALTH_MODULE:
        try:
            system_health = get_system_health()
            return DetailedHealthStatus(
                status=system_health.status,
                timestamp=system_health.timestamp,
                environment=system_health.environment,
                version=settings.version,
                backend=settings.backend_platform,
                components=[
                    ComponentHealth(
                        name=component.name,
                        status=component.status,
                        details=component.details,
                        message=component.message,
                        last_checked=component.last_checked,
                    )
                    for component in system_health.components
                ],
            )
        except Exception as e:
            # Fall back to basic component checks if error occurs
            return _fallback_health_check(settings, str(e), start_time)
    else:
        # Fall back to basic component checks
        return _fallback_health_check(settings, "Health module not available", start_time)


def _fallback_health_check(settings, error_message, start_time):
    """Fallback health check with basic component information."""
    # Get environment variables
    platform_env = os.environ.get("PLATFORM", "unknown")
    
    # Get version information from version module
    version_info = get_version_info()
    
    # Basic system info
    components = [
        ComponentHealth(
            name="api",
            status="ok",
            details={
                "response_time_ms": round((time.time() - start_time) * 1000, 2),
                "python_version": sys.version,
                "platform": platform.platform(),
            },
            message="API is running",
        ),
        ComponentHealth(
            name="system",
            status="ok",
            details={
                "hostname": platform.node(),
                "system": platform.system(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
            },
            message="System information",
        ),
    ]
    
    # Add database component (basic check)
    try:
        # Basic database check (not actually connecting)
        has_db_config = all([
            settings.db.host,
            settings.db.port,
            settings.db.user,
            settings.db.password,
        ])
        
        db_status = "warning" if has_db_config else "error"
        db_message = "Database configuration present but not validated" if has_db_config else "Database configuration missing"
        
        components.append(
            ComponentHealth(
                name="database",
                status=db_status,
                details={
                    "host": settings.db.host or "not set",
                    "port": settings.db.port or "not set",
                    "user": bool(settings.db.user),
                    "connected": False,
                },
                message=db_message,
            )
        )
    except Exception as e:
        components.append(
            ComponentHealth(
                name="database",
                status="error",
                details={"error": str(e)},
                message="Error checking database configuration",
            )
        )
    
    # Add backend-specific component
    if platform_env == "together_ai":
        components.append(
            ComponentHealth(
                name="together_ai",
                status="ok" if os.environ.get("TOGETHER_API_KEY") else "error",
                details={
                    "model": os.environ.get("TOGETHER_MODEL_NAME", "not set"),
                    "api_configured": bool(os.environ.get("TOGETHER_API_KEY")),
                },
                message="Together.ai configuration present" if os.environ.get("TOGETHER_API_KEY") else "Together.ai API key missing",
            )
        )
    elif platform_env == "nvidia_launchpad":
        components.append(
            ComponentHealth(
                name="nvidia_launchpad",
                status="ok",
                details={
                    "triton_server": os.environ.get("TRITON_SERVER_URL", "not set"),
                    "enable_tensorrt": os.environ.get("USE_TENSORRT", "true").lower() == "true",
                },
                message="NVIDIA LaunchPad configuration",
            )
        )
    
    # Determine overall status
    statuses = [component.status for component in components]
    overall_status = "error" if "error" in statuses else "warning" if "warning" in statuses else "ok"
    
    return DetailedHealthStatus(
        status=overall_status,
        timestamp=datetime.utcnow().isoformat(),
        environment=settings.api.environment,
        version=VERSION,
        backend=platform_env,
        components=components,
    )


@router.get(
    "/database", 
    summary="Database health check",
    description="Check the health and connectivity of the SAP HANA Cloud database connection",
    responses={
        200: {
            "description": "Database health information",
            "content": {
                "application/json": {
                    "examples": {
                        "connected": {
                            "value": {
                                "status": "ok",
                                "timestamp": "2023-06-05T12:34:56.789012",
                                "details": {
                                    "host": "hana-host.hanacloud.ondemand.com",
                                    "port": "443",
                                    "user": True,
                                    "connected": True,
                                    "connection_pool": {
                                        "active": 1,
                                        "idle": 2,
                                        "max": 5
                                    },
                                    "latency_ms": 15.7
                                },
                                "message": "Database connection successful"
                            }
                        },
                        "disconnected": {
                            "value": {
                                "status": "error",
                                "timestamp": "2023-06-05T12:34:56.789012",
                                "details": {
                                    "host": "hana-host.hanacloud.ondemand.com",
                                    "port": "443",
                                    "user": True,
                                    "connected": False,
                                    "error": "Connection refused"
                                },
                                "message": "Failed to connect to database"
                            }
                        }
                    }
                }
            }
        }
    }
)
async def database_health():
    """
    Check the health of the database connection.
    
    This endpoint attempts to connect to the SAP HANA Cloud database and verify the connection.
    It provides information about:
    - Connection status
    - Connection pool statistics
    - Connection latency
    - Database version (if connected)
    
    Returns:
        dict: Database status information and connectivity details
    """
    settings = get_settings()
    
    # If health module is available, use it for detailed database health
    if HAS_HEALTH_MODULE:
        try:
            system_health = get_system_health()
            for component in system_health.components:
                if component.name == "database":
                    return {
                        "status": component.status,
                        "timestamp": datetime.utcnow().isoformat(),
                        "details": component.details,
                        "message": component.message,
                    }
        except Exception:
            pass  # Fall back to basic check
    
    # Basic database check (not actually connecting)
    has_db_config = all([
        settings.db.host,
        settings.db.port,
        settings.db.user,
        settings.db.password,
    ])
    
    return {
        "status": "warning" if has_db_config else "error",
        "timestamp": datetime.utcnow().isoformat(),
        "details": {
            "host": settings.db.host or "not set",
            "port": settings.db.port or "not set",
            "user": bool(settings.db.user),
            "connected": False,
        },
        "message": "Database configuration present but not validated" if has_db_config else "Database configuration missing",
    }


@router.get(
    "/metrics",
    summary="API metrics",
    description="Get API usage and performance metrics in Prometheus format",
    response_description="Prometheus-formatted metrics data",
    responses={
        200: {
            "description": "Prometheus metrics data",
            "content": {
                "text/plain": {
                    "example": """
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{endpoint="/query"} 125
api_requests_total{endpoint="/texts"} 84
api_requests_total{endpoint="/health/ping"} 532
# HELP api_request_duration_seconds Request duration in seconds
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{endpoint="/query",le="0.1"} 42
api_request_duration_seconds_bucket{endpoint="/query",le="0.5"} 78
api_request_duration_seconds_bucket{endpoint="/query",le="1.0"} 98
api_request_duration_seconds_bucket{endpoint="/query",le="+Inf"} 125
# HELP gpu_memory_usage_bytes GPU memory usage in bytes
# TYPE gpu_memory_usage_bytes gauge
gpu_memory_usage_bytes{device="0"} 2147483648
# HELP embedding_generation_duration_seconds Embedding generation duration in seconds
# TYPE embedding_generation_duration_seconds histogram
embedding_generation_duration_seconds_bucket{model="all-MiniLM-L6-v2",le="0.1"} 125
embedding_generation_duration_seconds_bucket{model="all-MiniLM-L6-v2",le="0.5"} 356
embedding_generation_duration_seconds_bucket{model="all-MiniLM-L6-v2",le="1.0"} 478
embedding_generation_duration_seconds_bucket{model="all-MiniLM-L6-v2",le="+Inf"} 502
                    """
                }
            }
        }
    }
)
async def metrics():
    """
    Get API usage and performance metrics in Prometheus format.
    
    This endpoint provides metrics for:
    - Request counts by endpoint
    - Request duration by endpoint
    - GPU memory usage
    - Embedding generation performance
    - Database connection statistics
    - Error counts by type
    
    The metrics are in Prometheus text format for easy integration with
    monitoring systems like Prometheus and Grafana.
    
    Returns:
        str: Prometheus-formatted metrics data
    """
    try:
        # Import the metrics module - we'll import it here to avoid circular imports
        from langchain_hana.monitoring.metrics import get_prometheus_metrics
        metrics_data = get_prometheus_metrics()
        return metrics_data
    except ImportError:
        # If monitoring module isn't available, provide basic metrics
        prometheus_metrics = []
        
        # API info metric
        prometheus_metrics.append("# HELP api_info API information")
        prometheus_metrics.append("# TYPE api_info gauge")
        prometheus_metrics.append(f'api_info{{version="{VERSION}"}} 1')
        
        # System uptime
        uptime = time.time() - config.api.start_time if hasattr(config.api, 'start_time') else 0
        prometheus_metrics.append("# HELP api_uptime_seconds API uptime in seconds")
        prometheus_metrics.append("# TYPE api_uptime_seconds gauge")
        prometheus_metrics.append(f'api_uptime_seconds {uptime}')
        
        # Basic system metrics
        prometheus_metrics.append("# HELP system_info System information")
        prometheus_metrics.append("# TYPE system_info gauge")
        prometheus_metrics.append(f'system_info{{platform="{platform.system()}"}} 1')
        
        # Return as plain text
        return "\n".join(prometheus_metrics)