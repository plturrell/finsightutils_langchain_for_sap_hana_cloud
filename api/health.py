"""
Health check API endpoints for monitoring system status.
"""

import os
import platform
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from config import get_settings

# Import conditionally to handle missing dependencies
try:
    from langchain_hana.monitoring.health import get_system_health
    HAS_HEALTH_MODULE = True
except ImportError:
    HAS_HEALTH_MODULE = False

router = APIRouter(prefix="/health", tags=["health"])


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


@router.get("/ping", summary="Simple health check")
async def ping():
    """
    Simple health check that returns a 200 OK response if the API is running.
    
    Returns:
        dict: Status information
    """
    settings = get_settings()
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment,
        "version": settings.version,
        "backend": settings.backend_platform,
    }


@router.get("/complete", summary="Complete health check", response_model=DetailedHealthStatus)
async def complete_health():
    """
    Comprehensive health check with all system components.
    
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
            settings.hana_host,
            settings.hana_port,
            settings.hana_user,
            settings.hana_password,
        ])
        
        db_status = "warning" if has_db_config else "error"
        db_message = "Database configuration present but not validated" if has_db_config else "Database configuration missing"
        
        components.append(
            ComponentHealth(
                name="database",
                status=db_status,
                details={
                    "host": settings.hana_host or "not set",
                    "port": settings.hana_port or "not set",
                    "user": bool(settings.hana_user),
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
    if settings.backend_platform == "together_ai":
        components.append(
            ComponentHealth(
                name="together_ai",
                status="ok" if os.environ.get("TOGETHER_API_KEY") else "error",
                details={
                    "model": settings.together_model_name or "not set",
                    "api_configured": bool(os.environ.get("TOGETHER_API_KEY")),
                },
                message="Together.ai configuration present" if os.environ.get("TOGETHER_API_KEY") else "Together.ai API key missing",
            )
        )
    elif settings.backend_platform == "nvidia_launchpad":
        components.append(
            ComponentHealth(
                name="nvidia_launchpad",
                status="ok",
                details={
                    "triton_server": settings.triton_server_url or "not set",
                    "enable_tensorrt": settings.enable_tensorrt,
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
        environment=settings.environment,
        version=settings.version,
        backend=settings.backend_platform,
        components=components,
    )


@router.get("/database", summary="Database health check")
async def database_health():
    """
    Check the health of the database connection.
    
    Returns:
        dict: Database status information
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
        settings.hana_host,
        settings.hana_port,
        settings.hana_user,
        settings.hana_password,
    ])
    
    return {
        "status": "warning" if has_db_config else "error",
        "timestamp": datetime.utcnow().isoformat(),
        "details": {
            "host": settings.hana_host or "not set",
            "port": settings.hana_port or "not set",
            "user": bool(settings.hana_user),
            "connected": False,
        },
        "message": "Database configuration present but not validated" if has_db_config else "Database configuration missing",
    }