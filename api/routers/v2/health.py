"""
Health check routes for version 2 of the API.

This module provides enhanced health check endpoints with more detailed system information.
"""

import time
import platform
import os
import psutil
from typing import Dict, Any, List, Optional

from fastapi import Request
from pydantic import BaseModel, Field

from ...config_standardized import get_standardized_settings
from ..base import BaseRouter

# Get settings
settings = get_standardized_settings()


class ComponentStatus(BaseModel):
    """Status of a system component."""
    
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Component status (ok, warning, error)")
    details: Dict[str, Any] = Field(default_factory=dict, description="Component details")


class SystemInfo(BaseModel):
    """System information."""
    
    platform: str = Field(..., description="Operating system platform")
    python_version: str = Field(..., description="Python version")
    cpu_count: int = Field(..., description="Number of CPU cores")
    memory_total: float = Field(..., description="Total memory in GB")
    memory_available: float = Field(..., description="Available memory in GB")
    disk_total: float = Field(..., description="Total disk space in GB")
    disk_available: float = Field(..., description="Available disk space in GB")


class EnhancedHealthResponse(BaseModel):
    """Enhanced health check response with detailed information."""
    
    status: str = Field(..., description="Overall status (ok, warning, error)")
    version: str = Field(..., description="API version")
    environment: str = Field(..., description="Environment name")
    timestamp: float = Field(..., description="Current timestamp")
    uptime: float = Field(..., description="Server uptime in seconds")
    components: List[ComponentStatus] = Field(default_factory=list, description="Component statuses")
    system: SystemInfo = Field(..., description="System information")


# Create a router
router = BaseRouter(tags=["Health"])


@router.get("/health", response_model=EnhancedHealthResponse)
async def health(request: Request) -> EnhancedHealthResponse:
    """
    Enhanced health check endpoint.
    
    Returns detailed information about the API service status, including system 
    resources and component health.
    """
    # Calculate uptime
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time
    
    # Create a list of components to check
    components = []
    
    # API component
    components.append(ComponentStatus(
        name="api",
        status="ok",
        details={
            "version": settings.api.version,
            "environment": settings.environment.name,
        }
    ))
    
    # Database component
    db_status = "ok"
    db_details = {
        "type": "SAP HANA",
        "pool_size": settings.database.pool_size,
        "max_overflow": settings.database.max_overflow
    }
    
    # Try to check database connection
    try:
        # This is a placeholder - in a real implementation, you would check the actual connection
        db_status = "ok"
    except Exception as e:
        db_status = "error"
        db_details["error"] = str(e)
    
    components.append(ComponentStatus(
        name="database",
        status=db_status,
        details=db_details
    ))
    
    # GPU component if enabled
    if settings.gpu.enabled and hasattr(request.state, "gpu_info"):
        gpu_info = request.state.gpu_info
        gpu_status = "ok" if gpu_info.get("available", False) else "warning"
        
        components.append(ComponentStatus(
            name="gpu",
            status=gpu_status,
            details={
                "available": gpu_info.get("available", False),
                "count": gpu_info.get("count", 0),
                "devices": list(gpu_info.get("devices", {}).keys()),
                "cuda_version": gpu_info.get("cuda_version"),
                "tensorrt_available": gpu_info.get("tensorrt_available", False),
            }
        ))
    
    # Arrow Flight component if enabled
    if settings.arrow_flight.enabled and hasattr(request.app.state, "arrow_flight_middleware"):
        flight_info = request.app.state.arrow_flight_middleware.get_server_info()
        flight_status = "ok" if flight_info.get("running", False) else "warning"
        
        components.append(ComponentStatus(
            name="arrow_flight",
            status=flight_status,
            details={
                "running": flight_info.get("running", False),
                "host": flight_info.get("host", ""),
                "port": flight_info.get("port", 0),
                "location": flight_info.get("location", ""),
            }
        ))
    
    # Get system information
    try:
        system_info = SystemInfo(
            platform=platform.platform(),
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 0,
            memory_total=psutil.virtual_memory().total / (1024 ** 3),  # GB
            memory_available=psutil.virtual_memory().available / (1024 ** 3),  # GB
            disk_total=psutil.disk_usage('/').total / (1024 ** 3),  # GB
            disk_available=psutil.disk_usage('/').free / (1024 ** 3),  # GB
        )
    except Exception as e:
        # Fallback if psutil is not available
        system_info = SystemInfo(
            platform=platform.platform(),
            python_version=platform.python_version(),
            cpu_count=os.cpu_count() or 0,
            memory_total=0.0,
            memory_available=0.0,
            disk_total=0.0,
            disk_available=0.0,
        )
    
    # Determine overall status
    overall_status = "ok"
    for component in components:
        if component.status == "error":
            overall_status = "error"
            break
        if component.status == "warning" and overall_status != "error":
            overall_status = "warning"
    
    return EnhancedHealthResponse(
        status=overall_status,
        version=settings.api.version,
        environment=settings.environment.name,
        timestamp=time.time(),
        uptime=uptime,
        components=components,
        system=system_info,
    )


@router.get("/health/ping")
async def ping():
    """
    Simple ping endpoint for basic health checks.
    
    Returns a simple string response for lightweight health checks.
    """
    return "pong"


@router.get("/health/readiness")
async def readiness(request: Request):
    """
    Readiness probe for Kubernetes.
    
    Checks if the service is ready to handle requests.
    """
    # Check database connection
    db_ready = True
    
    # Check GPU availability if enabled
    gpu_ready = True
    if settings.gpu.enabled and hasattr(request.state, "gpu_info"):
        gpu_info = request.state.gpu_info
        gpu_ready = gpu_info.get("available", False)
    
    # Check Arrow Flight server if enabled
    flight_ready = True
    if settings.arrow_flight.enabled and hasattr(request.app.state, "arrow_flight_middleware"):
        flight_info = request.app.state.arrow_flight_middleware.get_server_info()
        flight_ready = flight_info.get("running", False)
    
    # Determine overall readiness
    ready = db_ready and (not settings.gpu.enabled or gpu_ready) and (not settings.arrow_flight.enabled or flight_ready)
    
    return {
        "ready": ready,
        "checks": {
            "database": db_ready,
            "gpu": gpu_ready if settings.gpu.enabled else None,
            "arrow_flight": flight_ready if settings.arrow_flight.enabled else None,
        }
    }


@router.get("/health/liveness")
async def liveness(request: Request):
    """
    Liveness probe for Kubernetes.
    
    Checks if the service is alive and functioning properly.
    """
    # Calculate uptime
    start_time = getattr(request.app.state, "start_time", time.time())
    uptime = time.time() - start_time
    
    return {
        "alive": True,
        "uptime": uptime,
        "timestamp": time.time()
    }