"""
Health check routes for version 1 of the API.

This module provides basic health check and status endpoints for monitoring.
"""

import logging
import time
import os
from typing import Dict, Any

from fastapi import Depends, Request
from pydantic import BaseModel, Field

from ...config_standardized import get_standardized_settings
from ...utils.standardized_exceptions import DatabaseException
from ..base import BaseRouter
from ..dependencies import get_current_user, get_db_status

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Models
class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="API status (ok or error)")
    timestamp: float = Field(..., description="Current timestamp")
    environment: str = Field(..., description="Current environment")
    version: str = Field(..., description="API version")


class StatusResponse(BaseModel):
    """Model for detailed status response."""
    status: str = Field(..., description="Overall status (ok or error)")
    components: list = Field(..., description="Status of individual components")


# Create router
router = BaseRouter(tags=["Health"])


@router.get("/ping", response_model=str)
async def ping():
    """
    Simple ping endpoint for basic connectivity checks.
    
    Returns a string 'pong' to indicate the API is responsive.
    """
    return "pong"


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Basic health check endpoint.
    
    Returns basic health information about the API.
    """
    return HealthResponse(
        status="ok",
        timestamp=time.time(),
        environment=settings.environment.name,
        version=settings.api.version
    )


@router.get("/status", response_model=StatusResponse)
async def status(
    request: Request,
    db_status: Dict[str, Any] = Depends(get_db_status)
):
    """
    Detailed status check endpoint.
    
    Returns detailed status information about all components of the system.
    """
    components = [
        {
            "name": "api",
            "status": "ok",
            "details": {
                "uptime": time.time() - request.app.state.start_time if hasattr(request.app.state, "start_time") else 0,
                "environment": settings.environment.name,
                "version": settings.api.version
            }
        },
        {
            "name": "database",
            "status": db_status.get("status", "unknown"),
            "details": {
                "type": "SAP HANA Cloud",
                "connection": db_status.get("connection_info", "unknown"),
                "message": db_status.get("message", "")
            }
        }
    ]
    
    # Add GPU component if GPU middleware is available
    if hasattr(request.app.state, "gpu_middleware"):
        gpu_info = request.app.state.gpu_middleware.get_gpu_info()
        components.append({
            "name": "gpu",
            "status": "ok" if gpu_info.get("available", False) else "not_available",
            "details": {
                "available": gpu_info.get("available", False),
                "device_count": gpu_info.get("device_count", 0),
                "devices": gpu_info.get("devices", []),
                "tensorrt_available": gpu_info.get("tensorrt_available", False)
            }
        })
    
    # Add Arrow Flight component if available
    if hasattr(request.app.state, "arrow_flight_middleware"):
        flight_info = request.app.state.arrow_flight_middleware.get_server_info()
        components.append({
            "name": "arrow_flight",
            "status": "ok" if flight_info.get("running", False) else "not_running",
            "details": {
                "host": flight_info.get("host", ""),
                "port": flight_info.get("port", 0),
                "location": flight_info.get("location", ""),
                "running": flight_info.get("running", False)
            }
        })
    
    # Determine overall status
    overall_status = "ok"
    for component in components:
        if component["status"] != "ok":
            if component["name"] in ["api", "database"]:  # Critical components
                overall_status = "error"
                break
    
    return StatusResponse(
        status=overall_status,
        components=components
    )