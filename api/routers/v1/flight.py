"""
Arrow Flight routes for version 1 of the API.

This module provides routes for interacting with the Arrow Flight server,
enabling high-performance transfer of vector embeddings.
"""

import logging
import json
import threading
import time
import os
from typing import Dict, List, Any, Optional, Union, Tuple

from fastapi import Depends, Request, HTTPException

from ...config_standardized import get_standardized_settings
from ...models.base_standardized import APIResponse
from ...models.flight_models import (
    FlightQueryRequest,
    FlightQueryResponse,
    FlightUploadRequest,
    FlightUploadResponse,
    FlightListResponse,
    FlightInfoResponse,
)
from ...utils.standardized_exceptions import ArrowFlightException
from ..base import BaseRouter
from ..dependencies import get_current_user, get_arrow_flight_info

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Create router
router = BaseRouter(tags=["Arrow Flight"])


@router.get("/info", response_model=FlightInfoResponse)
async def flight_info(
    request: Request,
    flight_info: Dict[str, Any] = Depends(get_arrow_flight_info),
):
    """
    Get information about the Arrow Flight server.
    
    Returns basic details about the Flight server configuration and status.
    """
    try:
        # Check if Arrow Flight middleware is available
        if hasattr(request.app.state, "arrow_flight_middleware"):
            server_info = request.app.state.arrow_flight_middleware.get_server_info()
            return FlightInfoResponse(
                host=server_info.get("host", "localhost"),
                port=server_info.get("port", 8815),
                location=server_info.get("location", ""),
                status="running" if server_info.get("running", False) else "stopped"
            )
        
        # Fall back to settings if middleware not available
        return FlightInfoResponse(
            host=settings.arrow_flight.host,
            port=settings.arrow_flight.port,
            location=settings.arrow_flight.location,
            status="unknown"
        )
    except Exception as e:
        logger.error(f"Error getting Flight server info: {str(e)}")
        raise ArrowFlightException(
            detail=f"Error getting Flight server info: {str(e)}",
            suggestion="Check that the Arrow Flight server is running"
        )


@router.post("/start", response_model=FlightInfoResponse)
async def start_server(
    request: Request,
    flight_info: Dict[str, Any] = Depends(get_arrow_flight_info),
):
    """
    Start the Arrow Flight server if it's not already running.
    
    Initializes and starts the Flight server if it's not currently active.
    """
    try:
        # Check if Arrow Flight middleware is available
        if hasattr(request.app.state, "arrow_flight_middleware"):
            # Restart the server if not running
            if not flight_info.get("running", False):
                request.app.state.arrow_flight_middleware.restart_server()
            
            # Get updated server info
            server_info = request.app.state.arrow_flight_middleware.get_server_info()
            return FlightInfoResponse(
                host=server_info.get("host", "localhost"),
                port=server_info.get("port", 8815),
                location=server_info.get("location", ""),
                status="running" if server_info.get("running", False) else "stopped"
            )
        
        # If middleware not available, raise an exception
        raise ArrowFlightException(
            detail="Arrow Flight middleware not available",
            suggestion="Check your server configuration"
        )
    except ArrowFlightException:
        raise
    except Exception as e:
        logger.error(f"Error starting Flight server: {str(e)}")
        raise ArrowFlightException(
            detail=f"Error starting Flight server: {str(e)}",
            suggestion="Check the server logs for more information"
        )


@router.post("/query", response_model=FlightQueryResponse)
async def flight_query(
    request: Request,
    query_request: FlightQueryRequest,
    flight_info: Dict[str, Any] = Depends(get_arrow_flight_info),
):
    """
    Create a Flight query for retrieving vectors.
    
    Returns a ticket and location that can be used by a Flight client to retrieve vectors.
    """
    try:
        # Check if Arrow Flight middleware is available
        if not hasattr(request.app.state, "arrow_flight_middleware"):
            raise ArrowFlightException(
                detail="Arrow Flight middleware not available",
                suggestion="Check your server configuration"
            )
        
        # Check if server is running
        if not flight_info.get("running", False):
            raise ArrowFlightException(
                detail="Arrow Flight server is not running",
                suggestion="Start the server first with /api/v1/flight/start"
            )
        
        # Create ticket for the query
        ticket_data = {
            "table": query_request.table_name,
            "filter": query_request.filter,
            "limit": query_request.limit,
            "offset": query_request.offset
        }
        
        ticket = json.dumps(ticket_data)
        
        return FlightQueryResponse(
            ticket=ticket,
            location=flight_info.get("location", ""),
            schema=None  # Schema will be determined by client
        )
    except ArrowFlightException:
        raise
    except Exception as e:
        logger.error(f"Error creating Flight query: {str(e)}")
        raise ArrowFlightException(
            detail=f"Error creating Flight query: {str(e)}",
            suggestion="Check that the table exists and the query parameters are valid"
        )


@router.post("/upload", response_model=FlightUploadResponse)
async def flight_upload(
    request: Request,
    upload_request: FlightUploadRequest,
    flight_info: Dict[str, Any] = Depends(get_arrow_flight_info),
):
    """
    Create a Flight descriptor for uploading vectors.
    
    Returns a descriptor and location that can be used by a Flight client to upload vectors.
    """
    try:
        # Check if Arrow Flight middleware is available
        if not hasattr(request.app.state, "arrow_flight_middleware"):
            raise ArrowFlightException(
                detail="Arrow Flight middleware not available",
                suggestion="Check your server configuration"
            )
        
        # Check if server is running
        if not flight_info.get("running", False):
            raise ArrowFlightException(
                detail="Arrow Flight server is not running",
                suggestion="Start the server first with /api/v1/flight/start"
            )
        
        # Create descriptor for the upload
        descriptor_data = {
            "table": upload_request.table_name,
            "create_if_not_exists": upload_request.create_if_not_exists
        }
        
        descriptor = json.dumps(descriptor_data)
        
        return FlightUploadResponse(
            descriptor=descriptor,
            location=flight_info.get("location", "")
        )
    except ArrowFlightException:
        raise
    except Exception as e:
        logger.error(f"Error creating Flight upload descriptor: {str(e)}")
        raise ArrowFlightException(
            detail=f"Error creating Flight upload descriptor: {str(e)}",
            suggestion="Check that the table name is valid"
        )


@router.get("/list", response_model=FlightListResponse)
async def flight_list(
    request: Request,
    flight_info: Dict[str, Any] = Depends(get_arrow_flight_info),
):
    """
    List available vector collections.
    
    Returns a list of available collections that can be accessed using Flight.
    """
    try:
        # Check if Arrow Flight middleware is available
        if not hasattr(request.app.state, "arrow_flight_middleware"):
            raise ArrowFlightException(
                detail="Arrow Flight middleware not available",
                suggestion="Check your server configuration"
            )
        
        # Check if server is running
        if not flight_info.get("running", False):
            raise ArrowFlightException(
                detail="Arrow Flight server is not running",
                suggestion="Start the server first with /api/v1/flight/start"
            )
        
        # Use middleware to get collections
        collections = []
        
        # Try to import Flight client
        try:
            import pyarrow.flight as flight
            client = flight.connect(flight_info.get("location", ""))
            
            # List available flights
            for flight_info in client.list_flights():
                if flight_info.descriptor.type == flight.DescriptorType.PATH:
                    collection_name = flight_info.descriptor.path[0].decode()
                    collections.append({
                        "name": collection_name,
                        "total_records": flight_info.total_records
                    })
        except ImportError:
            raise ArrowFlightException(
                detail="Arrow Flight module not available",
                suggestion="Install pyarrow with Flight support"
            )
        except Exception as e:
            raise ArrowFlightException(
                detail=f"Error listing Flight collections: {str(e)}",
                suggestion="Check the server logs for more information"
            )
        
        return FlightListResponse(
            collections=collections,
            location=flight_info.get("location", "")
        )
    except ArrowFlightException:
        raise
    except Exception as e:
        logger.error(f"Error listing Flight collections: {str(e)}")
        raise ArrowFlightException(
            detail=f"Error listing Flight collections: {str(e)}",
            suggestion="Check that the Flight server is running and accessible"
        )