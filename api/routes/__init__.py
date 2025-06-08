"""
API routes for SAP HANA Cloud integration API.

This module provides FastAPI route definitions.
"""

# Import route definitions here
from .optimization import router as optimization_router
from .update import router as update_router

__all__ = [
    "optimization_router",
    "update_router"
]