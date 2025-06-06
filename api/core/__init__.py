"""
Core components for SAP HANA Cloud integration API.

This module provides core functionality for the API and exports the main application 
entry point and Vercel serverless handler.
"""

from api.core.main import app, handler

# Export for direct import from api.core
__all__ = [
    "app",
    "handler"
]