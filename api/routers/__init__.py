"""
API routers for the SAP HANA LangChain Integration API.

This package contains versioned API routers for the application.
"""

from fastapi import APIRouter

from .v1 import router as v1_router
from .v2 import router as v2_router

# Create a parent router that includes all version routers
router = APIRouter()

# Include version-specific routers
router.include_router(v1_router, prefix="/v1")
router.include_router(v2_router, prefix="/v2")

__all__ = ["router"]