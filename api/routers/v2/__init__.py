"""
API routers for version 2 of the SAP HANA LangChain Integration API.

This module includes all v2 routers with enhanced functionality, optimizations,
and additional features not available in v1.
"""

from fastapi import APIRouter

from .gpu import router as gpu_router
from .health import router as health_router
from .tensorrt import router as tensorrt_router

# Create a parent router for v2 that includes all feature routers
router = APIRouter(tags=["v2"])

# Include feature-specific routers
router.include_router(health_router)
router.include_router(gpu_router)
router.include_router(tensorrt_router)

# NOTE: Additional v2 routers will be added as they are implemented
# The following routers are planned for future implementation:
# - analytics
# - data_pipeline
# - embeddings
# - flight
# - reasoning
# - vector_operations

__all__ = ["router"]