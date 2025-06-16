"""
API routers for version 1 of the API.
"""

from fastapi import APIRouter

from .analytics import router as analytics_router
from .data_pipeline import router as data_pipeline_router
from .developer import router as developer_router
from .embeddings import router as embeddings_router
from .financial_embeddings import router as financial_embeddings_router
from .flight import router as flight_router
from .health import router as health_router
from .optimization import router as optimization_router
from .reasoning import router as reasoning_router
from .update import router as update_router
from .vector_operations import router as vector_operations_router

# Create a parent router for v1 that includes all feature routers
router = APIRouter(tags=["v1"])

# Include feature-specific routers
router.include_router(health_router)
router.include_router(analytics_router)
router.include_router(data_pipeline_router)
router.include_router(developer_router)
router.include_router(embeddings_router)
router.include_router(financial_embeddings_router)
router.include_router(flight_router)
router.include_router(optimization_router)
router.include_router(reasoning_router)
router.include_router(update_router)
router.include_router(vector_operations_router)

__all__ = ["router"]