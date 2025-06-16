# Versioned API Structure

This document describes the standardized versioned API structure for the SAP HANA LangChain Integration API.

## Overview

The API is structured using a versioned approach, with separate namespaces for each API version. This allows for backward compatibility while enabling the addition of new features and improvements.

```
api/
├── routers/
│   ├── __init__.py         # Combines all versioned routers
│   ├── base.py             # Base router classes and utilities
│   ├── dependencies.py     # Common router dependencies
│   ├── v1/                 # Version 1 routers
│   │   ├── __init__.py     # Combines all v1 routers
│   │   ├── analytics.py    # Analytics endpoints
│   │   ├── data_pipeline.py # Data pipeline endpoints
│   │   ├── developer.py    # Developer tools endpoints
│   │   ├── embeddings.py   # Embeddings endpoints
│   │   ├── financial_embeddings.py # Financial embeddings endpoints
│   │   ├── flight.py       # Arrow Flight endpoints
│   │   ├── health.py       # Health check endpoints
│   │   ├── optimization.py # Optimization endpoints
│   │   ├── reasoning.py    # Reasoning endpoints
│   │   ├── update.py       # Document update endpoints
│   │   └── vector_operations.py # Vector operations endpoints
│   └── v2/                 # Version 2 routers (enhanced functionality)
│       ├── __init__.py     # Combines all v2 routers
│       ├── gpu.py          # GPU management endpoints
│       ├── health.py       # Enhanced health check endpoints
│       └── tensorrt.py     # TensorRT optimization endpoints
└── main_standardized.py    # Main application entry point
```

## URL Structure

API endpoints follow this URL structure:

```
/api/v{version}/{feature}/{endpoint}
```

For example:
- `/api/v1/health/status` - Basic health status (v1)
- `/api/v2/health/status` - Enhanced health status with more details (v2)
- `/api/v1/flight/info` - Arrow Flight server info (v1)
- `/api/v2/gpu/info` - GPU information (v2 only)

## Router Classes

The API uses three base router classes to ensure consistent behavior:

1. **BaseRouter**: Base router with common configuration and response wrapping
   - All responses are wrapped in a standard APIResponse format
   - Common error responses are defined
   - Consistent behavior across all routes

2. **AuthenticatedRouter**: Requires authentication for all routes
   - Extends BaseRouter
   - Adds authentication dependency to all routes
   - Supports JWT tokens and API keys

3. **AdminRouter**: Requires admin privileges for all routes
   - Extends BaseRouter
   - Adds admin authentication dependency
   - For administrative endpoints only

## Versioning Strategy

1. **Version 1 (v1)**
   - Base functionality
   - Core features and endpoints
   - Stable API contracts

2. **Version 2 (v2)**
   - Enhanced functionality
   - GPU acceleration and optimizations
   - TensorRT integration
   - More detailed responses
   - Additional parameters and options

## Adding New Endpoints

To add a new endpoint:

1. Determine the appropriate version (v1 or v2)
2. Create or update the feature-specific router in the appropriate version directory
3. Add the router to the version's `__init__.py` file
4. Implement the endpoint using the appropriate router class
5. Add proper documentation and type hints

Example:

```python
# api/routers/v2/new_feature.py
from ..base import BaseRouter
from pydantic import BaseModel, Field

router = BaseRouter(
    prefix="/new-feature",
    tags=["New Feature"]
)

class NewFeatureResponse(BaseModel):
    result: str = Field(..., description="Result of the operation")

@router.get("/endpoint", response_model=NewFeatureResponse)
async def endpoint():
    """
    New feature endpoint description.
    
    Returns:
        NewFeatureResponse: Response model
    """
    return NewFeatureResponse(result="Success")
```

Then add to `api/routers/v2/__init__.py`:

```python
from .new_feature import router as new_feature_router
# ...
router.include_router(new_feature_router)
```

## Testing

Use the `test_versioned_api.py` script to test the API endpoints:

```bash
python test_versioned_api.py
```

## Running the API

Use the `run_standardized_api.sh` script to run the API:

```bash
./run_standardized_api.sh
```

## Best Practices

1. Always use the appropriate base router class for your needs
2. Add comprehensive docstrings to all endpoints
3. Define proper response models with field descriptions
4. Use dependencies for common functionality
5. Ensure backward compatibility when updating v1 endpoints
6. Add new features to v2 when they would break v1 compatibility
7. Use type hints consistently
8. Validate inputs and handle errors gracefully
9. Write tests for new endpoints