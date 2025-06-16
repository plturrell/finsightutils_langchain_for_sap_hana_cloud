"""
Standardized FastAPI backend for SAP HANA LangChain Integration with Arrow Flight support.

This module provides a standardized FastAPI application with versioned API routes,
consistent middleware components, and a standardized configuration system.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uvicorn
from fastapi import FastAPI, Request, Depends, APIRouter
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config_standardized import get_standardized_settings
from .middleware import (
    MIDDLEWARE_REGISTRY,
    AuthMiddleware,
    ErrorHandlerMiddleware,
    GPUMiddleware,
    LoggingMiddleware,
    RateLimitMiddleware,
    RequestIdMiddleware,
    ResponseWrapperMiddleware,
    SecurityHeadersMiddleware,
    TelemetryMiddleware,
    ArrowFlightMiddleware,
    configure_cors,
)
from .middleware.response_wrapper_middleware import configure_response_wrapper
from .models.base_standardized import APIResponse, ErrorResponse
from .utils.standardized_exceptions import BaseAPIException
from .routers import router as api_router

# Get settings
settings = get_standardized_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper(), logging.INFO),
    format=settings.logging.format,
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.api.name,
    description=settings.api.description,
    version=settings.api.version,
    docs_url=settings.api.docs_url,
    redoc_url=settings.api.redoc_url,
    openapi_url=settings.api.openapi_url,
)

# Add the API router
app.include_router(api_router, prefix="/api")

# Add middleware (in reverse order of execution)
# Note: FastAPI executes middleware in reverse order (last to first)
for middleware in reversed(MIDDLEWARE_REGISTRY):
    if callable(middleware) and not isinstance(middleware, type):
        # Function middleware (like configure_cors)
        middleware(app)
    else:
        # Class middleware
        app.add_middleware(middleware)

# Configure response wrapper for all routes
configure_response_wrapper(app)

# Set start time for uptime tracking
app.state.start_time = time.time()

# Templates
templates_dir = Path(__file__).parent.parent / "templates"
if templates_dir.exists() and templates_dir.is_dir():
    templates = Jinja2Templates(directory=str(templates_dir))
else:
    # Create a simple templates directory if it doesn't exist
    templates_dir = Path(__file__).parent / "templates"
    templates_dir.mkdir(exist_ok=True)
    
    # Create a simple index.html template
    index_template = templates_dir / "index.html"
    if not index_template.exists():
        with open(index_template, "w") as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1 {
            color: #0066cc;
        }
        .api-info {
            background-color: #f5f5f5;
            border-radius: 5px;
            padding: 20px;
            margin: 20px 0;
        }
        .api-link {
            display: inline-block;
            background-color: #0066cc;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 20px;
        }
        .api-link:hover {
            background-color: #004c99;
        }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    <div class="api-info">
        <p>Version: <strong>{{ version }}</strong></p>
        <p>Environment: <strong>{{ environment }}</strong></p>
        <p>This API provides LangChain integration with SAP HANA Cloud and Arrow Flight protocol support.</p>
    </div>
    <a href="/api/docs" class="api-link">API Documentation</a>
</body>
</html>""")
    
    templates = Jinja2Templates(directory=str(templates_dir))

# Root endpoint to serve the frontend or a welcome page
@app.get("/", include_in_schema=False)
async def root(request: Request):
    """Serve the frontend application or a welcome page."""
    frontend_path = Path(__file__).parent.parent / "frontend-static" / "index.html"
    if frontend_path.exists():
        return FileResponse(frontend_path)
    else:
        # Fallback to a simple HTML page using templates
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "title": settings.api.name,
                "version": settings.api.version,
                "environment": settings.environment.name,
            }
        )

# Health check endpoint at the root level
@app.get("/health", include_in_schema=True, tags=["Health"])
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "version": settings.api.version,
        "timestamp": time.time(),
        "environment": settings.environment.name,
    }

# Mount static files if available
static_path = Path(__file__).parent.parent / "static"
if static_path.exists() and static_path.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

# Serve the application
if __name__ == "__main__":
    # Create an instance of each middleware to attach to app.state
    # This makes the middleware instances available to the application
    # and allows accessing their methods and properties
    app.state.gpu_middleware = GPUMiddleware(app)
    app.state.arrow_flight_middleware = ArrowFlightMiddleware(app)
    
    # Run the application
    uvicorn.run(
        "main_standardized:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.environment.development_mode,
    )