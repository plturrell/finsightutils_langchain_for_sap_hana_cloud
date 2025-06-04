"""
Ultra minimal FastAPI app with no external dependencies.
"""

from fastapi import FastAPI
import sys

app = FastAPI()

@app.get("/")
async def root():
    """Root endpoint with minimal functionality."""
    return {
        "message": "Ultra minimal API is working",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    }