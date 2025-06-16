"""
Financial Embeddings API Routes

This module provides API routes for financial embeddings visualization.
"""

import os
import json
import logging
from typing import Dict, Any, List

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# Set up logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(
    prefix="/financial-embeddings",
    tags=["financial-embeddings"],
    responses={404: {"description": "Not found"}},
)

# Data models
class FinancialEmbedding(BaseModel):
    """Financial embedding with visualization data."""
    points: List[List[float]]
    metadata: List[Dict[str, Any]]
    similarities: List[float]
    clusters: List[int]
    total_vectors: int

# Root directory for data files
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_FILE = os.path.join(ROOT_DIR, "financial_visualization_data.json")

def get_visualization_data() -> Dict[str, Any]:
    """
    Load visualization data from the pre-extracted file.
    
    Returns:
        Dict[str, Any]: The visualization data.
    
    Raises:
        HTTPException: If the data file is not found or is invalid.
    """
    try:
        if not os.path.exists(DATA_FILE):
            logger.error(f"Data file not found: {DATA_FILE}")
            raise HTTPException(status_code=404, detail="Financial embeddings data not found")
        
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        
        return data
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in data file: {DATA_FILE}")
        raise HTTPException(status_code=500, detail="Invalid financial embeddings data")
    except Exception as e:
        logger.error(f"Error loading visualization data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading visualization data: {str(e)}")

@router.get("/visualization-data", response_model=Dict[str, Any])
def visualization_data():
    """
    Get financial embeddings visualization data.
    
    Returns:
        Dict[str, Any]: The visualization data including points, metadata, similarities, and clusters.
    """
    data = get_visualization_data()
    logger.info(f"Returning visualization data with {data.get('total_vectors', 0)} vectors")
    return data

@router.get("/status")
def status():
    """
    Check the status of the financial embeddings API.
    
    Returns:
        Dict[str, Any]: The status information.
    """
    try:
        data = get_visualization_data()
        return {
            "status": "ok",
            "vectors_available": data.get('total_vectors', 0),
            "data_file": os.path.basename(DATA_FILE)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }