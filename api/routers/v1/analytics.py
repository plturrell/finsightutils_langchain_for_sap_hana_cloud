"""
Analytics routes for version 1 of the API.

This module provides analytics and monitoring endpoints for the application.
"""

import logging
from typing import Dict, List, Optional, Any
import time
import json
from datetime import datetime, timedelta
import random

from fastapi import Depends, Request, HTTPException, Path, Body
from pydantic import BaseModel, Field

from ...config_standardized import get_standardized_settings
from ...utils.standardized_exceptions import DatabaseException
from ...utils.error_utils import create_context_aware_error
from ..base import BaseRouter
from ..dependencies import get_current_user
from ...db import get_connection

# Get settings
settings = get_standardized_settings()

# Setup logging
logger = logging.getLogger(__name__)

# Models
class RecentQuery(BaseModel):
    """Model for a recently executed query."""
    id: int = Field(..., description="Unique identifier for the query")
    query: str = Field(..., description="The text of the query")
    timestamp: str = Field(..., description="Timestamp when the query was executed")
    results_count: int = Field(..., description="Number of results returned")
    execution_time: int = Field(..., description="Execution time in milliseconds")


class PerformanceStats(BaseModel):
    """Model for performance statistics."""
    name: str = Field(..., description="Name of the time period (month)")
    queries: int = Field(..., description="Number of queries executed")
    avgTime: float = Field(..., description="Average execution time in milliseconds")


class PerformanceComparison(BaseModel):
    """Model for performance comparison between CPU, GPU, and TensorRT."""
    name: str = Field(..., description="Name of the operation type")
    CPU: float = Field(..., description="Average execution time on CPU")
    GPU: float = Field(..., description="Average execution time on GPU")
    TensorRT: float = Field(..., description="Average execution time with TensorRT")


# Create router
router = BaseRouter(tags=["Analytics"])


# Helper Functions
async def get_recent_queries_from_db(limit: int = 10) -> List[RecentQuery]:
    """Get recent queries from the database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Execute query to get recent queries from the database
        query = """
        SELECT 
            ROW_NUMBER() OVER (ORDER BY QUERY_TIMESTAMP DESC) as ID,
            QUERY_TEXT, 
            QUERY_TIMESTAMP, 
            RESULTS_COUNT, 
            EXECUTION_TIME 
        FROM QUERY_LOGS 
        ORDER BY QUERY_TIMESTAMP DESC 
        LIMIT ?
        """
        cursor.execute(query, (limit,))
        
        # Process results
        results = []
        for row in cursor.fetchall():
            results.append(
                RecentQuery(
                    id=row[0],
                    query=row[1],
                    timestamp=row[2].isoformat() if isinstance(row[2], datetime) else str(row[2]),
                    results_count=row[3],
                    execution_time=row[4]
                )
            )
        
        return results
    except Exception as e:
        logger.error(f"Error getting recent queries: {e}")
        # Return empty list on error
        return []


async def get_performance_stats_from_db(months: int = 6) -> List[PerformanceStats]:
    """Get performance statistics from the database."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30 * months)
        
        # Execute query to get performance statistics from the database
        query = """
        SELECT 
            FORMAT_TIMESTAMP(QUERY_TIMESTAMP, 'yyyy-MM') as MONTH,
            COUNT(*) as QUERY_COUNT,
            AVG(EXECUTION_TIME) as AVG_TIME
        FROM QUERY_LOGS 
        WHERE QUERY_TIMESTAMP BETWEEN ? AND ?
        GROUP BY FORMAT_TIMESTAMP(QUERY_TIMESTAMP, 'yyyy-MM')
        ORDER BY MONTH
        """
        cursor.execute(query, (start_date, end_date))
        
        # Process results
        results = []
        for row in cursor.fetchall():
            month_year = row[0]
            month_name = datetime.strptime(month_year, "%Y-%m").strftime("%b")
            
            results.append(
                PerformanceStats(
                    name=month_name,
                    queries=row[1],
                    avgTime=float(row[2])
                )
            )
        
        return results
    except Exception as e:
        logger.error(f"Error getting performance stats: {e}")
        # Return empty list on error
        return []


async def get_performance_comparison_from_db() -> List[PerformanceComparison]:
    """Get performance comparison data for CPU vs GPU vs TensorRT."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Execute query to get performance comparison data
        query = """
        SELECT 
            OPERATION_TYPE,
            AVG(CPU_TIME) as CPU_AVG,
            AVG(GPU_TIME) as GPU_AVG,
            AVG(TENSORRT_TIME) as TENSORRT_AVG
        FROM PERFORMANCE_BENCHMARKS
        GROUP BY OPERATION_TYPE
        """
        cursor.execute(query)
        
        # Process results
        results = []
        for row in cursor.fetchall():
            results.append(
                PerformanceComparison(
                    name=row[0],
                    CPU=float(row[1]),
                    GPU=float(row[2]),
                    TensorRT=float(row[3])
                )
            )
        
        return results
    except Exception as e:
        logger.error(f"Error getting performance comparison: {e}")
        # Return empty list on error
        return []


# Endpoints
@router.get("/recent-queries")
async def get_recent_queries(
    request: Request,
    limit: int = 10
) -> Dict[str, List[RecentQuery]]:
    """
    Get recent queries executed in the system.
    
    Returns a list of recent queries with their execution details.
    """
    try:
        queries = await get_recent_queries_from_db(limit)
        return {"data": queries}
    except Exception as e:
        error_context = {
            "operation": "get_recent_queries",
            "details": str(e)
        }
        logger.error(f"Error in get_recent_queries: {e}", extra={"context": error_context})
        raise HTTPException(
            status_code=500,
            detail=create_context_aware_error(
                message="Failed to retrieve recent queries",
                operation="analytics",
                context=error_context
            )
        )


@router.get("/performance")
async def get_performance_stats(
    request: Request,
    months: int = 6
) -> Dict[str, List[PerformanceStats]]:
    """
    Get performance statistics over time.
    
    Returns a list of performance statistics by month.
    """
    try:
        stats = await get_performance_stats_from_db(months)
        return {"data": stats}
    except Exception as e:
        error_context = {
            "operation": "get_performance_stats",
            "details": str(e)
        }
        logger.error(f"Error in get_performance_stats: {e}", extra={"context": error_context})
        raise HTTPException(
            status_code=500,
            detail=create_context_aware_error(
                message="Failed to retrieve performance statistics",
                operation="analytics",
                context=error_context
            )
        )


@router.get("/query-count")
async def get_query_count(
    request: Request
) -> Dict[str, Dict[str, int]]:
    """
    Get total query count.
    
    Returns the total number of queries processed.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "SELECT COUNT(*) FROM QUERY_LOGS"
        cursor.execute(query)
        count = cursor.fetchone()[0]
        
        return {"data": {"count": count}}
    except Exception as e:
        error_context = {
            "operation": "get_query_count",
            "details": str(e)
        }
        logger.error(f"Error getting query count: {e}", extra={"context": error_context})
        # Return default data instead of raising an exception
        return {"data": {"count": 0}}


@router.get("/response-time")
async def get_average_response_time(
    request: Request
) -> Dict[str, Dict[str, float]]:
    """
    Get average response time for queries.
    
    Returns the average execution time for all queries.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = "SELECT AVG(EXECUTION_TIME) FROM QUERY_LOGS"
        cursor.execute(query)
        avg_time = cursor.fetchone()[0]
        
        return {"data": {"avgTime": float(avg_time) if avg_time else 0}}
    except Exception as e:
        error_context = {
            "operation": "get_average_response_time",
            "details": str(e)
        }
        logger.error(f"Error getting average response time: {e}", extra={"context": error_context})
        # Return default data instead of raising an exception
        return {"data": {"avgTime": 0}}


@router.get("/performance-comparison")
async def get_performance_comparison(
    request: Request
) -> Dict[str, List[PerformanceComparison]]:
    """
    Get performance comparison between CPU, GPU, and TensorRT.
    
    Returns a list of performance comparisons by operation type.
    """
    try:
        comparisons = await get_performance_comparison_from_db()
        return {"data": comparisons}
    except Exception as e:
        error_context = {
            "operation": "get_performance_comparison",
            "details": str(e)
        }
        logger.error(f"Error in get_performance_comparison: {e}", extra={"context": error_context})
        raise HTTPException(
            status_code=500,
            detail=create_context_aware_error(
                message="Failed to retrieve performance comparison data",
                operation="analytics",
                context=error_context
            )
        )