"""Metrics endpoints for dashboard API.

This module provides endpoints for overview, security, and performance metrics.
"""

from fastapi import APIRouter

from src.dashboard.api.dependencies import StorageDep

router = APIRouter(prefix="/api/metrics", tags=["metrics"])


@router.get("/overview")
async def get_overview_metrics(
    time_range: str = "24h",
    storage: StorageDep = None
):
    """Get overview metrics.
    
    TODO: Implement in Phase 2
    """
    return {
        "message": "Not implemented yet",
        "endpoint": "/api/metrics/overview",
        "time_range": time_range
    }


@router.get("/security")
async def get_security_metrics(
    time_range: str = "7d",
    storage: StorageDep = None
):
    """Get security metrics.
    
    TODO: Implement in Phase 2
    """
    return {
        "message": "Not implemented yet",
        "endpoint": "/api/metrics/security",
        "time_range": time_range
    }


@router.get("/performance")
async def get_performance_metrics(
    time_range: str = "24h",
    storage: StorageDep = None
):
    """Get performance metrics.
    
    TODO: Implement in Phase 2
    """
    return {
        "message": "Not implemented yet",
        "endpoint": "/api/metrics/performance",
        "time_range": time_range
    }

