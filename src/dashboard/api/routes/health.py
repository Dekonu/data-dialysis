"""Health check endpoint for dashboard API."""

import logging
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from src.dashboard.api.dependencies import StorageDep
from src.dashboard.models.health import DatabaseHealth, HealthResponse
from src.domain.ports import StoragePort

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["health"])


async def check_database_health(storage: StoragePort) -> DatabaseHealth:
    """Check database connection health.
    
    Parameters:
        storage: Storage adapter instance
        
    Returns:
        DatabaseHealth: Database health status
        
    Security Impact:
        - Only checks connectivity, no sensitive data exposed
    """
    db_type = storage.db_config.db_type if hasattr(storage, 'db_config') else "unknown"
    
    try:
        # Try a simple query to check connectivity
        start_time = time.time()
        
        # Use query method if available, otherwise just check connection
        if hasattr(storage, 'query'):
            result = storage.query("SELECT 1")
            if result.is_success():
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                return DatabaseHealth(
                    status="connected",
                    type=db_type,
                    response_time_ms=round(response_time, 2)
                )
            else:
                # Query failed - database is disconnected
                logger.warning(f"Database query failed: {result.error}")
                return DatabaseHealth(
                    status="disconnected",
                    type=db_type,
                    response_time_ms=None
                )
        
        # Fallback: just check if storage is initialized
        if hasattr(storage, '_initialized') and storage._initialized:
            return DatabaseHealth(
                status="connected",
                type=db_type,
                response_time_ms=None
            )
        
        return DatabaseHealth(
            status="disconnected",
            type=db_type,
            response_time_ms=None
        )
        
    except Exception as e:
        logger.warning(f"Database health check failed: {str(e)}")
        return DatabaseHealth(
            status="disconnected",
            type=db_type,
            response_time_ms=None
        )


@router.get("/health", response_model=HealthResponse)
async def health_check(storage: StorageDep) -> HealthResponse:
    """Health check endpoint.
    
    This endpoint provides system health status including database connectivity.
    Used by monitoring tools and load balancers.
    
    Parameters:
        storage: Storage adapter (injected via dependency)
        
    Returns:
        HealthResponse: System health status
        
    Security Impact:
        - No sensitive information exposed
        - Safe for public health checks
    """
    try:
        # Check database health
        db_health = await check_database_health(storage)
        
        # Determine overall status
        if db_health.status == "connected":
            overall_status = "healthy"
        elif db_health.status == "disconnected":
            overall_status = "unhealthy"
        else:
            overall_status = "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version="1.0.0",
            database=db_health
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Health check failed"
        )

