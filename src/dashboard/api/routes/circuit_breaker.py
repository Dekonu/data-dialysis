"""Circuit breaker status endpoint for dashboard API."""

import logging

from fastapi import APIRouter, HTTPException

from src.dashboard.api.dependencies import StorageDep
from src.dashboard.services.circuit_breaker_service import CircuitBreakerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/circuit-breaker", tags=["circuit-breaker"])


@router.get("/status")
async def get_circuit_breaker_status(storage: StorageDep):
    """Get circuit breaker status.
    
    Returns the current status of the circuit breaker, including:
    - Whether it's open or closed
    - Current failure rate
    - Configuration thresholds
    - Processing statistics
    
    Note: Circuit breaker state is calculated from recent audit logs
    if no active circuit breaker instance is available.
    """
    try:
        logger.debug("Circuit breaker status endpoint called")
        service = CircuitBreakerService(storage)
        result = service.get_status()
        
        if result.is_success():
            logger.debug("Circuit breaker status retrieved successfully")
            return result.value
        else:
            error_msg = str(result.error)
            logger.error(f"Circuit breaker service error: {error_msg}")
            raise HTTPException(
                status_code=500, 
                detail=f"Circuit breaker service error: {error_msg}"
            )
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        error_msg = f"Unexpected error in circuit breaker endpoint: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get circuit breaker status: {str(e)}"
        )

