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
        service = CircuitBreakerService(storage)
        result = service.get_status()
        
        if result.is_success():
            return result.value
        else:
            logger.error(f"Circuit breaker service error: {result.error}")
            raise HTTPException(status_code=500, detail=str(result.error))
    except Exception as e:
        logger.error(f"Unexpected error in circuit breaker endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get circuit breaker status: {str(e)}")

