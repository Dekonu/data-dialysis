"""Circuit breaker status endpoint for dashboard API."""

from fastapi import APIRouter, HTTPException

from src.dashboard.api.dependencies import StorageDep
from src.dashboard.services.circuit_breaker_service import CircuitBreakerService

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
    service = CircuitBreakerService(storage)
    result = service.get_status()
    
    if result.is_success():
        return result.value
    else:
        raise HTTPException(status_code=500, detail=str(result.error))

