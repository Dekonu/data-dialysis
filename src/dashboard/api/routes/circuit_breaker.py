"""Circuit breaker status endpoint for dashboard API."""

from fastapi import APIRouter

from src.dashboard.api.dependencies import StorageDep

router = APIRouter(prefix="/api/circuit-breaker", tags=["circuit-breaker"])


@router.get("/status")
async def get_circuit_breaker_status(storage: StorageDep = None):
    """Get circuit breaker status.
    
    TODO: Implement in Phase 3
    """
    return {
        "message": "Not implemented yet",
        "endpoint": "/api/circuit-breaker/status"
    }

