"""Health check models for dashboard API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class DatabaseHealth(BaseModel):
    """Database health status model.
    
    Attributes:
        status: Connection status
        type: Database type (duckdb or postgresql)
        response_time_ms: Database response time in milliseconds (optional)
    """
    status: Literal["connected", "disconnected"]
    type: str
    response_time_ms: float | None = Field(None, description="Database response time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response model.
    
    Attributes:
        status: Overall system status
        timestamp: Current timestamp
        version: Application version
        database: Database health information
    """
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Current UTC timestamp")
    version: str = Field(default="1.0.0", description="Application version")
    database: DatabaseHealth

