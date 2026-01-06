"""WebSocket message models for real-time updates.

This module defines Pydantic models for WebSocket messages sent between
the server and clients for real-time dashboard updates.
"""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.dashboard.models.circuit_breaker import CircuitBreakerStatus
from src.dashboard.models.metrics import OverviewMetrics, PerformanceMetrics, SecurityMetrics


class WebSocketMessage(BaseModel):
    """Base WebSocket message model."""
    
    type: str = Field(..., description="Message type identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(), description="Message timestamp")


class ConnectionMessage(WebSocketMessage):
    """Message sent when WebSocket connection is established."""
    
    type: Literal["connection"] = "connection"
    message: str = Field(..., description="Connection status message")
    server_time: datetime = Field(default_factory=lambda: datetime.now(), description="Server timestamp")


class HeartbeatMessage(WebSocketMessage):
    """Heartbeat message to keep connection alive."""
    
    type: Literal["heartbeat"] = "heartbeat"


class MetricsUpdateMessage(WebSocketMessage):
    """Message containing metrics update."""
    
    type: Literal["metrics_update"] = "metrics_update"
    data: OverviewMetrics = Field(..., description="Overview metrics data")


class SecurityMetricsUpdateMessage(WebSocketMessage):
    """Message containing security metrics update."""
    
    type: Literal["security_metrics_update"] = "security_metrics_update"
    data: SecurityMetrics = Field(..., description="Security metrics data")


class PerformanceMetricsUpdateMessage(WebSocketMessage):
    """Message containing performance metrics update."""
    
    type: Literal["performance_metrics_update"] = "performance_metrics_update"
    data: PerformanceMetrics = Field(..., description="Performance metrics data")


class CircuitBreakerUpdateMessage(WebSocketMessage):
    """Message containing circuit breaker status update."""
    
    type: Literal["circuit_breaker_update"] = "circuit_breaker_update"
    data: CircuitBreakerStatus = Field(..., description="Circuit breaker status")


class ErrorMessage(WebSocketMessage):
    """Error message sent to client."""
    
    type: Literal["error"] = "error"
    error: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Error type identifier")


# Union type for all possible WebSocket messages
WebSocketMessageType = (
    ConnectionMessage
    | HeartbeatMessage
    | MetricsUpdateMessage
    | SecurityMetricsUpdateMessage
    | PerformanceMetricsUpdateMessage
    | CircuitBreakerUpdateMessage
    | ErrorMessage
)

