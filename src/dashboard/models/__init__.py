"""Dashboard Pydantic models."""

from src.dashboard.models.health import HealthResponse, DatabaseHealth
from src.dashboard.models.metrics import (
    OverviewMetrics,
    SecurityMetrics,
    PerformanceMetrics,
    IngestionMetrics,
    RecordMetrics,
    RedactionSummary,
    CircuitBreakerStatus as MetricsCircuitBreakerStatus
)
from src.dashboard.models.audit import (
    AuditLogEntry,
    AuditLogsResponse,
    PaginationMeta,
    RedactionLogEntry,
    RedactionLogsResponse
)
from src.dashboard.models.circuit_breaker import CircuitBreakerStatus
from src.dashboard.models.websocket import (
    WebSocketMessage,
    ConnectionMessage,
    HeartbeatMessage,
    MetricsUpdateMessage,
    SecurityMetricsUpdateMessage,
    PerformanceMetricsUpdateMessage,
    CircuitBreakerUpdateMessage,
    ErrorMessage,
    WebSocketMessageType,
)

