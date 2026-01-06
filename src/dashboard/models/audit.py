"""Pydantic models for audit log endpoints.

This module defines the response models for audit log and redaction log queries.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AuditLogEntry(BaseModel):
    """Single audit log entry."""
    
    audit_id: str = Field(..., description="Unique audit log identifier")
    event_type: str = Field(..., description="Type of event (e.g., BULK_PERSISTENCE, REDACTION)")
    event_timestamp: datetime = Field(..., description="When the event occurred")
    record_id: Optional[str] = Field(None, description="Related record identifier")
    transformation_hash: Optional[str] = Field(None, description="Hash of the transformation")
    details: Optional[dict[str, Any]] = Field(None, description="Additional event details (JSON)")
    source_adapter: Optional[str] = Field(None, description="Source adapter that generated the event")
    severity: Optional[str] = Field(None, description="Severity level (INFO, WARNING, ERROR, CRITICAL)")
    table_name: Optional[str] = Field(None, description="Name of the table affected (for persistence events)")
    row_count: Optional[int] = Field(None, description="Number of rows processed (None for singular records, integer for bulk)")


class PaginationMeta(BaseModel):
    """Pagination metadata."""
    
    total: int = Field(..., description="Total number of records")
    limit: int = Field(..., description="Number of records per page")
    offset: int = Field(..., description="Current offset")
    has_next: bool = Field(..., description="Whether there are more records")
    has_previous: bool = Field(..., description="Whether there are previous records")


class AuditLogsResponse(BaseModel):
    """Response model for audit logs query."""
    
    logs: list[AuditLogEntry] = Field(..., description="List of audit log entries")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")


class RedactionLogEntry(BaseModel):
    """Single redaction log entry."""
    
    log_id: str = Field(..., description="Unique redaction log identifier")
    field_name: str = Field(..., description="Name of the field that was redacted")
    original_hash: str = Field(..., description="Hash of the original value")
    timestamp: datetime = Field(..., description="When the redaction occurred")
    rule_triggered: str = Field(..., description="Redaction rule that was triggered")
    record_id: Optional[str] = Field(None, description="Related record identifier")
    source_adapter: Optional[str] = Field(None, description="Source adapter")
    ingestion_id: Optional[str] = Field(None, description="Ingestion batch identifier")
    redacted_value: Optional[str] = Field(None, description="The redacted value")
    original_value_length: Optional[int] = Field(None, description="Length of original value")


class RedactionLogsResponse(BaseModel):
    """Response model for redaction logs query."""
    
    logs: list[RedactionLogEntry] = Field(..., description="List of redaction log entries")
    pagination: PaginationMeta = Field(..., description="Pagination metadata")
    summary: dict[str, Any] = Field(..., description="Summary statistics")


class ExportFormat(BaseModel):
    """Export format options."""
    
    format: str = Field(..., description="Export format (json, csv)")

