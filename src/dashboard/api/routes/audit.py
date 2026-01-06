"""Audit log endpoints for dashboard API.

This module provides endpoints for querying audit logs and redaction logs.
"""

import csv
import io
import json
from datetime import datetime, timezone
from typing import Iterator, Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from src.dashboard.api.dependencies import StorageDep
from src.dashboard.services.audit_service import AuditService

router = APIRouter(prefix="/api", tags=["audit"])


@router.get("/audit-logs")
async def get_audit_logs(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    severity: Optional[str] = Query(None, description="Filter by severity (INFO, WARNING, ERROR, CRITICAL)"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    start_date: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter by end date (ISO format)"),
    source_adapter: Optional[str] = Query(None, description="Filter by source adapter"),
    sort_by: str = Query("event_timestamp", description="Field to sort by"),
    sort_order: str = Query("DESC", pattern="^(ASC|DESC)$", description="Sort order"),
    storage: StorageDep = None
):
    """Get audit logs with filtering and pagination.
    
    Returns paginated audit log entries with optional filtering by:
    - Severity level
    - Event type
    - Date range
    - Source adapter
    
    Supports sorting and pagination.
    """
    service = AuditService(storage)
    
    # Parse dates if provided
    start_dt = None
    end_dt = None
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid start_date format: {start_date}")
    
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid end_date format: {end_date}")
    
    result = service.get_audit_logs(
        limit=limit,
        offset=offset,
        severity=severity,
        event_type=event_type,
        start_date=start_dt,
        end_date=end_dt,
        source_adapter=source_adapter,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    if result.is_success():
        return result.value
    else:
        raise HTTPException(status_code=500, detail=str(result.error))


@router.get("/redaction-logs")
async def get_redaction_logs(
    field_name: Optional[str] = Query(None, description="Filter by field name"),
    time_range: str = Query("24h", pattern="^(1h|24h|7d|30d)$", description="Time range for filtering"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    rule_triggered: Optional[str] = Query(None, description="Filter by redaction rule"),
    source_adapter: Optional[str] = Query(None, description="Filter by source adapter"),
    ingestion_id: Optional[str] = Query(None, description="Filter by ingestion ID"),
    sort_by: str = Query("timestamp", description="Field to sort by"),
    sort_order: str = Query("DESC", pattern="^(ASC|DESC)$", description="Sort order"),
    storage: StorageDep = None
):
    """Get redaction logs with filtering.
    
    Returns paginated redaction log entries with optional filtering by:
    - Field name
    - Time range (1h, 24h, 7d, 30d)
    - Redaction rule
    - Source adapter
    - Ingestion ID
    
    Includes summary statistics grouped by field, rule, and adapter.
    """
    service = AuditService(storage)
    
    result = service.get_redaction_logs(
        field_name=field_name,
        time_range=time_range,
        limit=limit,
        offset=offset,
        rule_triggered=rule_triggered,
        source_adapter=source_adapter,
        ingestion_id=ingestion_id,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    if result.is_success():
        return result.value
    else:
        raise HTTPException(status_code=500, detail=str(result.error))


@router.get("/audit-logs/export")
async def export_audit_logs(
    format: str = Query("json", pattern="^(json|csv)$", description="Export format"),
    severity: Optional[str] = Query(None, description="Filter by severity"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    start_date: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter by end date (ISO format)"),
    storage: StorageDep = None
):
    """Export audit logs in JSON or CSV format.
    
    Returns all matching audit logs (no pagination limit) in the requested format.
    """
    service = AuditService(storage)
    
    # Parse dates if provided
    start_dt = None
    end_dt = None
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid start_date format: {start_date}")
    
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid end_date format: {end_date}")
    
    # Get all logs (large limit)
    result = service.get_audit_logs(
        limit=10000,  # Large limit for export
        offset=0,
        severity=severity,
        event_type=event_type,
        start_date=start_dt,
        end_date=end_dt,
        sort_by="event_timestamp",
        sort_order="DESC"
    )
    
    if not result.is_success():
        raise HTTPException(status_code=500, detail=str(result.error))
    
    response_data = result.value
    
    if format == "json":
        # Convert datetime objects to ISO strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        json_data = json.dumps(
            [log.model_dump() for log in response_data.logs],
            default=serialize_datetime,
            indent=2
        )
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return Response(
            content=json_data,
            media_type="application/json",
            headers={
                "Content-Disposition": f'attachment; filename="audit_logs_{timestamp}.json"'
            }
        )
    
    elif format == "csv":
        def generate_csv() -> Iterator[str]:
            """Generator function to stream CSV rows incrementally.
            
            This avoids loading the entire CSV into memory, making it
            memory-efficient for large exports.
            
            Yields:
                str: CSV rows as strings (including newlines)
            """
            # Create a StringIO buffer for CSV writer
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            
            # Write header row
            writer.writerow([
                "audit_id", "event_type", "event_timestamp", "record_id",
                "transformation_hash", "details", "source_adapter", "severity"
            ])
            yield buffer.getvalue()
            buffer.seek(0)
            buffer.truncate(0)
            
            # Write data rows incrementally
            for log in response_data.logs:
                details_str = ""
                if log.details:
                    details_str = json.dumps(log.details)
                
                writer.writerow([
                    log.audit_id,
                    log.event_type,
                    log.event_timestamp.isoformat() if log.event_timestamp else "",
                    log.record_id or "",
                    log.transformation_hash or "",
                    details_str,
                    log.source_adapter or "",
                    log.severity or ""
                ])
                
                # Yield the row and clear buffer for next iteration
                yield buffer.getvalue()
                buffer.seek(0)
                buffer.truncate(0)
            
            buffer.close()
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return StreamingResponse(
            generate_csv(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="audit_logs_{timestamp}.csv"'
            }
        )

