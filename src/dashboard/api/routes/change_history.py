"""Change History API endpoints.

This module provides endpoints for querying change data capture (CDC) audit logs.
"""

import csv
import io
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import StreamingResponse

from src.dashboard.api.dependencies import StorageDep
from src.dashboard.services.change_history_service import ChangeHistoryService

router = APIRouter(prefix="/api", tags=["change-history"])


@router.get("/change-history")
async def get_change_history(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Number of records to skip"),
    table_name: Optional[str] = Query(None, description="Filter by table name (patients, encounters, observations)"),
    record_id: Optional[str] = Query(None, description="Filter by specific record ID"),
    field_name: Optional[str] = Query(None, description="Filter by field name"),
    change_type: Optional[str] = Query(None, description="Filter by change type (INSERT, UPDATE, DELETE)"),
    start_date: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter by end date (ISO format)"),
    ingestion_id: Optional[str] = Query(None, description="Filter by ingestion ID"),
    source_adapter: Optional[str] = Query(None, description="Filter by source adapter"),
    sort_by: str = Query("changed_at", description="Field to sort by"),
    sort_order: str = Query("DESC", pattern="^(ASC|DESC)$", description="Sort order"),
    storage: StorageDep = None
):
    """Get change history with filtering and pagination.
    
    Returns paginated change audit log entries with optional filtering by:
    - Table name
    - Record ID
    - Field name
    - Change type (INSERT, UPDATE, DELETE)
    - Date range
    - Ingestion ID
    - Source adapter
    
    Supports sorting and pagination.
    """
    service = ChangeHistoryService(storage)
    
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
    
    result = service.get_change_history(
        limit=limit,
        offset=offset,
        table_name=table_name,
        record_id=record_id,
        field_name=field_name,
        change_type=change_type,
        start_date=start_dt,
        end_date=end_dt,
        ingestion_id=ingestion_id,
        source_adapter=source_adapter,
        sort_by=sort_by,
        sort_order=sort_order
    )
    
    if result.is_success():
        return result.value
    else:
        raise HTTPException(status_code=500, detail=str(result.error))


@router.get("/change-history/summary")
async def get_change_summary(
    time_range: str = Query("24h", pattern="^(1h|24h|7d|30d)$", description="Time range for summary"),
    table_name: Optional[str] = Query(None, description="Filter by table name"),
    storage: StorageDep = None
):
    """Get summary of changes for a time range.
    
    Returns aggregated statistics about changes including:
    - Total number of changes
    - Number of unique records affected
    - Number of tables affected
    - Number of fields changed
    - Breakdown by change type (INSERT, UPDATE, DELETE)
    """
    service = ChangeHistoryService(storage)
    
    result = service.get_change_summary(
        time_range=time_range,
        table_name=table_name
    )
    
    if result.is_success():
        return result.value
    else:
        raise HTTPException(status_code=500, detail=str(result.error))


@router.get("/change-history/record/{table_name}/{record_id}")
async def get_record_change_history(
    table_name: str,
    record_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of changes to return"),
    storage: StorageDep = None
):
    """Get complete change history for a specific record.
    
    Returns all changes (field-level) for a specific record, ordered by timestamp.
    This is useful for viewing the complete audit trail of a single record.
    """
    service = ChangeHistoryService(storage)
    
    result = service.get_record_change_history(
        table_name=table_name,
        record_id=record_id,
        limit=limit
    )
    
    if result.is_success():
        return {"changes": result.value, "table_name": table_name, "record_id": record_id}
    else:
        raise HTTPException(status_code=500, detail=str(result.error))


@router.get("/change-history/export")
async def export_change_history(
    format: str = Query("csv", pattern="^(csv|json)$", description="Export format"),
    table_name: Optional[str] = Query(None, description="Filter by table name"),
    start_date: Optional[str] = Query(None, description="Filter by start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="Filter by end date (ISO format)"),
    change_type: Optional[str] = Query(None, description="Filter by change type"),
    storage: StorageDep = None
):
    """Export change history to CSV or JSON.
    
    Exports change audit logs in the specified format with optional filtering.
    Useful for compliance reporting and data analysis.
    """
    service = ChangeHistoryService(storage)
    
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
    
    # Get all matching records (with high limit for export)
    result = service.get_change_history(
        limit=10000,  # High limit for exports
        offset=0,
        table_name=table_name,
        start_date=start_dt,
        end_date=end_dt,
        change_type=change_type,
        sort_by="changed_at",
        sort_order="DESC"
    )
    
    if not result.is_success():
        raise HTTPException(status_code=500, detail=str(result.error))
    
    changes = result.value.get("changes", [])
    
    if format == "csv":
        # Generate CSV
        output = io.StringIO()
        if changes:
            fieldnames = [
                'change_id', 'table_name', 'record_id', 'field_name',
                'old_value', 'new_value', 'change_type', 'changed_at',
                'ingestion_id', 'source_adapter', 'changed_by'
            ]
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(changes)
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=change_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )
    else:  # JSON
        import json
        return Response(
            content=json.dumps(changes, indent=2, default=str),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename=change_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            }
        )
