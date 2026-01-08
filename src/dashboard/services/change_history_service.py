"""Change History Service.

This service provides methods for querying change audit logs (CDC) with
filtering, pagination, and search capabilities.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any

from src.domain.ports import Result, StoragePort
from src.dashboard.services.connection_helper import get_db_connection

logger = logging.getLogger(__name__)


class ChangeHistoryService:
    """Service for querying change audit logs."""
    
    def __init__(self, storage: StoragePort):
        """Initialize change history service.
        
        Parameters:
            storage: Storage adapter instance
        """
        self.storage = storage
    
    def _parse_time_range(self, time_range: str, end_time: datetime) -> datetime:
        """Parse time range string to start time."""
        time_range = time_range.lower()
        
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
            return end_time - timedelta(hours=hours)
        elif time_range.endswith('d'):
            days = int(time_range[:-1])
            return end_time - timedelta(days=days)
        else:
            logger.warning(f"Unknown time range format: {time_range}, defaulting to 24h")
            return end_time - timedelta(hours=24)
    
    def get_change_history(
        self,
        limit: int = 100,
        offset: int = 0,
        table_name: Optional[str] = None,
        record_id: Optional[str] = None,
        field_name: Optional[str] = None,
        change_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        ingestion_id: Optional[str] = None,
        source_adapter: Optional[str] = None,
        sort_by: str = "changed_at",
        sort_order: str = "DESC"
    ) -> Result[Dict[str, Any]]:
        """Get change history with filtering and pagination.
        
        Parameters:
            limit: Maximum number of records to return
            offset: Number of records to skip
            table_name: Filter by table name (patients, encounters, observations)
            record_id: Filter by specific record ID
            field_name: Filter by field name
            change_type: Filter by change type (INSERT, UPDATE, DELETE)
            start_date: Filter by start date
            end_date: Filter by end date
            ingestion_id: Filter by ingestion ID
            source_adapter: Filter by source adapter
            sort_by: Field to sort by (changed_at, table_name, record_id, field_name)
            sort_order: Sort order (ASC, DESC)
            
        Returns:
            Result containing paginated change history and total count
        """
        try:
            if not hasattr(self.storage, '_get_connection'):
                return Result.failure_result(
                    Exception("Storage adapter does not support query operations"),
                    error_type="ChangeHistoryError"
                )
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return Result.failure_result(
                        Exception("Storage adapter does not support query operations"),
                        error_type="ChangeHistoryError"
                    )
                
                # Build WHERE clause
                where_conditions = []
                params = []
                
                if table_name:
                    where_conditions.append("table_name = ?")
                    params.append(table_name)
                
                if record_id:
                    where_conditions.append("record_id = ?")
                    params.append(record_id)
                
                if field_name:
                    where_conditions.append("field_name = ?")
                    params.append(field_name)
                
                if change_type:
                    where_conditions.append("change_type = ?")
                    params.append(change_type)
                
                if start_date:
                    where_conditions.append("changed_at >= ?")
                    params.append(start_date)
                
                if end_date:
                    where_conditions.append("changed_at <= ?")
                    params.append(end_date)
                
                if ingestion_id:
                    where_conditions.append("ingestion_id = ?")
                    params.append(ingestion_id)
                
                if source_adapter:
                    where_conditions.append("source_adapter = ?")
                    params.append(source_adapter)
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                # Validate sort_by
                valid_sort_fields = ["changed_at", "table_name", "record_id", "field_name", "change_type"]
                if sort_by not in valid_sort_fields:
                    sort_by = "changed_at"
                
                # Validate sort_order
                if sort_order.upper() not in ["ASC", "DESC"]:
                    sort_order = "DESC"
                
                # Get total count
                count_query = f"SELECT COUNT(*) as total FROM change_audit_log WHERE {where_clause}"
                count_result = conn.execute(count_query, params).fetchone()
                total = count_result[0] if count_result and count_result[0] else 0
                
                # Get paginated results
                query = f"""
                    SELECT 
                        change_id,
                        table_name,
                        record_id,
                        field_name,
                        old_value,
                        new_value,
                        change_type,
                        changed_at,
                        ingestion_id,
                        source_adapter,
                        changed_by
                    FROM change_audit_log
                    WHERE {where_clause}
                    ORDER BY {sort_by} {sort_order}
                    LIMIT ? OFFSET ?
                """
                
                query_params = params + [limit, offset]
                results = conn.execute(query, query_params).fetchall()
                
                # Convert to list of dictionaries
                columns = [
                    'change_id', 'table_name', 'record_id', 'field_name',
                    'old_value', 'new_value', 'change_type', 'changed_at',
                    'ingestion_id', 'source_adapter', 'changed_by'
                ]
                
                changes = []
                for row in results:
                    change_dict = {}
                    for i, col in enumerate(columns):
                        value = row[i] if i < len(row) else None
                        # Convert datetime to ISO format string
                        if isinstance(value, datetime):
                            value = value.isoformat()
                        change_dict[col] = value
                    changes.append(change_dict)
                
                return Result.success_result({
                    "changes": changes,
                    "total": total,
                    "limit": limit,
                    "offset": offset
                })
                
        except Exception as e:
            logger.error(f"Failed to get change history: {str(e)}", exc_info=True)
            return Result.failure_result(
                e,
                error_type="ChangeHistoryError"
            )
    
    def get_change_summary(
        self,
        time_range: str = "24h",
        table_name: Optional[str] = None
    ) -> Result[Dict[str, Any]]:
        """Get summary of changes for a time range.
        
        Parameters:
            time_range: Time range string (1h, 24h, 7d, 30d)
            table_name: Optional table name filter
            
        Returns:
            Result containing change summary statistics
        """
        try:
            end_time = datetime.now(timezone.utc).replace(tzinfo=None)
            start_time = self._parse_time_range(time_range, end_time)
            
            if not hasattr(self.storage, '_get_connection'):
                return Result.failure_result(
                    Exception("Storage adapter does not support query operations"),
                    error_type="ChangeHistoryError"
                )
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return Result.failure_result(
                        Exception("Storage adapter does not support query operations"),
                        error_type="ChangeHistoryError"
                    )
                
                # Build WHERE clause
                where_conditions = ["changed_at >= ?", "changed_at <= ?"]
                params = [start_time, end_time]
                
                if table_name:
                    where_conditions.append("table_name = ?")
                    params.append(table_name)
                
                where_clause = " AND ".join(where_conditions)
                
                # Get summary statistics
                summary_query = f"""
                    SELECT 
                        COUNT(*) as total_changes,
                        COUNT(DISTINCT record_id) as unique_records,
                        COUNT(DISTINCT table_name) as tables_affected,
                        COUNT(DISTINCT field_name) as fields_changed,
                        COUNT(CASE WHEN change_type = 'INSERT' THEN 1 END) as inserts,
                        COUNT(CASE WHEN change_type = 'UPDATE' THEN 1 END) as updates,
                        COUNT(CASE WHEN change_type = 'DELETE' THEN 1 END) as deletes
                    FROM change_audit_log
                    WHERE {where_clause}
                """
                
                result = conn.execute(summary_query, params).fetchone()
                
                if result:
                    summary = {
                        "total_changes": result[0] or 0,
                        "unique_records": result[1] or 0,
                        "tables_affected": result[2] or 0,
                        "fields_changed": result[3] or 0,
                        "inserts": result[4] or 0,
                        "updates": result[5] or 0,
                        "deletes": result[6] or 0,
                        "time_range": time_range,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat()
                    }
                    return Result.success_result(summary)
                else:
                    return Result.success_result({
                        "total_changes": 0,
                        "unique_records": 0,
                        "tables_affected": 0,
                        "fields_changed": 0,
                        "inserts": 0,
                        "updates": 0,
                        "deletes": 0,
                        "time_range": time_range,
                        "start_time": start_time.isoformat(),
                        "end_time": end_time.isoformat()
                    })
                
        except Exception as e:
            logger.error(f"Failed to get change summary: {str(e)}", exc_info=True)
            return Result.failure_result(
                e,
                error_type="ChangeHistoryError"
            )
    
    def get_record_change_history(
        self,
        table_name: str,
        record_id: str,
        limit: int = 100
    ) -> Result[List[Dict[str, Any]]]:
        """Get complete change history for a specific record.
        
        Parameters:
            table_name: Name of the table
            record_id: Primary key of the record
            limit: Maximum number of changes to return
            
        Returns:
            Result containing list of changes for the record
        """
        result = self.get_change_history(
            limit=limit,
            offset=0,
            table_name=table_name,
            record_id=record_id,
            sort_by="changed_at",
            sort_order="DESC"
        )
        
        if result.is_success():
            return Result.success_result(result.value.get("changes", []))
        else:
            return result
