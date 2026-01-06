"""Audit log service for dashboard API.

This service provides methods to query audit logs and redaction logs
with filtering, pagination, and sorting capabilities.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from src.domain.ports import Result, StoragePort
from src.dashboard.models.audit import (
    AuditLogEntry,
    AuditLogsResponse,
    PaginationMeta,
    RedactionLogEntry,
    RedactionLogsResponse,
)

logger = logging.getLogger(__name__)


class AuditService:
    """Service for querying audit logs and redaction logs."""
    
    def __init__(self, storage: StoragePort):
        """Initialize AuditService.
        
        Parameters:
            storage: Storage adapter instance
        """
        self.storage = storage
    
    def get_audit_logs(
        self,
        limit: int = 100,
        offset: int = 0,
        severity: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source_adapter: Optional[str] = None,
        sort_by: str = "event_timestamp",
        sort_order: str = "DESC"
    ) -> Result[AuditLogsResponse]:
        """Get audit logs with filtering and pagination.
        
        Parameters:
            limit: Maximum number of records to return (1-1000)
            offset: Number of records to skip
            severity: Filter by severity (INFO, WARNING, ERROR, CRITICAL)
            event_type: Filter by event type
            start_date: Filter by start date (inclusive)
            end_date: Filter by end date (inclusive)
            source_adapter: Filter by source adapter
            sort_by: Field to sort by (default: event_timestamp)
            sort_order: Sort order (ASC or DESC)
            
        Returns:
            Result containing AuditLogsResponse or error
        """
        try:
            # Initialize schema if needed
            init_result = self.storage.initialize_schema()
            if not init_result.is_success():
                return init_result
            
            # Build query
            query = "SELECT * FROM audit_log WHERE 1=1"
            params = []
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            if start_date:
                query += " AND event_timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND event_timestamp <= ?"
                params.append(end_date)
            
            if source_adapter:
                query += " AND source_adapter = ?"
                params.append(source_adapter)
            
            # Validate sort_by to prevent SQL injection
            # Use a set to ensure we only use predefined, safe column names
            allowed_sort_fields = {
                "event_timestamp", "event_type", "severity", 
                "source_adapter", "record_id"
            }
            # Only use the sort_by value if it's in the allowed set, otherwise use default
            safe_sort_by = sort_by if sort_by in allowed_sort_fields else "event_timestamp"
            
            # Validate sort_order
            sort_order = sort_order.upper()
            if sort_order not in ["ASC", "DESC"]:
                sort_order = "DESC"
            
            # Get connection
            if not hasattr(self.storage, '_get_connection'):
                return Result.failure_result(
                    Exception("Storage adapter does not support query operations"),
                    error_type="AuditServiceError"
                )
            
            conn = self.storage._get_connection()
            
            # Get total count
            count_query = query.replace("SELECT *", "SELECT COUNT(*)")
            count_result = conn.execute(count_query, params).fetchone()
            total = count_result[0] if count_result else 0
            
            # Add sorting and pagination
            # Use validated column name from allowlist
            query += f" ORDER BY {safe_sort_by} {sort_order}"
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            result = conn.execute(query, params).fetchall()
            
            # Convert to AuditLogEntry objects
            logs = []
            if result:
                columns = [
                    "audit_id", "event_type", "event_timestamp", "record_id",
                    "transformation_hash", "details", "source_adapter", "severity"
                ]
                
                for row in result:
                    # Handle different database adapters (DuckDB vs PostgreSQL)
                    if isinstance(row, (list, tuple)):
                        row_dict = dict(zip(columns, row))
                    else:
                        row_dict = dict(row)
                    
                    # Parse details JSON if it's a string
                    details = row_dict.get("details")
                    if isinstance(details, str):
                        import json
                        try:
                            details = json.loads(details)
                        except (json.JSONDecodeError, TypeError):
                            details = None
                    
                    logs.append(AuditLogEntry(
                        audit_id=row_dict.get("audit_id", ""),
                        event_type=row_dict.get("event_type", ""),
                        event_timestamp=row_dict.get("event_timestamp"),
                        record_id=row_dict.get("record_id"),
                        transformation_hash=row_dict.get("transformation_hash"),
                        details=details,
                        source_adapter=row_dict.get("source_adapter"),
                        severity=row_dict.get("severity")
                    ))
            
            # Build pagination metadata
            pagination = PaginationMeta(
                total=total,
                limit=limit,
                offset=offset,
                has_next=(offset + limit) < total,
                has_previous=offset > 0
            )
            
            return Result.success_result(AuditLogsResponse(
                logs=logs,
                pagination=pagination
            ))
            
        except Exception as e:
            error_msg = f"Failed to get audit logs: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                Exception(error_msg),
                error_type="AuditServiceError"
            )
    
    def get_redaction_logs(
        self,
        field_name: Optional[str] = None,
        time_range: str = "24h",
        limit: int = 100,
        offset: int = 0,
        rule_triggered: Optional[str] = None,
        source_adapter: Optional[str] = None,
        ingestion_id: Optional[str] = None,
        sort_by: str = "timestamp",
        sort_order: str = "DESC"
    ) -> Result[RedactionLogsResponse]:
        """Get redaction logs with filtering and pagination.
        
        Parameters:
            field_name: Filter by field name
            time_range: Time range (1h, 24h, 7d, 30d)
            limit: Maximum number of records to return (1-1000)
            offset: Number of records to skip
            rule_triggered: Filter by redaction rule
            source_adapter: Filter by source adapter
            ingestion_id: Filter by ingestion ID
            sort_by: Field to sort by (default: timestamp)
            sort_order: Sort order (ASC or DESC)
            
        Returns:
            Result containing RedactionLogsResponse or error
        """
        try:
            # Initialize schema if needed
            init_result = self.storage.initialize_schema()
            if not init_result.is_success():
                return init_result
            
            # Calculate time range
            now = datetime.now(timezone.utc)
            time_deltas = {
                "1h": timedelta(hours=1),
                "24h": timedelta(hours=24),
                "7d": timedelta(days=7),
                "30d": timedelta(days=30)
            }
            start_date = now - time_deltas.get(time_range, timedelta(hours=24))
            
            # Build query
            query = "SELECT * FROM logs WHERE timestamp >= ?"
            params = [start_date]
            
            if field_name:
                query += " AND field_name = ?"
                params.append(field_name)
            
            if rule_triggered:
                query += " AND rule_triggered = ?"
                params.append(rule_triggered)
            
            if source_adapter:
                query += " AND source_adapter = ?"
                params.append(source_adapter)
            
            if ingestion_id:
                query += " AND ingestion_id = ?"
                params.append(ingestion_id)
            
            # Validate sort_by
            # Use a set to ensure we only use predefined, safe column names
            allowed_sort_fields = {
                "timestamp", "field_name", "rule_triggered",
                "source_adapter", "record_id"
            }
            # Only use the sort_by value if it's in the allowed set, otherwise use default
            safe_sort_by = sort_by if sort_by in allowed_sort_fields else "timestamp"
            
            # Validate sort_order
            sort_order = sort_order.upper()
            if sort_order not in ["ASC", "DESC"]:
                sort_order = "DESC"
            
            # Get connection
            if not hasattr(self.storage, '_get_connection'):
                return Result.failure_result(
                    Exception("Storage adapter does not support query operations"),
                    error_type="AuditServiceError"
                )
            
            conn = self.storage._get_connection()
            
            # Get total count
            count_query = query.replace("SELECT *", "SELECT COUNT(*)")
            count_result = conn.execute(count_query, params).fetchone()
            total = count_result[0] if count_result else 0
            
            # Get summary statistics
            summary_query = """
                SELECT 
                    COUNT(*) as total,
                    field_name,
                    rule_triggered,
                    source_adapter
                FROM logs
                WHERE timestamp >= ?
            """
            summary_params = [start_date]
            
            if field_name:
                summary_query += " AND field_name = ?"
                summary_params.append(field_name)
            
            summary_query += " GROUP BY field_name, rule_triggered, source_adapter"
            
            summary_result = conn.execute(summary_query, summary_params).fetchall()
            summary_stats = {
                "total_redactions": total,
                "by_field": {},
                "by_rule": {},
                "by_adapter": {}
            }
            
            if summary_result:
                for row in summary_result:
                    if isinstance(row, (list, tuple)):
                        count, field, rule, adapter = row
                    else:
                        count = row.get("total", 0)
                        field = row.get("field_name")
                        rule = row.get("rule_triggered")
                        adapter = row.get("source_adapter")
                    
                    if field:
                        summary_stats["by_field"][field] = summary_stats["by_field"].get(field, 0) + count
                    if rule:
                        summary_stats["by_rule"][rule] = summary_stats["by_rule"].get(rule, 0) + count
                    if adapter:
                        summary_stats["by_adapter"][adapter] = summary_stats["by_adapter"].get(adapter, 0) + count
            
            # Add sorting and pagination
            # Use validated column name from allowlist
            query += f" ORDER BY {safe_sort_by} {sort_order}"
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            # Execute query
            result = conn.execute(query, params).fetchall()
            
            # Convert to RedactionLogEntry objects
            logs = []
            if result:
                columns = [
                    "log_id", "field_name", "original_hash", "timestamp",
                    "rule_triggered", "record_id", "source_adapter",
                    "ingestion_id", "redacted_value", "original_value_length"
                ]
                
                for row in result:
                    if isinstance(row, (list, tuple)):
                        row_dict = dict(zip(columns, row))
                    else:
                        row_dict = dict(row)
                    
                    logs.append(RedactionLogEntry(
                        log_id=row_dict.get("log_id", ""),
                        field_name=row_dict.get("field_name", ""),
                        original_hash=row_dict.get("original_hash", ""),
                        timestamp=row_dict.get("timestamp"),
                        rule_triggered=row_dict.get("rule_triggered", ""),
                        record_id=row_dict.get("record_id"),
                        source_adapter=row_dict.get("source_adapter"),
                        ingestion_id=row_dict.get("ingestion_id"),
                        redacted_value=row_dict.get("redacted_value"),
                        original_value_length=row_dict.get("original_value_length")
                    ))
            
            # Build pagination metadata
            pagination = PaginationMeta(
                total=total,
                limit=limit,
                offset=offset,
                has_next=(offset + limit) < total,
                has_previous=offset > 0
            )
            
            return Result.success_result(RedactionLogsResponse(
                logs=logs,
                pagination=pagination,
                summary=summary_stats
            ))
            
        except Exception as e:
            error_msg = f"Failed to get redaction logs: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                Exception(error_msg),
                error_type="AuditServiceError"
            )

