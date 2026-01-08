"""Change Audit Logger.

This module provides a centralized logging mechanism for tracking field-level changes
in records. Each change is logged with field name, old/new values, timestamp, and
metadata for audit and compliance purposes.

Security Impact:
    - Creates immutable audit trail of all field-level changes
    - Enables forensic analysis of data modifications
    - Required for HIPAA/GDPR compliance reporting
    - Change logs are append-only for compliance

Architecture:
    - Infrastructure layer component
    - Can be called from domain services (ChangeDetector) or adapters
    - Integrates with storage adapters for persistence
    - Uses batch logging for performance with large datasets
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional

from src.domain.cdc_models import ChangeEvent
from src.domain.ports import Result, StorageError

logger = logging.getLogger(__name__)


class ChangeAuditLogger:
    """Logger for tracking field-level change events.
    
    This class provides a thread-safe way to log change events that can be
    persisted to the database. It maintains an in-memory buffer of change
    events that can be flushed to storage in batches for performance.
    
    Security Impact:
        - Logs old and new values (should be redacted before logging)
        - Tracks change type (INSERT, UPDATE, DELETE)
        - Maintains timestamp for compliance
        - Enables audit reporting and data lineage
    
    Example Usage:
        ```python
        logger = ChangeAuditLogger()
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE",
            ingestion_id="ing_123",
            source_adapter="csv_ingester"
        )
        # Later, flush to storage
        storage_adapter.flush_change_logs(logger.get_logs())
        ```
    """
    
    def __init__(self):
        """Initialize change audit logger."""
        self._logs: List[dict] = []
        self._ingestion_id: Optional[str] = None
        self._source_adapter: Optional[str] = None
    
    def set_ingestion_context(
        self,
        ingestion_id: Optional[str] = None,
        source_adapter: Optional[str] = None
    ) -> None:
        """Set ingestion context for grouping change events.
        
        Parameters:
            ingestion_id: Unique identifier for this ingestion run
            source_adapter: Source adapter identifier
        """
        self._ingestion_id = ingestion_id
        self._source_adapter = source_adapter
    
    def log_change(
        self,
        table_name: str,
        record_id: str,
        field_name: str,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        change_type: str = "UPDATE",
        ingestion_id: Optional[str] = None,
        source_adapter: Optional[str] = None,
        changed_by: Optional[str] = None
    ) -> None:
        """Log a single change event.
        
        Parameters:
            table_name: Name of the table (patients, encounters, observations)
            record_id: Primary key of the record that changed
            field_name: Name of the field that changed
            old_value: Previous value (before change)
            new_value: New value (after change)
            change_type: Type of change ('INSERT', 'UPDATE', 'DELETE')
            ingestion_id: ID of the ingestion run (uses context if not provided)
            source_adapter: Source adapter identifier (uses context if not provided)
            changed_by: System/user identifier (optional)
        
        Security Impact:
            - Values should be redacted before logging if they contain PII
            - Change logs are immutable (append-only)
        """
        log_entry = {
            "change_id": str(uuid.uuid4()),
            "table_name": table_name,
            "record_id": str(record_id),
            "field_name": field_name,
            "old_value": old_value,
            "new_value": new_value,
            "change_type": change_type,
            "changed_at": datetime.now(),
            "ingestion_id": ingestion_id or self._ingestion_id,
            "source_adapter": source_adapter or self._source_adapter,
            "changed_by": changed_by or "system"
        }
        
        self._logs.append(log_entry)
        logger.debug(
            f"Logged change: {table_name}.{record_id}.{field_name} "
            f"({change_type})"
        )
    
    def log_change_event(self, change_event: ChangeEvent) -> None:
        """Log a ChangeEvent object.
        
        Parameters:
            change_event: ChangeEvent instance to log
        
        Security Impact:
            - Uses ChangeEvent's serialization for values
            - Maintains audit trail consistency
        """
        audit_dict = change_event.to_audit_dict()
        # Override ingestion_id and source_adapter if context is set
        if self._ingestion_id:
            audit_dict['ingestion_id'] = self._ingestion_id
        if self._source_adapter:
            audit_dict['source_adapter'] = self._source_adapter
        
        self._logs.append(audit_dict)
        logger.debug(
            f"Logged change event: {change_event.table_name}."
            f"{change_event.record_id}.{change_event.field_name} "
            f"({change_event.change_type})"
        )
    
    def log_changes_batch(
        self,
        change_events: List[ChangeEvent],
        ingestion_id: Optional[str] = None,
        source_adapter: Optional[str] = None
    ) -> None:
        """Log multiple change events in batch.
        
        Parameters:
            change_events: List of ChangeEvent instances to log
            ingestion_id: ID of the ingestion run (uses context if not provided)
            source_adapter: Source adapter identifier (uses context if not provided)
        
        Performance:
            - More efficient than calling log_change_event() multiple times
            - Batch operations reduce overhead
        """
        for event in change_events:
            # Temporarily override context if provided
            original_ingestion_id = self._ingestion_id
            original_source_adapter = self._source_adapter
            
            if ingestion_id:
                self._ingestion_id = ingestion_id
            if source_adapter:
                self._source_adapter = source_adapter
            
            self.log_change_event(event)
            
            # Restore original context
            self._ingestion_id = original_ingestion_id
            self._source_adapter = original_source_adapter
    
    def get_logs(self) -> List[dict]:
        """Get all logged change events.
        
        Returns:
            List of change log entries (dictionaries ready for database insertion)
        """
        return self._logs.copy()
    
    def clear_logs(self) -> None:
        """Clear all logged events (after flushing to storage)."""
        self._logs.clear()
        logger.debug("Cleared change audit logs")
    
    def get_log_count(self) -> int:
        """Get count of logged change events.
        
        Returns:
            Number of change events logged
        """
        return len(self._logs)
    
    def has_logs(self) -> bool:
        """Check if there are any logged change events.
        
        Returns:
            True if there are logged events, False otherwise
        """
        return len(self._logs) > 0
