"""Change Data Capture (CDC) Models.

This module defines models for tracking field-level changes in records.
These models are used for audit trails, compliance reporting, and efficient updates.

Security Impact:
    - Change audit logs may contain PII (old/new values)
    - Ensure redaction is applied before logging changes
    - Change logs are immutable (append-only) for compliance

Architecture:
    - Pure domain models with zero infrastructure dependencies
    - Models are validated before use
    - Follows Hexagonal Architecture: Domain Core is isolated from Adapters
"""

import json
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChangeEvent(BaseModel):
    """Represents a single field-level change in a record.
    
    This model is used to track what changed, when, and from/to values
    for audit and compliance purposes.
    
    Parameters:
        table_name: Name of the table (patients, encounters, observations)
        record_id: Primary key of the record (patient_id, encounter_id, observation_id)
        field_name: Name of the field that changed
        old_value: Previous value (before change)
        new_value: New value (after change)
        change_type: Type of change ('INSERT', 'UPDATE', 'DELETE')
        changed_at: Timestamp when change occurred
        ingestion_id: ID of the ingestion run that caused this change
        source_adapter: Source adapter that provided the data
        changed_by: System/user identifier (optional)
    """
    
    table_name: str = Field(..., description="Name of the table")
    record_id: str = Field(..., description="Primary key of the record")
    field_name: str = Field(..., description="Name of the field that changed")
    old_value: Optional[Any] = Field(None, description="Previous value (before change)")
    new_value: Optional[Any] = Field(None, description="New value (after change)")
    change_type: str = Field(..., description="Type of change: INSERT, UPDATE, or DELETE")
    changed_at: datetime = Field(default_factory=datetime.now, description="Timestamp when change occurred")
    ingestion_id: Optional[str] = Field(None, description="ID of the ingestion run")
    source_adapter: Optional[str] = Field(None, description="Source adapter identifier")
    changed_by: Optional[str] = Field(None, description="System/user identifier")
    
    def to_audit_dict(self) -> dict:
        """Convert to dictionary for audit log insertion.
        
        Returns:
            Dictionary with serialized values suitable for database insertion
        """
        return {
            'change_id': str(uuid.uuid4()),
            'table_name': self.table_name,
            'record_id': self.record_id,
            'field_name': self.field_name,
            'old_value': self._serialize_value(self.old_value),
            'new_value': self._serialize_value(self.new_value),
            'change_type': self.change_type,
            'changed_at': self.changed_at,
            'ingestion_id': self.ingestion_id,
            'source_adapter': self.source_adapter,
            'changed_by': self.changed_by
        }
    
    def _serialize_value(self, value: Any) -> Optional[str]:
        """Serialize complex types to JSON string for database storage.
        
        Parameters:
            value: Value to serialize (can be None, str, list, dict, etc.)
            
        Returns:
            Serialized string representation or None
        """
        if value is None:
            return None
        
        # Handle complex types (lists, dicts)
        if isinstance(value, (list, dict)):
            try:
                return json.dumps(value)
            except (TypeError, ValueError):
                # Fallback to string representation if JSON serialization fails
                return str(value)
        
        # Handle pandas NaN/NaT
        try:
            import pandas as pd
            if pd.isna(value):
                return None
        except ImportError:
            pass
        
        # Convert to string for other types
        return str(value)
    
    model_config = {
        'frozen': False,  # Allow modification for ingestion_id, etc.
        'validate_assignment': True,
    }


class UpdateResult(BaseModel):
    """Result of a smart update operation with change detection.
    
    This model provides statistics about what was updated during an ingestion.
    
    Parameters:
        records_processed: Total number of records processed
        records_inserted: Number of new records inserted
        records_updated: Number of existing records updated
        records_unchanged: Number of records that had no changes
        fields_changed: Total number of fields that changed across all records
        changes_logged: Number of change events logged to audit trail
    """
    
    records_processed: int = Field(..., description="Total number of records processed")
    records_inserted: int = Field(0, description="Number of new records inserted")
    records_updated: int = Field(0, description="Number of existing records updated")
    records_unchanged: int = Field(0, description="Number of records with no changes")
    fields_changed: int = Field(0, description="Total number of fields that changed")
    changes_logged: int = Field(0, description="Number of change events logged")
    
    model_config = {
        'frozen': True,
    }
