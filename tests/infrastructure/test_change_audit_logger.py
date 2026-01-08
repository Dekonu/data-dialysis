"""Unit tests for ChangeAuditLogger."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from src.infrastructure.audit.change_audit_logger import ChangeAuditLogger
from src.domain.cdc_models import ChangeEvent


class TestChangeAuditLogger:
    """Test suite for ChangeAuditLogger."""
    
    def test_init(self):
        """Test ChangeAuditLogger initialization."""
        logger = ChangeAuditLogger()
        assert logger.get_log_count() == 0
        assert not logger.has_logs()
        assert logger._ingestion_id is None
        assert logger._source_adapter is None
    
    def test_set_ingestion_context(self):
        """Test setting ingestion context."""
        logger = ChangeAuditLogger()
        logger.set_ingestion_context(
            ingestion_id="test-ingestion-123",
            source_adapter="csv_ingester"
        )
        assert logger._ingestion_id == "test-ingestion-123"
        assert logger._source_adapter == "csv_ingester"
    
    def test_log_change_single(self):
        """Test logging a single change."""
        logger = ChangeAuditLogger()
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE"
        )
        
        assert logger.get_log_count() == 1
        assert logger.has_logs()
        
        logs = logger.get_logs()
        assert len(logs) == 1
        log_entry = logs[0]
        assert log_entry['table_name'] == "patients"
        assert log_entry['record_id'] == "P001"
        assert log_entry['field_name'] == "city"
        assert log_entry['old_value'] == "Boston"
        assert log_entry['new_value'] == "Cambridge"
        assert log_entry['change_type'] == "UPDATE"
        assert log_entry['changed_by'] == "system"
        assert 'change_id' in log_entry
        assert 'changed_at' in log_entry
    
    def test_log_change_with_context(self):
        """Test logging with ingestion context."""
        logger = ChangeAuditLogger()
        logger.set_ingestion_context(
            ingestion_id="ing-123",
            source_adapter="json_ingester"
        )
        
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE"
        )
        
        logs = logger.get_logs()
        assert logs[0]['ingestion_id'] == "ing-123"
        assert logs[0]['source_adapter'] == "json_ingester"
    
    def test_log_change_override_context(self):
        """Test that explicit ingestion_id/source_adapter override context."""
        logger = ChangeAuditLogger()
        logger.set_ingestion_context(
            ingestion_id="ing-123",
            source_adapter="json_ingester"
        )
        
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE",
            ingestion_id="ing-456",  # Override
            source_adapter="csv_ingester"  # Override
        )
        
        logs = logger.get_logs()
        assert logs[0]['ingestion_id'] == "ing-456"
        assert logs[0]['source_adapter'] == "csv_ingester"
    
    def test_log_change_insert(self):
        """Test logging an INSERT change."""
        logger = ChangeAuditLogger()
        logger.log_change(
            table_name="patients",
            record_id="P002",
            field_name="family_name",
            old_value=None,
            new_value="Smith",
            change_type="INSERT"
        )
        
        logs = logger.get_logs()
        assert logs[0]['change_type'] == "INSERT"
        assert logs[0]['old_value'] is None
        assert logs[0]['new_value'] == "Smith"
    
    def test_log_change_delete(self):
        """Test logging a DELETE change."""
        logger = ChangeAuditLogger()
        logger.log_change(
            table_name="patients",
            record_id="P003",
            field_name="status",
            old_value="active",
            new_value=None,
            change_type="DELETE"
        )
        
        logs = logger.get_logs()
        assert logs[0]['change_type'] == "DELETE"
        assert logs[0]['old_value'] == "active"
        assert logs[0]['new_value'] is None
    
    def test_log_change_event(self):
        """Test logging a ChangeEvent object."""
        logger = ChangeAuditLogger()
        
        change_event = ChangeEvent(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE",
            ingestion_id="ing-123",
            source_adapter="csv_ingester"
        )
        
        logger.log_change_event(change_event)
        
        assert logger.get_log_count() == 1
        logs = logger.get_logs()
        assert logs[0]['table_name'] == "patients"
        assert logs[0]['record_id'] == "P001"
        assert logs[0]['field_name'] == "city"
        assert logs[0]['old_value'] == "Boston"
        assert logs[0]['new_value'] == "Cambridge"
        assert logs[0]['change_type'] == "UPDATE"
        assert logs[0]['ingestion_id'] == "ing-123"
        assert logs[0]['source_adapter'] == "csv_ingester"
    
    def test_log_change_event_with_context(self):
        """Test logging ChangeEvent with context override."""
        logger = ChangeAuditLogger()
        logger.set_ingestion_context(
            ingestion_id="context-ing-123",
            source_adapter="context-adapter"
        )
        
        change_event = ChangeEvent(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE"
            # No ingestion_id or source_adapter in event
        )
        
        logger.log_change_event(change_event)
        
        logs = logger.get_logs()
        assert logs[0]['ingestion_id'] == "context-ing-123"
        assert logs[0]['source_adapter'] == "context-adapter"
    
    def test_log_changes_batch(self):
        """Test batch logging of multiple change events."""
        logger = ChangeAuditLogger()
        
        change_events = [
            ChangeEvent(
                table_name="patients",
                record_id="P001",
                field_name="city",
                old_value="Boston",
                new_value="Cambridge",
                change_type="UPDATE"
            ),
            ChangeEvent(
                table_name="patients",
                record_id="P002",
                field_name="phone",
                old_value="555-0101",
                new_value="555-0202",
                change_type="UPDATE"
            ),
            ChangeEvent(
                table_name="patients",
                record_id="P003",
                field_name="family_name",
                old_value=None,
                new_value="Johnson",
                change_type="INSERT"
            )
        ]
        
        logger.log_changes_batch(change_events)
        
        assert logger.get_log_count() == 3
        logs = logger.get_logs()
        assert len(logs) == 3
        assert logs[0]['record_id'] == "P001"
        assert logs[1]['record_id'] == "P002"
        assert logs[2]['record_id'] == "P003"
    
    def test_log_changes_batch_with_override(self):
        """Test batch logging with ingestion context override."""
        logger = ChangeAuditLogger()
        logger.set_ingestion_context(
            ingestion_id="context-ing",
            source_adapter="context-adapter"
        )
        
        change_events = [
            ChangeEvent(
                table_name="patients",
                record_id="P001",
                field_name="city",
                old_value="Boston",
                new_value="Cambridge",
                change_type="UPDATE"
            )
        ]
        
        # Override context for this batch
        logger.log_changes_batch(
            change_events,
            ingestion_id="batch-ing",
            source_adapter="batch-adapter"
        )
        
        logs = logger.get_logs()
        assert logs[0]['ingestion_id'] == "batch-ing"
        assert logs[0]['source_adapter'] == "batch-adapter"
        
        # Context should be restored
        assert logger._ingestion_id == "context-ing"
        assert logger._source_adapter == "context-adapter"
    
    def test_get_logs_returns_copy(self):
        """Test that get_logs() returns a copy, not the original list."""
        logger = ChangeAuditLogger()
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE"
        )
        
        logs1 = logger.get_logs()
        logs2 = logger.get_logs()
        
        # Should be different objects
        assert logs1 is not logs2
        # But should have same content
        assert len(logs1) == len(logs2)
        assert logs1[0]['record_id'] == logs2[0]['record_id']
        
        # Modifying one shouldn't affect the other
        logs1.append({'test': 'data'})
        assert len(logger.get_logs()) == 1  # Original unchanged
    
    def test_clear_logs(self):
        """Test clearing logged events."""
        logger = ChangeAuditLogger()
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE"
        )
        
        assert logger.get_log_count() == 1
        assert logger.has_logs()
        
        logger.clear_logs()
        
        assert logger.get_log_count() == 0
        assert not logger.has_logs()
        assert len(logger.get_logs()) == 0
    
    def test_get_log_count(self):
        """Test getting log count."""
        logger = ChangeAuditLogger()
        assert logger.get_log_count() == 0
        
        for i in range(5):
            logger.log_change(
                table_name="patients",
                record_id=f"P{i:03d}",
                field_name="city",
                old_value="Boston",
                new_value="Cambridge",
                change_type="UPDATE"
            )
        
        assert logger.get_log_count() == 5
    
    def test_has_logs(self):
        """Test checking if logs exist."""
        logger = ChangeAuditLogger()
        assert not logger.has_logs()
        
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE"
        )
        
        assert logger.has_logs()
        
        logger.clear_logs()
        assert not logger.has_logs()
    
    def test_log_change_with_changed_by(self):
        """Test logging with custom changed_by value."""
        logger = ChangeAuditLogger()
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE",
            changed_by="admin_user"
        )
        
        logs = logger.get_logs()
        assert logs[0]['changed_by'] == "admin_user"
    
    def test_log_change_complex_values(self):
        """Test logging with complex values (lists, etc.)."""
        logger = ChangeAuditLogger()
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="given_names",
            old_value='["John"]',
            new_value='["John", "Michael"]',
            change_type="UPDATE"
        )
        
        logs = logger.get_logs()
        assert logs[0]['old_value'] == '["John"]'
        assert logs[0]['new_value'] == '["John", "Michael"]'
    
    def test_multiple_changes_same_record(self):
        """Test logging multiple changes for the same record."""
        logger = ChangeAuditLogger()
        
        # Log multiple field changes for same record
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="city",
            old_value="Boston",
            new_value="Cambridge",
            change_type="UPDATE"
        )
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="postal_code",
            old_value="02101",
            new_value="02139",
            change_type="UPDATE"
        )
        logger.log_change(
            table_name="patients",
            record_id="P001",
            field_name="phone",
            old_value="555-0101",
            new_value="555-9999",
            change_type="UPDATE"
        )
        
        assert logger.get_log_count() == 3
        logs = logger.get_logs()
        assert all(log['record_id'] == "P001" for log in logs)
        assert {log['field_name'] for log in logs} == {"city", "postal_code", "phone"}
