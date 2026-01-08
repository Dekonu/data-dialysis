"""Unit tests for ChangeDetector service."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from src.domain.services.change_detector import ChangeDetector
from src.domain.cdc_models import ChangeEvent


class TestChangeDetector:
    """Test suite for ChangeDetector service."""
    
    def test_init(self):
        """Test ChangeDetector initialization."""
        detector = ChangeDetector(
            ingestion_id="test-ingestion-123",
            source_adapter="csv_ingester"
        )
        assert detector.ingestion_id == "test-ingestion-123"
        assert detector.source_adapter == "csv_ingester"
    
    def test_detect_changes_vectorized_no_changes(self):
        """Test change detection when no fields have changed."""
        detector = ChangeDetector()
        
        # Create merged DataFrame with no changes
        merged_df = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'family_name_old': ['Smith', 'Jones'],
            'family_name_new': ['Smith', 'Jones'],
            'given_names_old': [['John'], ['Jane']],
            'given_names_new': [['John'], ['Jane']],
            '_merge': ['both', 'both']
        })
        
        changes_df = detector.detect_changes_vectorized(
            merged_df, 'patients', 'patient_id'
        )
        
        assert changes_df.empty
    
    def test_detect_changes_vectorized_single_field_change(self):
        """Test change detection for a single field change."""
        detector = ChangeDetector()
        
        # Create merged DataFrame with one field changed
        merged_df = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'family_name_old': ['Smith', 'Jones'],
            'family_name_new': ['Smith', 'Johnson'],  # P002 changed
            'given_names_old': [['John'], ['Jane']],
            'given_names_new': [['John'], ['Jane']],
            '_merge': ['both', 'both']
        })
        
        changes_df = detector.detect_changes_vectorized(
            merged_df, 'patients', 'patient_id'
        )
        
        assert len(changes_df) == 1
        assert changes_df.iloc[0]['record_id'] == 'P002'
        assert changes_df.iloc[0]['field_name'] == 'family_name'
        assert changes_df.iloc[0]['old_value'] == 'Jones'
        assert changes_df.iloc[0]['new_value'] == 'Johnson'
        assert changes_df.iloc[0]['change_type'] == 'UPDATE'
    
    def test_detect_changes_vectorized_multiple_field_changes(self):
        """Test change detection for multiple field changes."""
        detector = ChangeDetector()
        
        # Create merged DataFrame with multiple fields changed
        merged_df = pd.DataFrame({
            'patient_id': ['P001'],
            'family_name_old': ['Smith'],
            'family_name_new': ['Smith-Jones'],  # Changed
            'given_names_old': [['John']],
            'given_names_new': [['John', 'Michael']],  # Changed
            'city_old': ['Boston'],
            'city_new': ['Boston'],  # Not changed
            '_merge': ['both']
        })
        
        changes_df = detector.detect_changes_vectorized(
            merged_df, 'patients', 'patient_id'
        )
        
        assert len(changes_df) == 2
        assert set(changes_df['field_name'].tolist()) == {'family_name', 'given_names'}
    
    def test_detect_changes_vectorized_nan_handling(self):
        """Test change detection handles NaN values correctly."""
        detector = ChangeDetector()
        
        # Create merged DataFrame with NaN values
        merged_df = pd.DataFrame({
            'observation_id': ['OBS001', 'OBS002'],
            'value_old': ['100', np.nan],
            'value_new': ['100', '200'],  # OBS002 changed from NaN to '200'
            'unit_old': [np.nan, 'mg/dL'],
            'unit_new': ['mg/dL', 'mg/dL'],  # OBS001 changed from NaN to 'mg/dL'
            '_merge': ['both', 'both']
        })
        
        changes_df = detector.detect_changes_vectorized(
            merged_df, 'observations', 'observation_id'
        )
        
        # Should detect changes where NaN -> value or value -> NaN
        assert len(changes_df) == 2
        assert set(changes_df['field_name'].tolist()) == {'value', 'unit'}
    
    def test_detect_changes_vectorized_none_to_value(self):
        """Test change detection for None to value transitions."""
        detector = ChangeDetector()
        
        merged_df = pd.DataFrame({
            'encounter_id': ['ENC001'],
            'diagnosis_codes_old': [None],
            'diagnosis_codes_new': [['E11.9', 'I10']],  # Changed from None to list
            '_merge': ['both']
        })
        
        changes_df = detector.detect_changes_vectorized(
            merged_df, 'encounters', 'encounter_id'
        )
        
        assert len(changes_df) == 1
        assert changes_df.iloc[0]['old_value'] is None
        assert changes_df.iloc[0]['new_value'] == ['E11.9', 'I10']
    
    def test_detect_changes_vectorized_empty_dataframe(self):
        """Test change detection with empty DataFrame."""
        detector = ChangeDetector()
        
        empty_df = pd.DataFrame()
        changes_df = detector.detect_changes_vectorized(
            empty_df, 'patients', 'patient_id'
        )
        
        assert changes_df.empty
        assert list(changes_df.columns) == [
            'table_name', 'record_id', 'field_name', 'old_value', 'new_value', 'change_type'
        ]
    
    def test_changes_df_to_events(self):
        """Test conversion of changes DataFrame to ChangeEvent objects."""
        detector = ChangeDetector(
            ingestion_id="test-123",
            source_adapter="csv_ingester"
        )
        
        changes_df = pd.DataFrame({
            'table_name': ['patients', 'patients'],
            'record_id': ['P001', 'P002'],
            'field_name': ['family_name', 'city'],
            'old_value': ['Smith', 'Boston'],
            'new_value': ['Smith-Jones', 'Cambridge'],
            'change_type': ['UPDATE', 'UPDATE']
        })
        
        events = detector.changes_df_to_events(changes_df)
        
        assert len(events) == 2
        assert all(isinstance(e, ChangeEvent) for e in events)
        assert events[0].ingestion_id == "test-123"
        assert events[0].source_adapter == "csv_ingester"
        assert events[0].table_name == "patients"
        assert events[0].record_id == "P001"
    
    def test_generate_insert_changes(self):
        """Test generation of INSERT change events for new records."""
        detector = ChangeDetector(
            ingestion_id="test-123",
            source_adapter="csv_ingester"
        )
        
        df = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'family_name': ['Smith', 'Jones'],
            'given_names': [['John'], ['Jane']],
            'city': ['Boston', 'New York']
        })
        
        events = detector.generate_insert_changes(df, 'patients', 'patient_id')
        
        # Should have 3 fields * 2 records = 6 events (excluding patient_id)
        assert len(events) == 6
        assert all(e.change_type == 'INSERT' for e in events)
        assert all(e.old_value is None for e in events)
        assert all(e.ingestion_id == "test-123" for e in events)
        
        # Check that all fields are covered
        field_names = {e.field_name for e in events}
        assert field_names == {'family_name', 'given_names', 'city'}
    
    def test_generate_insert_changes_empty_dataframe(self):
        """Test INSERT change generation with empty DataFrame."""
        detector = ChangeDetector()
        
        empty_df = pd.DataFrame(columns=['patient_id', 'family_name'])
        events = detector.generate_insert_changes(empty_df, 'patients', 'patient_id')
        
        assert events == []
    
    def test_values_equal_nan(self):
        """Test values_equal method with NaN values."""
        assert ChangeDetector.values_equal(np.nan, np.nan) is True
        assert ChangeDetector.values_equal(np.nan, 'value') is False
        assert ChangeDetector.values_equal('value', np.nan) is False
    
    def test_values_equal_none(self):
        """Test values_equal method with None values."""
        assert ChangeDetector.values_equal(None, None) is True
        assert ChangeDetector.values_equal(None, 'value') is False
        assert ChangeDetector.values_equal('value', None) is False
    
    def test_values_equal_arrays(self):
        """Test values_equal method with arrays/lists."""
        assert ChangeDetector.values_equal([1, 2, 3], [1, 2, 3]) is True
        assert ChangeDetector.values_equal([1, 2], [1, 2, 3]) is False
        assert ChangeDetector.values_equal([1, 2], [2, 1]) is False
    
    def test_values_equal_dicts(self):
        """Test values_equal method with dictionaries."""
        assert ChangeDetector.values_equal({'a': 1}, {'a': 1}) is True
        assert ChangeDetector.values_equal({'a': 1}, {'a': 2}) is False
        assert ChangeDetector.values_equal({'a': 1}, {'b': 1}) is False
    
    def test_values_equal_strings(self):
        """Test values_equal method with strings."""
        assert ChangeDetector.values_equal('test', 'test') is True
        assert ChangeDetector.values_equal('test', 'Test') is False
        assert ChangeDetector.values_equal('test1', 'test2') is False
    
    def test_to_audit_dict(self):
        """Test ChangeEvent.to_audit_dict() serialization."""
        event = ChangeEvent(
            table_name='patients',
            record_id='P001',
            field_name='family_name',
            old_value='Smith',
            new_value='Smith-Jones',
            change_type='UPDATE',
            ingestion_id='test-123',
            source_adapter='csv_ingester'
        )
        
        audit_dict = event.to_audit_dict()
        
        assert 'change_id' in audit_dict
        assert audit_dict['table_name'] == 'patients'
        assert audit_dict['record_id'] == 'P001'
        assert audit_dict['field_name'] == 'family_name'
        assert audit_dict['old_value'] == 'Smith'
        assert audit_dict['new_value'] == 'Smith-Jones'
        assert audit_dict['change_type'] == 'UPDATE'
        assert audit_dict['ingestion_id'] == 'test-123'
        assert audit_dict['source_adapter'] == 'csv_ingester'
    
    def test_to_audit_dict_complex_types(self):
        """Test ChangeEvent.to_audit_dict() with complex types (lists, dicts)."""
        event = ChangeEvent(
            table_name='patients',
            record_id='P001',
            field_name='given_names',
            old_value=['John'],
            new_value=['John', 'Michael'],
            change_type='UPDATE'
        )
        
        audit_dict = event.to_audit_dict()
        
        # Complex types should be serialized to JSON strings
        assert isinstance(audit_dict['old_value'], str)
        assert isinstance(audit_dict['new_value'], str)
        # Verify it's valid JSON
        import json
        assert json.loads(audit_dict['old_value']) == ['John']
        assert json.loads(audit_dict['new_value']) == ['John', 'Michael']
    
    def test_to_audit_dict_none_values(self):
        """Test ChangeEvent.to_audit_dict() with None values."""
        event = ChangeEvent(
            table_name='patients',
            record_id='P001',
            field_name='family_name',
            old_value=None,
            new_value='Smith',
            change_type='INSERT'
        )
        
        audit_dict = event.to_audit_dict()
        
        assert audit_dict['old_value'] is None
        assert audit_dict['new_value'] == 'Smith'
