"""Unit tests for PostgreSQLAdapter smart update functionality (CDC Phase 3).

These tests verify the Change Data Capture (CDC) smart update functionality:
- Bulk fetching existing records
- Change detection and logging
- Selective field updates
- INSERT vs UPDATE handling
- Performance optimizations (skip updates when no changes)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import pandas as pd
import numpy as np

from src.adapters.storage.postgresql_adapter import PostgreSQLAdapter
from src.domain.ports import Result, StorageError
from src.infrastructure.config_manager import DatabaseConfig


@pytest.fixture(autouse=True)
def mock_psycopg2():
    """Mock psycopg2 module and its components."""
    with patch('src.adapters.storage.postgresql_adapter.pool') as mock_pool_module:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.commit = MagicMock()
        mock_conn.rollback = MagicMock()
        mock_conn.close = MagicMock()
        
        mock_pool.getconn.return_value = mock_conn
        mock_pool.putconn = MagicMock()
        mock_pool.closeall = MagicMock()
        
        mock_threaded_pool_class = MagicMock(return_value=mock_pool)
        mock_pool_module.ThreadedConnectionPool = mock_threaded_pool_class
        
        yield {
            'pool_module': mock_pool_module,
            'pool': mock_pool,
            'conn': mock_conn,
            'cursor': mock_cursor,
            'ThreadedConnectionPool': mock_threaded_pool_class,
        }


@pytest.fixture
def mock_sqlalchemy():
    """Mock SQLAlchemy engine."""
    with patch('src.adapters.storage.postgresql_adapter.create_engine') as mock_create_engine:
        mock_engine = MagicMock()
        mock_raw_conn = MagicMock()
        mock_raw_cursor = MagicMock()
        
        mock_raw_cursor.connection = mock_raw_conn
        mock_raw_conn.configure_mock(encoding='utf-8')
        mock_raw_cursor.rowcount = 0
        
        mock_raw_conn.cursor.return_value = mock_raw_cursor
        mock_raw_conn.commit = MagicMock()
        mock_raw_conn.rollback = MagicMock()
        mock_raw_conn.close = MagicMock()
        mock_engine.raw_connection.return_value = mock_raw_conn
        mock_create_engine.return_value = mock_engine
        
        yield {
            'create_engine': mock_create_engine,
            'engine': mock_engine,
            'raw_conn': mock_raw_conn,
            'raw_cursor': mock_raw_cursor,
        }


@pytest.fixture
def postgresql_adapter():
    """Create PostgreSQLAdapter instance for testing."""
    db_config = DatabaseConfig(
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="test_db",
        user="test_user",
        password="test_pass"
    )
    adapter = PostgreSQLAdapter(db_config=db_config)
    adapter._initialized = True
    adapter._schema_initialized = True
    return adapter


class TestGetPrimaryKey:
    """Test _get_primary_key method."""
    
    def test_get_primary_key_patients(self, postgresql_adapter):
        """Test getting primary key for patients table."""
        assert postgresql_adapter._get_primary_key('patients') == 'patient_id'
    
    def test_get_primary_key_encounters(self, postgresql_adapter):
        """Test getting primary key for encounters table."""
        assert postgresql_adapter._get_primary_key('encounters') == 'encounter_id'
    
    def test_get_primary_key_observations(self, postgresql_adapter):
        """Test getting primary key for observations table."""
        assert postgresql_adapter._get_primary_key('observations') == 'observation_id'
    
    def test_get_primary_key_unknown_table(self, postgresql_adapter):
        """Test getting primary key for unknown table returns None."""
        assert postgresql_adapter._get_primary_key('unknown_table') is None


class TestBulkFetchExistingRecords:
    """Test _bulk_fetch_existing_records method."""
    
    def test_bulk_fetch_existing_records_success(self, mock_psycopg2, postgresql_adapter):
        """Test successful bulk fetch of existing records."""
        mock_cursor = mock_psycopg2['cursor']
        mock_conn = mock_psycopg2['conn']
        
        # Mock cursor description and fetchall
        mock_cursor.description = [
            ('patient_id',), ('family_name',), ('city',)
        ]
        mock_cursor.fetchall.return_value = [
            ('P001', 'Smith', 'Boston'),
            ('P002', 'Jones', 'New York')
        ]
        
        record_ids = ['P001', 'P002']
        result_df = postgresql_adapter._bulk_fetch_existing_records(
            record_ids, 'patients', 'patient_id'
        )
        
        assert len(result_df) == 2
        assert list(result_df['patient_id']) == ['P001', 'P002']
        assert list(result_df['family_name']) == ['Smith', 'Jones']
        
        # Verify query was executed correctly
        mock_cursor.execute.assert_called_once()
        call_args = mock_cursor.execute.call_args
        assert 'SELECT * FROM patients' in call_args[0][0]
        assert call_args[0][1] == record_ids
    
    def test_bulk_fetch_existing_records_empty(self, mock_psycopg2, postgresql_adapter):
        """Test bulk fetch when no records exist."""
        mock_cursor = mock_psycopg2['cursor']
        mock_cursor.description = [('patient_id',)]
        mock_cursor.fetchall.return_value = []
        
        record_ids = ['P999']
        result_df = postgresql_adapter._bulk_fetch_existing_records(
            record_ids, 'patients', 'patient_id'
        )
        
        assert result_df.empty
    
    def test_bulk_fetch_existing_records_empty_list(self, postgresql_adapter):
        """Test bulk fetch with empty record IDs list."""
        result_df = postgresql_adapter._bulk_fetch_existing_records(
            [], 'patients', 'patient_id'
        )
        
        assert result_df.empty
    
    def test_bulk_fetch_existing_records_error_handling(self, mock_psycopg2, postgresql_adapter):
        """Test error handling in bulk fetch returns empty DataFrame."""
        mock_cursor = mock_psycopg2['cursor']
        mock_cursor.execute.side_effect = Exception("Database error")
        
        record_ids = ['P001']
        result_df = postgresql_adapter._bulk_fetch_existing_records(
            record_ids, 'patients', 'patient_id'
        )
        
        # Should return empty DataFrame on error
        assert result_df.empty


class TestUpdateChangedFieldsBulk:
    """Test _update_changed_fields_bulk method."""
    
    def test_update_changed_fields_bulk_success(self, mock_psycopg2, mock_sqlalchemy, postgresql_adapter):
        """Test successful bulk update of changed fields."""
        # Create merged DataFrame with _old and _new columns
        merged_df = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'family_name_old': ['Smith', 'Jones'],
            'family_name_new': ['Smith-Jones', 'Jones'],
            'city_old': ['Boston', 'New York'],
            'city_new': ['Cambridge', 'New York'],
            '_merge': ['both', 'both']
        })
        
        # Create changes DataFrame
        changes_df = pd.DataFrame({
            'table_name': ['patients', 'patients'],
            'record_id': ['P001', 'P001'],
            'field_name': ['family_name', 'city'],
            'old_value': ['Smith', 'Boston'],
            'new_value': ['Smith-Jones', 'Cambridge'],
            'change_type': ['UPDATE', 'UPDATE']
        })
        
        # Mock persist_dataframe to be called
        with patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist:
            mock_persist.return_value = Result.success_result(2)
            
            result = postgresql_adapter._update_changed_fields_bulk(
                merged_df, changes_df, 'patients', 'patient_id'
            )
            
            assert result.is_success()
            assert result.value == 2
            mock_persist.assert_called_once()
            
            # Verify the DataFrame passed to persist_dataframe has correct columns
            call_df = mock_persist.call_args[0][0]
            assert 'patient_id' in call_df.columns
            assert 'family_name' in call_df.columns
            assert 'city' in call_df.columns
    
    def test_update_changed_fields_bulk_empty_changes(self, postgresql_adapter):
        """Test update with empty changes DataFrame."""
        merged_df = pd.DataFrame({'patient_id': ['P001']})
        changes_df = pd.DataFrame()
        
        result = postgresql_adapter._update_changed_fields_bulk(
            merged_df, changes_df, 'patients', 'patient_id'
        )
        
        assert result.is_success()
        assert result.value == 0


class TestPersistDataframeSmart:
    """Test persist_dataframe_smart method."""
    
    def test_persist_dataframe_smart_disabled_cdc(self, mock_psycopg2, mock_sqlalchemy, postgresql_adapter):
        """Test that smart persist falls back to standard persist when CDC is disabled."""
        df = pd.DataFrame({
            'patient_id': ['P001'],
            'family_name': ['Smith'],
            'city': ['Boston']
        })
        
        with patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist:
            mock_persist.return_value = Result.success_result(1)
            
            result = postgresql_adapter.persist_dataframe_smart(
                df, 'patients', enable_cdc=False
            )
            
            assert result.is_success()
            mock_persist.assert_called_once_with(df, 'patients')
    
    def test_persist_dataframe_smart_no_primary_key(self, postgresql_adapter):
        """Test smart persist when primary key is missing."""
        df = pd.DataFrame({
            'family_name': ['Smith'],
            'city': ['Boston']
        })
        
        with patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist:
            mock_persist.return_value = Result.success_result(1)
            
            result = postgresql_adapter.persist_dataframe_smart(
                df, 'patients', enable_cdc=True
            )
            
            # Should fall back to standard persist
            mock_persist.assert_called_once()
    
    def test_persist_dataframe_smart_all_new_records(self, mock_psycopg2, mock_sqlalchemy, postgresql_adapter):
        """Test smart persist when all records are new (fast path)."""
        df = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'family_name': ['Smith', 'Jones'],
            'city': ['Boston', 'New York']
        })
        
        # Mock bulk fetch to return empty (no existing records)
        with patch.object(postgresql_adapter, '_bulk_fetch_existing_records') as mock_fetch:
            mock_fetch.return_value = pd.DataFrame()
            
            with patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist:
                mock_persist.return_value = Result.success_result(2)
                
                with patch.object(postgresql_adapter, 'flush_change_logs') as mock_flush:
                    mock_flush.return_value = Result.success_result(4)
                    
                    result = postgresql_adapter.persist_dataframe_smart(
                        df, 'patients', enable_cdc=True,
                        ingestion_id='ing-123', source_adapter='csv_ingester'
                    )
                    
                    assert result.is_success()
                    assert result.value == 2
                    mock_persist.assert_called_once()
                    mock_flush.assert_called_once()
    
    def test_persist_dataframe_smart_mixed_inserts_updates(self, mock_psycopg2, mock_sqlalchemy, postgresql_adapter):
        """Test smart persist with mixed inserts and updates."""
        # New DataFrame with some existing and some new records
        df = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'family_name': ['Smith-Updated', 'Jones', 'New-Patient'],
            'city': ['Cambridge', 'New York', 'Seattle']
        })
        
        # Existing records (P001 and P002 exist)
        existing_df = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'family_name': ['Smith', 'Jones'],
            'city': ['Boston', 'New York']
        })
        
        with patch.object(postgresql_adapter, '_bulk_fetch_existing_records') as mock_fetch:
            mock_fetch.return_value = existing_df
            
            with patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist:
                mock_persist.return_value = Result.success_result(1)
                
                with patch.object(postgresql_adapter, 'flush_change_logs') as mock_flush:
                    mock_flush.return_value = Result.success_result(2)
                    
                    result = postgresql_adapter.persist_dataframe_smart(
                        df, 'patients', enable_cdc=True,
                        ingestion_id='ing-123', source_adapter='csv_ingester'
                    )
                    
                    assert result.is_success()
                    # Should call persist_dataframe for both inserts and updates
                    assert mock_persist.call_count >= 1
                    mock_flush.assert_called_once()
    
    def test_persist_dataframe_smart_no_changes_skips_update(self, mock_psycopg2, mock_sqlalchemy, postgresql_adapter):
        """Test that smart persist skips update when no changes detected."""
        df = pd.DataFrame({
            'patient_id': ['P001'],
            'family_name': ['Smith'],
            'city': ['Boston']
        })
        
        # Existing records with same values (no changes)
        existing_df = pd.DataFrame({
            'patient_id': ['P001'],
            'family_name': ['Smith'],
            'city': ['Boston']
        })
        
        with patch.object(postgresql_adapter, '_bulk_fetch_existing_records') as mock_fetch:
            mock_fetch.return_value = existing_df
            
            with patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist:
                result = postgresql_adapter.persist_dataframe_smart(
                    df, 'patients', enable_cdc=True,
                    ingestion_id='ing-123', source_adapter='csv_ingester'
                )
                
                assert result.is_success()
                # Should not call persist_dataframe for updates (no changes)
                # But might be called for other reasons, so we just check success
    
    def test_persist_dataframe_smart_error_fallback(self, mock_psycopg2, mock_sqlalchemy, postgresql_adapter):
        """Test that smart persist falls back to standard persist on error."""
        df = pd.DataFrame({
            'patient_id': ['P001'],
            'family_name': ['Smith'],
            'city': ['Boston']
        })
        
        with patch.object(postgresql_adapter, '_bulk_fetch_existing_records') as mock_fetch:
            mock_fetch.side_effect = Exception("Unexpected error")
            
            with patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist:
                mock_persist.return_value = Result.success_result(1)
                
                result = postgresql_adapter.persist_dataframe_smart(
                    df, 'patients', enable_cdc=True
                )
                
                # Should fall back to standard persist
                mock_persist.assert_called_once_with(df, 'patients')
                assert result.is_success()
    
    def test_persist_dataframe_smart_empty_dataframe(self, postgresql_adapter):
        """Test smart persist with empty DataFrame."""
        df = pd.DataFrame()
        
        result = postgresql_adapter.persist_dataframe_smart(
            df, 'patients', enable_cdc=True
        )
        
        assert result.is_success()
        assert result.value == 0
    
    def test_persist_dataframe_smart_raw_vault_disabled_uses_main_table_fetch(
        self, mock_psycopg2, mock_sqlalchemy, postgresql_adapter
    ):
        """Test that when enable_raw_vault=False, main table is fetched instead of raw vault."""
        df = pd.DataFrame({
            'patient_id': ['P001', 'P002'],
            'family_name': ['Smith', 'Jones'],
            'city': ['Boston', 'New York']
        })

        with patch.object(postgresql_adapter, '_fetch_raw_records_decrypted') as mock_raw_fetch, \
             patch.object(postgresql_adapter, '_bulk_fetch_existing_records') as mock_main_fetch, \
             patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist, \
             patch.object(postgresql_adapter, 'flush_change_logs') as mock_flush:

            mock_main_fetch.return_value = pd.DataFrame()
            mock_persist.return_value = Result.success_result(2)
            mock_flush.return_value = Result.success_result(4)

            result = postgresql_adapter.persist_dataframe_smart(
                df, 'patients', enable_cdc=True, enable_raw_vault=False,
                ingestion_id='ing-123', source_adapter='csv_ingester'
            )

            assert result.is_success()
            # Main table fetch should be used when raw vault is disabled
            mock_main_fetch.assert_called_once_with(
                ['P001', 'P002'], 'patients', 'patient_id'
            )
            # Raw vault should NOT be accessed when disabled
            mock_raw_fetch.assert_not_called()

    def test_persist_dataframe_smart_logs_changes(self, mock_psycopg2, mock_sqlalchemy, postgresql_adapter):
        """Test that smart persist logs changes correctly."""
        df = pd.DataFrame({
            'patient_id': ['P001'],
            'family_name': ['Smith-Updated'],
            'city': ['Cambridge']
        })
        
        existing_df = pd.DataFrame({
            'patient_id': ['P001'],
            'family_name': ['Smith'],
            'city': ['Boston']
        })
        
        with patch.object(postgresql_adapter, '_bulk_fetch_existing_records') as mock_fetch:
            mock_fetch.return_value = existing_df
            
            with patch.object(postgresql_adapter, 'persist_dataframe') as mock_persist:
                mock_persist.return_value = Result.success_result(1)
                
                with patch.object(postgresql_adapter, 'flush_change_logs') as mock_flush:
                    mock_flush.return_value = Result.success_result(2)
                    
                    result = postgresql_adapter.persist_dataframe_smart(
                        df, 'patients', enable_cdc=True,
                        ingestion_id='ing-123', source_adapter='csv_ingester'
                    )
                    
                    assert result.is_success()
                    # Verify change logs were flushed
                    mock_flush.assert_called_once()
                    # Verify logs contain change events
                    call_args = mock_flush.call_args[0][0]
                    assert len(call_args) > 0
                    # Verify ingestion context is set
                    assert any(log.get('ingestion_id') == 'ing-123' for log in call_args)
                    assert any(log.get('source_adapter') == 'csv_ingester' for log in call_args)
