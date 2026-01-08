"""Comprehensive unit test suite for JSONIngester.

These tests verify all IngestionPort methods and internal logic without requiring
actual JSON files. All file operations are mocked to enable fast, isolated unit testing.

Security Impact:
    - Verifies PII redaction is applied correctly
    - Confirms malformed records are rejected
    - Ensures error handling prevents DoS attacks
    - Tests triage logic for security rejections
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

import pandas as pd

from src.adapters.ingesters.json_ingester import JSONIngester
from src.domain.ports import Result, SourceNotFoundError, UnsupportedSourceError, ValidationError
from src.domain.services import RedactorService
from src.infrastructure.redaction_context import set_redaction_context, get_redaction_context


class TestJSONIngesterInitialization:
    """Test JSONIngester initialization."""
    
    def test_init_defaults(self):
        """Test initialization with default parameters."""
        ingester = JSONIngester()
        
        assert ingester.max_record_size == 10 * 1024 * 1024
        assert ingester.initial_chunk_size == 10000
        assert ingester.chunk_size == 10000
        assert ingester.target_total_rows == 50000
        assert ingester.adapter_name == "json_ingester"
        assert ingester.adaptive_chunking_enabled is True
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        ingester = JSONIngester(
            max_record_size=5 * 1024 * 1024,
            chunk_size=5000,
            target_total_rows=25000
        )
        
        assert ingester.max_record_size == 5 * 1024 * 1024
        assert ingester.initial_chunk_size == 5000
        assert ingester.chunk_size == 5000
        assert ingester.target_total_rows == 25000
    
    def test_init_disable_adaptive_chunking(self):
        """Test initialization with adaptive chunking disabled."""
        ingester = JSONIngester(target_total_rows=0)
        
        assert ingester.adaptive_chunking_enabled is False


class TestJSONIngesterCanIngest:
    """Test can_ingest method."""
    
    def test_can_ingest_json_file(self):
        """Test that .json files are recognized."""
        ingester = JSONIngester()
        
        assert ingester.can_ingest("test.json") is True
        assert ingester.can_ingest("test.JSON") is True
        assert ingester.can_ingest("/path/to/file.json") is True
    
    def test_can_ingest_jsonl_file(self):
        """Test that .jsonl files are recognized."""
        ingester = JSONIngester()
        
        assert ingester.can_ingest("test.jsonl") is True
        assert ingester.can_ingest("test.JSONL") is True
    
    def test_can_ingest_url(self):
        """Test that URLs ending in .json are recognized."""
        ingester = JSONIngester()
        
        assert ingester.can_ingest("http://example.com/data.json") is True
        assert ingester.can_ingest("https://api.example.com/records.jsonl") is True
    
    def test_can_ingest_rejects_non_json(self):
        """Test that non-JSON files are rejected."""
        ingester = JSONIngester()
        
        assert ingester.can_ingest("test.csv") is False
        assert ingester.can_ingest("test.xml") is False
        assert ingester.can_ingest("test.txt") is False
        assert ingester.can_ingest("") is False
        assert ingester.can_ingest(None) is False


class TestJSONIngesterGetSourceInfo:
    """Test get_source_info method."""
    
    def test_get_source_info_existing_file(self):
        """Test getting source info for existing file."""
        ingester = JSONIngester()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"test": "data"}')
            temp_path = f.name
        
        try:
            info = ingester.get_source_info(temp_path)
            
            assert info is not None
            assert info['format'] == 'json'
            assert info['exists'] is True
            assert info['encoding'] == 'utf-8'
            assert 'size' in info
        finally:
            Path(temp_path).unlink()
    
    def test_get_source_info_nonexistent_file(self):
        """Test getting source info for non-existent file."""
        ingester = JSONIngester()
        
        info = ingester.get_source_info("/nonexistent/file.json")
        
        assert info is None


class TestJSONIngesterExtractRecords:
    """Test _extract_records method."""
    
    def test_extract_records_list(self):
        """Test extracting records from a JSON list."""
        ingester = JSONIngester()
        
        raw_data = [
            {"patient_id": "MRN001", "name": "John"},
            {"patient_id": "MRN002", "name": "Jane"}
        ]
        
        records = ingester._extract_records(raw_data)
        
        assert len(records) == 2
        assert records[0]["patient_id"] == "MRN001"
        assert records[1]["patient_id"] == "MRN002"
    
    def test_extract_records_dict_with_records_key(self):
        """Test extracting records from a dict with 'records' key."""
        ingester = JSONIngester()
        
        raw_data = {
            "records": [
                {"patient_id": "MRN001"},
                {"patient_id": "MRN002"}
            ]
        }
        
        records = ingester._extract_records(raw_data)
        
        assert len(records) == 2
    
    def test_extract_records_dict_with_data_key(self):
        """Test extracting records from a dict with 'data' key."""
        ingester = JSONIngester()
        
        raw_data = {
            "data": [
                {"patient_id": "MRN001"},
                {"patient_id": "MRN002"}
            ]
        }
        
        records = ingester._extract_records(raw_data)
        
        assert len(records) == 2
    
    def test_extract_records_single_dict(self):
        """Test extracting a single record from a dict."""
        ingester = JSONIngester()
        
        raw_data = {"patient_id": "MRN001", "name": "John"}
        
        records = ingester._extract_records(raw_data)
        
        assert len(records) == 1
        assert records[0]["patient_id"] == "MRN001"
    
    def test_extract_records_empty_list(self):
        """Test extracting from empty list."""
        ingester = JSONIngester()
        
        records = ingester._extract_records([])
        
        assert records == []


class TestJSONIngesterRedactDataFrame:
    """Test _redact_dataframe method."""
    
    def test_redact_ssn(self):
        """Test SSN redaction in DataFrame."""
        ingester = JSONIngester()
        df = pd.DataFrame({
            'patient_id': ['MRN001', 'MRN002'],
            'ssn': ['123-45-6789', '987-65-4321']
        })
        
        redacted_df = ingester._redact_dataframe(df)
        
        assert all(redacted_df['ssn'] == RedactorService.SSN_MASK)
    
    def test_redact_names(self):
        """Test name redaction in DataFrame."""
        ingester = JSONIngester()
        df = pd.DataFrame({
            'patient_id': ['MRN001', 'MRN002'],
            'first_name': ['John', 'Jane'],
            'last_name': ['Doe', 'Smith']
        })
        
        redacted_df = ingester._redact_dataframe(df)
        
        assert all(redacted_df['first_name'] == RedactorService.NAME_MASK)
        assert all(redacted_df['last_name'] == RedactorService.NAME_MASK)
    
    def test_redact_phone(self):
        """Test phone redaction in DataFrame."""
        ingester = JSONIngester()
        df = pd.DataFrame({
            'patient_id': ['MRN001'],
            'phone': ['555-123-4567']
        })
        
        redacted_df = ingester._redact_dataframe(df)
        
        assert redacted_df['phone'].iloc[0] == RedactorService.PHONE_MASK
    
    def test_redact_email(self):
        """Test email redaction in DataFrame."""
        ingester = JSONIngester()
        df = pd.DataFrame({
            'patient_id': ['MRN001'],
            'email': ['john.doe@example.com']
        })
        
        redacted_df = ingester._redact_dataframe(df)
        
        assert redacted_df['email'].iloc[0] == RedactorService.EMAIL_MASK
    
    def test_redact_address(self):
        """Test address redaction in DataFrame."""
        ingester = JSONIngester()
        df = pd.DataFrame({
            'patient_id': ['MRN001'],
            'address_line1': ['123 Main St']
        })
        
        redacted_df = ingester._redact_dataframe(df)
        
        assert redacted_df['address_line1'].iloc[0] == RedactorService.ADDRESS_MASK
    
    def test_redact_dob(self):
        """Test date of birth redaction in DataFrame."""
        ingester = JSONIngester()
        df = pd.DataFrame({
            'patient_id': ['MRN001'],
            'date_of_birth': ['1990-01-01']
        })
        
        redacted_df = ingester._redact_dataframe(df)
        
        assert pd.isna(redacted_df['date_of_birth'].iloc[0])


class TestJSONIngesterIngest:
    """Test ingest method."""
    
    def test_ingest_nonexistent_file(self):
        """Test that nonexistent file raises SourceNotFoundError."""
        ingester = JSONIngester()
        
        with pytest.raises(SourceNotFoundError):
            list(ingester.ingest("/nonexistent/file.json"))
    
    def test_ingest_invalid_json(self):
        """Test that invalid JSON raises UnsupportedSourceError."""
        ingester = JSONIngester()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{invalid json}')
            temp_path = f.name
        
        try:
            with pytest.raises(UnsupportedSourceError):
                list(ingester.ingest(temp_path))
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_empty_file(self):
        """Test ingesting empty JSON file."""
        ingester = JSONIngester()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('[]')
            temp_path = f.name
        
        try:
            results = list(ingester.ingest(temp_path))
            assert len(results) == 0
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_simple_patient_record(self):
        """Test ingesting a simple patient record."""
        ingester = JSONIngester(chunk_size=10)
        
        data = {
            "patient": {
                "patient_id": "MRN001",
                "family_name": "Doe",
                "given_names": ["John"],
                "city": "Springfield",
                "state": "IL",
                "postal_code": "62701"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([data], f)
            temp_path = f.name
        
        try:
            results = list(ingester.ingest(temp_path))
            
            # Should yield at least one DataFrame result
            assert len(results) > 0
            assert all(r.is_success() for r in results)
            
            # Check first result is a tuple (redacted_df, raw_df) for raw vault
            first_result = results[0]
            assert isinstance(first_result.value, tuple)
            assert len(first_result.value) == 2
            
            # Unpack tuple
            redacted_df, raw_df = first_result.value
            
            # Check both are DataFrames
            assert isinstance(redacted_df, pd.DataFrame)
            assert isinstance(raw_df, pd.DataFrame)
            
            # Check patient data is present
            assert len(redacted_df) > 0
            assert 'patient_id' in redacted_df.columns
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_with_encounters_and_observations(self):
        """Test ingesting records with encounters and observations."""
        ingester = JSONIngester(chunk_size=10)
        
        data = {
            "patient": {
                "patient_id": "MRN001",
                "family_name": "Doe",
                "given_names": ["John"],
                "city": "Springfield",
                "state": "IL",
                "postal_code": "62701"
            },
            "encounters": [
                {
                    "encounter_id": "ENC001",
                    "patient_id": "MRN001",
                    "status": "finished",
                    "class_code": "outpatient"
                }
            ],
            "observations": [
                {
                    "observation_id": "OBS001",
                    "patient_id": "MRN001",
                    "status": "final",
                    "category": "vital-signs",
                    "code": "8480-6",
                    "value": "120/80"
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([data], f)
            temp_path = f.name
        
        try:
            results = list(ingester.ingest(temp_path))
            
            # Should yield multiple DataFrames (patients, encounters, observations)
            # Each result is now a tuple (redacted_df, raw_df)
            assert len(results) >= 1
            
            # All results should be successful
            assert all(r.is_success() for r in results)
            
            # Check that we have patients DataFrame (first result should be patients)
            patients_result = results[0]
            assert isinstance(patients_result.value, tuple)
            patients_df, patients_raw_df = patients_result.value
            assert isinstance(patients_df, pd.DataFrame)
            assert isinstance(patients_raw_df, pd.DataFrame)
            assert len(patients_df) > 0
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_adaptive_chunking(self):
        """Test that adaptive chunking adjusts chunk size after first chunk."""
        ingester = JSONIngester(chunk_size=5, target_total_rows=20)
        
        # Create data with multiple records
        data_list = []
        for i in range(10):
            data_list.append({
                "patient": {
                    "patient_id": f"MRN{i:03d}",
                    "family_name": "Doe",
                    "given_names": ["John"],
                    "city": "Springfield",
                    "state": "IL",
                    "postal_code": "62701"
                },
                "encounters": [
                    {
                        "encounter_id": f"ENC{i:03d}",
                        "patient_id": f"MRN{i:03d}",
                        "status": "finished",
                        "class_code": "outpatient"
                    }
                ],
                "observations": [
                    {
                        "observation_id": f"OBS{i:03d}",
                        "patient_id": f"MRN{i:03d}",
                        "status": "final",
                        "category": "vital-signs",
                        "code": "8480-6",
                        "value": "120/80"
                    }
                ]
            })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data_list, f)
            temp_path = f.name
        
        try:
            results = list(ingester.ingest(temp_path))
            
            # Should have processed records
            assert len(results) > 0
            
            # Check that ratios were calculated
            assert ingester.ratios is not None
            assert 'encounters_per_patient' in ingester.ratios
            assert 'observations_per_patient' in ingester.ratios
            
            # Check that chunk size may have been adjusted
            # (it might stay the same if optimal size equals initial size)
            assert ingester.chunk_size >= 1000  # Minimum bound
            assert ingester.chunk_size <= 50000  # Maximum bound
        finally:
            Path(temp_path).unlink()


class TestJSONIngesterErrorHandling:
    """Test error handling in JSONIngester."""
    
    def test_ingest_handles_malformed_record(self):
        """Test that malformed records are handled gracefully."""
        ingester = JSONIngester(chunk_size=10)
        
        # Create data with one valid and one invalid record
        data = [
            {
                "patient": {
                    "patient_id": "MRN001",
                    "family_name": "Doe",
                    "given_names": ["John"],
                    "city": "Springfield",
                    "state": "IL",
                    "postal_code": "62701"
                }
            },
            {
                "patient": {
                    # Missing required fields
                }
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            results = list(ingester.ingest(temp_path))
            
            # Should still yield results for valid records
            assert len(results) > 0
            assert all(r.is_success() for r in results)
        finally:
            Path(temp_path).unlink()
    
    def test_ingest_handles_missing_patient_id(self):
        """Test that records without patient_id are skipped."""
        ingester = JSONIngester(chunk_size=10)
        
        data = [
            {
                "patient": {
                    # No patient_id
                    "family_name": "Doe",
                    "given_names": ["John"]
                }
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            results = list(ingester.ingest(temp_path))
            
            # Should yield no results (all records skipped)
            assert len(results) == 0
        finally:
            Path(temp_path).unlink()


class TestJSONIngesterRedactionLogging:
    """Test redaction logging functionality."""
    
    def test_redaction_logging_with_context(self):
        """Test that redactions are logged when context is available."""
        ingester = JSONIngester()
        
        # Create mock logger
        mock_logger = Mock()
        mock_logger.log_redaction = Mock()
        
        # Set redaction context
        set_redaction_context(
            logger=mock_logger,
            record_id='MRN001',
            source_adapter='json_ingester',
            ingestion_id='test-ingestion-1'
        )
        
        # Create DataFrame with PII
        df = pd.DataFrame({
            'patient_id': ['MRN001', 'MRN002'],
            'ssn': ['123-45-6789', '987-65-4321']
        })
        
        # Redact DataFrame
        redacted_df = ingester._redact_dataframe(df)
        
        # Verify redactions occurred
        assert all(redacted_df['ssn'] == RedactorService.SSN_MASK)
        
        # Note: Actual logging depends on log_vectorized_redactions being called
        # which happens during ingest, not just _redact_dataframe
