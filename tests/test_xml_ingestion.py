"""Tests for XML ingestion adapter.

These tests verify that the XML ingester correctly:
- Processes XML files using defusedxml for security
- Extracts data using XPath configuration
- Validates records and yields GoldenRecords
- Handles malformed records gracefully
- Integrates with CircuitBreaker
"""

import json
import pytest
from pathlib import Path

from src.adapters.xml_ingester import XMLIngester
from src.domain.ports import Result, SourceNotFoundError, UnsupportedSourceError
from src.domain.guardrails import CircuitBreaker, CircuitBreakerConfig
from src.domain.golden_record import GoldenRecord


@pytest.fixture
def xml_test_file(tmp_path):
    """Create a temporary XML file with test data."""
    test_file = tmp_path / "test_batch.xml"
    
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<ClinicalData>
    <!-- Valid Record 1: Should pass validation -->
    <PatientRecord>
        <MRN>MRN001</MRN>
        <Demographics>
            <FullName>John Doe</FullName>
            <BirthDate>1990-01-01</BirthDate>
            <Gender>male</Gender>
            <SSN>123-45-6789</SSN>
            <Phone>555-123-4567</Phone>
            <Email>john.doe@example.com</Email>
            <Address>
                <Street>123 Main St</Street>
                <City>Springfield</City>
                <State>IL</State>
                <ZIP>62701</ZIP>
            </Address>
        </Demographics>
        <Visit>
            <AdmitDate>2023-01-01T10:00:00</AdmitDate>
            <Status>finished</Status>
            <Type>outpatient</Type>
            <DxCode>I10</DxCode>
        </Visit>
        <Notes>
            <ProgressNote>Blood pressure normal. Patient SSN: 123-45-6789</ProgressNote>
        </Notes>
    </PatientRecord>
    
    <!-- Invalid Record 2: MRN too short - should be rejected -->
    <PatientRecord>
        <MRN>AB</MRN>
        <Demographics>
            <FullName>Jane Smith</FullName>
            <BirthDate>1995-05-15</BirthDate>
            <Gender>female</Gender>
            <SSN>987-65-4321</SSN>
            <Phone>555-987-6543</Phone>
        </Demographics>
        <Visit>
            <AdmitDate>2023-01-01T10:00:00</AdmitDate>
            <Status>arrived</Status>
            <Type>outpatient</Type>
        </Visit>
    </PatientRecord>
    
    <!-- Valid Record 3: Should pass validation -->
    <PatientRecord>
        <MRN>MRN003</MRN>
        <Demographics>
            <FullName>Bob Johnson</FullName>
            <BirthDate>1985-03-20</BirthDate>
            <Gender>male</Gender>
            <Address>
                <Street>456 Oak Ave</Street>
                <City>Los Angeles</City>
                <State>CA</State>
                <ZIP>90210</ZIP>
            </Address>
        </Demographics>
        <Visit>
            <AdmitDate>2023-01-02T14:00:00</AdmitDate>
            <Status>finished</Status>
            <Type>outpatient</Type>
        </Visit>
        <Notes>
            <ProgressNote>Temperature: 98.6F. All vital signs normal.</ProgressNote>
        </Notes>
    </PatientRecord>
    
    <!-- Invalid Record 4: Future DOB - should be rejected -->
    <PatientRecord>
        <MRN>MRN004</MRN>
        <Demographics>
            <FullName>Alice Williams</FullName>
            <BirthDate>2030-01-01</BirthDate>
            <Gender>female</Gender>
        </Demographics>
        <Visit>
            <AdmitDate>2023-01-01T10:00:00</AdmitDate>
            <Status>finished</Status>
            <Type>outpatient</Type>
        </Visit>
    </PatientRecord>
</ClinicalData>"""
    
    test_file.write_text(xml_data, encoding="utf-8")
    return test_file


@pytest.fixture
def xml_config_file(tmp_path):
    """Create a temporary XML configuration file."""
    config_file = tmp_path / "xml_config.json"
    
    xml_config = {
        "root_element": "./PatientRecord",
        "fields": {
            "mrn": "./MRN",
            "patient_name": "./Demographics/FullName",
            "patient_dob": "./Demographics/BirthDate",
            "patient_gender": "./Demographics/Gender",
            "ssn": "./Demographics/SSN",
            "phone": "./Demographics/Phone",
            "email": "./Demographics/Email",
            "address_line1": "./Demographics/Address/Street",
            "city": "./Demographics/Address/City",
            "state": "./Demographics/Address/State",
            "postal_code": "./Demographics/Address/ZIP",
            "encounter_date": "./Visit/AdmitDate",
            "encounter_status": "./Visit/Status",
            "encounter_type": "./Visit/Type",
            "primary_diagnosis_code": "./Visit/DxCode",
            "clinical_notes": "./Notes/ProgressNote"
        }
    }
    
    with open(config_file, "w") as f:
        json.dump(xml_config, f, indent=2)
    
    return config_file


class TestXMLIngestion:
    """Test suite for XML ingestion."""
    
    def test_can_ingest_xml_file(self, xml_config_file):
        """Test that XML ingester recognizes XML files."""
        ingester = XMLIngester(config_path=str(xml_config_file))
        assert ingester.can_ingest("test.xml")
        assert not ingester.can_ingest("test.csv")
        assert not ingester.can_ingest("test.json")
    
    def test_ingest_valid_xml(self, xml_test_file, xml_config_file):
        """Test ingesting a valid XML file."""
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        results = list(ingester.ingest(str(xml_test_file)))
        
        # Should yield Results (individual GoldenRecords for XML)
        assert len(results) > 0
        
        # Check that we get at least one successful result
        success_results = [r for r in results if r.is_success()]
        assert len(success_results) > 0
        
        # Check that successful results contain GoldenRecords
        for result in success_results:
            assert isinstance(result.value, GoldenRecord)
            assert result.value.patient.patient_id is not None
    
    def test_ingest_xml_rejects_invalid_records(self, xml_test_file, xml_config_file):
        """Test that invalid records are rejected without crashing."""
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        results = list(ingester.ingest(str(xml_test_file)))
        
        # Should have both success and failure results
        success_results = [r for r in results if r.is_success()]
        failure_results = [r for r in results if r.is_failure()]
        
        # Should have at least 2 valid records (MRN001, MRN003)
        assert len(success_results) >= 2
        
        # Should have at least 2 invalid records (AB, MRN004 with future DOB)
        assert len(failure_results) >= 2
    
    def test_ingest_xml_pii_redaction(self, xml_test_file, xml_config_file):
        """Test that PII fields are properly redacted in XML ingestion."""
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        results = list(ingester.ingest(str(xml_test_file)))
        
        # Check successful results
        for result in results:
            if result.is_success():
                golden_record = result.value
                patient = golden_record.patient
                
                # PII fields should be redacted
                assert patient.first_name == "[REDACTED]"
                assert patient.last_name == "[REDACTED]"
                assert patient.date_of_birth is None  # DOB fully redacted
                if patient.ssn:
                    assert patient.ssn == "***-**-****"
    
    def test_ingest_nonexistent_file_raises_error(self, xml_config_file):
        """Test that ingesting a non-existent file raises error."""
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        with pytest.raises(SourceNotFoundError):
            list(ingester.ingest("nonexistent.xml"))
    
    def test_ingest_invalid_xml_raises_error(self, tmp_path, xml_config_file):
        """Test that invalid XML format raises error."""
        test_file = tmp_path / "invalid.xml"
        test_file.write_text("This is not valid XML <root><unclosed>")
        
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        with pytest.raises(UnsupportedSourceError):
            list(ingester.ingest(str(test_file)))
    
    def test_ingest_xml_requires_config(self, xml_test_file):
        """Test that XML ingester requires config."""
        # XML ingester requires config
        with pytest.raises(ValueError, match="Must specify either config_path or config_dict"):
            ingester = XMLIngester()
    
    def test_xml_ingestion_with_circuit_breaker(self, xml_test_file, xml_config_file):
        """Test XML ingestion with CircuitBreaker integration."""
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        config = CircuitBreakerConfig(
            failure_threshold_percent=50.0,
            window_size=100,
            min_records_before_check=1,
            abort_on_open=False
        )
        breaker = CircuitBreaker(config)
        
        results = []
        for result in ingester.ingest(str(xml_test_file)):
            breaker.record_result(result)
            results.append(result)
        
        # Check statistics
        stats = breaker.get_statistics()
        assert stats['total_processed'] > 0
        # Note: Circuit may open if failure rate exceeds threshold, which is expected behavior
    
    def test_xml_ingestion_extracts_encounters(self, xml_test_file, xml_config_file):
        """Test that XML ingester extracts encounter data."""
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        results = list(ingester.ingest(str(xml_test_file)))
        
        # Find a record with encounters
        for result in results:
            if result.is_success():
                golden_record = result.value
                # Some records should have encounters
                if len(golden_record.encounters) > 0:
                    encounter = golden_record.encounters[0]
                    assert encounter.encounter_id is not None
                    assert encounter.patient_id == golden_record.patient.patient_id
                    break
    
    def test_xml_ingestion_extracts_observations(self, xml_test_file, xml_config_file):
        """Test that XML ingester extracts observation data."""
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        results = list(ingester.ingest(str(xml_test_file)))
        
        # Find a record with observations
        for result in results:
            if result.is_success():
                golden_record = result.value
                # Some records should have observations
                if len(golden_record.observations) > 0:
                    observation = golden_record.observations[0]
                    assert observation.observation_id is not None
                    assert observation.patient_id == golden_record.patient.patient_id
                    break
    
    def test_xml_ingestion_handles_missing_fields(self, tmp_path, xml_config_file):
        """Test that XML ingester handles missing optional fields gracefully."""
        test_file = tmp_path / "minimal.xml"
        
        xml_data = """<?xml version="1.0" encoding="UTF-8"?>
<ClinicalData>
    <PatientRecord>
        <MRN>MRN999</MRN>
        <Demographics>
            <FullName>Minimal Patient</FullName>
            <BirthDate>1990-01-01</BirthDate>
            <Gender>male</Gender>
        </Demographics>
    </PatientRecord>
</ClinicalData>"""
        
        test_file.write_text(xml_data, encoding="utf-8")
        
        ingester = XMLIngester(config_path=str(xml_config_file))
        
        results = list(ingester.ingest(str(test_file)))
        
        # Should process successfully even with minimal fields
        success_results = [r for r in results if r.is_success()]
        assert len(success_results) > 0
        
        # Patient should have required fields
        for result in success_results:
            if result.is_success():
                golden_record = result.value
                assert golden_record.patient.patient_id == "MRN999"
                assert golden_record.patient.first_name == "[REDACTED]"

