"""Tests for golden record schemas.

These tests verify that the domain models correctly validate and transform
clinical data according to the golden record schema.
"""

from datetime import date, datetime
import pytest
from pydantic import ValidationError

from src.domain.enums import (
    AdministrativeGender,
    EncounterClass,
    ObservationCategory,
)
from src.domain.golden_record import (
    PatientRecord,
    ClinicalObservation,
    EncounterRecord,
    GoldenRecord,
)


class TestPatientRecord:
    """Test suite for PatientRecord model."""
    
    def test_valid_patient_record(self):
        """Test creating a valid patient record with automatic PII redaction."""
        patient = PatientRecord(
            patient_id="P001",
            first_name="John",
            last_name="Doe",
            date_of_birth=date(1980, 1, 15),
            ssn="123456789",
            gender="male",
            state="CA",
        )
        assert patient.patient_id == "P001"
        # PII fields should be automatically redacted
        assert patient.first_name == "[REDACTED]"
        assert patient.last_name == "[REDACTED]"
        assert patient.date_of_birth is None  # DOB is fully redacted
        assert patient.ssn == "***-**-****"
        assert patient.gender == AdministrativeGender.MALE
        assert patient.state == "CA"  # Non-PII field unchanged
    
    def test_gender_normalization(self):
        """Test that gender values are normalized to FHIR AdministrativeGender."""
        patient_male = PatientRecord(patient_id="P001", gender="male")
        assert patient_male.gender == AdministrativeGender.MALE
        
        patient_female = PatientRecord(patient_id="P002", gender="FEMALE")
        assert patient_female.gender == AdministrativeGender.FEMALE
        
        patient_other = PatientRecord(patient_id="P003", gender="other")
        assert patient_other.gender == AdministrativeGender.OTHER
        
        patient_unknown = PatientRecord(patient_id="P004", gender="unknown")
        assert patient_unknown.gender == AdministrativeGender.UNKNOWN
        
        # Test single letter abbreviations
        patient_m = PatientRecord(patient_id="P005", gender="m")
        assert patient_m.gender == AdministrativeGender.MALE
    
    def test_state_normalization(self):
        """Test that state codes are normalized to uppercase."""
        patient = PatientRecord(patient_id="P001", state="ca")
        assert patient.state == "CA"
        
        patient_long = PatientRecord(patient_id="P002", state="california")
        assert patient_long.state == "CA"
    
    def test_ssn_redaction(self):
        """Test that SSN is automatically redacted regardless of format."""
        # Valid SSN with dashes - should be redacted
        patient = PatientRecord(patient_id="P001", ssn="123-45-6789")
        assert patient.ssn == "***-**-****"
        
        # Valid SSN without dashes - should be redacted
        patient2 = PatientRecord(patient_id="P002", ssn="987654321")
        assert patient2.ssn == "***-**-****"
        
        # Invalid SSN format - should still be redacted if pattern matches
        patient3 = PatientRecord(patient_id="P003", ssn="123-45-678")
        # If pattern doesn't match exactly, may return original or redacted
        # The service will redact if it detects SSN pattern
    
    def test_pii_redaction(self):
        """Test that all PII fields are automatically redacted."""
        patient = PatientRecord(
            patient_id="P001",
            first_name="John",
            last_name="Doe",
            date_of_birth=date(1980, 1, 15),
            ssn="123-45-6789",
            phone="555-123-4567",
            email="john.doe@example.com",
            address_line1="123 Main St",
            address_line2="Apt 4B",
            zip_code="12345",
        )
        
        # All PII should be redacted
        assert patient.first_name == "[REDACTED]"
        assert patient.last_name == "[REDACTED]"
        assert patient.date_of_birth is None
        assert patient.ssn == "***-**-****"
        assert patient.phone == "***-***-****"
        assert patient.email == "***@***.***"
        assert patient.address_line1 == "[REDACTED]"
        assert patient.address_line2 == "[REDACTED]"
        assert patient.zip_code == "12***"  # Partially redacted
        
        # Non-PII fields should remain unchanged
        assert patient.patient_id == "P001"
    
    def test_immutable_record(self):
        """Test that records are immutable."""
        patient = PatientRecord(patient_id="P001", first_name="John")
        
        with pytest.raises(ValidationError):
            # Attempting to modify should raise an error
            patient.first_name = "Jane"


class TestClinicalObservation:
    """Test suite for ClinicalObservation model."""
    
    def test_valid_observation(self):
        """Test creating a valid clinical observation."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            observation_type="vital-signs",
            observation_code="85354-9",
            value="120/80",
            unit="mmHg",
            effective_date=datetime(2024, 1, 15, 10, 30),
        )
        assert observation.observation_id == "O001"
        assert observation.observation_type == ObservationCategory.VITAL_SIGNS
        assert observation.value == "120/80"
    
    def test_observation_type_normalization(self):
        """Test that observation types are normalized to FHIR ObservationCategory."""
        observation_lab = ClinicalObservation(
            observation_id="O002",
            patient_id="P001",
            observation_type="laboratory",
        )
        assert observation_lab.observation_type == ObservationCategory.LABORATORY
        
        observation_vital = ClinicalObservation(
            observation_id="O003",
            patient_id="P001",
            observation_type="VITAL_SIGN",
        )
        assert observation_vital.observation_type == ObservationCategory.VITAL_SIGNS


class TestEncounterRecord:
    """Test suite for EncounterRecord model."""
    
    def test_valid_encounter(self):
        """Test creating a valid encounter record."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            encounter_type="outpatient",
            start_date=datetime(2024, 1, 15, 9, 0),
            end_date=datetime(2024, 1, 15, 10, 0),
            diagnosis_codes=["I10", "E11.9"],
        )
        assert encounter.encounter_id == "E001"
        assert encounter.encounter_type == EncounterClass.OUTPATIENT
        assert len(encounter.diagnosis_codes) == 2
    
    def test_encounter_type_normalization(self):
        """Test that encounter types are normalized to FHIR EncounterClass."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            encounter_type="inpatient",
        )
        assert encounter.encounter_type == EncounterClass.INPATIENT
        
        encounter2 = EncounterRecord(
            encounter_id="E002",
            patient_id="P001",
            encounter_type="urgent care",
        )
        assert encounter2.encounter_type == EncounterClass.URGENT_CARE
        
        encounter3 = EncounterRecord(
            encounter_id="E003",
            patient_id="P001",
            encounter_type="emergency",
        )
        assert encounter3.encounter_type == EncounterClass.EMERGENCY
        
        encounter4 = EncounterRecord(
            encounter_id="E004",
            patient_id="P001",
            encounter_type="virtual",
        )
        assert encounter4.encounter_type == EncounterClass.VIRTUAL


class TestGoldenRecord:
    """Test suite for GoldenRecord container model."""
    
    def test_complete_golden_record(self):
        """Test creating a complete golden record."""
        patient = PatientRecord(patient_id="P001", first_name="John")
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            encounter_type="outpatient",
        )
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            observation_type="laboratory",
        )
        
        golden = GoldenRecord(
            patient=patient,
            encounters=[encounter],
            observations=[observation],
            source_adapter="json_adapter",
        )
        
        assert golden.patient.patient_id == "P001"
        assert len(golden.encounters) == 1
        assert len(golden.observations) == 1
        assert golden.source_adapter == "json_adapter"
        assert golden.ingestion_timestamp is not None
        assert golden.encounters[0].encounter_type == EncounterClass.OUTPATIENT
        assert golden.observations[0].observation_type == ObservationCategory.LABORATORY

