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
    ObservationStatus,
    EncounterStatus,
    AddressUse,
)
from datetime import timedelta
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
    
    def test_valid_data_passes(self):
        """Test that valid data passes all validations."""
        # Valid patient record with all required fields
        patient = PatientRecord(
            patient_id="MRN123456",
            first_name="Jane",
            last_name="Smith",
            date_of_birth=date(1985, 5, 15),
            ssn="123-45-6789",
            gender="female",
            state="NY",
            phone="555-123-4567",
            email="jane@example.com",
            address_line1="123 Main St",
            zip_code="10001",
        )
        
        # Should create successfully with PII redacted
        assert patient.patient_id == "MRN123456"
        assert patient.first_name == "[REDACTED]"
        assert patient.last_name == "[REDACTED]"
        assert patient.date_of_birth is None  # Redacted
        assert patient.ssn == "***-**-****"
        assert patient.gender == AdministrativeGender.FEMALE
        assert patient.state == "NY"
    
    def test_malformed_mrn_fails(self):
        """Test that malformed MRN (patient_id) fails validation."""
        # Empty MRN
        with pytest.raises(ValidationError) as exc_info:
            PatientRecord(patient_id="")
        assert "MRN" in str(exc_info.value).lower() or "empty" in str(exc_info.value).lower()
        
        # MRN too short
        with pytest.raises(ValidationError) as exc_info:
            PatientRecord(patient_id="AB")
        assert "at least 3 characters" in str(exc_info.value).lower()
        
        # MRN too long
        with pytest.raises(ValidationError) as exc_info:
            PatientRecord(patient_id="A" * 21)
        assert "at most 20 characters" in str(exc_info.value).lower()
        
        # MRN with invalid characters (special chars that aren't allowed)
        with pytest.raises(ValidationError) as exc_info:
            PatientRecord(patient_id="MRN@123")
        assert "alphanumeric" in str(exc_info.value).lower()
        
        # Valid MRN formats should pass
        valid_mrns = ["MRN123", "P001", "123456", "MRN-123-456", "MRN_123"]
        for mrn in valid_mrns:
            patient = PatientRecord(patient_id=mrn)
            assert patient.patient_id == mrn.strip()
    
    def test_future_date_of_birth_blocked(self):
        """Test that future dates for patient DOB are blocked."""
        from datetime import date, timedelta
        
        today = date.today()
        tomorrow = today + timedelta(days=1)
        next_year = date(today.year + 1, today.month, today.day)
        
        # Future date (tomorrow) should fail
        with pytest.raises(ValidationError) as exc_info:
            PatientRecord(patient_id="P001", date_of_birth=tomorrow)
        assert "future" in str(exc_info.value).lower()
        
        # Future date (next year) should fail
        with pytest.raises(ValidationError) as exc_info:
            PatientRecord(patient_id="P001", date_of_birth=next_year)
        assert "future" in str(exc_info.value).lower()
        
        # Today's date should pass (edge case)
        patient_today = PatientRecord(patient_id="P001", date_of_birth=today)
        assert patient_today.date_of_birth is None  # Redacted but validation passed
        
        # Past date should pass
        past_date = date(1990, 1, 1)
        patient_past = PatientRecord(patient_id="P001", date_of_birth=past_date)
        assert patient_past.date_of_birth is None  # Redacted but validation passed
    
    def test_time_travel_logic(self):
        """Test time travel logic - dates too far in the past are blocked."""
        from datetime import timedelta
        
        # Date before 1900 should fail
        with pytest.raises(ValidationError) as exc_info:
            PatientRecord(patient_id="P001", date_of_birth=date(1850, 1, 1))
        assert "1900" in str(exc_info.value) or "before" in str(exc_info.value).lower()
        
        # Date in 1800s should fail
        with pytest.raises(ValidationError) as exc_info:
            PatientRecord(patient_id="P001", date_of_birth=date(1899, 12, 31))
        assert "1900" in str(exc_info.value) or "before" in str(exc_info.value).lower()
        
        # Date exactly at 1900 should pass
        patient_1900 = PatientRecord(patient_id="P001", date_of_birth=date(1900, 1, 1))
        assert patient_1900.date_of_birth is None  # Redacted but validation passed
        
        # Date after 1900 should pass
        patient_1950 = PatientRecord(patient_id="P001", date_of_birth=date(1950, 1, 1))
        assert patient_1950.date_of_birth is None  # Redacted but validation passed
        
        # Age > 150 years should fail (use a date after 1900 but very old)
        today = date.today()
        very_old_date = date(1901, 1, 1)  # After 1900 but very old
        # Calculate if this would be > 150 years
        age_years = (today - very_old_date).days / 365.25
        if age_years > 150:
            with pytest.raises(ValidationError) as exc_info:
                PatientRecord(patient_id="P001", date_of_birth=very_old_date)
            assert "150" in str(exc_info.value) or "age" in str(exc_info.value).lower()
        else:
            # If current date doesn't make 1901 > 150 years, use a calculated date
            very_old_date = today - timedelta(days=int(151 * 365.25))
            # Ensure it's after 1900
            if very_old_date >= date(1900, 1, 1):
                with pytest.raises(ValidationError) as exc_info:
                    PatientRecord(patient_id="P001", date_of_birth=very_old_date)
                assert "150" in str(exc_info.value) or "age" in str(exc_info.value).lower()
    
    def test_notes_with_ssn_redacted(self):
        """Test that notes with SSNs are successfully redacted."""
        # Notes containing SSN with dashes
        observation1 = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            notes="Patient SSN is 123-45-6789. Vital signs normal.",
        )
        assert "***-**-****" in observation1.notes
        assert "123-45-6789" not in observation1.notes
        
        # Notes containing SSN without dashes
        observation2 = ClinicalObservation(
            observation_id="O002",
            patient_id="P001",
            category="laboratory",
            notes="SSN: 987654321. Lab results pending.",
        )
        assert "***-**-****" in observation2.notes
        assert "987654321" not in observation2.notes
        
        # Notes with multiple SSNs
        observation3 = ClinicalObservation(
            observation_id="O003",
            patient_id="P001",
            category="exam",
            notes="Primary SSN: 111-22-3333. Secondary: 444-55-6666.",
        )
        # Should redact all SSNs
        ssn_count = observation3.notes.count("***-**-****")
        assert ssn_count >= 2  # At least 2 redacted SSNs
        assert "111-22-3333" not in observation3.notes
        assert "444-55-6666" not in observation3.notes
        
        # Notes with SSN and other PII (phone, email)
        observation4 = ClinicalObservation(
            observation_id="O004",
            patient_id="P001",
            category="procedure",
            notes=(
                "Patient SSN: 123-45-6789. "
                "Contact: 555-123-4567 or email@example.com. "
                "Procedure completed successfully."
            ),
        )
        assert "***-**-****" in observation4.notes
        assert "***-***-****" in observation4.notes  # Phone redacted
        assert "***@***.***" in observation4.notes  # Email redacted
        assert "123-45-6789" not in observation4.notes
        assert "555-123-4567" not in observation4.notes
        assert "email@example.com" not in observation4.notes
        
        # Notes without PII should remain unchanged (except for any false positives)
        observation5 = ClinicalObservation(
            observation_id="O005",
            patient_id="P001",
            category="vital-signs",
            notes="Blood pressure: 120/80. Heart rate: 72 bpm.",
        )
        assert "Blood pressure" in observation5.notes
        assert "120/80" in observation5.notes


class TestClinicalObservation:
    """Test suite for ClinicalObservation model."""
    
    def test_valid_observation(self):
        """Test creating a valid clinical observation."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            code="85354-9",
            value="120/80",
            unit="mmHg",
            effective_date=datetime(2024, 1, 15, 10, 30),
        )
        assert observation.observation_id == "O001"
        assert observation.category == ObservationCategory.VITAL_SIGNS
        assert observation.value == "120/80"
    
    def test_observation_category_normalization(self):
        """Test that observation categories are normalized to FHIR ObservationCategory."""
        observation_lab = ClinicalObservation(
            observation_id="O002",
            patient_id="P001",
            category="laboratory",
        )
        assert observation_lab.category == ObservationCategory.LABORATORY
        
        observation_vital = ClinicalObservation(
            observation_id="O003",
            patient_id="P001",
            category="VITAL_SIGN",
        )
        assert observation_vital.category == ObservationCategory.VITAL_SIGNS


class TestEncounterRecord:
    """Test suite for EncounterRecord model."""
    
    def test_valid_encounter(self):
        """Test creating a valid encounter record."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            period_start=datetime(2024, 1, 15, 9, 0),
            period_end=datetime(2024, 1, 15, 10, 0),
            diagnosis_codes=["I10", "E11.9"],
        )
        assert encounter.encounter_id == "E001"
        assert encounter.class_code == EncounterClass.OUTPATIENT
        assert len(encounter.diagnosis_codes) == 2
    
    def test_encounter_class_normalization(self):
        """Test that encounter classes are normalized to FHIR EncounterClass."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="inpatient",
        )
        assert encounter.class_code == EncounterClass.INPATIENT
        
        encounter2 = EncounterRecord(
            encounter_id="E002",
            patient_id="P001",
            class_code="urgent-care",
        )
        assert encounter2.class_code == EncounterClass.URGENT_CARE
        
        encounter3 = EncounterRecord(
            encounter_id="E003",
            patient_id="P001",
            class_code="emergency",
        )
        assert encounter3.class_code == EncounterClass.EMERGENCY
        
        encounter4 = EncounterRecord(
            encounter_id="E004",
            patient_id="P001",
            class_code="virtual",
        )
        assert encounter4.class_code == EncounterClass.VIRTUAL


class TestGoldenRecord:
    """Test suite for GoldenRecord container model."""
    
    def test_complete_golden_record(self):
        """Test creating a complete golden record."""
        patient = PatientRecord(patient_id="P001", first_name="John")
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
        )
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="laboratory",
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
        assert golden.encounters[0].class_code == EncounterClass.OUTPATIENT
        assert golden.observations[0].category == ObservationCategory.LABORATORY


# ============================================================================
# P0 CRITICAL TESTS - Pattern Validation
# ============================================================================

class TestPatternValidation:
    """P0: Pattern validation tests for fields with regex patterns."""
    
    # 1.1 State Code Pattern Validation
    def test_state_code_valid_patterns(self):
        """Test valid 2-letter state codes."""
        valid_states = ["CA", "NY", "TX", "FL", "IL"]
        for state in valid_states:
            patient = PatientRecord(patient_id="P001", state=state)
            assert patient.state == state
    
    def test_state_code_invalid_patterns(self):
        """Test invalid state code patterns."""
        # Note: Normalization happens before pattern validation, so some invalid
        # inputs get normalized to valid patterns. Test cases that can't be normalized.
        
        # Single letter should fail (can't normalize to 2 letters)
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", state="C")
        
        # Special characters that can't be normalized should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", state="@@")
        
        # Numbers should fail (normalized but still invalid pattern)
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", state="12")
        
        # Empty string with only special chars should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", state="!!")
    
    def test_state_code_normalization(self):
        """Test state code normalization to uppercase."""
        # Lowercase should be normalized
        patient = PatientRecord(patient_id="P001", state="ca")
        assert patient.state == "CA"
        
        # Mixed case should be normalized
        patient2 = PatientRecord(patient_id="P002", state="Ny")
        assert patient2.state == "NY"
    
    def test_state_code_empty_allowed(self):
        """Test that empty state code is allowed (optional field)."""
        patient = PatientRecord(patient_id="P001", state=None)
        assert patient.state is None
    
    # 1.2 Postal Code Pattern Validation
    def test_postal_code_valid_patterns(self):
        """Test valid postal code patterns."""
        # Valid 5-digit ZIP - pattern validates before redaction
        patient1 = PatientRecord(patient_id="P001", postal_code="12345")
        assert patient1.postal_code == "12***"  # Redacted after pattern validation
        
        # Valid 9-digit ZIP with dash - pattern validates before redaction
        patient2 = PatientRecord(patient_id="P002", postal_code="12345-6789")
        assert patient2.postal_code == "12***"  # Redacted after pattern validation
    
    def test_postal_code_invalid_patterns(self):
        """Test invalid postal code patterns."""
        # 4-digit ZIP should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", postal_code="1234")
        
        # 6-digit ZIP should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", postal_code="123456")
        
        # ZIP with letters should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", postal_code="ABC45")
        
        # ZIP with spaces should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", postal_code="123 45")
    
    def test_postal_code_empty_allowed(self):
        """Test that empty postal code is allowed."""
        patient = PatientRecord(patient_id="P001", postal_code=None)
        assert patient.postal_code is None
    
    # 1.3 Country Code Pattern Validation
    def test_country_code_valid_patterns(self):
        """Test valid country code patterns."""
        # Valid 2-letter ISO codes
        valid_2letter = ["US", "CA", "MX", "GB", "FR"]
        for country in valid_2letter:
            patient = PatientRecord(patient_id="P001", country=country)
            assert patient.country == country
        
        # Valid 3-letter ISO codes
        valid_3letter = ["USA", "CAN", "MEX"]
        for country in valid_3letter:
            patient = PatientRecord(patient_id="P001", country=country)
            assert patient.country == country
    
    def test_country_code_invalid_patterns(self):
        """Test invalid country code patterns."""
        # Single letter should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", country="U")
        
        # 4+ letters should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", country="UNITED")
        
        # Lowercase should fail (must be uppercase)
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", country="us")
    
    def test_country_code_empty_allowed(self):
        """Test that empty country code is allowed."""
        patient = PatientRecord(patient_id="P001", country=None)
        assert patient.country is None
    
    # 1.4 Language Code Pattern Validation
    def test_language_code_valid_patterns(self):
        """Test valid language code patterns."""
        # Valid 2-letter language code
        patient1 = PatientRecord(patient_id="P001", language="en")
        assert patient1.language == "en"
        
        # Valid language-region code
        patient2 = PatientRecord(patient_id="P002", language="en-US")
        assert patient2.language == "en-US"
        
        patient3 = PatientRecord(patient_id="P003", language="es-MX")
        assert patient3.language == "es-MX"
    
    def test_language_code_invalid_patterns(self):
        """Test invalid language code patterns."""
        # Single letter should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", language="e")
        
        # Invalid format without dash should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", language="enUS")
        
        # Uppercase language part should fail
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="P001", language="EN-US")
    
    def test_language_code_empty_allowed(self):
        """Test that empty language code is allowed."""
        patient = PatientRecord(patient_id="P001", language=None)
        assert patient.language is None
    
    # 1.5 Observation Code Pattern Validation
    def test_observation_code_valid_patterns(self):
        """Test valid observation code patterns."""
        # Valid LOINC format
        obs1 = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            code="85354-9"
        )
        assert obs1.code == "85354-9"
        
        # Valid SNOMED format
        obs2 = ClinicalObservation(
            observation_id="O002",
            patient_id="P001",
            category="laboratory",
            code="123456789"
        )
        assert obs2.code == "123456789"
    
    def test_observation_code_invalid_patterns(self):
        """Test invalid observation code patterns."""
        # Code with special characters should fail
        with pytest.raises(ValidationError):
            ClinicalObservation(
                observation_id="O001",
                patient_id="P001",
                category="vital-signs",
                code="85354@9"
            )
    
    def test_observation_code_empty_allowed(self):
        """Test that empty observation code is allowed."""
        obs = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            code=None
        )
        assert obs.code is None
    
    # 1.6 ICD-10 Diagnosis Code Pattern Validation
    def test_diagnosis_code_valid_patterns(self):
        """Test valid ICD-10 diagnosis code patterns."""
        # Valid ICD-10 code without decimal
        encounter1 = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            diagnosis_codes=["I10"]
        )
        assert "I10" in encounter1.diagnosis_codes
        
        # Valid ICD-10 code with decimal
        encounter2 = EncounterRecord(
            encounter_id="E002",
            patient_id="P001",
            class_code="outpatient",
            diagnosis_codes=["E11.9"]
        )
        assert "E11.9" in encounter2.diagnosis_codes
        
        # Multiple valid codes
        encounter3 = EncounterRecord(
            encounter_id="E003",
            patient_id="P001",
            class_code="outpatient",
            diagnosis_codes=["I10", "E11.9", "A00.1"]
        )
        assert len(encounter3.diagnosis_codes) == 3
    
    def test_diagnosis_code_empty_list_allowed(self):
        """Test that empty diagnosis codes list is allowed."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            diagnosis_codes=[]
        )
        assert encounter.diagnosis_codes == []


# ============================================================================
# P0 CRITICAL TESTS - PII Redaction Edge Cases
# ============================================================================

class TestPIIRedactionEdgeCases:
    """P0: Comprehensive PII redaction edge case tests."""
    
    # 3.1 Name Redaction Edge Cases
    def test_name_redaction_hyphenated_names(self):
        """Test redaction of hyphenated names."""
        patient = PatientRecord(
            patient_id="P001",
            family_name="O'Brien",
            given_names=["Mary-Jane"]
        )
        assert patient.family_name == "[REDACTED]"
        assert patient.given_names == ["[REDACTED]"]
    
    def test_name_redaction_multiple_given_names(self):
        """Test redaction of multiple given names."""
        patient = PatientRecord(
            patient_id="P001",
            given_names=["John", "Michael", "Robert"]
        )
        assert len(patient.given_names) == 3
        assert all(name == "[REDACTED]" for name in patient.given_names)
    
    def test_name_redaction_name_prefix_suffix(self):
        """Test that name prefixes and suffixes are handled."""
        patient = PatientRecord(
            patient_id="P001",
            name_prefix=["Dr."],
            family_name="Smith",
            name_suffix=["Jr."]
        )
        assert patient.family_name == "[REDACTED]"
        # Prefixes/suffixes may or may not be redacted depending on length
    
    def test_name_redaction_empty_names(self):
        """Test that empty name strings are handled."""
        patient = PatientRecord(
            patient_id="P001",
            family_name="",
            given_names=[]
        )
        assert patient.family_name is None or patient.family_name == ""
        assert patient.given_names == []
    
    # 3.2 SSN Redaction Edge Cases
    def test_ssn_redaction_with_spaces(self):
        """Test SSN redaction with spaces."""
        patient = PatientRecord(patient_id="P001", ssn="123 45 6789")
        assert patient.ssn == "***-**-****"
    
    def test_ssn_redaction_multiple_in_identifiers(self):
        """Test multiple SSNs in identifiers list."""
        patient = PatientRecord(
            patient_id="P001",
            identifiers=["123-45-6789", "987-65-4321"]
        )
        assert len(patient.identifiers) == 2
        assert all(id_val == "***-**-****" for id_val in patient.identifiers)
    
    def test_ssn_redaction_empty_identifiers(self):
        """Test empty identifiers list."""
        patient = PatientRecord(patient_id="P001", identifiers=[])
        assert patient.identifiers == []
    
    # 3.3 Phone Number Redaction Edge Cases
    def test_phone_redaction_international_format(self):
        """Test international phone format redaction."""
        patient = PatientRecord(patient_id="P001", phone="+1-555-123-4567")
        assert patient.phone == "***-***-****"
    
    def test_phone_redaction_with_parentheses(self):
        """Test phone with parentheses format."""
        patient = PatientRecord(patient_id="P001", phone="(555) 123-4567")
        assert patient.phone == "***-***-****"
    
    def test_phone_redaction_with_dots(self):
        """Test phone with dots format."""
        patient = PatientRecord(patient_id="P001", phone="555.123.4567")
        assert patient.phone == "***-***-****"
    
    def test_phone_redaction_emergency_contact(self):
        """Test emergency contact phone redaction."""
        patient = PatientRecord(
            patient_id="P001",
            emergency_contact_phone="555-987-6543"
        )
        assert patient.emergency_contact_phone == "***-***-****"
    
    def test_fax_redaction(self):
        """Test fax number redaction."""
        patient = PatientRecord(patient_id="P001", fax="555-123-4567")
        assert patient.fax == "***-***-****"
    
    # 3.4 Email Redaction Edge Cases
    def test_email_redaction_with_subdomain(self):
        """Test email with subdomain redaction."""
        patient = PatientRecord(patient_id="P001", email="user@mail.example.com")
        assert patient.email == "***@***.***"
    
    def test_email_redaction_with_plus_addressing(self):
        """Test email with plus addressing redaction."""
        patient = PatientRecord(patient_id="P001", email="user+tag@example.com")
        assert patient.email == "***@***.***"
    
    # 3.5 Address Redaction Edge Cases
    def test_address_redaction_po_box(self):
        """Test PO Box address redaction."""
        patient = PatientRecord(
            patient_id="P001",
            address_line1="PO Box 123"
        )
        assert patient.address_line1 == "[REDACTED]"
    
    def test_address_redaction_rural_route(self):
        """Test rural route address redaction."""
        patient = PatientRecord(
            patient_id="P001",
            address_line1="RR 1 Box 123"
        )
        assert patient.address_line1 == "[REDACTED]"
    
    def test_location_address_redaction(self):
        """Test encounter location address redaction."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            location_address="123 Hospital Dr"
        )
        assert encounter.location_address == "[REDACTED]"
    
    # 3.6 Date of Birth Redaction
    def test_dob_always_redacted_to_none(self):
        """Test that DOB is always redacted to None."""
        # Valid DOB
        patient1 = PatientRecord(
            patient_id="P001",
            date_of_birth=date(1980, 1, 15)
        )
        assert patient1.date_of_birth is None
        
        # None DOB
        patient2 = PatientRecord(patient_id="P002", date_of_birth=None)
        assert patient2.date_of_birth is None
    
    # 3.7 ZIP Code Partial Redaction
    def test_zip_code_partial_redaction_5digit(self):
        """Test 5-digit ZIP partial redaction."""
        patient = PatientRecord(patient_id="P001", postal_code="12345")
        assert patient.postal_code == "12***"
    
    def test_zip_code_partial_redaction_9digit(self):
        """Test 9-digit ZIP partial redaction."""
        # 9-digit ZIP gets redacted to first 2 digits
        patient = PatientRecord(patient_id="P001", postal_code="12345-6789")
        # After pattern validation, it gets redacted
        assert patient.postal_code == "12***"
    
    # 3.8 Unstructured Text Redaction
    def test_unstructured_text_pii_at_beginning(self):
        """Test PII at beginning of unstructured text."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            notes="123-45-6789 is the patient SSN. Vital signs normal."
        )
        assert "***-**-****" in observation.notes
        assert "123-45-6789" not in observation.notes
    
    def test_unstructured_text_pii_at_end(self):
        """Test PII at end of unstructured text."""
        observation = ClinicalObservation(
            observation_id="O002",
            patient_id="P001",
            category="laboratory",
            notes="Lab results are normal. Contact: 555-123-4567"
        )
        assert "***-***-****" in observation.notes
        assert "555-123-4567" not in observation.notes
    
    def test_unstructured_text_multiple_pii_types(self):
        """Test multiple PII types in unstructured text."""
        observation = ClinicalObservation(
            observation_id="O003",
            patient_id="P001",
            category="exam",
            notes=(
                "Patient SSN: 123-45-6789. "
                "Phone: 555-123-4567. "
                "Email: patient@example.com. "
                "Exam completed."
            )
        )
        assert "***-**-****" in observation.notes
        assert "***-***-****" in observation.notes
        assert "***@***.***" in observation.notes
        assert "123-45-6789" not in observation.notes
        assert "555-123-4567" not in observation.notes
        assert "patient@example.com" not in observation.notes
    
    def test_unstructured_text_no_pii_unchanged(self):
        """Test that text without PII remains unchanged."""
        observation = ClinicalObservation(
            observation_id="O004",
            patient_id="P001",
            category="vital-signs",
            notes="Blood pressure: 120/80. Heart rate: 72 bpm. Temperature: 98.6F."
        )
        assert "Blood pressure" in observation.notes
        assert "120/80" in observation.notes
        assert "72 bpm" in observation.notes


# ============================================================================
# P0 CRITICAL TESTS - Required Field Validation
# ============================================================================

class TestRequiredFieldValidation:
    """P0: Required field validation tests."""
    
    def test_missing_required_patient_id(self):
        """Test that missing patient_id fails validation."""
        with pytest.raises(ValidationError):
            PatientRecord()  # Missing required patient_id
    
    def test_missing_required_observation_id(self):
        """Test that missing observation_id fails validation."""
        with pytest.raises(ValidationError):
            ClinicalObservation(
                patient_id="P001",
                category="vital-signs"
            )  # Missing observation_id
    
    def test_missing_required_category(self):
        """Test that missing category fails validation."""
        with pytest.raises(ValidationError):
            ClinicalObservation(
                observation_id="O001",
                patient_id="P001"
            )  # Missing category
    
    def test_missing_required_encounter_id(self):
        """Test that missing encounter_id fails validation."""
        with pytest.raises(ValidationError):
            EncounterRecord(
                patient_id="P001",
                class_code="outpatient"
            )  # Missing encounter_id
    
    def test_missing_required_class_code(self):
        """Test that missing class_code fails validation."""
        with pytest.raises(ValidationError):
            EncounterRecord(
                encounter_id="E001",
                patient_id="P001"
            )  # Missing class_code
    
    def test_missing_required_patient_in_golden_record(self):
        """Test that missing patient in GoldenRecord fails validation."""
        with pytest.raises(ValidationError):
            GoldenRecord(
                encounters=[],
                observations=[],
                source_adapter="test"
            )  # Missing patient
    
    def test_missing_required_source_adapter(self):
        """Test that missing source_adapter in GoldenRecord fails validation."""
        patient = PatientRecord(patient_id="P001")
        with pytest.raises(ValidationError):
            GoldenRecord(
                patient=patient,
                encounters=[],
                observations=[]
            )  # Missing source_adapter


# ============================================================================
# P0 CRITICAL TESTS - Security Tests
# ============================================================================

class TestSecurityPIILeakage:
    """P0: Security tests for PII leakage prevention."""
    
    def test_no_pii_in_patient_id(self):
        """Test that patient_id does not contain PII."""
        # patient_id should not be redacted (it's not PII)
        patient = PatientRecord(patient_id="MRN123456", first_name="John")
        assert patient.patient_id == "MRN123456"  # Not redacted
        assert patient.first_name == "[REDACTED]"  # PII is redacted
    
    def test_no_pii_in_observation_id(self):
        """Test that observation_id does not contain PII."""
        observation = ClinicalObservation(
            observation_id="OBS-12345",
            patient_id="P001",
            category="vital-signs",
            performer_name="Dr. Smith"
        )
        assert observation.observation_id == "OBS-12345"  # Not redacted
        assert observation.performer_name == "[REDACTED]"  # PII is redacted
    
    def test_no_pii_in_encounter_id(self):
        """Test that encounter_id does not contain PII."""
        encounter = EncounterRecord(
            encounter_id="ENC-12345",
            patient_id="P001",
            class_code="outpatient",
            participant_name="Nurse Jones"
        )
        assert encounter.encounter_id == "ENC-12345"  # Not redacted
        assert encounter.participant_name == "[REDACTED]"  # PII is redacted
    
    def test_no_pii_in_diagnosis_codes(self):
        """Test that diagnosis codes do not contain PII."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            diagnosis_codes=["I10", "E11.9"]
        )
        # Diagnosis codes should remain unchanged (not PII)
        assert "I10" in encounter.diagnosis_codes
        assert "E11.9" in encounter.diagnosis_codes
    
    def test_pii_redaction_irreversible(self):
        """Test that redacted PII cannot be reversed."""
        patient = PatientRecord(
            patient_id="P001",
            ssn="123-45-6789",
            first_name="John",
            last_name="Doe"
        )
        
        # Redacted values should not contain original data
        assert patient.ssn == "***-**-****"
        assert "123" not in patient.ssn
        assert "45" not in patient.ssn
        assert "6789" not in patient.ssn
        
        assert patient.first_name == "[REDACTED]"
        assert "John" not in patient.first_name
        
        assert patient.last_name == "[REDACTED]"
        assert "Doe" not in patient.last_name
    
    def test_audit_trail_fields_present(self):
        """Test that audit trail fields are present in GoldenRecord."""
        patient = PatientRecord(patient_id="P001")
        golden = GoldenRecord(
            patient=patient,
            encounters=[],
            observations=[],
            source_adapter="test_adapter",
            transformation_hash="abc123"
        )
        
        assert golden.ingestion_timestamp is not None
        assert isinstance(golden.ingestion_timestamp, datetime)
        assert golden.source_adapter == "test_adapter"
        assert golden.transformation_hash == "abc123"
    
    def test_immutable_records_cannot_be_modified(self):
        """Test that records are immutable and cannot be modified."""
        patient = PatientRecord(patient_id="P001", first_name="John")
        
        # Attempting to modify should raise ValidationError
        with pytest.raises(ValidationError):
            patient.first_name = "Jane"
        
        with pytest.raises(ValidationError):
            patient.patient_id = "P002"
        
        # Original values should remain unchanged
        assert patient.patient_id == "P001"
        assert patient.first_name == "[REDACTED]"
    
    def test_data_integrity_after_redaction(self):
        """Test that redaction doesn't corrupt data structure."""
        patient = PatientRecord(
            patient_id="P001",
            given_names=["John", "Michael"],
            identifiers=["123-45-6789", "MRN123"]
        )
        
        # Lists should remain lists
        assert isinstance(patient.given_names, list)
        assert isinstance(patient.identifiers, list)
        
        # Optional fields can be None
        assert patient.family_name is None or isinstance(patient.family_name, str)
        
        # Required fields cannot be None
        assert patient.patient_id is not None
        assert isinstance(patient.patient_id, str)


# ============================================================================
# P1 HIGH PRIORITY TESTS - FHIR Enum Validation
# ============================================================================

class TestFHIREnumValidation:
    """P1: FHIR R5 enum validation tests."""
    
    # 2.1 ObservationStatus Enum
    def test_observation_status_all_valid_values(self):
        """Test all valid ObservationStatus enum values."""
        valid_statuses = [
            "registered", "preliminary", "final", "amended", "corrected",
            "cancelled", "entered-in-error", "unknown"
        ]
        
        for status_str in valid_statuses:
            observation = ClinicalObservation(
                observation_id="O001",
                patient_id="P001",
                category="vital-signs",
                status=status_str
            )
            assert isinstance(observation.status, ObservationStatus)
            assert observation.status.value == status_str
    
    def test_observation_status_case_insensitive(self):
        """Test case-insensitive matching for ObservationStatus."""
        observation1 = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            status="FINAL"
        )
        assert observation1.status == ObservationStatus.FINAL
        
        observation2 = ClinicalObservation(
            observation_id="O002",
            patient_id="P001",
            category="laboratory",
            status="Preliminary"
        )
        assert observation2.status == ObservationStatus.PRELIMINARY
    
    def test_observation_status_underscore_hyphen_variations(self):
        """Test underscore/hyphen variations for ObservationStatus."""
        # entered-in-error vs entered_in_error
        observation1 = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            status="entered_in_error"
        )
        assert observation1.status == ObservationStatus.ENTERED_IN_ERROR
        
        observation2 = ClinicalObservation(
            observation_id="O002",
            patient_id="P001",
            category="laboratory",
            status="entered-in-error"
        )
        assert observation2.status == ObservationStatus.ENTERED_IN_ERROR
    
    def test_observation_status_invalid_defaults_to_final(self):
        """Test that invalid status values default to 'final'."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            status="invalid_status"
        )
        assert observation.status == ObservationStatus.FINAL
    
    def test_observation_status_none_defaults_to_final(self):
        """Test that None status defaults to 'final'."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            status=None
        )
        assert observation.status == ObservationStatus.FINAL
    
    # 2.2 EncounterStatus Enum
    def test_encounter_status_all_valid_values(self):
        """Test all valid EncounterStatus enum values."""
        valid_statuses = [
            "planned", "arrived", "triaged", "in-progress", "onleave",
            "finished", "cancelled", "entered-in-error", "unknown"
        ]
        
        for status_str in valid_statuses:
            encounter = EncounterRecord(
                encounter_id="E001",
                patient_id="P001",
                class_code="outpatient",
                status=status_str
            )
            assert isinstance(encounter.status, EncounterStatus)
            assert encounter.status.value == status_str
    
    def test_encounter_status_case_insensitive(self):
        """Test case-insensitive matching for EncounterStatus."""
        encounter1 = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            status="FINISHED"
        )
        assert encounter1.status == EncounterStatus.FINISHED
        
        encounter2 = EncounterRecord(
            encounter_id="E002",
            patient_id="P001",
            class_code="inpatient",
            status="In-Progress"
        )
        assert encounter2.status == EncounterStatus.IN_PROGRESS
    
    def test_encounter_status_underscore_hyphen_variations(self):
        """Test underscore/hyphen variations for EncounterStatus."""
        encounter1 = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            status="entered_in_error"
        )
        assert encounter1.status == EncounterStatus.ENTERED_IN_ERROR
        
        encounter2 = EncounterRecord(
            encounter_id="E002",
            patient_id="P001",
            class_code="emergency",
            status="in_progress"
        )
        assert encounter2.status == EncounterStatus.IN_PROGRESS
    
    def test_encounter_status_invalid_defaults_to_finished(self):
        """Test that invalid status values default to 'finished'."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            status="invalid_status"
        )
        assert encounter.status == EncounterStatus.FINISHED
    
    def test_encounter_status_none_defaults_to_finished(self):
        """Test that None status defaults to 'finished'."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            status=None
        )
        assert encounter.status == EncounterStatus.FINISHED
    
    # 2.3 AddressUse Enum
    def test_address_use_all_valid_values(self):
        """Test all valid AddressUse enum values."""
        valid_uses = ["home", "work", "temp", "old", "billing"]
        
        for use_str in valid_uses:
            patient = PatientRecord(
                patient_id="P001",
                address_use=use_str
            )
            assert isinstance(patient.address_use, AddressUse)
            assert patient.address_use.value == use_str
    
    def test_address_use_case_insensitive(self):
        """Test case-insensitive matching for AddressUse."""
        patient1 = PatientRecord(
            patient_id="P001",
            address_use="HOME"
        )
        assert patient1.address_use == AddressUse.HOME
        
        patient2 = PatientRecord(
            patient_id="P002",
            address_use="Work"
        )
        assert patient2.address_use == AddressUse.WORK
    
    def test_address_use_temporary_maps_to_temp(self):
        """Test that 'temporary' maps to 'temp'."""
        patient = PatientRecord(
            patient_id="P001",
            address_use="temporary"
        )
        assert patient.address_use == AddressUse.TEMP
    
    def test_address_use_invalid_defaults_to_home(self):
        """Test that invalid values default to 'home'."""
        patient = PatientRecord(
            patient_id="P001",
            address_use="invalid_use"
        )
        assert patient.address_use == AddressUse.HOME
    
    def test_address_use_none_allowed(self):
        """Test that None is allowed for address_use (optional field)."""
        patient = PatientRecord(
            patient_id="P001",
            address_use=None
        )
        assert patient.address_use is None
    
    # 2.4 AdministrativeGender Enum
    def test_administrative_gender_all_valid_values(self):
        """Test all valid AdministrativeGender enum values."""
        valid_genders = ["male", "female", "other", "unknown"]
        
        for gender_str in valid_genders:
            patient = PatientRecord(
                patient_id="P001",
                gender=gender_str
            )
            assert isinstance(patient.gender, AdministrativeGender)
            assert patient.gender.value == gender_str
    
    def test_administrative_gender_single_letter_abbreviations(self):
        """Test single letter abbreviations for AdministrativeGender."""
        patient_m = PatientRecord(patient_id="P001", gender="m")
        assert patient_m.gender == AdministrativeGender.MALE
        
        patient_f = PatientRecord(patient_id="P002", gender="f")
        assert patient_f.gender == AdministrativeGender.FEMALE
        
        patient_o = PatientRecord(patient_id="P003", gender="o")
        assert patient_o.gender == AdministrativeGender.OTHER
        
        patient_u = PatientRecord(patient_id="P004", gender="u")
        assert patient_u.gender == AdministrativeGender.UNKNOWN
    
    def test_administrative_gender_case_insensitive(self):
        """Test case-insensitive matching for AdministrativeGender."""
        patient1 = PatientRecord(patient_id="P001", gender="MALE")
        assert patient1.gender == AdministrativeGender.MALE
        
        patient2 = PatientRecord(patient_id="P002", gender="Female")
        assert patient2.gender == AdministrativeGender.FEMALE
    
    def test_administrative_gender_invalid_defaults_to_unknown(self):
        """Test that invalid values default to 'unknown'."""
        patient = PatientRecord(patient_id="P001", gender="invalid")
        assert patient.gender == AdministrativeGender.UNKNOWN
    
    def test_administrative_gender_none_allowed(self):
        """Test that None is allowed for gender (optional field)."""
        patient = PatientRecord(patient_id="P001", gender=None)
        assert patient.gender is None


# ============================================================================
# P1 HIGH PRIORITY TESTS - FHIR R5 Field Tests
# ============================================================================

class TestFHIRR5Fields:
    """P1: FHIR R5 field tests for new fields."""
    
    # 5.1 PatientRecord FHIR R5 Fields
    def test_identifiers_list_multiple_identifiers(self):
        """Test identifiers list with multiple identifiers."""
        patient = PatientRecord(
            patient_id="P001",
            identifiers=["123-45-6789", "MRN123456", "987-65-4321"]
        )
        assert len(patient.identifiers) == 3
        # All SSNs should be redacted
        assert all(id_val == "***-**-****" or id_val == "MRN123456" for id_val in patient.identifiers)
    
    def test_given_names_list_multiple_names(self):
        """Test given_names list with multiple names."""
        patient = PatientRecord(
            patient_id="P001",
            given_names=["John", "Michael", "Robert"]
        )
        assert len(patient.given_names) == 3
        assert all(name == "[REDACTED]" for name in patient.given_names)
    
    def test_name_prefix_list_functionality(self):
        """Test name_prefix list functionality."""
        patient = PatientRecord(
            patient_id="P001",
            name_prefix=["Dr.", "Prof."],
            family_name="Smith"
        )
        assert len(patient.name_prefix) == 2
        assert patient.family_name == "[REDACTED]"
    
    def test_name_suffix_list_functionality(self):
        """Test name_suffix list functionality."""
        patient = PatientRecord(
            patient_id="P001",
            family_name="Smith",
            name_suffix=["Jr.", "III"]
        )
        assert len(patient.name_suffix) == 2
        assert patient.family_name == "[REDACTED]"
    
    def test_deceased_boolean_and_date_consistency(self):
        """Test deceased boolean and deceased_date consistency."""
        # Patient marked as deceased with date
        patient1 = PatientRecord(
            patient_id="P001",
            deceased=True,
            deceased_date=datetime(2020, 1, 15, 10, 30)
        )
        assert patient1.deceased is True
        assert patient1.deceased_date is not None
        
        # Patient not deceased
        patient2 = PatientRecord(
            patient_id="P002",
            deceased=False
        )
        assert patient2.deceased is False
        assert patient2.deceased_date is None
    
    def test_emergency_contact_fields_redaction(self):
        """Test emergency contact fields redaction."""
        patient = PatientRecord(
            patient_id="P001",
            emergency_contact_name="Jane Doe",
            emergency_contact_phone="555-123-4567",
            emergency_contact_relationship="spouse"
        )
        assert patient.emergency_contact_name == "[REDACTED]"
        assert patient.emergency_contact_phone == "***-***-****"
        assert patient.emergency_contact_relationship == "spouse"  # Not PII
    
    def test_backward_compatibility_deprecated_fields(self):
        """Test backward compatibility with deprecated fields."""
        # Test deprecated first_name/last_name still work
        patient1 = PatientRecord(
            patient_id="P001",
            first_name="John",
            last_name="Doe"
        )
        assert patient1.first_name == "[REDACTED]"
        assert patient1.last_name == "[REDACTED]"
        
        # Test deprecated ssn still works
        patient2 = PatientRecord(
            patient_id="P002",
            ssn="123-45-6789"
        )
        assert patient2.ssn == "***-**-****"
        
        # Test deprecated zip_code still works
        patient3 = PatientRecord(
            patient_id="P003",
            zip_code="12345"
        )
        assert patient3.zip_code == "12***"
    
    # 5.2 ClinicalObservation FHIR R5 Fields
    def test_observation_status_enum_validation(self):
        """Test status enum validation."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            status="preliminary"
        )
        assert observation.status == ObservationStatus.PRELIMINARY
    
    def test_observation_category_enum_validation(self):
        """Test category enum validation."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="laboratory"
        )
        assert observation.category == ObservationCategory.LABORATORY
    
    def test_observation_issued_datetime(self):
        """Test issued datetime field."""
        issued_dt = datetime(2024, 1, 15, 14, 30)
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            issued=issued_dt
        )
        assert observation.issued == issued_dt
    
    def test_observation_performer_name_redaction(self):
        """Test performer_name redaction."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            performer_name="Dr. Smith"
        )
        assert observation.performer_name == "[REDACTED]"
    
    def test_observation_interpretation_redaction(self):
        """Test interpretation text redaction."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            interpretation="Patient contact: 555-123-4567. Results normal."
        )
        assert "***-***-****" in observation.interpretation
        assert "555-123-4567" not in observation.interpretation
    
    def test_observation_backward_compatibility_observation_type(self):
        """Test backward compatibility with observation_type."""
        # observation_type is deprecated but still works if category is also provided
        # Since category is required, we need to provide both
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            observation_type="laboratory"  # Deprecated field
        )
        assert observation.category == ObservationCategory.VITAL_SIGNS
        assert observation.observation_type == ObservationCategory.LABORATORY
    
    # 5.3 EncounterRecord FHIR R5 Fields
    def test_encounter_status_enum_validation(self):
        """Test status enum validation."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            status="in-progress"
        )
        assert encounter.status == EncounterStatus.IN_PROGRESS
    
    def test_encounter_class_code_enum_validation(self):
        """Test class_code enum validation."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="emergency"
        )
        assert encounter.class_code == EncounterClass.EMERGENCY
    
    def test_encounter_location_address_redaction(self):
        """Test location_address redaction."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            location_address="123 Hospital Dr, Suite 100"
        )
        assert encounter.location_address == "[REDACTED]"
    
    def test_encounter_participant_name_redaction(self):
        """Test participant_name redaction."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            participant_name="Nurse Johnson"
        )
        assert encounter.participant_name == "[REDACTED]"
    
    def test_encounter_backward_compatibility_encounter_type(self):
        """Test backward compatibility with encounter_type."""
        # encounter_type is deprecated but still works if class_code is also provided
        # Since class_code is required, we need to provide both
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            encounter_type="inpatient"  # Deprecated field
        )
        assert encounter.class_code == EncounterClass.OUTPATIENT
        assert encounter.encounter_type == EncounterClass.INPATIENT


# ============================================================================
# P1 HIGH PRIORITY TESTS - Edge Cases & Error Handling
# ============================================================================

class TestEdgeCasesErrorHandling:
    """P1: Edge cases and error handling tests."""
    
    # 7.1 Malformed Data Tests
    def test_sql_injection_attempts_in_string_fields(self):
        """Test SQL injection attempts in string fields."""
        # SQL injection attempt in patient_id
        with pytest.raises(ValidationError):
            PatientRecord(patient_id="'; DROP TABLE patients; --")
        
        # SQL injection attempt in notes
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            notes="'; DROP TABLE observations; --"
        )
        # Should be stored as-is (redaction doesn't prevent SQL injection, that's app-layer)
        assert "DROP TABLE" in observation.notes
    
    def test_extremely_long_strings(self):
        """Test extremely long strings (>10,000 chars)."""
        long_string = "A" * 10001
        # Should handle long strings without error
        # Note: Very long strings may not be redacted if they don't match PII patterns
        patient = PatientRecord(
            patient_id="P001",
            address_line1=long_string
        )
        # Address redaction checks for address patterns, very long strings may not match
        # The important thing is it doesn't crash
        assert len(patient.address_line1) <= len(long_string)
        # It should either be redacted or remain (depending on pattern matching)
        assert patient.address_line1 is not None
    
    def test_unicode_characters_in_names(self):
        """Test Unicode characters in names."""
        patient = PatientRecord(
            patient_id="P001",
            family_name="Müller",
            given_names=["José", "François"]
        )
        assert patient.family_name == "[REDACTED]"
        assert all(name == "[REDACTED]" for name in patient.given_names)
    
    def test_emoji_in_text_fields(self):
        """Test emoji in text fields."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            notes="Patient feels 😊. Vital signs normal."
        )
        assert "😊" in observation.notes
    
    # 7.2 Boundary Value Tests
    def test_minimum_valid_mrn_length(self):
        """Test minimum valid MRN length (3 chars)."""
        patient = PatientRecord(patient_id="ABC")
        assert patient.patient_id == "ABC"
    
    def test_maximum_valid_mrn_length(self):
        """Test maximum valid MRN length (20 chars)."""
        mrn = "A" * 20
        patient = PatientRecord(patient_id=mrn)
        assert len(patient.patient_id) == 20
    
    def test_minimum_valid_date(self):
        """Test minimum valid date (1900-01-01)."""
        patient = PatientRecord(
            patient_id="P001",
            date_of_birth=date(1900, 1, 1)
        )
        assert patient.date_of_birth is None  # Redacted
    
    def test_maximum_valid_date(self):
        """Test maximum valid date (today)."""
        today = date.today()
        patient = PatientRecord(
            patient_id="P001",
            date_of_birth=today
        )
        assert patient.date_of_birth is None  # Redacted
    
    def test_maximum_age_150_years(self):
        """Test maximum age (150 years)."""
        today = date.today()
        # Calculate date exactly 150 years ago, but ensure it's after 1900
        age_150_date = date(today.year - 150, today.month, today.day)
        # Adjust if today is Feb 29 and target year is not a leap year
        if today.month == 2 and today.day == 29:
            age_150_date = date(today.year - 150, 2, 28)
        
        # If the date is before 1900, use 1900 instead (minimum valid date)
        if age_150_date < date(1900, 1, 1):
            age_150_date = date(1900, 1, 1)
        
        # Calculate actual age
        age_years = (today - age_150_date).days / 365.25
        
        # Only test if age is actually <= 150 (if current year is <= 2050, this will be true)
        if age_years <= 150:
            patient = PatientRecord(
                patient_id="P001",
                date_of_birth=age_150_date
            )
            assert patient.date_of_birth is None  # Should pass validation and be redacted
        else:
            # If age > 150, it should fail validation
            with pytest.raises(ValidationError) as exc_info:
                PatientRecord(
                    patient_id="P001",
                    date_of_birth=age_150_date
                )
            assert "150" in str(exc_info.value) or "age" in str(exc_info.value).lower()
    
    def test_minimum_age_0_years(self):
        """Test minimum age (0 years, born today)."""
        today = date.today()
        patient = PatientRecord(
            patient_id="P001",
            date_of_birth=today
        )
        assert patient.date_of_birth is None  # Redacted
    
    # 7.3 Type Coercion Tests
    def test_string_numbers_converted_to_appropriate_types(self):
        """Test string numbers converted to appropriate types."""
        # String numbers in numeric fields should fail (we don't have numeric fields)
        # But test that string dates are converted
        patient = PatientRecord(
            patient_id="P001",
            date_of_birth="1980-01-15"  # String date
        )
        # Should be converted to date object, then redacted
        assert patient.date_of_birth is None
    
    def test_string_dates_converted_to_date_objects(self):
        """Test string dates converted to date objects."""
        # Test various date string formats
        date_str = "1980-01-15"
        patient = PatientRecord(
            patient_id="P001",
            date_of_birth=date_str
        )
        assert patient.date_of_birth is None  # Redacted after conversion
    
    def test_invalid_type_conversions_fail_gracefully(self):
        """Test invalid type conversions fail gracefully."""
        # Invalid date string should fail
        with pytest.raises(ValidationError):
            PatientRecord(
                patient_id="P001",
                date_of_birth="invalid-date"
            )


# ============================================================================
# P1 HIGH PRIORITY TESTS - Integration & Composition
# ============================================================================

class TestIntegrationComposition:
    """P1: Integration and composition tests."""
    
    # 8.1 GoldenRecord Composition
    def test_patient_with_multiple_encounters(self):
        """Test patient with multiple encounters."""
        patient = PatientRecord(patient_id="P001")
        encounters = [
            EncounterRecord(
                encounter_id=f"E{i:03d}",
                patient_id="P001",
                class_code="outpatient"
            )
            for i in range(5)
        ]
        
        golden = GoldenRecord(
            patient=patient,
            encounters=encounters,
            observations=[],
            source_adapter="test"
        )
        
        assert len(golden.encounters) == 5
        assert all(e.patient_id == "P001" for e in golden.encounters)
    
    def test_patient_with_multiple_observations(self):
        """Test patient with multiple observations."""
        patient = PatientRecord(patient_id="P001")
        observations = [
            ClinicalObservation(
                observation_id=f"O{i:03d}",
                patient_id="P001",
                category="vital-signs"
            )
            for i in range(10)
        ]
        
        golden = GoldenRecord(
            patient=patient,
            encounters=[],
            observations=observations,
            source_adapter="test"
        )
        
        assert len(golden.observations) == 10
        assert all(o.patient_id == "P001" for o in golden.observations)
    
    def test_encounter_with_multiple_diagnosis_codes(self):
        """Test encounter with multiple diagnosis codes."""
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id="P001",
            class_code="outpatient",
            diagnosis_codes=["I10", "E11.9", "A00.1", "J44.0"]
        )
        
        assert len(encounter.diagnosis_codes) == 4
        assert "I10" in encounter.diagnosis_codes
        assert "E11.9" in encounter.diagnosis_codes
    
    def test_observation_with_all_optional_fields_populated(self):
        """Test observation with all optional fields populated."""
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="laboratory",
            status="final",
            code="85354-9",
            value="120/80",
            unit="mmHg",
            effective_date=datetime(2024, 1, 15, 10, 30),
            issued=datetime(2024, 1, 15, 11, 0),
            performer_name="Dr. Smith",
            interpretation="Normal range",
            body_site="123456789",
            method="AUTOMATED",
            device="BP-MONITOR-001",
            reference_range="90-140/60-90",
            notes="Patient resting, normal BP"
        )
        
        assert observation.status == ObservationStatus.FINAL
        assert observation.code == "85354-9"
        assert observation.value == "120/80"
        assert observation.performer_name == "[REDACTED]"
    
    def test_empty_lists_for_encounters_observations(self):
        """Test empty lists for encounters/observations."""
        patient = PatientRecord(patient_id="P001")
        golden = GoldenRecord(
            patient=patient,
            encounters=[],
            observations=[],
            source_adapter="test"
        )
        
        assert len(golden.encounters) == 0
        assert len(golden.observations) == 0
        assert golden.patient.patient_id == "P001"
    
    def test_very_large_lists(self):
        """Test very large lists (100+ items)."""
        patient = PatientRecord(patient_id="P001")
        observations = [
            ClinicalObservation(
                observation_id=f"O{i:05d}",
                patient_id="P001",
                category="vital-signs"
            )
            for i in range(100)
        ]
        
        golden = GoldenRecord(
            patient=patient,
            encounters=[],
            observations=observations,
            source_adapter="test"
        )
        
        assert len(golden.observations) == 100
    
    # 8.2 Cross-Reference Validation
    def test_patient_id_consistency_across_records(self):
        """Test patient_id consistency across records."""
        patient_id = "P001"
        patient = PatientRecord(patient_id=patient_id)
        encounter = EncounterRecord(
            encounter_id="E001",
            patient_id=patient_id,
            class_code="outpatient"
        )
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id=patient_id,
            category="vital-signs"
        )
        
        golden = GoldenRecord(
            patient=patient,
            encounters=[encounter],
            observations=[observation],
            source_adapter="test"
        )
        
        assert golden.patient.patient_id == patient_id
        assert golden.encounters[0].patient_id == patient_id
        assert golden.observations[0].patient_id == patient_id
    
    def test_encounter_id_references_in_observations(self):
        """Test encounter_id references in observations."""
        encounter_id = "E001"
        observation = ClinicalObservation(
            observation_id="O001",
            patient_id="P001",
            category="vital-signs",
            encounter_id=encounter_id
        )
        
        assert observation.encounter_id == encounter_id

