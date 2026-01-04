"""Golden Record Schema Definitions.

This module defines the canonical data models (Golden Records) for clinical entities.
These schemas represent the "ideal" structure that all ingested data must conform to
after passing through the Safety Layer and Sieve.

Security Impact:
    - All PII fields are explicitly marked and will be redacted by the Sieve
    - Schema validation prevents malformed data from reaching persistence
    - Type safety enforced at runtime via Pydantic V2

Architecture:
    - Pure domain models with zero infrastructure dependencies
    - Models are immutable and validated before use
    - Follows Hexagonal Architecture: Domain Core is isolated from Adapters
"""

from datetime import date, datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain.enums import (
    AdministrativeGender,
    EncounterClass,
    ObservationCategory,
)
from src.domain.services import RedactorService


class PatientRecord(BaseModel):
    """Golden record for patient demographic information.
    
    Security Impact: Contains PII fields (name, ssn, dob) that must be redacted
    before persistence. The Sieve layer will mask these values.
    
    Parameters:
        patient_id: Unique identifier (non-PII, safe to persist)
        first_name: Patient first name (PII - will be redacted)
        last_name: Patient last name (PII - will be redacted)
        date_of_birth: Patient date of birth (PII - will be redacted)
        ssn: Social Security Number (PII - will be redacted)
        gender: Patient gender (standardized values)
        address_line1: Street address (PII - will be redacted)
        address_line2: Apartment/suite (PII - will be redacted)
        city: City name (may be redacted depending on policy)
        state: State code (standardized 2-letter code)
        zip_code: ZIP code (may be partially redacted)
        phone: Phone number (PII - will be redacted)
        email: Email address (PII - will be redacted)
    """
    
    patient_id: str = Field(..., description="Unique patient identifier")
    first_name: Optional[str] = Field(None, description="Patient first name (PII)")
    last_name: Optional[str] = Field(None, description="Patient last name (PII)")
    date_of_birth: Optional[date] = Field(None, description="Date of birth (PII)")
    ssn: Optional[str] = Field(None, description="Social Security Number (PII)")
    gender: Optional[AdministrativeGender] = Field(
        None, description="Gender (FHIR AdministrativeGender)"
    )
    address_line1: Optional[str] = Field(None, description="Street address (PII)")
    address_line2: Optional[str] = Field(None, description="Address line 2 (PII)")
    city: Optional[str] = Field(None, description="City name")
    state: Optional[str] = Field(None, description="State code (2-letter)")
    zip_code: Optional[str] = Field(None, description="ZIP code")
    phone: Optional[str] = Field(None, description="Phone number (PII)")
    email: Optional[str] = Field(None, description="Email address (PII)")
    
    @field_validator("gender", mode="before")
    @classmethod
    def validate_gender(cls, v) -> Optional[AdministrativeGender]:
        """Convert string values to FHIR AdministrativeGender enum.
        
        Accepts various string formats and normalizes to FHIR-compliant values.
        """
        if v is None:
            return v
        if isinstance(v, AdministrativeGender):
            return v
        
        v_str = str(v).strip().lower()
        # Map common variations to FHIR values
        mapping = {
            "m": AdministrativeGender.MALE,
            "male": AdministrativeGender.MALE,
            "f": AdministrativeGender.FEMALE,
            "female": AdministrativeGender.FEMALE,
            "o": AdministrativeGender.OTHER,
            "other": AdministrativeGender.OTHER,
            "u": AdministrativeGender.UNKNOWN,
            "unknown": AdministrativeGender.UNKNOWN,
        }
        return mapping.get(v_str, AdministrativeGender.UNKNOWN)
    
    @field_validator("state")
    @classmethod
    def validate_state(cls, v: Optional[str]) -> Optional[str]:
        """Normalize state codes to uppercase."""
        if v is None:
            return v
        return v.upper().strip()[:2] if len(v.strip()) >= 2 else v.upper().strip()
    
    @field_validator("ssn", mode="before")
    @classmethod
    def redact_ssn(cls, v: Optional[str]) -> Optional[str]:
        """Redact SSN using RedactorService.
        
        Security Impact: SSN is redacted before being stored in the model.
        """
        return RedactorService.redact_ssn(v)
    
    @field_validator("first_name", mode="before")
    @classmethod
    def redact_first_name(cls, v: Optional[str]) -> Optional[str]:
        """Redact first name using RedactorService.
        
        Security Impact: First name is redacted before being stored in the model.
        """
        return RedactorService.redact_name(v)
    
    @field_validator("last_name", mode="before")
    @classmethod
    def redact_last_name(cls, v: Optional[str]) -> Optional[str]:
        """Redact last name using RedactorService.
        
        Security Impact: Last name is redacted before being stored in the model.
        """
        return RedactorService.redact_name(v)
    
    @field_validator("date_of_birth", mode="before")
    @classmethod
    def redact_date_of_birth(cls, v) -> Optional[date]:
        """Redact date of birth using RedactorService.
        
        Security Impact: Date of birth is redacted (set to None) before being stored.
        """
        if v is None:
            return None
        # Convert to date if needed, then redact (always returns None)
        dob = v if isinstance(v, date) else None
        return RedactorService.redact_date_of_birth(dob)
    
    @field_validator("phone", mode="before")
    @classmethod
    def redact_phone(cls, v: Optional[str]) -> Optional[str]:
        """Redact phone number using RedactorService.
        
        Security Impact: Phone number is redacted before being stored in the model.
        """
        return RedactorService.redact_phone(v)
    
    @field_validator("email", mode="before")
    @classmethod
    def redact_email(cls, v: Optional[str]) -> Optional[str]:
        """Redact email address using RedactorService.
        
        Security Impact: Email address is redacted before being stored in the model.
        """
        return RedactorService.redact_email(v)
    
    @field_validator("address_line1", mode="before")
    @classmethod
    def redact_address_line1(cls, v: Optional[str]) -> Optional[str]:
        """Redact address line 1 using RedactorService.
        
        Security Impact: Address is redacted before being stored in the model.
        """
        return RedactorService.redact_address(v)
    
    @field_validator("address_line2", mode="before")
    @classmethod
    def redact_address_line2(cls, v: Optional[str]) -> Optional[str]:
        """Redact address line 2 using RedactorService.
        
        Security Impact: Address is redacted before being stored in the model.
        """
        return RedactorService.redact_address(v)
    
    @field_validator("zip_code", mode="before")
    @classmethod
    def redact_zip_code(cls, v: Optional[str]) -> Optional[str]:
        """Partially redact ZIP code using RedactorService.
        
        Security Impact: ZIP code is partially redacted (first 2 digits kept) for analytics.
        """
        return RedactorService.redact_zip_code(v)
    
    model_config = ConfigDict(
        frozen=True,  # Immutable records
        str_strip_whitespace=True,
    )


class ClinicalObservation(BaseModel):
    """Golden record for clinical measurements and observations.
    
    Security Impact: May contain PHI in notes field. The Sieve will redact
    any PII detected in unstructured text.
    
    Parameters:
        observation_id: Unique observation identifier
        patient_id: Reference to patient (foreign key)
        encounter_id: Reference to encounter/visit
        observation_type: Type of observation (e.g., "VITAL_SIGN", "LAB_RESULT")
        observation_code: Standardized code (e.g., LOINC, SNOMED)
        value: Numeric or text value of observation
        unit: Unit of measurement
        effective_date: When observation was taken
        notes: Unstructured clinical notes (may contain PII)
    """
    
    observation_id: str = Field(..., description="Unique observation identifier")
    patient_id: str = Field(..., description="Reference to patient")
    encounter_id: Optional[str] = Field(None, description="Reference to encounter")
    observation_type: ObservationCategory = Field(
        ..., description="Type of observation (FHIR ObservationCategory)"
    )
    observation_code: Optional[str] = Field(None, description="Standardized code (LOINC/SNOMED)")
    value: Optional[str] = Field(None, description="Observation value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    effective_date: Optional[datetime] = Field(None, description="When observation was taken")
    notes: Optional[str] = Field(None, description="Clinical notes (may contain PII)")
    
    @field_validator("notes", mode="before")
    @classmethod
    def redact_notes(cls, v: Optional[str]) -> Optional[str]:
        """Redact PII from clinical notes using RedactorService.
        
        Security Impact: Unstructured notes are scanned and PII is redacted
        before being stored in the model.
        """
        return RedactorService.redact_observation_notes(v)
    
    @field_validator("observation_type", mode="before")
    @classmethod
    def validate_observation_type(cls, v) -> ObservationCategory:
        """Convert string values to FHIR ObservationCategory enum."""
        if isinstance(v, ObservationCategory):
            return v
        
        v_str = str(v).strip().lower().replace("_", "-")
        # Map common variations to FHIR values
        mapping = {
            "vital-signs": ObservationCategory.VITAL_SIGNS,
            "vital_signs": ObservationCategory.VITAL_SIGNS,
            "vital sign": ObservationCategory.VITAL_SIGNS,
            "laboratory": ObservationCategory.LABORATORY,
            "lab": ObservationCategory.LABORATORY,
            "lab_result": ObservationCategory.LABORATORY,
            "imaging": ObservationCategory.IMAGING,
            "procedure": ObservationCategory.PROCEDURE,
            "survey": ObservationCategory.SURVEY,
            "exam": ObservationCategory.EXAM,
            "therapy": ObservationCategory.THERAPY,
        }
        # Try direct match first
        try:
            return ObservationCategory(v_str)
        except ValueError:
            return mapping.get(v_str, ObservationCategory.VITAL_SIGNS)
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
    )


class EncounterRecord(BaseModel):
    """Golden record for patient encounters/visits.
    
    Security Impact: Contains encounter dates and types. May reference
    PII indirectly through patient_id.
    
    Parameters:
        encounter_id: Unique encounter identifier
        patient_id: Reference to patient
        encounter_type: Type of encounter (e.g., "INPATIENT", "OUTPATIENT", "EMERGENCY")
        start_date: Encounter start date/time
        end_date: Encounter end date/time
        facility_name: Name of facility (may be redacted if contains location PII)
        diagnosis_codes: List of diagnosis codes (e.g., ICD-10)
    """
    
    encounter_id: str = Field(..., description="Unique encounter identifier")
    patient_id: str = Field(..., description="Reference to patient")
    encounter_type: EncounterClass = Field(
        ..., description="Type of encounter (FHIR EncounterClass)"
    )
    start_date: Optional[datetime] = Field(None, description="Encounter start")
    end_date: Optional[datetime] = Field(None, description="Encounter end")
    facility_name: Optional[str] = Field(None, description="Facility name")
    diagnosis_codes: list[str] = Field(default_factory=list, description="Diagnosis codes")
    
    @field_validator("encounter_type", mode="before")
    @classmethod
    def validate_encounter_type(cls, v) -> EncounterClass:
        """Convert string values to FHIR EncounterClass enum.
        
        Accepts various string formats and normalizes to FHIR-compliant values.
        """
        if isinstance(v, EncounterClass):
            return v
        
        v_str = str(v).strip().lower().replace("_", "-").replace(" ", "-")
        # Map common variations to FHIR values
        mapping = {
            "inpatient": EncounterClass.INPATIENT,
            "outpatient": EncounterClass.OUTPATIENT,
            "ambulatory": EncounterClass.AMBULATORY,
            "emergency": EncounterClass.EMERGENCY,
            "virtual": EncounterClass.VIRTUAL,
            "telehealth": EncounterClass.VIRTUAL,
            "observation": EncounterClass.OBSERVATION,
            "urgent-care": EncounterClass.URGENT_CARE,
            "urgent_care": EncounterClass.URGENT_CARE,
        }
        # Try direct match first
        try:
            return EncounterClass(v_str)
        except ValueError:
            return mapping.get(v_str, EncounterClass.OUTPATIENT)
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
    )


class GoldenRecord(BaseModel):
    """Container for a complete golden record set.
    
    This model represents a fully validated, standardized clinical record
    that has passed through the Safety Layer and Sieve. All PII has been
    redacted and schema validation has been enforced.
    
    Security Impact: This is the "safe" record that can be persisted.
    All PII fields in nested models have been processed by the Sieve.
    
    Parameters:
        patient: Patient demographic record (PII redacted)
        encounters: List of encounter records
        observations: List of clinical observations
        ingestion_timestamp: When this record was ingested and validated
        source_adapter: Which adapter provided the raw data
        transformation_hash: Hash of original data for audit trail
    """
    
    patient: PatientRecord = Field(..., description="Patient record (PII redacted)")
    encounters: list[EncounterRecord] = Field(
        default_factory=list,
        description="List of encounter records"
    )
    observations: list[ClinicalObservation] = Field(
        default_factory=list,
        description="List of clinical observations"
    )
    ingestion_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When record was ingested"
    )
    source_adapter: str = Field(..., description="Source adapter identifier")
    transformation_hash: Optional[str] = Field(
        None,
        description="Hash of original data for audit trail"
    )
    
    model_config = ConfigDict(frozen=True)

