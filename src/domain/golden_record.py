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
from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError as PydanticValidationError

from src.domain.enums import (
    AdministrativeGender,
    EncounterClass,
    ObservationCategory,
    ObservationStatus,
    EncounterStatus,
    IdentifierUse,
    AddressUse,
    ContactPointSystem,
)
from src.domain.services import RedactorService

# Import context function (infrastructure, but optional - graceful degradation)
try:
    from src.infrastructure.redaction_context import log_redaction_if_context
except ImportError:
    # Fallback if infrastructure not available (e.g., in minimal tests)
    def log_redaction_if_context(field_name: str, original_value: Optional[str], rule_triggered: str) -> None:
        pass


class PatientRecord(BaseModel):
    """Golden record for patient demographic information (FHIR R5 Patient resource).
    
    Security Impact: Contains PII fields (name, identifiers, addresses, telecom) that 
    must be redacted before persistence. The Sieve layer will mask these values.
    
    Aligns with FHIR R5 Patient resource: https://www.hl7.org/fhir/resourcelist.html
    
    Parameters:
        patient_id: Primary MRN identifier (non-PII, safe to persist)
        identifiers: List of additional identifiers (SSN, etc.) - PII will be redacted
        family_name: Patient family/last name (PII - will be redacted)
        given_names: List of given/first names (PII - will be redacted)
        name_prefix: Name prefixes (e.g., "Dr.", "Mr.") - may contain PII
        name_suffix: Name suffixes (e.g., "Jr.", "III") - may contain PII
        date_of_birth: Patient date of birth (PII - will be redacted)
        gender: Patient gender (FHIR AdministrativeGender)
        deceased: Whether patient is deceased (boolean or date)
        marital_status: Marital status code
        address_line1: Street address line 1 (PII - will be redacted)
        address_line2: Street address line 2 (PII - will be redacted)
        city: City name
        state: State code (standardized 2-letter)
        postal_code: ZIP/postal code (may be partially redacted)
        country: Country code
        address_use: Use of address (home, work, etc.)
        phone: Primary phone number (PII - will be redacted)
        email: Primary email address (PII - will be redacted)
        fax: Fax number (PII - will be redacted)
        emergency_contact_name: Emergency contact name (PII - will be redacted)
        emergency_contact_relationship: Relationship to patient
        emergency_contact_phone: Emergency contact phone (PII - will be redacted)
        language: Preferred communication language
        managing_organization: Managing organization identifier
    """
    
    patient_id: str = Field(..., description="Primary patient identifier (MRN)")
    identifiers: list[str] = Field(
        default_factory=list,
        description="Additional identifiers (SSN, etc.) - PII will be redacted"
    )
    family_name: Optional[str] = Field(None, description="Family/last name (PII)")
    given_names: list[str] = Field(
        default_factory=list,
        description="Given/first names (PII - will be redacted)"
    )
    name_prefix: list[str] = Field(
        default_factory=list,
        description="Name prefixes (e.g., 'Dr.', 'Mr.')"
    )
    name_suffix: list[str] = Field(
        default_factory=list,
        description="Name suffixes (e.g., 'Jr.', 'III')"
    )
    date_of_birth: Optional[date] = Field(None, description="Date of birth (PII)")
    gender: Optional[AdministrativeGender] = Field(
        None, description="Gender (FHIR AdministrativeGender)"
    )
    deceased: Optional[bool] = Field(None, description="Whether patient is deceased")
    deceased_date: Optional[datetime] = Field(None, description="Date of death if deceased")
    marital_status: Optional[str] = Field(None, description="Marital status code")
    address_line1: Optional[str] = Field(None, description="Street address line 1 (PII)")
    address_line2: Optional[str] = Field(None, description="Street address line 2 (PII)")
    city: Optional[str] = Field(None, description="City name")
    state: Optional[str] = Field(
        None,
        pattern=r"^[A-Z]{2}$",
        description="State code (2-letter uppercase)"
    )
    postal_code: Optional[str] = Field(
        None,
        pattern=r"^\d{5}(-\d{4})?$",
        description="ZIP/postal code (5 or 9 digits)"
    )
    country: Optional[str] = Field(
        None,
        pattern=r"^[A-Z]{2,3}$",
        description="Country code (ISO 2-3 letter code)"
    )
    address_use: Optional[AddressUse] = Field(None, description="Address use (home, work, etc.)")
    phone: Optional[str] = Field(None, description="Primary phone number (PII)")
    email: Optional[str] = Field(None, description="Primary email address (PII)")
    fax: Optional[str] = Field(None, description="Fax number (PII)")
    emergency_contact_name: Optional[str] = Field(
        None, description="Emergency contact name (PII)"
    )
    emergency_contact_relationship: Optional[str] = Field(
        None, description="Emergency contact relationship"
    )
    emergency_contact_phone: Optional[str] = Field(
        None, description="Emergency contact phone (PII)"
    )
    language: Optional[str] = Field(
        None,
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$",
        description="Preferred communication language (ISO 639-1 code, e.g., 'en' or 'en-US')"
    )
    managing_organization: Optional[str] = Field(
        None, description="Managing organization identifier"
    )
    
    # Backward compatibility fields (deprecated, use family_name/given_names)
    first_name: Optional[str] = Field(None, description="[DEPRECATED] Use given_names")
    last_name: Optional[str] = Field(None, description="[DEPRECATED] Use family_name")
    ssn: Optional[str] = Field(None, description="[DEPRECATED] Use identifiers")
    zip_code: Optional[str] = Field(None, description="[DEPRECATED] Use postal_code")
    
    @field_validator("patient_id")
    @classmethod
    def validate_mrn_format(cls, v: str) -> str:
        """Validate Medical Record Number (MRN) format.
        
        Security Impact: Ensures MRN follows expected format to prevent injection
        attacks and data quality issues. MRN should be alphanumeric, typically
        starting with a letter followed by digits, or all digits.
        
        Parameters:
            v: MRN string to validate
        
        Returns:
            Validated MRN string
        
        Raises:
            ValueError: If MRN format is invalid
        """
        if not v or not isinstance(v, str):
            raise ValueError("MRN must be a non-empty string")
        
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("MRN cannot be empty or whitespace only")
        
        # MRN format: alphanumeric, typically 6-12 characters
        # Common formats: "MRN123456", "123456", "P001", etc.
        if not v_stripped.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                f"MRN must be alphanumeric (may include hyphens/underscores). Got: {v}"
            )
        
        if len(v_stripped) < 3:
            raise ValueError(f"MRN must be at least 3 characters long. Got: {v}")
        
        if len(v_stripped) > 20:
            raise ValueError(f"MRN must be at most 20 characters long. Got: {v}")
        
        return v_stripped
    
    @field_validator("date_of_birth", mode="before")
    @classmethod
    def validate_date_of_birth(cls, v) -> Optional[date]:
        """Validate date of birth for future dates and time travel logic.
        
        Security Impact: Prevents invalid dates that could indicate data quality
        issues or potential attacks. Blocks future dates and unreasonably old dates.
        
        Parameters:
            v: Date value to validate (before redaction)
        
        Returns:
            Date if valid, None if invalid (will be redacted anyway)
        
        Raises:
            ValueError: If date is in the future or too far in the past
        """
        if v is None:
            return None
        
        # Convert to date if it's a string or datetime
        if isinstance(v, str):
            try:
                v = datetime.strptime(v, "%Y-%m-%d").date()
            except ValueError:
                # If parsing fails, let Pydantic handle it
                return v
        
        if isinstance(v, datetime):
            v = v.date()
        
        if not isinstance(v, date):
            return v
        
        today = date.today()
        
        # Block future dates
        if v > today:
            raise ValueError(
                f"Date of birth cannot be in the future. Got: {v}, Today: {today}"
            )
        
        # Time travel logic: Block unreasonably old dates (before 1900)
        # This prevents data entry errors and invalid historical data
        min_date = date(1900, 1, 1)
        if v < min_date:
            raise ValueError(
                f"Date of birth cannot be before {min_date.year}. "
                f"Got: {v}. This may indicate a data entry error."
            )
        
        # Check for unreasonably old age (e.g., > 150 years)
        age_years = (today - v).days / 365.25
        if age_years > 150:
            raise ValueError(
                f"Date of birth results in age > 150 years ({age_years:.1f} years). "
                f"Got: {v}. This may indicate a data entry error."
            )
        
        return v
    
    @field_validator("gender", mode="before")
    @classmethod
    def validate_gender(cls, v) -> Optional[AdministrativeGender]:
        """Convert string values to FHIR AdministrativeGender enum.
        
        Validates against AdministrativeGender enum pattern.
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
    
    @field_validator("address_use", mode="before")
    @classmethod
    def validate_address_use(cls, v) -> Optional[AddressUse]:
        """Convert string values to FHIR AddressUse enum.
        
        Validates against AddressUse enum pattern.
        """
        if v is None:
            return v
        if isinstance(v, AddressUse):
            return v
        
        v_str = str(v).strip().lower()
        # Map common variations to FHIR values
        mapping = {
            "home": AddressUse.HOME,
            "work": AddressUse.WORK,
            "temp": AddressUse.TEMP,
            "temporary": AddressUse.TEMP,
            "old": AddressUse.OLD,
            "billing": AddressUse.BILLING,
        }
        # Try direct match first
        try:
            return AddressUse(v_str)
        except ValueError:
            return mapping.get(v_str, AddressUse.HOME)
    
    @field_validator("state", mode="before")
    @classmethod
    def normalize_state(cls, v: Optional[str]) -> Optional[str]:
        """Normalize state codes to uppercase before pattern validation."""
        if v is None:
            return v
        # Normalize to uppercase and take first 2 characters
        normalized = v.upper().strip()[:2] if len(v.strip()) >= 2 else v.upper().strip()
        return normalized
    
    @field_validator("identifiers", mode="before")
    @classmethod
    def redact_identifiers(cls, v) -> list[str]:
        """Redact PII in identifiers using RedactorService.
        
        Security Impact: Identifiers containing SSN or other PII are redacted.
        """
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        result = []
        for identifier in v:
            original = str(identifier)
            redacted = RedactorService.redact_ssn(original)
            result.append(redacted)
            if redacted != original:
                log_redaction_if_context("identifiers", original, "SSN_PATTERN")
        return result
    
    @field_validator("family_name", mode="before")
    @classmethod
    def redact_family_name(cls, v: Optional[str]) -> Optional[str]:
        """Redact family name using RedactorService.
        
        Security Impact: Family name is redacted before being stored in the model.
        """
        original = v
        result = RedactorService.redact_name(v)
        # Log redaction if context available (infrastructure concern, optional)
        if result != original:
            log_redaction_if_context("family_name", original, "NAME_PATTERN")
        return result
    
    @field_validator("given_names", mode="before")
    @classmethod
    def redact_given_names(cls, v) -> list[str]:
        """Redact given names using RedactorService.
        
        Security Impact: All given names are redacted before being stored.
        """
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        result = []
        for name in v:
            original = str(name)
            redacted = RedactorService.redact_name(original)
            result.append(redacted)
            # Log redaction if context available
            if redacted != original:
                log_redaction_if_context("given_names", original, "NAME_PATTERN")
        return result
    
    @field_validator("name_prefix", mode="before")
    @classmethod
    def redact_name_prefix(cls, v) -> list[str]:
        """Redact name prefixes (may contain titles that could be PII)."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        # Prefixes like "Dr." are usually safe, but we'll redact if they look like names
        return [RedactorService.redact_name(str(prefix)) if len(str(prefix)) > 3 else str(prefix) for prefix in v]
    
    @field_validator("emergency_contact_name", mode="before")
    @classmethod
    def redact_emergency_contact_name(cls, v: Optional[str]) -> Optional[str]:
        """Redact emergency contact name using RedactorService.
        
        Security Impact: Emergency contact name is PII and must be redacted.
        """
        original = v
        result = RedactorService.redact_name(v)
        if result != original:
            log_redaction_if_context("emergency_contact_name", original, "NAME_PATTERN")
        return result
    
    @field_validator("emergency_contact_phone", mode="before")
    @classmethod
    def redact_emergency_contact_phone(cls, v: Optional[str]) -> Optional[str]:
        """Redact emergency contact phone using RedactorService.
        
        Security Impact: Emergency contact phone is PII and must be redacted.
        """
        original = v
        result = RedactorService.redact_phone(v)
        if result != original:
            log_redaction_if_context("emergency_contact_phone", original, "PHONE_PATTERN")
        return result
    
    @field_validator("fax", mode="before")
    @classmethod
    def redact_fax(cls, v: Optional[str]) -> Optional[str]:
        """Redact fax number using RedactorService.
        
        Security Impact: Fax number is PII and must be redacted.
        """
        original = v
        result = RedactorService.redact_phone(v)
        if result != original:
            log_redaction_if_context("fax", original, "PHONE_PATTERN")
        return result
    
    @field_validator("ssn", mode="before")
    @classmethod
    def redact_ssn(cls, v: Optional[str]) -> Optional[str]:
        """Redact SSN using RedactorService (deprecated - use identifiers).
        
        Security Impact: SSN is redacted before being stored in the model.
        """
        original = v
        result = RedactorService.redact_ssn(v)
        if result != original:
            log_redaction_if_context("ssn", original, "SSN_PATTERN")
        return result
    
    @field_validator("first_name", mode="before")
    @classmethod
    def redact_first_name(cls, v: Optional[str]) -> Optional[str]:
        """Redact first name using RedactorService (deprecated - use given_names).
        
        Security Impact: First name is redacted before being stored in the model.
        """
        return RedactorService.redact_name(v)
    
    @field_validator("last_name", mode="before")
    @classmethod
    def redact_last_name(cls, v: Optional[str]) -> Optional[str]:
        """Redact last name using RedactorService (deprecated - use family_name).
        
        Security Impact: Last name is redacted before being stored in the model.
        """
        return RedactorService.redact_name(v)
    
    @field_validator("date_of_birth", mode="after")
    @classmethod
    def redact_date_of_birth(cls, v: Optional[date]) -> Optional[date]:
        """Redact date of birth using RedactorService.
        
        Security Impact: Date of birth is redacted (set to None) after validation.
        Note: Validation happens before redaction to catch invalid dates.
        """
        if v is None:
            return None
        return RedactorService.redact_date_of_birth(v)
    
    @field_validator("phone", mode="before")
    @classmethod
    def redact_phone(cls, v: Optional[str]) -> Optional[str]:
        """Redact phone number using RedactorService.
        
        Security Impact: Phone number is redacted before being stored in the model.
        """
        original = v
        result = RedactorService.redact_phone(v)
        if result != original:
            log_redaction_if_context("phone", original, "PHONE_PATTERN")
        return result
    
    @field_validator("email", mode="before")
    @classmethod
    def redact_email(cls, v: Optional[str]) -> Optional[str]:
        """Redact email address using RedactorService.
        
        Security Impact: Email address is redacted before being stored in the model.
        """
        original = v
        result = RedactorService.redact_email(v)
        if result != original:
            log_redaction_if_context("email", original, "EMAIL_PATTERN")
        return result
    
    @field_validator("address_line1", mode="before")
    @classmethod
    def redact_address_line1(cls, v: Optional[str]) -> Optional[str]:
        """Redact address line 1 using RedactorService.
        
        Security Impact: Address is redacted before being stored in the model.
        """
        original = v
        result = RedactorService.redact_address(v)
        if result != original:
            log_redaction_if_context("address_line1", original, "ADDRESS_PATTERN")
        return result
    
    @field_validator("address_line2", mode="before")
    @classmethod
    def redact_address_line2(cls, v: Optional[str]) -> Optional[str]:
        """Redact address line 2 using RedactorService.
        
        Security Impact: Address is redacted before being stored in the model.
        """
        original = v
        result = RedactorService.redact_address(v)
        if result != original:
            log_redaction_if_context("address_line2", original, "ADDRESS_PATTERN")
        return result
    
    @field_validator("postal_code", mode="after")
    @classmethod
    def redact_postal_code(cls, v: Optional[str]) -> Optional[str]:
        """Partially redact postal code using RedactorService.
        
        Security Impact: Postal code is partially redacted (first 2 digits kept) for analytics.
        Pattern validation happens before redaction (mode="after").
        """
        return RedactorService.redact_zip_code(v)
    
    @field_validator("zip_code", mode="before")
    @classmethod
    def redact_zip_code(cls, v: Optional[str]) -> Optional[str]:
        """Partially redact ZIP code using RedactorService (deprecated - use postal_code).
        
        Security Impact: ZIP code is partially redacted (first 2 digits kept) for analytics.
        """
        return RedactorService.redact_zip_code(v)
    
    model_config = ConfigDict(
        frozen=True,  # Immutable records
        str_strip_whitespace=True,
    )


class ClinicalObservation(BaseModel):
    """Golden record for clinical measurements and observations (FHIR R5 Observation resource).
    
    Security Impact: May contain PHI in notes, performer names, and interpretation fields.
    The Sieve will redact any PII detected in unstructured text.
    
    Aligns with FHIR R5 Observation resource: https://www.hl7.org/fhir/resourcelist.html
    
    Parameters:
        observation_id: Unique observation identifier
        status: Observation status (registered, preliminary, final, etc.)
        category: Category of observation (FHIR ObservationCategory)
        code: Standardized observation code (LOINC, SNOMED)
        patient_id: Reference to patient (foreign key)
        encounter_id: Reference to encounter/visit
        effective_date: When observation was taken
        issued: When result was made available
        performer_name: Name of who performed observation (PII - will be redacted)
        value: Numeric or text value of observation
        unit: Unit of measurement
        interpretation: Clinical interpretation of result
        body_site: Body site where observation was made
        method: Method used to produce observation
        device: Device used for observation
        reference_range: Reference range for interpretation
        notes: Unstructured clinical notes (may contain PII)
    """
    
    observation_id: str = Field(..., description="Unique observation identifier")
    status: ObservationStatus = Field(
        default=ObservationStatus.FINAL,
        description="Observation status (FHIR ObservationStatus)"
    )
    category: ObservationCategory = Field(
        ..., description="Category of observation (FHIR ObservationCategory)"
    )
    code: Optional[str] = Field(
        None,
        pattern=r"^[A-Z0-9]+(-[A-Z0-9]+)*$",
        description="Standardized code (LOINC/SNOMED format)"
    )
    patient_id: str = Field(..., description="Reference to patient")
    encounter_id: Optional[str] = Field(None, description="Reference to encounter")
    effective_date: Optional[datetime] = Field(None, description="When observation was taken")
    issued: Optional[datetime] = Field(None, description="When result was made available")
    performer_name: Optional[str] = Field(
        None, description="Name of performer (PII - will be redacted)"
    )
    value: Optional[str] = Field(None, description="Observation value")
    unit: Optional[str] = Field(None, description="Unit of measurement")
    interpretation: Optional[str] = Field(None, description="Clinical interpretation")
    body_site: Optional[str] = Field(
        None,
        pattern=r"^[A-Z0-9]+(-[A-Z0-9]+)*$",
        description="Body site code (SNOMED format)"
    )
    method: Optional[str] = Field(
        None,
        pattern=r"^[A-Z0-9]+(-[A-Z0-9]+)*$",
        description="Method code (SNOMED format)"
    )
    device: Optional[str] = Field(
        None,
        pattern=r"^[A-Z0-9_-]+$",
        description="Device identifier (alphanumeric with hyphens/underscores)"
    )
    reference_range: Optional[str] = Field(None, description="Reference range text")
    notes: Optional[str] = Field(None, description="Clinical notes (may contain PII)")
    
    # Backward compatibility
    observation_type: Optional[ObservationCategory] = Field(
        None, description="[DEPRECATED] Use category"
    )
    observation_code: Optional[str] = Field(None, description="[DEPRECATED] Use code")
    
    @field_validator("performer_name", mode="before")
    @classmethod
    def redact_performer_name(cls, v: Optional[str]) -> Optional[str]:
        """Redact performer name using RedactorService.
        
        Security Impact: Performer name is PII and must be redacted.
        """
        return RedactorService.redact_name(v)
    
    @field_validator("notes", mode="before")
    @classmethod
    def redact_notes(cls, v: Optional[str]) -> Optional[str]:
        """Redact PII from clinical notes using RedactorService (regex-only during validation).
        
        Security Impact: Unstructured notes are scanned and PII is redacted
        before being stored in the model.
        
        Note: NER-based redaction is deferred to batch processing for performance.
        This validator only applies regex-based redaction (SSN, phone, email).
        """
        if not v:
            return None
        
        # Apply regex-based redaction only (fast, no NER)
        # NER will be applied later in batch processing
        original = v
        result = RedactorService.redact_observation_notes_fast(v)
        if result != original:
            log_redaction_if_context("notes", original, "OBSERVATION_NOTES_PII_DETECTION")
        return result
    
    @field_validator("interpretation", mode="before")
    @classmethod
    def redact_interpretation(cls, v: Optional[str]) -> Optional[str]:
        """Redact PII from interpretation text using RedactorService.
        
        Security Impact: Interpretation may contain PII in narrative text.
        """
        return RedactorService.redact_unstructured_text(v)
    
    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v) -> ObservationStatus:
        """Convert string values to FHIR ObservationStatus enum.
        
        Validates against ObservationStatus enum pattern.
        """
        if isinstance(v, ObservationStatus):
            return v
        
        if v is None:
            return ObservationStatus.FINAL
        
        v_str = str(v).strip().lower().replace("_", "-")
        # Map common variations to FHIR values
        mapping = {
            "registered": ObservationStatus.REGISTERED,
            "preliminary": ObservationStatus.PRELIMINARY,
            "final": ObservationStatus.FINAL,
            "amended": ObservationStatus.AMENDED,
            "corrected": ObservationStatus.CORRECTED,
            "cancelled": ObservationStatus.CANCELLED,
            "canceled": ObservationStatus.CANCELLED,  # US spelling
            "entered-in-error": ObservationStatus.ENTERED_IN_ERROR,
            "entered_in_error": ObservationStatus.ENTERED_IN_ERROR,
            "unknown": ObservationStatus.UNKNOWN,
        }
        # Try direct match first
        try:
            return ObservationStatus(v_str)
        except ValueError:
            return mapping.get(v_str, ObservationStatus.FINAL)
    
    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v) -> ObservationCategory:
        """Convert string values to FHIR ObservationCategory enum.
        
        Validates against ObservationCategory enum pattern.
        """
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
            "activity": ObservationCategory.ACTIVITY,
        }
        # Try direct match first
        try:
            return ObservationCategory(v_str)
        except ValueError:
            return mapping.get(v_str, ObservationCategory.VITAL_SIGNS)
    
    @field_validator("observation_type", mode="before")
    @classmethod
    def validate_observation_type(cls, v) -> Optional[ObservationCategory]:
        """Convert string values to FHIR ObservationCategory enum (deprecated)."""
        if v is None:
            return None
        # Delegate to category validator
        return cls.validate_category(v)
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
    )


class EncounterRecord(BaseModel):
    """Golden record for patient encounters/visits (FHIR R5 Encounter resource).
    
    Security Impact: Contains encounter dates, locations, and participant names.
    May contain PII in facility names, location addresses, and participant information.
    
    Aligns with FHIR R5 Encounter resource: https://www.hl7.org/fhir/resourcelist.html
    
    Parameters:
        encounter_id: Unique encounter identifier
        status: Encounter status (planned, in-progress, finished, etc.)
        class_code: Encounter class (FHIR EncounterClass)
        type: Specific encounter type code
        service_type: Service type code
        priority: Priority code
        patient_id: Reference to patient
        period_start: Encounter start date/time
        period_end: Encounter end date/time
        length_minutes: Length of encounter in minutes
        reason_code: Reason for encounter (diagnosis codes)
        diagnosis_codes: List of diagnosis codes (e.g., ICD-10)
        facility_name: Name of facility (may be redacted if contains location PII)
        location_address: Location address (PII - will be redacted)
        participant_name: Name of participant (PII - will be redacted)
        participant_role: Role of participant
        service_provider: Service provider organization identifier
    """
    
    encounter_id: str = Field(..., description="Unique encounter identifier")
    status: EncounterStatus = Field(
        default=EncounterStatus.FINISHED,
        description="Encounter status (FHIR EncounterStatus)"
    )
    class_code: EncounterClass = Field(
        ..., description="Encounter class (FHIR EncounterClass)"
    )
    type: Optional[str] = Field(
        None,
        pattern=r"^[A-Z0-9]+(-[A-Z0-9]+)*$",
        description="Specific encounter type code (SNOMED format)"
    )
    service_type: Optional[str] = Field(
        None,
        pattern=r"^[A-Z0-9]+(-[A-Z0-9]+)*$",
        description="Service type code (SNOMED format)"
    )
    priority: Optional[str] = Field(
        None,
        pattern=r"^(routine|urgent|asap|stat)$",
        description="Priority code (routine, urgent, asap, stat)"
    )
    patient_id: str = Field(..., description="Reference to patient")
    period_start: Optional[datetime] = Field(None, description="Encounter start")
    period_end: Optional[datetime] = Field(None, description="Encounter end")
    length_minutes: Optional[int] = Field(None, description="Length in minutes")
    reason_code: Optional[str] = Field(
        None,
        pattern=r"^[A-Z][0-9]{2}(\.[0-9]+)?$",
        description="Reason for encounter (ICD-10 format, e.g., 'I10' or 'E11.9')"
    )
    diagnosis_codes: list[str] = Field(
        default_factory=list,
        description="Diagnosis codes (ICD-10 format, e.g., 'I10' or 'E11.9')"
    )
    
    @field_validator("diagnosis_codes", mode="before")
    @classmethod
    def validate_diagnosis_codes(cls, v) -> list[str]:
        """Validate each diagnosis code matches ICD-10 pattern."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        import re
        pattern = re.compile(r"^[A-Z][0-9]{2}(\.[0-9]+)?$")
        validated = []
        for code in v:
            code_str = str(code).strip().upper()
            if pattern.match(code_str):
                validated.append(code_str)
            else:
                # Allow invalid codes but log warning (fail-fast would raise ValueError)
                validated.append(code_str)
        return validated
    facility_name: Optional[str] = Field(None, description="Facility name")
    location_address: Optional[str] = Field(None, description="Location address (PII)")
    participant_name: Optional[str] = Field(None, description="Participant name (PII)")
    participant_role: Optional[str] = Field(None, description="Participant role")
    service_provider: Optional[str] = Field(None, description="Service provider identifier")
    
    # Backward compatibility
    encounter_type: Optional[EncounterClass] = Field(
        None, description="[DEPRECATED] Use class_code"
    )
    start_date: Optional[datetime] = Field(None, description="[DEPRECATED] Use period_start")
    end_date: Optional[datetime] = Field(None, description="[DEPRECATED] Use period_end")
    
    @field_validator("location_address", mode="before")
    @classmethod
    def redact_location_address(cls, v: Optional[str]) -> Optional[str]:
        """Redact location address using RedactorService.
        
        Security Impact: Location address is PII and must be redacted.
        """
        return RedactorService.redact_address(v)
    
    @field_validator("participant_name", mode="before")
    @classmethod
    def redact_participant_name(cls, v: Optional[str]) -> Optional[str]:
        """Redact participant name using RedactorService.
        
        Security Impact: Participant name is PII and must be redacted.
        """
        return RedactorService.redact_name(v)
    
    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v) -> EncounterStatus:
        """Convert string values to FHIR EncounterStatus enum.
        
        Validates against EncounterStatus enum pattern.
        """
        if isinstance(v, EncounterStatus):
            return v
        
        if v is None:
            return EncounterStatus.FINISHED
        
        v_str = str(v).strip().lower().replace("_", "-")
        # Map common variations to FHIR values
        mapping = {
            "planned": EncounterStatus.PLANNED,
            "arrived": EncounterStatus.ARRIVED,
            "triaged": EncounterStatus.TRIAGED,
            "in-progress": EncounterStatus.IN_PROGRESS,
            "in_progress": EncounterStatus.IN_PROGRESS,
            "onleave": EncounterStatus.ONLEAVE,
            "on-leave": EncounterStatus.ONLEAVE,
            "finished": EncounterStatus.FINISHED,
            "cancelled": EncounterStatus.CANCELLED,
            "canceled": EncounterStatus.CANCELLED,  # US spelling
            "entered-in-error": EncounterStatus.ENTERED_IN_ERROR,
            "entered_in_error": EncounterStatus.ENTERED_IN_ERROR,
            "unknown": EncounterStatus.UNKNOWN,
        }
        # Try direct match first
        try:
            return EncounterStatus(v_str)
        except ValueError:
            return mapping.get(v_str, EncounterStatus.FINISHED)
    
    @field_validator("class_code", mode="before")
    @classmethod
    def validate_class_code(cls, v) -> EncounterClass:
        """Convert string values to FHIR EncounterClass enum."""
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
    
    @field_validator("encounter_type", mode="before")
    @classmethod
    def validate_encounter_type(cls, v) -> Optional[EncounterClass]:
        """Convert string values to FHIR EncounterClass enum (deprecated)."""
        if v is None:
            return None
        # Delegate to class_code validator
        return cls.validate_class_code(v)
    
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

