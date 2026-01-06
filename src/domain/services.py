"""PII Redaction Service.

This module provides the RedactorService class responsible for identifying and masking
Personally Identifiable Information (PII) in clinical data. This service follows the
Single Responsibility Principle by isolating all security and redaction logic from
data models.

Security Impact:
    - Prevents PII from being persisted in unredacted form
    - Uses regex patterns to identify structured PII (SSN, phone, email)
    - Uses NLP-based NER for person names in unstructured text (when available)
    - Handles unstructured text redaction for clinical notes
    - All redaction methods are pure functions (no side effects)

Architecture:
    - Pure domain service with zero infrastructure dependencies
    - Stateless service that can be used across the application
    - Follows Hexagonal Architecture: Core security logic isolated from infrastructure
    - Uses NER adapter via port interface (dependency injection)
"""

import logging
import re
from datetime import date, datetime
from typing import Optional, Union, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from src.domain.ports import NERPort

logger = logging.getLogger(__name__)


class RedactorService:
    """Service for identifying and masking PII in clinical records.
    
    This service provides methods to redact various types of PII including:
    - Social Security Numbers (SSN)
    - Phone numbers
    - Email addresses
    - Names (first and last)
    - Addresses
    - Dates of birth
    - Unstructured text containing PII (uses NER when available)
    
    All methods are pure functions that take input and return redacted output
    without side effects.
    
    NER Integration:
        - Uses NER adapter (via NERPort) for person name detection in unstructured text
        - Falls back to regex if NER unavailable
        - NER adapter can be injected via set_ner_adapter() for testing
    """
    
    # Class variable for NER adapter (lazy-loaded, dependency injection)
    _ner_adapter: Optional['NERPort'] = None
    
    # Regex patterns for PII detection
    SSN_PATTERN = re.compile(r'\b\d{3}-?\d{2}-?\d{4}\b')
    PHONE_PATTERN = re.compile(
        r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    )
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    )
    # Pattern for names (common first/last names - basic detection)
    NAME_PATTERN = re.compile(
        r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    )
    
    # Redaction masks
    SSN_MASK = "***-**-****"
    PHONE_MASK = "***-***-****"
    EMAIL_MASK = "***@***.***"
    NAME_MASK = "[REDACTED]"
    ADDRESS_MASK = "[REDACTED]"
    DATE_MASK = "****-**-**"
    
    @classmethod
    def set_ner_adapter(cls, adapter: Optional['NERPort']) -> None:
        """Set NER adapter for name extraction.
        
        This allows dependency injection of NER implementation.
        Follows Hexagonal Architecture pattern.
        
        Parameters:
            adapter: NER adapter instance (implements NERPort) or None to disable
        
        Security Impact:
            - Enables testing with mock NER adapters
            - Allows swapping NER implementations without changing domain logic
        """
        cls._ner_adapter = adapter
    
    @classmethod
    def _get_ner_adapter(cls) -> Optional['NERPort']:
        """Get NER adapter (lazy initialization).
        
        Returns:
            NER adapter if available, None otherwise
        
        Security Impact:
            - Lazy loading prevents unnecessary model loading
            - Graceful fallback to regex if NER unavailable
        """
        if cls._ner_adapter is None:
            # Try to initialize default SpaCy adapter
            try:
                from src.infrastructure.ner.spacy_adapter import SpaCyNERAdapter
                # Get model name from settings (if available)
                try:
                    from src.infrastructure.settings import settings
                    model_name = getattr(settings, 'spacy_model', 'en_core_web_sm')
                except (ImportError, AttributeError):
                    model_name = 'en_core_web_sm'
                
                # Check if NER is enabled
                try:
                    from src.infrastructure.settings import settings
                    ner_enabled = getattr(settings, 'ner_enabled', True)
                except (ImportError, AttributeError):
                    ner_enabled = True
                
                if ner_enabled:
                    cls._ner_adapter = SpaCyNERAdapter(model_name=model_name)
                else:
                    cls._ner_adapter = None
                    logger.debug("NER is disabled via settings")
            except ImportError:
                logger.debug("SpaCy not available, using regex-only redaction")
                cls._ner_adapter = None
            except Exception as e:
                logger.warning(f"Failed to initialize NER adapter: {e}")
                cls._ner_adapter = None
        
        return cls._ner_adapter
    
    @staticmethod
    def redact_ssn(value: Union[Optional[str], 'pd.Series']) -> Union[Optional[str], 'pd.Series']:
        """Redact Social Security Number.
        
        Security Impact: Masks SSN to prevent identity theft and HIPAA violations.
        Handles both formatted (123-45-6789) and unformatted (123456789) SSNs.
        Supports both scalar values (backward compatible) and pandas Series (vectorized).
        
        Parameters:
            value: SSN string to redact (may contain dashes or spaces) or pandas Series
        
        Returns:
            Redacted SSN string/Series or None if input is None
        """
        # Vectorized operation for pandas Series
        if isinstance(value, pd.Series):
            # Use vectorized string operations (fastest)
            # Handle NaN values properly
            result = value.copy().astype(str)
            # Replace 'nan' strings with empty strings for processing
            result = result.replace('nan', '')
            
            # Create mask for values that match SSN pattern
            mask = result.str.contains(RedactorService.SSN_PATTERN.pattern, regex=True, na=False)
            # Also check for 9-digit numbers without separators
            cleaned = result.str.replace(r'[-\s]', '', regex=True)
            nine_digit_mask = cleaned.str.isdigit() & (cleaned.str.len() == 9)
            
            # Apply redaction where either condition is true
            result[mask | nine_digit_mask] = RedactorService.SSN_MASK
            # Restore original NaN values
            result[value.isna()] = None
            return result
        
        # Scalar operation (backward compatible)
        if not value:
            return None
        
        # Remove separators and validate format
        cleaned = re.sub(r'[-\s]', '', str(value))
        if cleaned.isdigit() and len(cleaned) == 9:
            return RedactorService.SSN_MASK
        
        # If format is suspicious but contains digits, still redact
        if RedactorService.SSN_PATTERN.search(str(value)):
            return RedactorService.SSN_MASK
        
        return value
    
    @staticmethod
    def redact_phone(value: Union[Optional[str], 'pd.Series']) -> Union[Optional[str], 'pd.Series']:
        """Redact phone number.
        
        Security Impact: Masks phone numbers to prevent contact information exposure.
        Handles various formats including (123) 456-7890, 123-456-7890, etc.
        Supports both scalar values (backward compatible) and pandas Series (vectorized).
        
        Parameters:
            value: Phone number string to redact or pandas Series
        
        Returns:
            Redacted phone number string/Series or None if input is None
        """
        # Vectorized operation for pandas Series
        if isinstance(value, pd.Series):
            result = value.copy().astype(str)
            result = result.replace('nan', '')
            mask = result.str.contains(RedactorService.PHONE_PATTERN.pattern, regex=True, na=False)
            result[mask] = RedactorService.PHONE_MASK
            result[value.isna()] = None
            return result
        
        # Scalar operation (backward compatible)
        if not value:
            return None
        
        value_str = str(value).strip()
        
        # Security: Redact if it contains XSS patterns
        xss_patterns = ['<script', '<img', '<iframe', 'javascript:', 'onerror=', 'onload=']
        if any(pattern in value_str.lower() for pattern in xss_patterns):
            return RedactorService.PHONE_MASK
        
        if RedactorService.PHONE_PATTERN.search(value_str):
            return RedactorService.PHONE_MASK
        
        return value
    
    @staticmethod
    def redact_email(value: Union[Optional[str], 'pd.Series']) -> Union[Optional[str], 'pd.Series']:
        """Redact email address.
        
        Security Impact: Masks email addresses to prevent contact information exposure
        and reduce phishing attack vectors.
        Supports both scalar values (backward compatible) and pandas Series (vectorized).
        
        Parameters:
            value: Email address string to redact or pandas Series
        
        Returns:
            Redacted email address string/Series or None if input is None
        """
        # Vectorized operation for pandas Series
        if isinstance(value, pd.Series):
            result = value.copy().astype(str)
            result = result.replace('nan', '')
            mask = result.str.contains(RedactorService.EMAIL_PATTERN.pattern, regex=True, na=False)
            result[mask] = RedactorService.EMAIL_MASK
            result[value.isna()] = None
            return result
        
        # Scalar operation (backward compatible)
        if not value:
            return None
        
        value_str = str(value).strip()
        
        # Security: Redact if it contains XSS patterns
        xss_patterns = ['<script', '<img', '<iframe', 'javascript:', 'onerror=', 'onload=']
        if any(pattern in value_str.lower() for pattern in xss_patterns):
            return RedactorService.EMAIL_MASK
        
        if RedactorService.EMAIL_PATTERN.search(value_str):
            return RedactorService.EMAIL_MASK
        
        return value
    
    @staticmethod
    def redact_name(value: Union[Optional[str], 'pd.Series']) -> Union[Optional[str], 'pd.Series']:
        """Redact person name.
        
        Security Impact: Masks names to prevent patient identification.
        This is a basic implementation; for production, consider NLP-based NER.
        Supports both scalar values (backward compatible) and pandas Series (vectorized).
        
        Parameters:
            value: Name string to redact or pandas Series
        
        Returns:
            Redacted name string/Series or None if input is None
        """
        # Vectorized operation for pandas Series
        if isinstance(value, pd.Series):
            result = value.copy().astype(str)
            result = result.replace('nan', '')
            value_str = result.str.strip()
            # Security: Redact if it contains XSS patterns
            xss_patterns = ['<script', '<img', '<iframe', 'javascript:', 'onerror=', 'onload=']
            xss_mask = value_str.str.lower().str.contains('|'.join(xss_patterns), na=False, regex=False)
            result[xss_mask] = RedactorService.NAME_MASK
            
            # Simple heuristic: if it looks like a name (capitalized, 1-3 words)
            name_mask = (
                (value_str.str.len() > 0) &
                (value_str.str[0].str.isupper()) &
                (value_str.str.split().str.len() <= 3) &
                (~xss_mask)  # Don't double-redact XSS patterns
            )
            result[name_mask] = RedactorService.NAME_MASK
            result[value.isna()] = None
            return result
        
        # Scalar operation (backward compatible)
        if not value:
            return None
        
        value_str = str(value).strip()
        if not value_str:
            return None
        
        # Security: Redact if it contains XSS patterns (security concern, not just PII)
        xss_patterns = ['<script', '<img', '<iframe', 'javascript:', 'onerror=', 'onload=']
        if any(pattern in value_str.lower() for pattern in xss_patterns):
            return RedactorService.NAME_MASK
        
        # Simple heuristic: if it looks like a name (capitalized words), redact it
        if value_str[0].isupper() and len(value_str.split()) <= 3:
            return RedactorService.NAME_MASK
        
        return value
    
    @staticmethod
    def redact_address(value: Union[Optional[str], 'pd.Series']) -> Union[Optional[str], 'pd.Series']:
        """Redact street address.
        
        Security Impact: Masks addresses to prevent location-based identification.
        Supports both scalar values (backward compatible) and pandas Series (vectorized).
        
        Parameters:
            value: Address string to redact or pandas Series
        
        Returns:
            Redacted address string/Series or None if input is None
        """
        # Vectorized operation for pandas Series
        if isinstance(value, pd.Series):
            result = value.copy().astype(str)
            result = result.replace('nan', '')
            value_str = result.str.strip()
            # Addresses typically contain numbers and street names
            mask = value_str.str.contains(r'\d', regex=True, na=False)  # Contains digits
            result[mask] = RedactorService.ADDRESS_MASK
            result[value.isna()] = None
            return result
        
        # Scalar operation (backward compatible)
        if not value:
            return None
        
        # Addresses typically contain numbers and street names
        value_str = str(value).strip()
        if value_str and any(char.isdigit() for char in value_str):
            return RedactorService.ADDRESS_MASK
        
        return value
    
    @staticmethod
    def redact_date_of_birth(value: Optional[date]) -> Optional[date]:
        """Redact date of birth by returning None.
        
        Security Impact: Removes DOB to prevent age-based identification and
        reduce re-identification risk when combined with other data.
        
        Parameters:
            value: Date of birth to redact
        
        Returns:
            None (DOB is fully redacted)
        """
        return None
    
    @staticmethod
    def redact_zip_code(value: Union[Optional[str], 'pd.Series']) -> Union[Optional[str], 'pd.Series']:
        """Partially redact ZIP code (keep first 2 digits for analytics).
        
        Security Impact: Partially masks ZIP to balance privacy with geographic
        analytics needs. Full ZIP can be used for re-identification.
        Supports both scalar values (backward compatible) and pandas Series (vectorized).
        
        Parameters:
            value: ZIP code string to redact or pandas Series
        
        Returns:
            Partially redacted ZIP (e.g., "12***") or None if input is None
        """
        # Vectorized operation for pandas Series
        if isinstance(value, pd.Series):
            result = value.copy().astype(str)
            result = result.replace('nan', '')
            value_str = result.str.strip()
            # Remove dashes for processing
            cleaned = value_str.str.replace("-", "", regex=False)
            # Keep first 2 digits, mask the rest for valid ZIP codes (5+ digits)
            mask = cleaned.str.isdigit() & (cleaned.str.len() >= 5)
            result[mask] = value_str[mask].str[:2] + "***"
            result[value.isna()] = None
            return result
        
        # Scalar operation (backward compatible)
        if not value:
            return None
        
        value_str = str(value).strip()
        # Remove dashes for processing
        cleaned = value_str.replace("-", "")
        
        # Keep first 2 digits, mask the rest
        if cleaned.isdigit() and len(cleaned) >= 5:
            return value_str[:2] + "***"
        
        return value
    
    @staticmethod
    def redact_unstructured_text(value: Union[Optional[str], 'pd.Series']) -> Union[Optional[str], 'pd.Series']:
        """Redact PII from unstructured text (e.g., clinical notes).
        
        Security Impact: Scans unstructured text for PII patterns and redacts them.
        Uses NLP-based Named Entity Recognition (NER) for person names when available,
        falls back to regex if NER unavailable. Always uses regex for structured PII
        (SSN, phone, email).
        Supports both scalar values (backward compatible) and pandas Series (vectorized).
        
        Parameters:
            value: Unstructured text that may contain PII or pandas Series
        
        Returns:
            Text with PII redacted or None if input is None
        """
        # Vectorized operation for pandas Series
        if isinstance(value, pd.Series):
            result = value.astype(str).copy()
            result = result.replace('nan', '')
            # Redact SSNs (always use regex - fast and accurate)
            result = result.str.replace(RedactorService.SSN_PATTERN.pattern, RedactorService.SSN_MASK, regex=True)
            # Redact phone numbers (always use regex)
            result = result.str.replace(RedactorService.PHONE_PATTERN.pattern, RedactorService.PHONE_MASK, regex=True)
            # Redact email addresses (always use regex)
            result = result.str.replace(RedactorService.EMAIL_PATTERN.pattern, RedactorService.EMAIL_MASK, regex=True)
            
            # Use NER for person names (if available)
            ner_adapter = RedactorService._get_ner_adapter()
            if ner_adapter and ner_adapter.is_available():
                # Apply NER-based name redaction to each text
                def redact_with_ner(text: str) -> str:
                    if not text or text == 'nan':
                        return text
                    return RedactorService._redact_names_with_ner(text, ner_adapter)
                
                result = result.apply(redact_with_ner)
            
            result[value.isna()] = None
            return result
        
        # Scalar operation (backward compatible)
        if not value:
            return None
        
        text = str(value)
        
        # Redact SSNs (always use regex - fast and accurate)
        text = RedactorService.SSN_PATTERN.sub(RedactorService.SSN_MASK, text)
        
        # Redact phone numbers (always use regex)
        text = RedactorService.PHONE_PATTERN.sub(RedactorService.PHONE_MASK, text)
        
        # Redact email addresses (always use regex)
        text = RedactorService.EMAIL_PATTERN.sub(RedactorService.EMAIL_MASK, text)
        
        # Use NER for person names (if available)
        ner_adapter = RedactorService._get_ner_adapter()
        if ner_adapter and ner_adapter.is_available():
            text = RedactorService._redact_names_with_ner(text, ner_adapter)
        
        return text
    
    @staticmethod
    def _redact_names_with_ner(text: str, ner_adapter: 'NERPort') -> str:
        """Redact person names using NER adapter.
        
        Parameters:
            text: Text to redact
            ner_adapter: NER adapter instance
        
        Returns:
            Text with person names redacted
        
        Security Impact:
            - Uses NLP to identify person names more accurately than regex
            - Redacts names in-place, preserving text structure
            - Handles errors gracefully (returns original text on failure)
        """
        try:
            person_names = ner_adapter.extract_person_names(text)
            
            # Redact names in reverse order (to preserve character positions)
            for name, start, end in reversed(person_names):
                text = text[:start] + RedactorService.NAME_MASK + text[end:]
            
            return text
        except Exception as e:
            logger.warning(f"Error redacting names with NER: {e}, using original text")
            return text
    
    @staticmethod
    def redact_patient_record(patient_data: dict) -> dict:
        """Redact all PII fields in a patient record dictionary (FHIR R5 compliant).
        
        Security Impact: Applies redaction to all known PII fields in a patient record.
        This is a convenience method that applies all redaction rules at once.
        Supports both legacy fields and FHIR R5 fields.
        
        Parameters:
            patient_data: Dictionary containing patient record fields
        
        Returns:
            Dictionary with all PII fields redacted
        """
        redacted = patient_data.copy()
        
        # FHIR R5 name fields
        if "family_name" in redacted:
            redacted["family_name"] = RedactorService.redact_name(
                redacted.get("family_name")
            )
        
        if "given_names" in redacted:
            given_names = redacted.get("given_names", [])
            if isinstance(given_names, list):
                redacted["given_names"] = [
                    RedactorService.redact_name(name) for name in given_names
                ]
            else:
                redacted["given_names"] = [RedactorService.redact_name(given_names)]
        
        # Legacy name fields (backward compatibility)
        if "first_name" in redacted:
            redacted["first_name"] = RedactorService.redact_name(
                redacted.get("first_name")
            )
        
        if "last_name" in redacted:
            redacted["last_name"] = RedactorService.redact_name(
                redacted.get("last_name")
            )
        
        # FHIR R5 identifiers
        if "identifiers" in redacted:
            identifiers = redacted.get("identifiers", [])
            if isinstance(identifiers, list):
                redacted["identifiers"] = [
                    RedactorService.redact_ssn(str(identifier)) for identifier in identifiers
                ]
            else:
                redacted["identifiers"] = [RedactorService.redact_ssn(str(identifiers))]
        
        # Legacy SSN field
        if "ssn" in redacted:
            redacted["ssn"] = RedactorService.redact_ssn(redacted.get("ssn"))
        
        # Date of birth
        if "date_of_birth" in redacted:
            dob = redacted.get("date_of_birth")
            if isinstance(dob, (date, datetime, str)):
                redacted["date_of_birth"] = None
        
        # FHIR R5 telecom fields
        if "phone" in redacted:
            redacted["phone"] = RedactorService.redact_phone(redacted.get("phone"))
        
        if "email" in redacted:
            redacted["email"] = RedactorService.redact_email(redacted.get("email"))
        
        if "fax" in redacted:
            redacted["fax"] = RedactorService.redact_phone(redacted.get("fax"))
        
        # FHIR R5 address fields
        if "address_line1" in redacted:
            redacted["address_line1"] = RedactorService.redact_address(
                redacted.get("address_line1")
            )
        
        if "address_line2" in redacted:
            redacted["address_line2"] = RedactorService.redact_address(
                redacted.get("address_line2")
            )
        
        if "postal_code" in redacted:
            redacted["postal_code"] = RedactorService.redact_zip_code(
                redacted.get("postal_code")
            )
        
        # Legacy zip_code field
        if "zip_code" in redacted:
            redacted["zip_code"] = RedactorService.redact_zip_code(
                redacted.get("zip_code")
            )
        
        # FHIR R5 emergency contact fields
        if "emergency_contact_name" in redacted:
            redacted["emergency_contact_name"] = RedactorService.redact_name(
                redacted.get("emergency_contact_name")
            )
        
        if "emergency_contact_phone" in redacted:
            redacted["emergency_contact_phone"] = RedactorService.redact_phone(
                redacted.get("emergency_contact_phone")
            )
        
        return redacted
    
    @staticmethod
    def redact_observation_notes(notes: Optional[str]) -> Optional[str]:
        """Redact PII from clinical observation notes.
        
        Security Impact: Removes PII from unstructured clinical notes to prevent
        patient identification through narrative text.
        
        Parameters:
            notes: Clinical notes text that may contain PII
        
        Returns:
            Notes with PII redacted or None if input is None
        """
        return RedactorService.redact_unstructured_text(notes)

