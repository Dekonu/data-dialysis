"""PII Redaction Service.

This module provides the RedactorService class responsible for identifying and masking
Personally Identifiable Information (PII) in clinical data. This service follows the
Single Responsibility Principle by isolating all security and redaction logic from
data models.

Security Impact:
    - Prevents PII from being persisted in unredacted form
    - Uses regex patterns to identify structured PII (SSN, phone, email)
    - Handles unstructured text redaction for clinical notes
    - All redaction methods are pure functions (no side effects)

Architecture:
    - Pure domain service with zero infrastructure dependencies
    - Stateless service that can be used across the application
    - Follows Hexagonal Architecture: Core security logic isolated from infrastructure
"""

import re
from datetime import date, datetime
from typing import Optional, Union
import pandas as pd


class RedactorService:
    """Service for identifying and masking PII in clinical records.
    
    This service provides methods to redact various types of PII including:
    - Social Security Numbers (SSN)
    - Phone numbers
    - Email addresses
    - Names (first and last)
    - Addresses
    - Dates of birth
    - Unstructured text containing PII
    
    All methods are pure functions that take input and return redacted output
    without side effects.
    """
    
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
        This is a basic regex-based implementation. For production, consider
        NLP-based Named Entity Recognition (NER) for better accuracy.
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
            # Redact SSNs
            result = result.str.replace(RedactorService.SSN_PATTERN.pattern, RedactorService.SSN_MASK, regex=True)
            # Redact phone numbers
            result = result.str.replace(RedactorService.PHONE_PATTERN.pattern, RedactorService.PHONE_MASK, regex=True)
            # Redact email addresses
            result = result.str.replace(RedactorService.EMAIL_PATTERN.pattern, RedactorService.EMAIL_MASK, regex=True)
            # Note: Name redaction in unstructured text is complex and would benefit
            # from NLP/NER. This basic implementation may miss names or over-redact.
            # For production, integrate with SpaCy or similar NER library.
            result[value.isna()] = None
            return result
        
        # Scalar operation (backward compatible)
        if not value:
            return None
        
        text = str(value)
        
        # Redact SSNs
        text = RedactorService.SSN_PATTERN.sub(RedactorService.SSN_MASK, text)
        
        # Redact phone numbers
        text = RedactorService.PHONE_PATTERN.sub(RedactorService.PHONE_MASK, text)
        
        # Redact email addresses
        text = RedactorService.EMAIL_PATTERN.sub(RedactorService.EMAIL_MASK, text)
        
        # Note: Name redaction in unstructured text is complex and would benefit
        # from NLP/NER. This basic implementation may miss names or over-redact.
        # For production, integrate with SpaCy or similar NER library.
        
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

