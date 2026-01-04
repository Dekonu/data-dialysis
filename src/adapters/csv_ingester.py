"""CSV Data Ingestion Adapter.

This adapter implements the IngestionPort contract for CSV data sources.
It provides robust error handling and triage to prevent pipeline failures
from bad data, with configurable column mapping for different CSV formats.

Security Impact:
    - Triage logic identifies and rejects malicious or malformed records
    - Bad records are logged as security rejections for audit trail
    - Each record is wrapped in try/except to prevent DoS attacks
    - PII redaction is applied before validation
    - Handles large CSV files efficiently with streaming

Architecture:
    - Implements IngestionPort (Hexagonal Architecture)
    - Configurable column mapping via dictionary or header detection
    - Isolated from domain core - only depends on ports and models
    - Streaming pattern prevents memory exhaustion
    - Fail-safe design: bad records don't crash the pipeline
"""

import csv
import logging
import hashlib
from pathlib import Path
from typing import Iterator, Optional, Dict, Any, List
from datetime import datetime

from pydantic import ValidationError as PydanticValidationError

from src.domain.ports import (
    IngestionPort,
    Result,
    SourceNotFoundError,
    ValidationError,
    TransformationError,
    UnsupportedSourceError,
)
from src.domain.golden_record import (
    GoldenRecord,
    PatientRecord,
    ClinicalObservation,
    EncounterRecord,
)
from src.domain.services import RedactorService

# Configure logging for security rejections
logger = logging.getLogger(__name__)


class CSVIngester(IngestionPort):
    """CSV ingestion adapter with configurable column mapping and fail-safe error handling.
    
    This adapter reads CSV files and transforms them into GoldenRecord objects using
    configurable column mappings. It supports different CSV structures (with/without headers,
    different delimiters) through flexible configuration.
    
    Key Features:
        - Configurable column mapping: Map CSV columns to domain model fields
        - Header detection: Automatically detects headers or uses provided mapping
        - Triage: Identifies and rejects bad records without crashing
        - Fail-safe: Each record wrapped in try/except to prevent DoS
        - Security logging: Bad records logged as security rejections
        - Streaming: Processes records one-by-one to save memory
        - PII redaction: Applies redaction before validation
        - Multiple delimiters: Supports comma, tab, semicolon, pipe
    
    Security Impact:
        - Malformed records are logged and rejected, not processed
        - DoS protection via per-record error isolation
        - Audit trail of all rejected records
    
    Column Mapping Format:
        {
            "patient_id": "MRN",
            "first_name": "FirstName",
            "last_name": "LastName",
            "date_of_birth": "DOB",
            "gender": "Gender",
            ...
        }
    """
    
    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        has_header: bool = True,
        delimiter: str = ',',
        max_record_size: int = 10 * 1024 * 1024
    ):
        """Initialize CSV ingester.
        
        Parameters:
            column_mapping: Dictionary mapping domain model fields to CSV column names
                          If None, will attempt to auto-detect from header row
            has_header: Whether CSV file has a header row (default: True)
            delimiter: CSV delimiter character (default: ',', also supports '\t', ';', '|')
            max_record_size: Maximum size of a single record in bytes (default: 10MB)
                            Prevents memory exhaustion from oversized records
        """
        self.column_mapping = column_mapping or {}
        self.has_header = has_header
        self.delimiter = delimiter
        self.max_record_size = max_record_size
        self.adapter_name = "csv_ingester"
    
    def can_ingest(self, source: str) -> bool:
        """Check if this adapter can handle the given source.
        
        Parameters:
            source: Source identifier (file path or URL)
        
        Returns:
            bool: True if source is a CSV file, False otherwise
        """
        if not source:
            return False
        
        # Check file extension
        source_path = Path(source)
        if source_path.suffix.lower() in ('.csv', '.tsv'):
            return True
        
        # Check if it's a URL ending in .csv or .tsv
        if source.lower().endswith('.csv') or source.lower().endswith('.tsv'):
            return True
        
        return False
    
    def get_source_info(self, source: str) -> Optional[dict]:
        """Get metadata about the CSV source.
        
        Parameters:
            source: Source identifier
        
        Returns:
            Optional[dict]: Metadata dictionary or None if unavailable
        """
        try:
            source_path = Path(source)
            if source_path.exists():
                stat = source_path.stat()
                # Try to detect delimiter from file extension
                delimiter = '\t' if source_path.suffix.lower() == '.tsv' else ','
                
                return {
                    'format': 'csv',
                    'size': stat.st_size,
                    'encoding': 'utf-8',
                    'exists': True,
                    'delimiter': delimiter,
                    'has_header': self.has_header,
                }
        except (OSError, ValueError):
            pass
        
        return None
    
    def ingest(self, source: str) -> Iterator[Result[GoldenRecord]]:
        """Ingest CSV data and yield Result objects containing GoldenRecord.
        
        This method implements triage logic to handle bad data gracefully:
        1. Each record is wrapped in try/except to prevent DoS
        2. Bad records are logged as security rejections and returned as Result.failure_result()
        3. Pipeline continues processing valid records
        4. Valid records are returned as Result.success_result()
        
        Parameters:
            source: Path to CSV file
        
        Yields:
            Result[GoldenRecord]: Result object containing either:
                - Success: Validated, PII-redacted golden record
                - Failure: Error information (error message, type, details)
        
        Raises:
            SourceNotFoundError: If source file doesn't exist
            UnsupportedSourceError: If source is not valid CSV
        """
        # Validate source exists
        source_path = Path(source)
        if not source_path.exists():
            raise SourceNotFoundError(
                f"CSV source not found: {source}",
                source=source
            )
        
        # Auto-detect delimiter if TSV
        delimiter = '\t' if source_path.suffix.lower() == '.tsv' else self.delimiter
        
        # Check file size to prevent memory exhaustion
        file_size = source_path.stat().st_size
        if file_size > self.max_record_size * 100:
            logger.warning(
                f"Large CSV file detected: {source} ({file_size} bytes). "
                "Processing may be slow."
            )
        
        try:
            # Detect delimiter if not specified
            if delimiter == ',':
                # Try to auto-detect delimiter from first line
                with open(source_path, 'r', encoding='utf-8', newline='') as sample_file:
                    sample = sample_file.read(1024)
                    sniffer = csv.Sniffer()
                    try:
                        detected_delimiter = sniffer.sniff(sample, delimiters=',;\t|').delimiter
                        delimiter = detected_delimiter
                    except:
                        delimiter = ','  # Default to comma
            
            # Open CSV file with appropriate encoding and delimiter
            with open(source_path, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f, delimiter=delimiter) if self.has_header else csv.reader(f, delimiter=delimiter)
                
                # If no header, use column mapping or create default mapping
                if not self.has_header:
                    if not self.column_mapping:
                        raise UnsupportedSourceError(
                            f"CSV file {source} has no header and no column mapping provided",
                            source=source,
                            adapter=self.adapter_name
                        )
                    # For files without headers, we'll use positional mapping
                    # This requires knowing the column order
                    reader = csv.reader(f, delimiter=delimiter)
                
                # Process header row if present
                column_mapping = self.column_mapping.copy() if self.column_mapping else {}
                
                if self.has_header and isinstance(reader, csv.DictReader):
                    # Auto-detect column mapping from header if not provided
                    if not column_mapping:
                        column_mapping = self._auto_detect_column_mapping(reader.fieldnames or [])
                    else:
                        # Normalize column names (case-insensitive, strip whitespace)
                        header_map = {col.strip().lower(): col for col in (reader.fieldnames or [])}
                        # Update column_mapping to use actual header names
                        normalized_mapping = {}
                        for domain_field, csv_col in column_mapping.items():
                            csv_col_lower = csv_col.lower().strip()
                            if csv_col_lower in header_map:
                                normalized_mapping[domain_field] = header_map[csv_col_lower]
                            else:
                                normalized_mapping[domain_field] = csv_col
                        column_mapping = normalized_mapping
                
                # Process each record with triage
                record_count = 0
                rejected_count = 0
                
                for row in reader:
                    record_count += 1
                    
                    # Triage: Wrap each record in try/except to prevent DoS
                    try:
                        # Convert row to dictionary
                        if isinstance(row, dict):
                            record_data = row
                        else:
                            # Row is a list (no header), map by position
                            record_data = self._map_row_by_position(row, column_mapping)
                        
                        # Check record size to prevent memory exhaustion
                        record_str = str(record_data)
                        if len(record_str.encode('utf-8')) > self.max_record_size:
                            error = TransformationError(
                                f"Record {record_count} exceeds maximum size ({self.max_record_size} bytes)",
                                source=source,
                                raw_data={"size": len(record_str), "record_index": record_count}
                            )
                            rejected_count += 1
                            self._log_security_rejection(
                                source=source,
                                record_index=record_count,
                                error=error,
                                raw_record={"record_index": record_count}
                            )
                            yield Result.failure_result(
                                error,
                                error_type="TransformationError",
                                error_details={
                                    "source": source,
                                    "record_index": record_count,
                                    "size": len(record_str),
                                    "max_size": self.max_record_size
                                }
                            )
                            continue
                        
                        # Transform and validate record
                        golden_record = self._triage_and_transform(
                            record_data,
                            column_mapping,
                            source,
                            record_count
                        )
                        
                        # Yield success result
                        yield Result.success_result(golden_record)
                        
                    except (ValidationError, TransformationError) as e:
                        # Security rejection: Log and return failure result
                        rejected_count += 1
                        self._log_security_rejection(
                            source=source,
                            record_index=record_count,
                            error=e,
                            raw_record=row if isinstance(row, dict) else {"record_index": record_count}
                        )
                        yield Result.failure_result(
                            e,
                            error_type=type(e).__name__,
                            error_details={
                                "source": source,
                                "record_index": record_count,
                                **(e.details if hasattr(e, 'details') and e.details else {}),
                                **(e.raw_data if hasattr(e, 'raw_data') and e.raw_data else {})
                            }
                        )
                        continue
                        
                    except Exception as e:
                        # Unexpected error: Log as security concern and return failure result
                        rejected_count += 1
                        logger.error(
                            f"Unexpected error processing record {record_count} from {source}: {str(e)}",
                            exc_info=True,
                            extra={
                                'source': source,
                                'record_index': record_count,
                                'error_type': type(e).__name__,
                            }
                        )
                        yield Result.failure_result(
                            e,
                            error_type=type(e).__name__,
                            error_details={
                                "source": source,
                                "record_index": record_count
                            }
                        )
                        continue
                
                # Log ingestion summary
                if record_count > 0:
                    logger.info(
                        f"CSV ingestion complete: {source} - "
                        f"{record_count - rejected_count} accepted, {rejected_count} rejected"
                    )
                    
        except csv.Error as e:
            raise UnsupportedSourceError(
                f"Invalid CSV format in {source}: {str(e)}",
                source=source,
                adapter=self.adapter_name
            )
        except Exception as e:
            raise SourceNotFoundError(
                f"Cannot read CSV source {source}: {str(e)}",
                source=source
            )
    
    def _auto_detect_column_mapping(self, headers: List[str]) -> Dict[str, str]:
        """Auto-detect column mapping from CSV headers.
        
        This method attempts to match CSV column names to domain model fields
        using common naming patterns and case-insensitive matching.
        
        Parameters:
            headers: List of CSV column names from header row
        
        Returns:
            dict: Mapping from domain model fields to CSV column names
        """
        mapping = {}
        
        # Common field name variations
        field_variations = {
            'patient_id': ['patient_id', 'patientid', 'mrn', 'medical_record_number', 'id', 'record_id'],
            'first_name': ['first_name', 'firstname', 'fname', 'given_name', 'givenname'],
            'last_name': ['last_name', 'lastname', 'lname', 'family_name', 'familyname', 'surname'],
            'date_of_birth': ['date_of_birth', 'dateofbirth', 'dob', 'birth_date', 'birthdate'],
            'gender': ['gender', 'sex'],
            'ssn': ['ssn', 'social_security_number', 'socialsecuritynumber'],
            'phone': ['phone', 'phone_number', 'phonenumber', 'telephone', 'tel'],
            'email': ['email', 'email_address', 'emailaddress'],
            'address_line1': ['address', 'address_line1', 'addressline1', 'street', 'street_address'],
            'city': ['city'],
            'state': ['state', 'state_code', 'statecode'],
            'postal_code': ['postal_code', 'postalcode', 'zip', 'zip_code', 'zipcode'],
        }
        
        # Normalize headers (lowercase, strip)
        normalized_headers = {h.lower().strip(): h for h in headers}
        
        # Match headers to domain fields
        for domain_field, variations in field_variations.items():
            for variation in variations:
                if variation.lower() in normalized_headers:
                    mapping[domain_field] = normalized_headers[variation.lower()]
                    break
        
        return mapping
    
    def _map_row_by_position(self, row: List[str], column_mapping: Dict[str, str]) -> Dict[str, Any]:
        """Map CSV row (list) to dictionary using positional mapping.
        
        This is used when CSV has no header row. The column_mapping should
        specify the position (index) for each field, or we use a default order.
        
        Parameters:
            row: CSV row as list of values
            column_mapping: Mapping from domain fields to CSV column indices or names
        
        Returns:
            dict: Record data dictionary
        """
        # If column_mapping uses indices, use them directly
        # Otherwise, assume standard order
        record_data = {}
        
        # Standard field order (if mapping not provided with indices)
        standard_order = [
            'patient_id', 'first_name', 'last_name', 'date_of_birth',
            'gender', 'ssn', 'phone', 'email', 'address_line1',
            'city', 'state', 'postal_code'
        ]
        
        for i, domain_field in enumerate(standard_order):
            if i < len(row) and domain_field in column_mapping:
                value = row[i].strip() if row[i] else None
                if value:
                    record_data[domain_field] = value
        
        return record_data
    
    def _triage_and_transform(
        self,
        record_data: Dict[str, Any],
        column_mapping: Dict[str, str],
        source: str,
        record_index: int
    ) -> GoldenRecord:
        """Triage and transform CSV record data into a GoldenRecord.
        
        This method implements the triage logic:
        1. Map CSV columns to domain model fields
        2. Pre-validation checks (structure, required fields)
        3. Transformation to domain models
        4. PII redaction (automatic via model validators)
        5. Pydantic validation (Safety Layer)
        
        Parameters:
            record_data: CSV record as dictionary (column name -> value)
            column_mapping: Mapping from domain fields to CSV column names
            source: Source identifier
            record_index: Index of record in source (for logging)
        
        Returns:
            GoldenRecord: Validated, PII-redacted golden record
        
        Raises:
            TransformationError: If transformation fails
            ValidationError: If validation fails
        """
        # Triage: Pre-validation checks
        if not isinstance(record_data, dict):
            raise TransformationError(
                f"Record {record_index} is not a dictionary",
                source=source,
                raw_data={"record_index": record_index, "type": type(record_data).__name__}
            )
        
        # Map CSV columns to domain model fields
        patient_data = {}
        for domain_field, csv_column in column_mapping.items():
            if csv_column in record_data:
                value = record_data[csv_column]
                # Clean up value (strip whitespace, handle empty strings)
                if isinstance(value, str):
                    value = value.strip()
                    if not value:  # Empty string becomes None
                        value = None
                patient_data[domain_field] = value
        
        # Check for required patient identifier
        if 'patient_id' not in patient_data or not patient_data['patient_id']:
            raise TransformationError(
                f"Record {record_index} missing required patient_id",
                source=source,
                raw_data={"record_index": record_index, "keys": list(record_data.keys())}
            )
        
        try:
            # Transform patient record (PII redaction happens automatically via validators)
            patient = PatientRecord(**patient_data)
            
            # CSV typically has one record per row, so encounters and observations are empty
            # unless the CSV has additional columns for them
            encounters = []
            observations = []
            
            # Check for encounter data in CSV (if columns exist)
            encounter_data = self._extract_encounter_data(record_data, column_mapping)
            if encounter_data:
                try:
                    encounter = EncounterRecord(**encounter_data)
                    encounters.append(encounter)
                except PydanticValidationError as e:
                    logger.warning(
                        f"Encounter validation failed in record {record_index}: {str(e)}",
                        extra={'source': source, 'record_index': record_index}
                    )
            
            # Check for observation data in CSV (if columns exist)
            observation_data = self._extract_observation_data(record_data, column_mapping)
            if observation_data:
                try:
                    observation = ClinicalObservation(**observation_data)
                    observations.append(observation)
                except PydanticValidationError as e:
                    logger.warning(
                        f"Observation validation failed in record {record_index}: {str(e)}",
                        extra={'source': source, 'record_index': record_index}
                    )
            
            # Generate transformation hash for audit trail
            transformation_hash = self._generate_hash(record_data)
            
            # Construct GoldenRecord (final validation)
            golden_record = GoldenRecord(
                patient=patient,
                encounters=encounters,
                observations=observations,
                source_adapter=self.adapter_name,
                transformation_hash=transformation_hash
            )
            
            return golden_record
            
        except PydanticValidationError as e:
            # Validation error: Convert to domain ValidationError
            error_details = {
                'record_index': record_index,
                'validation_errors': str(e),
                'error_count': len(e.errors()) if hasattr(e, 'errors') else 0,
            }
            raise ValidationError(
                f"Record {record_index} failed validation: {str(e)}",
                source=source,
                details=error_details
            )
        except Exception as e:
            # Transformation error: Wrap in domain exception
            raise TransformationError(
                f"Record {record_index} transformation failed: {str(e)}",
                source=source,
                raw_data={"record_index": record_index}
            )
    
    def _extract_encounter_data(
        self,
        record_data: Dict[str, Any],
        column_mapping: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Extract encounter data from CSV record if present.
        
        Parameters:
            record_data: CSV record dictionary
            column_mapping: Column mapping dictionary
        
        Returns:
            Optional[dict]: Encounter data dictionary or None
        """
        encounter_fields = [
            'encounter_id', 'class_code', 'period_start', 'period_end',
            'diagnosis_codes', 'status'
        ]
        
        encounter_data = {}
        for field in encounter_fields:
            # Check if field is in column_mapping (reverse lookup)
            csv_col = None
            for domain_field, csv_column in column_mapping.items():
                if domain_field == field and csv_column in record_data:
                    csv_col = csv_column
                    break
            
            if csv_col and record_data.get(csv_col):
                value = record_data[csv_col]
                if isinstance(value, str):
                    value = value.strip()
                encounter_data[field] = value if value else None
        
        return encounter_data if encounter_data else None
    
    def _extract_observation_data(
        self,
        record_data: Dict[str, Any],
        column_mapping: Dict[str, str]
    ) -> Optional[Dict[str, Any]]:
        """Extract observation data from CSV record if present.
        
        Parameters:
            record_data: CSV record dictionary
            column_mapping: Column mapping dictionary
        
        Returns:
            Optional[dict]: Observation data dictionary or None
        """
        observation_fields = [
            'observation_id', 'category', 'code', 'value', 'unit',
            'effective_date', 'status'
        ]
        
        observation_data = {}
        for field in observation_fields:
            # Check if field is in column_mapping (reverse lookup)
            csv_col = None
            for domain_field, csv_column in column_mapping.items():
                if domain_field == field and csv_column in record_data:
                    csv_col = csv_column
                    break
            
            if csv_col and record_data.get(csv_col):
                value = record_data[csv_col]
                if isinstance(value, str):
                    value = value.strip()
                observation_data[field] = value if value else None
        
        return observation_data if observation_data else None
    
    def _generate_hash(self, record_data: Dict[str, Any]) -> str:
        """Generate hash of CSV record for audit trail.
        
        Parameters:
            record_data: CSV record dictionary
        
        Returns:
            str: SHA-256 hash of the record
        """
        # Sort keys for consistent hashing
        record_str = str(sorted(record_data.items()))
        return hashlib.sha256(record_str.encode('utf-8')).hexdigest()
    
    def _log_security_rejection(
        self,
        source: str,
        record_index: int,
        error: Exception,
        raw_record: Dict[str, Any]
    ) -> None:
        """Log a security rejection for audit trail.
        
        Parameters:
            source: Source identifier
            record_index: Index of rejected record
            error: Exception that caused rejection
            raw_record: Raw record data (may be truncated)
        """
        # Truncate raw_record for logging (prevent log bloat)
        truncated_record = self._truncate_for_logging(raw_record)
        
        logger.warning(
            f"SECURITY REJECTION: Record {record_index} from {source} rejected",
            extra={
                'rejection_type': 'validation_failure',
                'source': source,
                'record_index': record_index,
                'error_type': type(error).__name__,
                'error_message': str(error),
                'raw_record_preview': truncated_record,
            }
        )
    
    def _truncate_for_logging(self, data: Dict[str, Any], max_size: int = 500) -> Dict[str, Any]:
        """Truncate data dictionary for safe logging.
        
        Parameters:
            data: Data dictionary to truncate
            max_size: Maximum size in characters
        
        Returns:
            dict: Truncated dictionary safe for logging
        """
        data_str = str(data)
        if len(data_str) <= max_size:
            return data
        
        # Truncate and add indicator
        truncated = dict(list(data.items())[:5])  # Keep first 5 items
        truncated['_truncated'] = True
        truncated['_original_size'] = len(data_str)
        return truncated

