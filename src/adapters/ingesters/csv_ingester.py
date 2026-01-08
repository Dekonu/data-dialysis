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
from typing import Iterator, Optional, Dict, Any, List, Union
from datetime import datetime

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from src.domain.ports import (
    IngestionPort,
    Result,
    SourceNotFoundError,
    ValidationError,
    TransformationError,
    UnsupportedSourceError,
)
from src.domain.utils import should_use_parallel, initialize_pandarallel_if_needed
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
        max_record_size: int = 10 * 1024 * 1024,
        chunk_size: int = 10000,
        target_total_rows: int = 50000
    ):
        """Initialize CSV ingester.
        
        Parameters:
            column_mapping: Dictionary mapping domain model fields to CSV column names
                          If None, will attempt to auto-detect from header row
            has_header: Whether CSV file has a header row (default: True)
            delimiter: CSV delimiter character (default: ',', also supports '\t', ';', '|')
            max_record_size: Maximum size of a single record in bytes (default: 10MB)
                            Prevents memory exhaustion from oversized records
            chunk_size: Initial number of rows to process per chunk (default: 10000)
                       Will be adjusted adaptively after first chunk if target_total_rows is set
            target_total_rows: Target total rows per chunk (default: 50000)
                              If set, chunk_size will be adjusted after first chunk to achieve this target
                              Set to 0 to disable adaptive chunking
        """
        self.column_mapping = column_mapping or {}
        self.has_header = has_header
        self.delimiter = delimiter
        self.max_record_size = max_record_size
        self.initial_chunk_size = chunk_size
        self.chunk_size = chunk_size
        self.target_total_rows = target_total_rows
        self.adapter_name = "csv_ingester"
        
        # Adaptive chunk sizing state
        self.adaptive_chunking_enabled = target_total_rows > 0
        self.chunk_size_adjusted = False  # Track if we've adjusted chunk size
        
        # CSV type detection (set during ingestion)
        self._detected_csv_type = 'patients'  # Default
        
        # Initialize pandarallel if needed (one-time setup)
        initialize_pandarallel_if_needed(progress_bar=False)
    
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
    
    def ingest(self, source: str) -> Iterator[Result[Union[GoldenRecord, pd.DataFrame]]]:
        """Ingest CSV data and yield Result objects containing DataFrames (chunked processing).
        
        This method uses pandas chunked reading for vectorized processing:
        1. Reads CSV in chunks (default: 10,000 rows per chunk)
        2. Applies vectorized PII redaction to entire chunks
        3. Validates records per-row (required for Pydantic)
        4. Tracks failures at chunk level for CircuitBreaker
        5. Yields Result[pd.DataFrame] with validated, redacted data
        
        Parameters:
            source: Path to CSV file
        
        Yields:
            Result[pd.DataFrame]: Result object containing either:
                - Success: DataFrame with validated, PII-redacted records
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
                "Processing in chunks for memory efficiency."
            )
        
        try:
            # Read first chunk to detect column mapping
            first_chunk = pd.read_csv(
                source_path,
                nrows=1,
                delimiter=delimiter,
                header=0 if self.has_header else None,
                encoding='utf-8',
                low_memory=False
            )
            
            # Detect CSV type (patients, encounters, observations)
            csv_type = self._detect_csv_type(first_chunk.columns.tolist())
            logger.info(f"Detected CSV type: {csv_type} for {source}")
            
            # Determine column mapping
            column_mapping = self.column_mapping.copy() if self.column_mapping else {}
            if self.has_header:
                if not column_mapping:
                    # Auto-detect from headers using detected CSV type
                    column_mapping = self._auto_detect_column_mapping(first_chunk.columns.tolist(), csv_type)
                else:
                    # Normalize column names (case-insensitive, strip whitespace)
                    header_map = {col.strip().lower(): col for col in first_chunk.columns}
                    normalized_mapping = {}
                    for domain_field, csv_col in column_mapping.items():
                        csv_col_lower = csv_col.lower().strip()
                        if csv_col_lower in header_map:
                            normalized_mapping[domain_field] = header_map[csv_col_lower]
                        else:
                            normalized_mapping[domain_field] = csv_col
                    column_mapping = normalized_mapping
            else:
                if not column_mapping:
                    raise UnsupportedSourceError(
                        f"CSV file {source} has no header and no column mapping provided",
                        source=source,
                        adapter=self.adapter_name
                    )
            
            # Store CSV type for use in validation
            self._detected_csv_type = csv_type
            
            # Process CSV in chunks using pandas
            chunk_count = 0
            total_processed = 0
            total_rejected = 0
            current_chunk_size = self.chunk_size
            
            # Create iterator for reading CSV chunks
            chunk_iterator = pd.read_csv(
                source_path,
                chunksize=current_chunk_size,
                delimiter=delimiter,
                header=0 if self.has_header else None,
                encoding='utf-8',
                low_memory=False
            )
            
            for chunk_df in chunk_iterator:
                chunk_count += 1
                
                try:
                    # Map CSV columns to domain model fields
                    mapped_df = self._map_dataframe_columns(chunk_df, column_mapping)
                    
                    # Capture original DataFrame BEFORE redaction (for raw vault)
                    original_df = mapped_df.copy()
                    
                    # Apply vectorized PII redaction
                    redacted_df = self._redact_dataframe(mapped_df)
                    
                    # Validate and filter valid records
                    validated_df, failed_indices = self._validate_dataframe_chunk(
                        redacted_df,
                        source,
                        chunk_count,
                        csv_type=self._detected_csv_type
                    )
                    
                    # Filter original_df to match validated_df (same rows)
                    # This ensures raw_df has the same rows as redacted_df after validation
                    # Note: validated_df is created from valid_records with a new default index,
                    # so we need to filter original_df by excluding failed_indices
                    if validated_df.empty:
                        # All records failed validation, original_df should be empty too
                        original_df = original_df.iloc[0:0].copy()
                    elif len(failed_indices) > 0:
                        # Some records failed - remove failed indices from original_df
                        # Then reindex to match validated_df (which has a new default index)
                        original_df = original_df.drop(failed_indices)
                        # Reset index to match validated_df's default integer index
                        original_df = original_df.reset_index(drop=True)
                        # validated_df also has a default integer index, so they should align now
                    else:
                        # No failures - just reset index to match validated_df
                        original_df = original_df.reset_index(drop=True)
                    
                    # Track failures
                    num_failed = len(failed_indices)
                    num_valid = len(validated_df)
                    total_processed += len(chunk_df)
                    total_rejected += num_failed
                    
                    # Adaptive chunk sizing: Adjust after first chunk based on validation success rate
                    if self.adaptive_chunking_enabled and chunk_count == 1 and num_valid > 0:
                        # Calculate validation success rate
                        success_rate = num_valid / len(chunk_df) if len(chunk_df) > 0 else 1.0
                        
                        # Adjust chunk size to achieve target_total_rows of valid records
                        # Account for validation failures
                        if success_rate > 0:
                            optimal_chunk_size = int(self.target_total_rows / success_rate)
                            # Ensure minimum chunk size (at least 1000) and maximum (no more than 50000)
                            optimal_chunk_size = max(1000, min(optimal_chunk_size, 50000))
                            
                            if optimal_chunk_size != current_chunk_size:
                                logger.info(
                                    f"Adaptive chunk sizing activated for CSV: "
                                    f"Initial chunk size: {self.initial_chunk_size}, "
                                    f"Optimal chunk size: {optimal_chunk_size} "
                                    f"(target: {self.target_total_rows} valid rows, "
                                    f"success rate: {success_rate:.2%})"
                                )
                                current_chunk_size = optimal_chunk_size
                                self.chunk_size = optimal_chunk_size  # Update instance variable
                                self.chunk_size_adjusted = True
                                # Note: The iterator is already created with initial chunk_size
                                # The adjustment will apply to future file reads
                    
                    # Log failures if any
                    if num_failed > 0:
                        logger.warning(
                            f"Chunk {chunk_count} from {source}: {num_failed} records failed validation, "
                            f"{num_valid} records passed"
                        )
                    
                    # Yield success result with validated DataFrame and original DataFrame (for raw vault)
                    # Return as tuple: (redacted_df, raw_df)
                    if len(validated_df) > 0:
                        yield Result.success_result((validated_df, original_df))
                    
                    # Yield failure result if entire chunk failed
                    if num_failed == len(chunk_df) and num_valid == 0:
                        yield Result.failure_result(
                            ValidationError(
                                f"Chunk {chunk_count} from {source} failed validation: all {num_failed} records invalid",
                                source=source,
                                details={"chunk": chunk_count, "failed_count": num_failed}
                            ),
                            error_type="ValidationError",
                            error_details={
                                "source": source,
                                "chunk": chunk_count,
                                "failed_count": num_failed,
                                "total_in_chunk": len(chunk_df)
                            }
                        )
                        
                except Exception as e:
                    # Chunk-level error: Log and yield failure
                    total_processed += len(chunk_df)
                    total_rejected += len(chunk_df)
                    logger.error(
                        f"Error processing chunk {chunk_count} from {source}: {str(e)}",
                        exc_info=True,
                        extra={
                            'source': source,
                            'chunk': chunk_count,
                            'error_type': type(e).__name__,
                        }
                    )
                    yield Result.failure_result(
                        e,
                        error_type=type(e).__name__,
                        error_details={
                            "source": source,
                            "chunk": chunk_count,
                            "chunk_size": len(chunk_df)
                        }
                    )
                    continue
            
            # Log ingestion summary
            if total_processed > 0:
                logger.info(
                    f"CSV ingestion complete: {source} - "
                    f"{total_processed - total_rejected} accepted, {total_rejected} rejected "
                    f"({chunk_count} chunks processed)"
                )
                    
        except pd.errors.EmptyDataError:
            raise UnsupportedSourceError(
                f"CSV file {source} is empty",
                source=source,
                adapter=self.adapter_name
            )
        except pd.errors.ParserError as e:
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
    
    def _detect_csv_type(self, headers: List[str]) -> str:
        """Detect CSV type based on headers.
        
        Parameters:
            headers: List of CSV column names from header row
        
        Returns:
            str: 'patients', 'encounters', 'observations', or 'unknown'
        """
        normalized_headers = [h.lower().strip() for h in headers]
        
        # Check for specific IDs first (most reliable indicator)
        if 'observation_id' in normalized_headers:
            return 'observations'
        elif 'encounter_id' in normalized_headers:
            return 'encounters'
        elif 'patient_id' in normalized_headers or 'mrn' in normalized_headers:
            # Check if it has patient fields (not just patient_id)
            patient_fields = ['first_name', 'last_name', 'firstname', 'lastname', 'fname', 'lname']
            if any(field in normalized_headers for field in patient_fields):
                return 'patients'
            # If only patient_id, could be encounters or observations referencing patient
            # Check for encounter or observation fields
            if 'encounter_id' in normalized_headers or 'class_code' in normalized_headers:
                return 'encounters'
            elif 'observation_id' in normalized_headers or 'category' in normalized_headers:
                return 'observations'
            # Default to patients if only patient_id
            return 'patients'
        
        return 'unknown'
    
    def _auto_detect_column_mapping(self, headers: List[str], csv_type: str = 'patients') -> Dict[str, str]:
        """Auto-detect column mapping from CSV headers.
        
        This method attempts to match CSV column names to domain model fields
        using common naming patterns and case-insensitive matching.
        
        Parameters:
            headers: List of CSV column names from header row
            csv_type: Type of CSV ('patients', 'encounters', 'observations')
        
        Returns:
            dict: Mapping from domain model fields to CSV column names
        """
        mapping = {}
        
        # Common field name variations by type
        if csv_type == 'patients':
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
        elif csv_type == 'encounters':
            field_variations = {
                'encounter_id': ['encounter_id', 'encounterid', 'enc_id'],
                'patient_id': ['patient_id', 'patientid', 'mrn', 'medical_record_number'],
                'class_code': ['class_code', 'classcode', 'class', 'encounter_class'],
                'status': ['status', 'encounter_status'],
                'period_start': ['period_start', 'start', 'start_date', 'startdate', 'admission_date'],
                'period_end': ['period_end', 'end', 'end_date', 'enddate', 'discharge_date'],
                'diagnosis_codes': ['diagnosis_codes', 'diagnosiscodes', 'diagnosis', 'icd_codes', 'icd10'],
            }
        elif csv_type == 'observations':
            field_variations = {
                'observation_id': ['observation_id', 'observationid', 'obs_id'],
                'patient_id': ['patient_id', 'patientid', 'mrn', 'medical_record_number'],
                'category': ['category', 'observation_category'],
                'code': ['code', 'observation_code', 'loinc_code', 'loinc'],
                'value': ['value', 'result_value', 'numeric_value'],
                'unit': ['unit', 'result_unit', 'measurement_unit'],
                'effective_date': ['effective_date', 'effectivedate', 'observation_date', 'result_date'],
                'status': ['status', 'observation_status'],
                'notes': ['notes', 'note', 'comment', 'interpretation'],
            }
        else:
            # Default to patient fields
            field_variations = {
                'patient_id': ['patient_id', 'patientid', 'mrn', 'medical_record_number', 'id', 'record_id'],
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
    
    def _map_dataframe_columns(self, df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
        """Map CSV DataFrame columns to domain model field names.
        
        Parameters:
            df: Raw CSV DataFrame
            column_mapping: Mapping from domain fields to CSV column names
        
        Returns:
            DataFrame with columns renamed to domain field names
        """
        # Create reverse mapping (CSV column -> domain field)
        reverse_mapping = {csv_col: domain_field for domain_field, csv_col in column_mapping.items()}
        
        # Rename columns that are in the mapping
        df_mapped = df.copy()
        df_mapped = df_mapped.rename(columns=reverse_mapping)
        
        # Keep only columns that are in domain model (drop unmapped columns)
        domain_columns = list(column_mapping.keys())
        existing_columns = [col for col in domain_columns if col in df_mapped.columns]
        df_mapped = df_mapped[existing_columns]
        
        return df_mapped
    
    def _redact_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply vectorized PII redaction to DataFrame columns.
        
        Note: Some fields (like postal_code) are not redacted here because
        they need to pass pattern validation first. They will be redacted
        by the Pydantic validators during model creation.
        
        Parameters:
            df: DataFrame with domain model column names
        
        Returns:
            DataFrame with PII fields redacted (where safe to do so)
        """
        df_redacted = df.copy()
        
        # Get redaction context for logging (if available)
        try:
            from src.infrastructure.redaction_context import get_redaction_context
            context = get_redaction_context()
            logger_available = context is not None and context.get('logger') is not None
        except (ImportError, AttributeError):
            logger_available = False
        
        # Helper function to log redactions from vectorized operations
        def log_vectorized_redactions(series: pd.Series, field_name: str, rule_triggered: str, original_series: pd.Series) -> None:
            """Log redactions that occurred during vectorized operations."""
            if not logger_available:
                return
            
            # Find rows where redaction occurred (value changed and matches mask)
            if field_name == 'ssn':
                mask = RedactorService.SSN_MASK
            elif field_name in ['phone', 'fax']:
                mask = RedactorService.PHONE_MASK
            elif field_name == 'email':
                mask = RedactorService.EMAIL_MASK
            elif field_name in ['first_name', 'last_name', 'family_name', 'given_names', 'emergency_contact_name']:
                mask = RedactorService.NAME_MASK
            elif field_name in ['address_line1', 'address_line2']:
                mask = RedactorService.ADDRESS_MASK
            else:
                return  # Unknown field type
            
            # Find indices where redaction occurred
            redacted_mask = (series == mask) & (original_series.notna()) & (original_series != mask)
            if redacted_mask.any():
                logger = context.get('logger')
                source_adapter = context.get('source_adapter')
                ingestion_id = context.get('ingestion_id')
                
                # Log each redaction
                for idx in original_series[redacted_mask].index:
                    # Use .loc instead of .iloc since idx is a DataFrame index label, not positional
                    original_value = str(original_series.loc[idx])
                    # Get record_id from DataFrame if available
                    # Try encounter_id, observation_id, or patient_id
                    record_id = None
                    if 'encounter_id' in df_redacted.columns and idx in df_redacted.index:
                        record_id = str(df_redacted.loc[idx, 'encounter_id'])
                    elif 'observation_id' in df_redacted.columns and idx in df_redacted.index:
                        record_id = str(df_redacted.loc[idx, 'observation_id'])
                    elif 'patient_id' in df_redacted.columns and idx in df_redacted.index:
                        record_id = str(df_redacted.loc[idx, 'patient_id'])
                    
                    logger.log_redaction(
                        field_name=field_name,
                        original_value=original_value,
                        rule_triggered=rule_triggered,
                        record_id=record_id,
                        source_adapter=source_adapter
                    )
        
        # Apply vectorized redaction to PII columns that don't have pattern constraints
        if 'ssn' in df_redacted.columns:
            original_ssn = df_redacted['ssn'].copy()
            df_redacted['ssn'] = RedactorService.redact_ssn(df_redacted['ssn'])
            log_vectorized_redactions(df_redacted['ssn'], 'ssn', 'SSN_PATTERN', original_ssn)
        
        if 'first_name' in df_redacted.columns:
            original_first_name = df_redacted['first_name'].copy()
            df_redacted['first_name'] = RedactorService.redact_name(df_redacted['first_name'])
            log_vectorized_redactions(df_redacted['first_name'], 'first_name', 'NAME_PATTERN', original_first_name)
        
        if 'last_name' in df_redacted.columns:
            original_last_name = df_redacted['last_name'].copy()
            df_redacted['last_name'] = RedactorService.redact_name(df_redacted['last_name'])
            log_vectorized_redactions(df_redacted['last_name'], 'last_name', 'NAME_PATTERN', original_last_name)
        
        if 'phone' in df_redacted.columns:
            original_phone = df_redacted['phone'].copy()
            df_redacted['phone'] = RedactorService.redact_phone(df_redacted['phone'])
            log_vectorized_redactions(df_redacted['phone'], 'phone', 'PHONE_PATTERN', original_phone)
        
        if 'email' in df_redacted.columns:
            original_email = df_redacted['email'].copy()
            df_redacted['email'] = RedactorService.redact_email(df_redacted['email'])
            log_vectorized_redactions(df_redacted['email'], 'email', 'EMAIL_PATTERN', original_email)
        
        if 'address_line1' in df_redacted.columns:
            original_address = df_redacted['address_line1'].copy()
            df_redacted['address_line1'] = RedactorService.redact_address(df_redacted['address_line1'])
            log_vectorized_redactions(df_redacted['address_line1'], 'address_line1', 'ADDRESS_PATTERN', original_address)
        
        # Note: postal_code is NOT redacted here because it needs to pass pattern validation first
        # It will be redacted by the Pydantic validator after validation
        
        # Date of birth is always redacted (set to None)
        # Note: DOB redaction is logged by the Pydantic validator
        if 'date_of_birth' in df_redacted.columns:
            df_redacted['date_of_birth'] = None
        
        return df_redacted
    
    def _validate_dataframe_chunk(
        self,
        df: pd.DataFrame,
        source: str,
        chunk_number: int,
        csv_type: str = 'patients'
    ) -> tuple[pd.DataFrame, List[int]]:
        """Validate DataFrame chunk and return valid records + failed indices.
        
        Parameters:
            df: DataFrame with redacted data
            source: Source identifier
            chunk_number: Chunk number for logging
            csv_type: Type of CSV ('patients', 'encounters', 'observations')
        
        Returns:
            Tuple of (validated DataFrame, list of failed row indices)
        """
        valid_records = []
        failed_indices = []
        
        # Validate each row individually (required for Pydantic)
        for idx, row in df.iterrows():
            try:
                # Convert row to dictionary
                row_dict = row.to_dict()
                
                # Clean up NaN values and convert types (pandas reads missing values as NaN/float)
                # This is critical: Pydantic expects None for Optional fields, not NaN
                for key, value in row_dict.items():
                    # Skip list/array values (like identifiers) - pd.isna() doesn't work on them
                    if isinstance(value, (list, tuple)):
                        continue
                    
                    # Convert NaN values to None FIRST (Pydantic expects None for Optional fields)
                    # Check both pd.isna() to catch all NaN cases (including float NaN)
                    if pd.isna(value):
                        row_dict[key] = None
                        continue  # Skip further processing for NaN values
                    
                    # Handle specific field conversions for non-NaN values
                    if key == 'postal_code' and pd.notna(value):
                        # Convert postal_code to string
                        row_dict[key] = str(value)
                    elif isinstance(value, float) and key in ['state', 'city']:
                        # Convert float to string for string fields
                        row_dict[key] = str(int(value)) if value == int(value) else str(value)
                    elif isinstance(value, float) and key in ['value', 'unit']:
                        # Convert float to string for observation value/unit fields
                        # This handles cases where CSV has numeric values that should be strings
                        row_dict[key] = str(int(value)) if value == int(value) else str(value)
                
                # Handle diagnosis_codes (convert comma-separated string to list)
                if 'diagnosis_codes' in row_dict and pd.notna(row_dict.get('diagnosis_codes')):
                    if isinstance(row_dict['diagnosis_codes'], str):
                        # Split comma-separated values and strip whitespace
                        row_dict['diagnosis_codes'] = [code.strip() for code in row_dict['diagnosis_codes'].split(',') if code.strip()]
                
                # Check for required IDs based on CSV type
                if csv_type == 'encounters':
                    if 'encounter_id' not in row_dict or pd.isna(row_dict.get('encounter_id')) or not row_dict.get('encounter_id'):
                        failed_indices.append(idx)
                        continue
                    record_id = row_dict.get('encounter_id')
                elif csv_type == 'observations':
                    if 'observation_id' not in row_dict or pd.isna(row_dict.get('observation_id')) or not row_dict.get('observation_id'):
                        failed_indices.append(idx)
                        continue
                    record_id = row_dict.get('observation_id')
                else:  # patients
                    if 'patient_id' not in row_dict or pd.isna(row_dict.get('patient_id')) or not row_dict.get('patient_id'):
                        failed_indices.append(idx)
                        continue
                    record_id = row_dict.get('patient_id')
                
                # Set record_id in context for this row (if available)
                # This allows validators to log redactions with the correct record_id
                try:
                    from src.infrastructure.redaction_context import set_redaction_context, get_redaction_context
                    context = get_redaction_context()
                    if context and record_id:
                        # Update context with this record's ID
                        set_redaction_context(
                            logger=context.get('logger'),
                            record_id=str(record_id),
                            source_adapter=context.get('source_adapter'),
                            ingestion_id=context.get('ingestion_id')
                        )
                    elif not context:
                        # Context not set - this means redaction logging won't work
                        # This is expected in tests but should be set in production
                        logger.debug("Redaction context not available - redaction logging disabled")
                except (ImportError, AttributeError) as e:
                    logger.debug(f"Could not set redaction context: {e}")
                    pass  # Context not available - skip (e.g., in tests)
                
                # Validate based on CSV type
                if csv_type == 'encounters':
                    # Create EncounterRecord
                    encounter = EncounterRecord(**{k: v for k, v in row_dict.items() if k in EncounterRecord.model_fields})
                    # Store validated record as dict for DataFrame reconstruction
                    encounter_dict = encounter.model_dump(exclude_none=False)
                    valid_records.append(encounter_dict)
                elif csv_type == 'observations':
                    # Create ClinicalObservation
                    observation = ClinicalObservation(**{k: v for k, v in row_dict.items() if k in ClinicalObservation.model_fields})
                    # Store validated record as dict for DataFrame reconstruction
                    observation_dict = observation.model_dump(exclude_none=False)
                    valid_records.append(observation_dict)
                else:  # patients
                    # Create PatientRecord
                    patient = PatientRecord(**{k: v for k, v in row_dict.items() if k in PatientRecord.model_fields})
                    # Store validated record as dict for DataFrame reconstruction
                    patient_dict = patient.model_dump(exclude_none=False)
                    # Add source_adapter for database persistence
                    patient_dict['source_adapter'] = self.adapter_name
                    valid_records.append(patient_dict)
                
            except (PydanticValidationError, ValidationError, TransformationError) as e:
                failed_indices.append(idx)
                # Log individual failures (but don't spam logs)
                if len(failed_indices) <= 10:  # Only log first 10 failures per chunk
                    logger.warning(
                        f"Record {idx} in chunk {chunk_number} from {source} failed validation: {str(e)}",
                        extra={'source': source, 'chunk': chunk_number, 'row_index': idx}
                    )
            except Exception as e:
                failed_indices.append(idx)
                logger.error(
                    f"Unexpected error validating record {idx} in chunk {chunk_number} from {source}: {str(e)}",
                    exc_info=True,
                    extra={'source': source, 'chunk': chunk_number, 'row_index': idx}
                )
        
        # Create DataFrame from valid records
        if valid_records:
            validated_df = pd.DataFrame(valid_records)
        else:
            validated_df = pd.DataFrame()
        
        return validated_df, failed_indices
    
    def _generate_hash_from_row(self, row_dict: Dict[str, Any]) -> str:
        """Generate hash from row dictionary for audit trail.
        
        Parameters:
            row_dict: Row data as dictionary
        
        Returns:
            str: SHA-256 hash
        """
        record_str = str(sorted(row_dict.items()))
        return hashlib.sha256(record_str.encode('utf-8')).hexdigest()
    
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

