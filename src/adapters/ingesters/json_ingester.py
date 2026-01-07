"""JSON Data Ingestion Adapter.

This adapter implements the IngestionPort contract for JSON data sources.
It provides robust error handling and triage to prevent pipeline failures
from bad data.

Security Impact:
    - Triage logic identifies and rejects malicious or malformed records
    - Bad records are logged as security rejections for audit trail
    - Each record is wrapped in try/except to prevent DoS attacks
    - PII redaction is applied before validation

Architecture:
    - Implements IngestionPort (Hexagonal Architecture)
    - Isolated from domain core - only depends on ports and models
    - Streaming pattern prevents memory exhaustion
    - Fail-safe design: bad records don't crash the pipeline
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Iterator, Optional, Any, Union, List, Dict
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


class JSONIngester(IngestionPort):
    """JSON ingestion adapter with triage and fail-safe error handling.
    
    This adapter reads JSON files and transforms them into GoldenRecord objects.
    It implements robust error handling to prevent bad data from crashing the pipeline.
    
    Key Features:
        - Triage: Identifies and rejects bad records without crashing
        - Fail-safe: Each record wrapped in try/except to prevent DoS
        - Security logging: Bad records logged as security rejections
        - Streaming: Processes records one-by-one to save memory
        - PII redaction: Applies redaction before validation
    
    Security Impact:
        - Malformed records are logged and rejected, not processed
        - DoS protection via per-record error isolation
        - Audit trail of all rejected records
    """
    
    def __init__(self, max_record_size: int = 10 * 1024 * 1024, chunk_size: int = 10000):
        """Initialize JSON ingester.
        
        Parameters:
            max_record_size: Maximum size of a single record in bytes (default: 10MB)
                            Prevents memory exhaustion from oversized records
            chunk_size: Number of records to process per chunk (default: 10000)
                       Larger chunks = better vectorization, but more memory usage
                       TODO: Future enhancement - calculate optimal chunk size based on available memory
        """
        self.max_record_size = max_record_size
        self.chunk_size = chunk_size
        self.adapter_name = "json_ingester"
        
        # Initialize pandarallel if needed (one-time setup)
        initialize_pandarallel_if_needed(progress_bar=False)
    
    def can_ingest(self, source: str) -> bool:
        """Check if this adapter can handle the given source.
        
        Parameters:
            source: Source identifier (file path or URL)
        
        Returns:
            bool: True if source is a JSON file, False otherwise
        """
        if not source:
            return False
        
        # Check file extension
        source_path = Path(source)
        if source_path.suffix.lower() in ('.json', '.jsonl'):
            return True
        
        # Check if it's a URL ending in .json
        if source.lower().endswith('.json') or source.lower().endswith('.jsonl'):
            return True
        
        return False
    
    def get_source_info(self, source: str) -> Optional[dict]:
        """Get metadata about the JSON source.
        
        Parameters:
            source: Source identifier
        
        Returns:
            Optional[dict]: Metadata dictionary or None if unavailable
        """
        try:
            source_path = Path(source)
            if source_path.exists():
                stat = source_path.stat()
                return {
                    'format': 'json',
                    'size': stat.st_size,
                    'encoding': 'utf-8',
                    'exists': True,
                }
        except (OSError, ValueError):
            pass
        
        return None
    
    def ingest(self, source: str) -> Iterator[Result[Union[GoldenRecord, pd.DataFrame]]]:
        """Ingest JSON data and yield Result objects containing DataFrames (chunked processing).
        
        This method uses pandas for vectorized processing:
        1. Loads JSON data and converts to DataFrame
        2. Processes in chunks (default: 10,000 records per chunk)
        3. Applies vectorized PII redaction to entire chunks
        4. Validates records per-row (required for Pydantic)
        5. Tracks failures at chunk level for CircuitBreaker
        6. Yields Result[pd.DataFrame] with validated, redacted data
        
        Parameters:
            source: Path to JSON file
        
        Yields:
            Result[pd.DataFrame]: Result object containing either:
                - Success: DataFrame with validated, PII-redacted records
                - Failure: Error information (error message, type, details)
        
        Raises:
            SourceNotFoundError: If source file doesn't exist
            UnsupportedSourceError: If source is not valid JSON
        """
        # Validate source exists
        source_path = Path(source)
        if not source_path.exists():
            raise SourceNotFoundError(
                f"JSON source not found: {source}",
                source=source
            )
        
        # Check file size to prevent memory exhaustion
        file_size = source_path.stat().st_size
        if file_size > self.max_record_size * 100:
            logger.warning(
                f"Large JSON file detected: {source} ({file_size} bytes). "
                "Processing in chunks for memory efficiency."
            )
        
        try:
            # Load JSON data
            with open(source_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            raise UnsupportedSourceError(
                f"Invalid JSON format in {source}: {str(e)}",
                source=source,
                adapter=self.adapter_name
            )
        except Exception as e:
            raise SourceNotFoundError(
                f"Cannot read JSON source {source}: {str(e)}",
                source=source
            )
        
        # Handle different JSON structures and normalize to list
        records = self._extract_records(raw_data)
        
        if not records:
            logger.warning(f"No records found in {source}")
            return
        
        # Convert to DataFrame for vectorized processing
        # Store encounters/observations by patient_id for later extraction
        self._patient_encounters = {}  # patient_id -> list of encounter dicts
        self._patient_observations = {}  # patient_id -> list of observation dicts
        
        try:
            # Normalize nested JSON structure (patient, encounters, observations)
            normalized_records = []
            for record in records:
                # Extract patient data (flatten nested structure)
                patient_data = record.get('patient', {})
                if not isinstance(patient_data, dict):
                    continue  # Skip invalid records
                
                patient_id = patient_data.get('patient_id')
                if not patient_id:
                    continue  # Skip records without patient_id
                
                # Flatten patient data to top level
                flat_record = patient_data.copy()
                
                # Add metadata
                flat_record['_source_record_index'] = len(normalized_records)
                flat_record['_has_encounters'] = bool(record.get('encounters'))
                flat_record['_has_observations'] = bool(record.get('observations'))
                
                # Store encounters and observations for this patient
                if record.get('encounters'):
                    self._patient_encounters[patient_id] = record.get('encounters', [])
                if record.get('observations'):
                    self._patient_observations[patient_id] = record.get('observations', [])
                
                normalized_records.append(flat_record)
            
            if not normalized_records:
                logger.warning(f"No valid patient records found in {source}")
                return
            
            # Convert to DataFrame
            df_all = pd.DataFrame(normalized_records)
            
        except Exception as e:
            raise TransformationError(
                f"Failed to convert JSON to DataFrame: {str(e)}",
                source=source,
                raw_data={"error": str(e)}
            )
        
        # Process DataFrame in chunks
        chunk_count = 0
        total_processed = 0
        total_rejected = 0
        
        for start_idx in range(0, len(df_all), self.chunk_size):
            chunk_count += 1
            end_idx = min(start_idx + self.chunk_size, len(df_all))
            chunk_df = df_all.iloc[start_idx:end_idx].copy()
            
            try:
                # Apply vectorized PII redaction
                redacted_df = self._redact_dataframe(chunk_df)
                
                # Validate and filter valid records
                validated_df, failed_indices, encounters_df, observations_df = self._validate_dataframe_chunk(
                    redacted_df,
                    source,
                    chunk_count
                )
                
                # Track failures
                num_failed = len(failed_indices)
                num_valid = len(validated_df)
                total_processed += len(chunk_df)
                total_rejected += num_failed
                
                # Log failures if any
                if num_failed > 0:
                    logger.warning(
                        f"Chunk {chunk_count} from {source}: {num_failed} records failed validation, "
                        f"{num_valid} records passed"
                    )
                
                # Yield success result with validated patient DataFrame
                if len(validated_df) > 0:
                    yield Result.success_result(validated_df)
                
                # Yield encounters DataFrame if present
                if len(encounters_df) > 0:
                    yield Result.success_result(encounters_df)
                
                # Yield observations DataFrame if present
                if len(observations_df) > 0:
                    logger.info(f"Yielding observations DataFrame with {len(observations_df)} rows for chunk {chunk_count}")
                    yield Result.success_result(observations_df)
                else:
                    # Log if observations were expected but not found
                    total_obs_in_dict = sum(len(obs_list) for obs_list in self._patient_observations.values())
                    if total_obs_in_dict > 0:
                        logger.warning(
                            f"Chunk {chunk_count}: Expected observations but DataFrame is empty. "
                            f"Total observations in dictionary: {total_obs_in_dict}, "
                            f"Patients in chunk: {len(chunk_df)}"
                        )
                
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
                f"JSON ingestion complete: {source} - "
                f"{total_processed - total_rejected} accepted, {total_rejected} rejected "
                f"({chunk_count} chunks processed)"
            )
    
    def _redact_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply vectorized PII redaction to DataFrame columns.
        
        Note: Some fields (like postal_code) are not redacted here because
        they need to pass pattern validation first. They will be redacted
        by the Pydantic validators during model creation.
        
        Parameters:
            df: DataFrame with patient data
        
        Returns:
            DataFrame with PII fields redacted (where safe to do so)
        """
        df_redacted = df.copy()
        
        # Apply vectorized redaction to PII columns that don't have pattern constraints
        if 'ssn' in df_redacted.columns:
            df_redacted['ssn'] = RedactorService.redact_ssn(df_redacted['ssn'])
        
        if 'first_name' in df_redacted.columns:
            df_redacted['first_name'] = RedactorService.redact_name(df_redacted['first_name'])
        
        if 'last_name' in df_redacted.columns:
            df_redacted['last_name'] = RedactorService.redact_name(df_redacted['last_name'])
        
        if 'phone' in df_redacted.columns:
            df_redacted['phone'] = RedactorService.redact_phone(df_redacted['phone'])
        
        if 'email' in df_redacted.columns:
            df_redacted['email'] = RedactorService.redact_email(df_redacted['email'])
        
        if 'address_line1' in df_redacted.columns:
            df_redacted['address_line1'] = RedactorService.redact_address(df_redacted['address_line1'])
        
        # Note: postal_code is NOT redacted here because it needs to pass pattern validation first
        # It will be redacted by the Pydantic validator after validation
        
        # Date of birth is always redacted (set to None)
        if 'date_of_birth' in df_redacted.columns:
            df_redacted['date_of_birth'] = None
        
        return df_redacted
    
    def _validate_dataframe_chunk(
        self,
        df: pd.DataFrame,
        source: str,
        chunk_number: int
    ) -> tuple[pd.DataFrame, List[int]]:
        """Validate DataFrame chunk and return valid records + failed indices.
        
        Parameters:
            df: DataFrame with redacted data
            source: Source identifier
            chunk_number: Chunk number for logging
        
        Returns:
            Tuple of (validated DataFrame, list of failed row indices)
        """
        valid_records = []
        failed_indices = []
        encounters_records = []  # Store encounters for this chunk
        observations_records = []  # Store observations for this chunk
        
        # Validate each row individually (required for Pydantic)
        for idx, row in df.iterrows():
            try:
                # Convert row to dictionary
                row_dict = row.to_dict()
                
                # Clean up NaN values and convert types
                for key, value in row_dict.items():
                    # Skip list/array values (like identifiers) - pd.isna() doesn't work on them
                    if isinstance(value, (list, tuple)):
                        continue
                    
                    if pd.isna(value):
                        row_dict[key] = None
                    elif isinstance(value, float) and pd.isna(value):
                        row_dict[key] = None
                    elif key == 'postal_code' and pd.notna(value):
                        # Convert postal_code to string
                        row_dict[key] = str(value)
                    elif isinstance(value, float) and not pd.isna(value) and key in ['state', 'city']:
                        # Convert float to string for string fields
                        row_dict[key] = str(int(value)) if value == int(value) else str(value)
                
                # Check for required patient_id
                if 'patient_id' not in row_dict or pd.isna(row_dict.get('patient_id')) or not row_dict.get('patient_id'):
                    failed_indices.append(idx)
                    continue
                
                # Set record_id in context for this row (if available)
                patient_id = row_dict.get('patient_id')
                if patient_id:
                    try:
                        from src.infrastructure.redaction_context import set_redaction_context, get_redaction_context
                        context = get_redaction_context()
                        if context:
                            set_redaction_context(
                                logger=context.get('logger'),
                                record_id=str(patient_id),
                                source_adapter=context.get('source_adapter'),
                                ingestion_id=context.get('ingestion_id')
                            )
                    except (ImportError, AttributeError):
                        pass  # Context not available - skip
                
                # Create PatientRecord (validates and applies additional redaction via validators)
                patient = PatientRecord(**{k: v for k, v in row_dict.items() if k in PatientRecord.model_fields})
                
                # Extract encounters and observations for this patient
                patient_id = patient.patient_id
                encounters = []
                observations = []
                
                # Create EncounterRecord objects from stored encounters
                if patient_id in self._patient_encounters:
                    for enc_data in self._patient_encounters[patient_id]:
                        try:
                            # Map JSON fields to EncounterRecord fields
                            encounter_dict = {
                                'encounter_id': enc_data.get('encounter_id'),
                                'patient_id': patient_id,
                                'status': enc_data.get('status'),
                                'class_code': enc_data.get('class_code'),
                                'type': enc_data.get('type'),
                                'service_type': enc_data.get('service_type'),
                                'priority': enc_data.get('priority'),
                                'period_start': self._parse_datetime(enc_data.get('period_start')),
                                'period_end': self._parse_datetime(enc_data.get('period_end')),
                                'length_minutes': enc_data.get('length_minutes'),
                                'reason_code': enc_data.get('reason_code'),
                                'diagnosis_codes': enc_data.get('diagnosis_codes', []),
                                'facility_name': enc_data.get('facility_name'),
                                'location_address': enc_data.get('location_address'),
                                'participant_name': enc_data.get('participant_name'),
                                'participant_role': enc_data.get('participant_role'),
                                'service_provider': enc_data.get('service_provider'),
                            }
                            # Filter out None values for optional fields
                            encounter_dict = {k: v for k, v in encounter_dict.items() if v is not None or k in ['encounter_id', 'patient_id', 'class_code']}
                            if encounter_dict.get('encounter_id'):
                                encounter = EncounterRecord(**{k: v for k, v in encounter_dict.items() if k in EncounterRecord.model_fields})
                                encounters.append(encounter)
                        except Exception as e:
                            logger.warning(f"Failed to create EncounterRecord for patient {patient_id}: {str(e)}")
                
                # Create ClinicalObservation objects from stored observations
                if patient_id in self._patient_observations:
                    for obs_data in self._patient_observations[patient_id]:
                        try:
                            # Map JSON fields to ClinicalObservation fields
                            observation_dict = {
                                'observation_id': obs_data.get('observation_id'),
                                'patient_id': patient_id,
                                'encounter_id': obs_data.get('encounter_id'),
                                'status': obs_data.get('status'),
                                'category': obs_data.get('category'),
                                'code': obs_data.get('code'),
                                'effective_date': self._parse_datetime(obs_data.get('effective_date')),
                                'issued': self._parse_datetime(obs_data.get('issued')),
                                'performer_name': obs_data.get('performer_name'),
                                'value': obs_data.get('value'),
                                'unit': obs_data.get('unit'),
                                'interpretation': obs_data.get('interpretation'),
                                'body_site': obs_data.get('body_site'),
                                'method': obs_data.get('method'),
                                'device': obs_data.get('device'),
                                'reference_range': obs_data.get('reference_range'),
                                'notes': obs_data.get('notes'),
                            }
                            
                            # Validate required fields before proceeding
                            if not observation_dict.get('observation_id'):
                                logger.warning(f"Observation missing observation_id for patient {patient_id}, skipping")
                                continue
                            
                            if not observation_dict.get('category'):
                                logger.warning(f"Observation {observation_dict.get('observation_id')} missing category, skipping")
                                continue
                            
                            # Filter to only include fields that exist in ClinicalObservation model
                            # Keep all fields (including None) that are in the model
                            filtered_dict = {k: v for k, v in observation_dict.items() if k in ClinicalObservation.model_fields}
                            
                            # Create ClinicalObservation object
                            observation = ClinicalObservation(**filtered_dict)
                            observations.append(observation)
                        except Exception as e:
                            logger.warning(
                                f"Failed to create ClinicalObservation for patient {patient_id}, observation_id={obs_data.get('observation_id')}: {str(e)}",
                                exc_info=True
                            )
                
                # Create GoldenRecord with encounters and observations
                golden_record = GoldenRecord(
                    patient=patient,
                    encounters=encounters,
                    observations=observations,
                    source_adapter=self.adapter_name,
                    transformation_hash=self._generate_hash_from_row(row_dict)
                )
                
                # Store validated record as dict for DataFrame reconstruction
                # Include all fields from the validated patient record
                patient_dict = patient.model_dump(exclude_none=False)
                # Add GoldenRecord-level fields that are needed for database persistence
                patient_dict['source_adapter'] = golden_record.source_adapter
                patient_dict['transformation_hash'] = golden_record.transformation_hash
                valid_records.append(patient_dict)
                
                # Store encounters and observations for separate DataFrame creation
                for encounter in encounters:
                    enc_dict = encounter.model_dump(exclude_none=False)
                    enc_dict['ingestion_timestamp'] = golden_record.ingestion_timestamp
                    enc_dict['source_adapter'] = golden_record.source_adapter
                    enc_dict['transformation_hash'] = golden_record.transformation_hash
                    encounters_records.append(enc_dict)
                
                for observation in observations:
                    obs_dict = observation.model_dump(exclude_none=False)
                    obs_dict['ingestion_timestamp'] = golden_record.ingestion_timestamp
                    obs_dict['source_adapter'] = golden_record.source_adapter
                    obs_dict['transformation_hash'] = golden_record.transformation_hash
                    observations_records.append(obs_dict)
                
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
        
        # Create DataFrames for encounters and observations
        encounters_df = pd.DataFrame(encounters_records) if encounters_records else pd.DataFrame()
        observations_df = pd.DataFrame(observations_records) if observations_records else pd.DataFrame()
        
        # Batch process observation notes with NER (performance optimization)
        if not observations_df.empty and 'notes' in observations_df.columns:
            try:
                # Get NER adapter for batch processing
                from src.domain.services import RedactorService
                ner_adapter = RedactorService._get_ner_adapter()
                
                if ner_adapter and ner_adapter.is_available():
                    # Extract notes column for batch processing
                    notes_list = observations_df['notes'].fillna('').astype(str).tolist()
                    
                    # Batch redact notes with NER
                    logger.debug(
                        f"Batch processing {len(notes_list)} observation notes with NER "
                        f"for chunk {chunk_number}"
                    )
                    redacted_notes = RedactorService.redact_observation_notes_batch(
                        notes_list, 
                        ner_adapter=ner_adapter
                    )
                    
                    # Update DataFrame with batch-redacted notes
                    observations_df['notes'] = redacted_notes
                    
                    logger.debug(
                        f"Completed batch NER processing for {len(notes_list)} notes "
                        f"in chunk {chunk_number}"
                    )
                else:
                    logger.debug(
                        f"NER not available, skipping batch NER processing for chunk {chunk_number}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error in batch NER processing for chunk {chunk_number}: {e}. "
                    f"Notes were redacted with regex-only during validation.",
                    exc_info=True
                )
                # Continue with regex-only redacted notes (already done in validator)
        
        # Log DataFrame creation for debugging
        if len(observations_records) > 0:
            logger.debug(
                f"Created observations DataFrame with {len(observations_df)} rows from {len(observations_records)} records. "
                f"Columns: {list(observations_df.columns) if not observations_df.empty else 'empty'}"
            )
        elif len(observations_records) == 0 and len(self._patient_observations) > 0:
            # Log warning if we have observations in the dictionary but none in records
            total_obs = sum(len(obs_list) for obs_list in self._patient_observations.values())
            logger.warning(
                f"Chunk {chunk_number}: {total_obs} observations in dictionary but 0 in records. "
                f"Patients processed: {len(valid_records)}"
            )
        
        return validated_df, failed_indices, encounters_df, observations_df
    
    def _parse_datetime(self, value: Optional[Any]) -> Optional[datetime]:
        """Parse datetime from various formats.
        
        Parameters:
            value: String, datetime, or None
            
        Returns:
            datetime object or None
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            # Try common datetime formats
            formats = [
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
            logger.warning(f"Could not parse datetime: {value}")
        return None
    
    def _generate_hash_from_row(self, row_dict: Dict[str, Any]) -> str:
        """Generate hash from row dictionary for audit trail.
        
        Parameters:
            row_dict: Row data as dictionary
        
        Returns:
            str: SHA-256 hash
        """
        record_str = json.dumps(row_dict, sort_keys=True)
        return hashlib.sha256(record_str.encode('utf-8')).hexdigest()
    
    def _extract_records(self, raw_data: Any) -> list[dict]:
        """Extract records from various JSON structures.
        
        Handles:
        - Array of records: [{"patient": {...}, ...}, ...]
        - Single record: {"patient": {...}, ...}
        - JSONL format: Not directly supported, but could be extended
        
        Parameters:
            raw_data: Parsed JSON data
        
        Returns:
            list[dict]: List of record dictionaries
        """
        if isinstance(raw_data, list):
            return raw_data
        elif isinstance(raw_data, dict):
            # Single record
            return [raw_data]
        else:
            raise UnsupportedSourceError(
                f"Unsupported JSON structure: expected array or object, got {type(raw_data).__name__}",
                source="unknown",
                adapter=self.adapter_name
            )
    
    def _triage_and_transform(
        self,
        raw_record: dict,
        source: str,
        record_index: int
    ) -> GoldenRecord:
        """Triage and transform a raw JSON record into a GoldenRecord.
        
        This method implements the triage logic:
        1. Pre-validation checks (structure, required fields)
        2. Transformation to domain models
        3. PII redaction (automatic via model validators)
        4. Pydantic validation (Safety Layer)
        
        Parameters:
            raw_record: Raw JSON record dictionary
            source: Source identifier
            record_index: Index of record in source (for logging)
        
        Returns:
            GoldenRecord: Validated, PII-redacted golden record
        
        Raises:
            TransformationError: If transformation fails
            ValidationError: If validation fails
        """
        # Triage: Pre-validation checks
        if not isinstance(raw_record, dict):
            raise TransformationError(
                f"Record {record_index} is not a dictionary",
                source=source,
                raw_data={"record_index": record_index, "type": type(raw_record).__name__}
            )
        
        # Check for required patient data
        if 'patient' not in raw_record:
            raise TransformationError(
                f"Record {record_index} missing required 'patient' field",
                source=source,
                raw_data={"record_index": record_index, "keys": list(raw_record.keys())}
            )
        
        try:
            # Transform patient record (PII redaction happens automatically via validators)
            patient_data = raw_record.get('patient', {})
            if not isinstance(patient_data, dict):
                raise TransformationError(
                    f"Record {record_index} patient field is not a dictionary",
                    source=source,
                    raw_data={"record_index": record_index}
                )
            
            patient = PatientRecord(**patient_data)
            
            # Transform encounters (optional)
            encounters = []
            for enc_data in raw_record.get('encounters', []):
                try:
                    encounter = EncounterRecord(**enc_data)
                    encounters.append(encounter)
                except PydanticValidationError as e:
                    # Log individual encounter validation failure but continue
                    logger.warning(
                        f"Encounter validation failed in record {record_index}: {str(e)}",
                        extra={'source': source, 'record_index': record_index}
                    )
                    continue
            
            # Transform observations (optional)
            observations = []
            for obs_data in raw_record.get('observations', []):
                try:
                    observation = ClinicalObservation(**obs_data)
                    observations.append(observation)
                except PydanticValidationError as e:
                    # Log individual observation validation failure but continue
                    logger.warning(
                        f"Observation validation failed in record {record_index}: {str(e)}",
                        extra={'source': source, 'record_index': record_index}
                    )
                    continue
            
            # Generate transformation hash for audit trail
            transformation_hash = self._generate_hash(raw_record)
            
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
    
    def _generate_hash(self, raw_record: dict) -> str:
        """Generate hash of raw record for audit trail.
        
        Parameters:
            raw_record: Raw JSON record
        
        Returns:
            str: SHA-256 hash of the record
        """
        record_str = json.dumps(raw_record, sort_keys=True)
        return hashlib.sha256(record_str.encode('utf-8')).hexdigest()
    
    def _log_security_rejection(
        self,
        source: str,
        record_index: int,
        error: Exception,
        raw_record: dict
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
    
    def _truncate_for_logging(self, data: dict, max_size: int = 500) -> dict:
        """Truncate data dictionary for safe logging.
        
        Parameters:
            data: Data dictionary to truncate
            max_size: Maximum size in characters
        
        Returns:
            dict: Truncated dictionary safe for logging
        """
        data_str = json.dumps(data)
        if len(data_str) <= max_size:
            return data
        
        # Truncate and add indicator
        truncated = json.loads(data_str[:max_size])
        if isinstance(truncated, dict):
            truncated['_truncated'] = True
            truncated['_original_size'] = len(data_str)
        return truncated

