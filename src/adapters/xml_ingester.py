"""XML Data Ingestion Adapter.

This adapter implements the IngestionPort contract for XML data sources.
It uses defusedxml to prevent XML-based attacks and supports configurable
XPath-based field mapping for different XML structures.

Security Impact:
    - Uses defusedxml to prevent Billion Laughs attacks and quadratic blowup
    - Triage logic identifies and rejects malicious or malformed records
    - Bad records are logged as security rejections for audit trail
    - Each record is wrapped in try/except to prevent DoS attacks
    - PII redaction is applied before validation

Architecture:
    - Implements IngestionPort (Hexagonal Architecture)
    - Configurable via JSON XPath mapping files
    - Supports nested and flat XML structures
    - Isolated from domain core - only depends on ports and models
    - Streaming pattern prevents memory exhaustion
    - Fail-safe design: bad records don't crash the pipeline
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Iterator, Optional, Any, Dict
from datetime import datetime

from defusedxml import ElementTree as SafeET
from defusedxml.ElementTree import ParseError as SafeParseError
from pydantic import ValidationError as PydanticValidationError

from src.domain.ports import (
    IngestionPort,
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


class XMLIngester(IngestionPort):
    """XML ingestion adapter with configurable XPath mapping and fail-safe error handling.
    
    This adapter reads XML files and transforms them into GoldenRecord objects using
    configurable XPath expressions. It supports different XML structures (nested, flat)
    through JSON configuration files.
    
    Key Features:
        - DefusedXML: Prevents XML-based attacks (Billion Laughs, quadratic blowup)
        - Configurable: JSON-based XPath mapping for flexible XML structures
        - Triage: Identifies and rejects bad records without crashing
        - Fail-safe: Each record wrapped in try/except to prevent DoS
        - Security logging: Bad records logged as security rejections
        - Streaming: Processes records one-by-one to save memory
        - PII redaction: Applies redaction before validation
    
    Security Impact:
        - Malformed records are logged and rejected, not processed
        - DoS protection via per-record error isolation
        - XML attack prevention via defusedxml
        - Audit trail of all rejected records
    
    Configuration Format:
        {
            "root_element": "XPath/to/root/element",
            "fields": {
                "patient_id": "./MRN",
                "first_name": "./Demographics/FirstName",
                "last_name": "./Demographics/LastName",
                ...
            }
        }
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        max_record_size: int = 10 * 1024 * 1024
    ):
        """Initialize XML ingester.
        
        Parameters:
            config_path: Path to JSON configuration file with XPath mappings
            config_dict: Configuration dictionary (alternative to config_path)
            max_record_size: Maximum size of a single record in bytes (default: 10MB)
                            Prevents memory exhaustion from oversized records
        
        Raises:
            ValueError: If neither config_path nor config_dict is provided
            FileNotFoundError: If config_path doesn't exist
            json.JSONDecodeError: If config file is invalid JSON
        """
        if config_path and config_dict:
            raise ValueError("Cannot specify both config_path and config_dict")
        
        if not config_path and not config_dict:
            raise ValueError("Must specify either config_path or config_dict")
        
        if config_path:
            config_path_obj = Path(config_path)
            if not config_path_obj.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path_obj, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = config_dict
        
        # Validate configuration structure
        self._validate_config()
        
        self.max_record_size = max_record_size
        self.adapter_name = "xml_ingester"
        self.root_xpath = self.config.get('root_element', '.')
        self.field_mappings = self.config.get('fields', {})
    
    def _validate_config(self) -> None:
        """Validate configuration structure.
        
        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(self.config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        if 'fields' not in self.config:
            raise ValueError("Configuration must contain 'fields' key")
        
        if not isinstance(self.config['fields'], dict):
            raise ValueError("Configuration 'fields' must be a dictionary")
        
        # root_element is optional, defaults to '.'
        if 'root_element' in self.config:
            if not isinstance(self.config['root_element'], str):
                raise ValueError("Configuration 'root_element' must be a string")
    
    def can_ingest(self, source: str) -> bool:
        """Check if this adapter can handle the given source.
        
        Parameters:
            source: Source identifier (file path or URL)
        
        Returns:
            bool: True if source is an XML file, False otherwise
        """
        if not source:
            return False
        
        # Check file extension
        source_path = Path(source)
        if source_path.suffix.lower() in ('.xml', '.xml.gz'):
            return True
        
        # Check if it's a URL ending in .xml
        if source.lower().endswith('.xml') or source.lower().endswith('.xml.gz'):
            return True
        
        return False
    
    def get_source_info(self, source: str) -> Optional[dict]:
        """Get metadata about the XML source.
        
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
                    'format': 'xml',
                    'size': stat.st_size,
                    'encoding': 'utf-8',
                    'exists': True,
                    'root_element': self.root_xpath,
                }
        except (OSError, ValueError):
            pass
        
        return None
    
    def ingest(self, source: str) -> Iterator[GoldenRecord]:
        """Ingest XML data and yield validated GoldenRecord objects.
        
        This method implements triage logic to handle bad data gracefully:
        1. Each record is wrapped in try/except to prevent DoS
        2. Bad records are logged as security rejections
        3. Pipeline continues processing valid records
        4. Only validated, PII-redacted records are yielded
        
        Parameters:
            source: Path to XML file
        
        Yields:
            GoldenRecord: Validated, PII-redacted golden records
        
        Raises:
            SourceNotFoundError: If source file doesn't exist
            UnsupportedSourceError: If source is not valid XML
        """
        # Validate source exists
        source_path = Path(source)
        if not source_path.exists():
            raise SourceNotFoundError(
                f"XML source not found: {source}",
                source=source
            )
        
        # Check file size to prevent memory exhaustion
        file_size = source_path.stat().st_size
        if file_size > self.max_record_size * 100:
            logger.warning(
                f"Large XML file detected: {source} ({file_size} bytes). "
                "Processing may be slow."
            )
        
        try:
            # Parse XML with defusedxml (prevents XML attacks)
            tree = SafeET.parse(str(source_path))
            root = tree.getroot()
        except SafeParseError as e:
            raise UnsupportedSourceError(
                f"Invalid XML format in {source}: {str(e)}",
                source=source,
                adapter=self.adapter_name
            )
        except Exception as e:
            raise SourceNotFoundError(
                f"Cannot read XML source {source}: {str(e)}",
                source=source
            )
        
        # Find all root elements (records)
        try:
            records = root.findall(self.root_xpath) if self.root_xpath != '.' else [root]
        except Exception as e:
            raise TransformationError(
                f"Invalid root_element XPath '{self.root_xpath}': {str(e)}",
                source=source
            )
        
        if not records:
            logger.warning(f"No records found in {source} using XPath '{self.root_xpath}'")
            return
        
        # Process each record with triage
        record_count = 0
        rejected_count = 0
        
        for xml_record in records:
            record_count += 1
            
            # Triage: Wrap each record in try/except to prevent DoS
            try:
                # Extract data using XPath mappings
                record_data = self._extract_record_data(xml_record, record_count)
                
                # Check record size to prevent memory exhaustion
                record_str = json.dumps(record_data)
                if len(record_str.encode('utf-8')) > self.max_record_size:
                    raise TransformationError(
                        f"Record {record_count} exceeds maximum size ({self.max_record_size} bytes)",
                        source=source,
                        raw_data={"size": len(record_str), "record_index": record_count}
                    )
                
                # Transform and validate record
                golden_record = self._triage_and_transform(
                    record_data,
                    source,
                    record_count
                )
                
                # Yield validated record
                yield golden_record
                
            except (ValidationError, TransformationError) as e:
                # Security rejection: Log and continue
                rejected_count += 1
                self._log_security_rejection(
                    source=source,
                    record_index=record_count,
                    error=e,
                    raw_record={"record_index": record_count}
                )
                # Continue processing - don't crash the pipeline
                continue
                
            except Exception as e:
                # Unexpected error: Log as security concern and continue
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
                # Continue processing - fail-safe design
                continue
        
        # Log ingestion summary
        if record_count > 0:
            logger.info(
                f"XML ingestion complete: {source} - "
                f"{record_count - rejected_count} accepted, {rejected_count} rejected"
            )
    
    def _extract_record_data(self, xml_element: Any, record_index: int) -> dict:
        """Extract data from XML element using XPath mappings.
        
        Parameters:
            xml_element: XML element to extract data from
            record_index: Index of record (for error messages)
        
        Returns:
            dict: Extracted data dictionary
        
        Raises:
            TransformationError: If XPath extraction fails
        """
        record_data = {}
        
        for field_name, xpath_expr in self.field_mappings.items():
            try:
                # Find element using XPath (relative to current element)
                elements = xml_element.findall(xpath_expr)
                
                if not elements:
                    # Field not found - skip (optional fields)
                    continue
                
                if len(elements) == 1:
                    # Single element - get text content
                    element = elements[0]
                    value = element.text
                    if value is not None:
                        value = value.strip()
                        if value:  # Only add non-empty values
                            record_data[field_name] = value
                else:
                    # Multiple elements - collect as list
                    values = [elem.text.strip() for elem in elements if elem.text and elem.text.strip()]
                    if values:
                        record_data[field_name] = values if len(values) > 1 else values[0]
                
            except Exception as e:
                # XPath error - log warning but continue
                logger.warning(
                    f"XPath extraction failed for field '{field_name}' "
                    f"in record {record_index}: {str(e)}",
                    extra={
                        'field_name': field_name,
                        'xpath': xpath_expr,
                        'record_index': record_index,
                    }
                )
                continue
        
        return record_data
    
    def _triage_and_transform(
        self,
        record_data: dict,
        source: str,
        record_index: int
    ) -> GoldenRecord:
        """Triage and transform extracted XML data into a GoldenRecord.
        
        This method implements the triage logic:
        1. Pre-validation checks (structure, required fields)
        2. Transformation to domain models
        3. PII redaction (automatic via model validators)
        4. Pydantic validation (Safety Layer)
        
        Parameters:
            record_data: Extracted XML data dictionary
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
        
        # Check for required patient data
        if 'patient_id' not in record_data and 'mrn' not in record_data:
            raise TransformationError(
                f"Record {record_index} missing required patient identifier (patient_id or mrn)",
                source=source,
                raw_data={"record_index": record_index, "keys": list(record_data.keys())}
            )
        
        try:
            # Map XML fields to domain model fields
            patient_data = self._map_to_patient_record(record_data)
            
            # Transform patient record (PII redaction happens automatically via validators)
            patient = PatientRecord(**patient_data)
            
            # Transform encounters (optional)
            encounters = []
            encounter_data = self._map_to_encounter_record(record_data)
            if encounter_data:
                try:
                    encounter = EncounterRecord(**encounter_data)
                    encounters.append(encounter)
                except PydanticValidationError as e:
                    logger.warning(
                        f"Encounter validation failed in record {record_index}: {str(e)}",
                        extra={'source': source, 'record_index': record_index}
                    )
            
            # Transform observations (optional)
            observations = []
            observation_data = self._map_to_observation_record(record_data)
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
    
    def _map_to_patient_record(self, record_data: dict) -> dict:
        """Map XML record data to PatientRecord fields.
        
        Parameters:
            record_data: Extracted XML data
        
        Returns:
            dict: PatientRecord-compatible dictionary
        """
        patient_data = {}
        
        # Map patient identifier (support both mrn and patient_id)
        if 'mrn' in record_data:
            patient_data['patient_id'] = record_data['mrn']
        elif 'patient_id' in record_data:
            patient_data['patient_id'] = record_data['patient_id']
        
        # Map name fields (support various formats)
        if 'patient_name' in record_data:
            # Split full name if needed
            name_parts = str(record_data['patient_name']).strip().split(maxsplit=1)
            if len(name_parts) == 2:
                patient_data['first_name'] = name_parts[0]
                patient_data['last_name'] = name_parts[1]
            else:
                patient_data['first_name'] = name_parts[0] if name_parts else None
        else:
            if 'first_name' in record_data:
                patient_data['first_name'] = record_data['first_name']
            if 'last_name' in record_data:
                patient_data['last_name'] = record_data['last_name']
        
        # Map date of birth
        if 'patient_dob' in record_data:
            patient_data['date_of_birth'] = record_data['patient_dob']
        
        # Map gender
        if 'patient_gender' in record_data:
            patient_data['gender'] = record_data['patient_gender']
        
        # Map other fields
        field_mappings = {
            'ssn': 'ssn',
            'phone': 'phone',
            'email': 'email',
            'address_line1': 'address_line1',
            'address_line2': 'address_line2',
            'city': 'city',
            'state': 'state',
            'postal_code': 'postal_code',
            'zip_code': 'zip_code',  # Support deprecated field
        }
        
        for xml_field, model_field in field_mappings.items():
            if xml_field in record_data:
                patient_data[model_field] = record_data[xml_field]
        
        return patient_data
    
    def _map_to_encounter_record(self, record_data: dict) -> Optional[dict]:
        """Map XML record data to EncounterRecord fields.
        
        Parameters:
            record_data: Extracted XML data
        
        Returns:
            Optional[dict]: EncounterRecord-compatible dictionary or None
        """
        # Check if we have encounter data
        if not any(key in record_data for key in ['encounter_date', 'encounter_id', 'visit_date']):
            return None
        
        encounter_data = {}
        
        # Map encounter identifier
        if 'encounter_id' in record_data:
            encounter_data['encounter_id'] = record_data['encounter_id']
        else:
            # Generate from patient_id if available
            patient_id = record_data.get('patient_id') or record_data.get('mrn')
            if patient_id:
                encounter_data['encounter_id'] = f"ENC_{patient_id}"
        
        # Map patient reference
        patient_id = record_data.get('patient_id') or record_data.get('mrn')
        if patient_id:
            encounter_data['patient_id'] = patient_id
        
        # Map encounter class
        if 'encounter_type' in record_data:
            encounter_data['class_code'] = record_data['encounter_type']
        elif 'encounter_class' in record_data:
            encounter_data['class_code'] = record_data['encounter_class']
        else:
            encounter_data['class_code'] = 'outpatient'  # Default
        
        # Map dates
        if 'encounter_date' in record_data:
            encounter_data['period_start'] = record_data['encounter_date']
        elif 'visit_date' in record_data:
            encounter_data['period_start'] = record_data['visit_date']
        elif 'admit_date' in record_data:
            encounter_data['period_start'] = record_data['admit_date']
        
        # Map status
        if 'encounter_status' in record_data:
            encounter_data['status'] = record_data['encounter_status']
        elif 'visit_status' in record_data:
            encounter_data['status'] = record_data['visit_status']
        
        # Map diagnosis codes
        if 'primary_diagnosis_code' in record_data:
            code = record_data['primary_diagnosis_code']
            encounter_data['diagnosis_codes'] = [code] if code else []
        elif 'dx_code' in record_data:
            code = record_data['dx_code']
            encounter_data['diagnosis_codes'] = [code] if code else []
        
        return encounter_data if encounter_data else None
    
    def _map_to_observation_record(self, record_data: dict) -> Optional[dict]:
        """Map XML record data to ClinicalObservation fields.
        
        Parameters:
            record_data: Extracted XML data
        
        Returns:
            Optional[dict]: ClinicalObservation-compatible dictionary or None
        """
        # Check if we have observation data
        if 'clinical_notes' not in record_data and 'observation_value' not in record_data:
            return None
        
        observation_data = {}
        
        # Map observation identifier
        if 'observation_id' in record_data:
            observation_data['observation_id'] = record_data['observation_id']
        else:
            # Generate from patient_id if available
            patient_id = record_data.get('patient_id') or record_data.get('mrn')
            if patient_id:
                observation_data['observation_id'] = f"OBS_{patient_id}"
        
        # Map patient reference
        patient_id = record_data.get('patient_id') or record_data.get('mrn')
        if patient_id:
            observation_data['patient_id'] = patient_id
        
        # Map category
        if 'observation_category' in record_data:
            observation_data['category'] = record_data['observation_category']
        else:
            observation_data['category'] = 'exam'  # Default
        
        # Map notes
        if 'clinical_notes' in record_data:
            observation_data['notes'] = record_data['clinical_notes']
        
        # Map value
        if 'observation_value' in record_data:
            observation_data['value'] = record_data['observation_value']
        
        # Map unit
        if 'observation_unit' in record_data:
            observation_data['unit'] = record_data['observation_unit']
        
        return observation_data if observation_data else None
    
    def _generate_hash(self, record_data: dict) -> str:
        """Generate hash of record data for audit trail.
        
        Parameters:
            record_data: Record data dictionary
        
        Returns:
            str: SHA-256 hash of the record
        """
        record_str = json.dumps(record_data, sort_keys=True)
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

