"""XML Data Ingestion Adapter.

This adapter implements the IngestionPort contract for XML data sources.
It uses a hybrid approach combining lxml for performance with security protections
similar to defusedxml, supporting both streaming and traditional parsing modes.

Security Impact:
    - Uses secure streaming parser to prevent XML-based attacks
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
    - Automatic mode selection based on file size
"""

import json
import logging
import hashlib
import gc
from pathlib import Path
from typing import Iterator, Optional, Any, Dict
from datetime import datetime

from defusedxml import ElementTree as SafeET
from defusedxml.ElementTree import ParseError as SafeParseError
from pydantic import ValidationError as PydanticValidationError

# Check if lxml is available for streaming and XPath compilation
try:
    from lxml import etree
    LXML_AVAILABLE = True
except ImportError:
    LXML_AVAILABLE = False
    etree = None

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
from src.domain.field_mapping import FieldMapper

# Configure logging for security rejections
logger = logging.getLogger(__name__)


class XMLIngester(IngestionPort):
    """XML ingestion adapter with configurable XPath mapping and fail-safe error handling.
    
    This adapter processes XML files using XPath expressions to extract data into
    GoldenRecord format. It supports both streaming (for large files) and traditional
    (for small files) parsing modes with automatic selection based on file size.
    
    Key Features:
        - Automatic mode selection: Streaming for large files, traditional for small files
        - Security: Secure streaming parser with event/depth limits
        - Performance: Pre-compiled XPath expressions for faster extraction
        - Fail-safe: Each record wrapped in try/except to prevent DoS
        - PII redaction: Applied before validation via model validators
    
    Security Features:
        - Uses secure streaming parser to prevent XML-based attacks
        - Streaming parser with event/depth limits
        - Record size limits to prevent memory exhaustion
        - PII redaction before validation
    
    Configuration Format:
        {
            "root_element": "./PatientRecord",
            "fields": {
                "patient_id": "./MRN",
                "first_name": "./Demographics/FirstName",
                "last_name": "./Demographics/LastName"
            }
        }
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        field_mapping_path: Optional[str] = None,
        field_mapping_dict: Optional[Dict[str, Any]] = None,
        max_record_size: int = 10 * 1024 * 1024,
        streaming_enabled: Optional[bool] = None,
        streaming_threshold: Optional[int] = None
    ):
        """Initialize XML ingester.
        
        Parameters:
            config_path: Path to JSON configuration file with XPath mappings
            config_dict: Configuration dictionary (alternative to config_path)
            field_mapping_path: Path to JSON file with field name mappings (optional)
            field_mapping_dict: Field mapping dictionary (alternative to field_mapping_path)
            max_record_size: Maximum size of a single record in bytes (default: 10MB)
            streaming_enabled: Enable streaming mode (None = auto-detect based on file size)
            streaming_threshold: File size threshold for auto-enabling streaming (bytes)
        
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
        
        # Initialize FieldMapper with default configuration
        # Default mappings are embedded in FieldMapper and can be overridden by field_mapping_path or field_mapping_dict
        try:
            self.field_mapper = FieldMapper(
                mapping_config_path=field_mapping_path,
                mapping_config_dict=field_mapping_dict
            )
        except Exception as e:
            logger.error(
                f"Error initializing FieldMapper: {str(e)}. "
                "Using default field mapping configuration."
            )
            # Fallback to default only (no override)
            self.field_mapper = FieldMapper()
        
        self.max_record_size = max_record_size
        self.adapter_name = "xml_ingester"
        self.root_xpath = self.config.get('root_element', '.')
        self.field_mappings = self.config.get('fields', {})
        
        # Pre-compile XPath expressions for performance
        self._compiled_xpaths = {}
        self._has_compiled_xpaths = False
        if LXML_AVAILABLE:
            try:
                for field_name, xpath_expr in self.field_mappings.items():
                    # Clean XPath: remove leading ./ for relative paths
                    clean_path = xpath_expr.lstrip('./')
                    self._compiled_xpaths[field_name] = etree.XPath(clean_path)
                self._has_compiled_xpaths = True
            except Exception as e:
                logger.warning(f"Could not pre-compile XPaths (falling back to string lookup): {e}")
                self._has_compiled_xpaths = False
        
        # Streaming configuration
        from src.infrastructure.settings import settings
        self.streaming_enabled = streaming_enabled
        self.streaming_threshold = streaming_threshold or settings.xml_streaming_threshold
        
        # Initialize streaming parser if streaming might be used
        self._streaming_parser = None
        if self.streaming_enabled is not False:  # True or None (auto-detect)
            try:
                from src.infrastructure.xml_streaming_parser import StreamingXMLParser
                self._streaming_parser = StreamingXMLParser(
                    max_events=settings.xml_max_events,
                    max_depth=settings.xml_max_depth,
                    huge_tree=False  # Keep False for security
                )
            except ImportError:
                logger.warning(
                    "lxml not available, falling back to non-streaming mode. "
                    "Install lxml for streaming support: pip install lxml"
                )
                self.streaming_enabled = False
    
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
    
    def _get_record_tag_from_config(self) -> Optional[str]:
        """Extract record tag name from root_element XPath.
        
        Returns:
            Optional[str]: Tag name if root_element is a simple tag path, None otherwise
        """
        if not self.root_xpath or self.root_xpath == '.':
            return None
        
        # Try to extract tag name from XPath
        # Simple case: "./PatientRecord" or "PatientRecord"
        xpath = self.root_xpath.strip()
        if xpath.startswith('./'):
            xpath = xpath[2:]
        if xpath.startswith('//'):
            xpath = xpath[2:]
        
        # If it's a simple tag name (no slashes, no predicates), use it
        if '/' not in xpath and '[' not in xpath:
            return xpath
        
        return None
    
    def _should_use_streaming(self, source: str) -> bool:
        """Determine if streaming should be used for this source.
        
        Parameters:
            source: Source file path
        
        Returns:
            bool: True if streaming should be used
        
        Logic:
            - If streaming_enabled is explicitly True: Always use streaming
            - If streaming_enabled is explicitly False: Never use streaming
            - If streaming_enabled is None (auto-detect): Check file size against threshold
        """
        # Explicitly disabled
        if self.streaming_enabled is False:
            return False
        
        # Streaming parser not available
        if not self._streaming_parser:
            return False
        
        # Explicitly enabled - use streaming regardless of file size
        if self.streaming_enabled is True:
            return True
        
        # Auto-detect mode (streaming_enabled is None) - check file size
        try:
            source_path = Path(source)
            if source_path.exists():
                file_size = source_path.stat().st_size
                return file_size >= self.streaming_threshold
        except (OSError, ValueError):
            pass
        
        return False
    
    def ingest(self, source: str) -> Iterator[Result[GoldenRecord]]:
        """Ingest XML data and yield Result objects containing GoldenRecord.
        
        This method automatically selects streaming or non-streaming mode based on:
        - File size (streaming for large files)
        - Configuration settings
        - Availability of lxml library
        
        Parameters:
            source: Path to XML file
        
        Yields:
            Result[GoldenRecord]: Result object containing either:
                - Success: Validated, PII-redacted golden record
                - Failure: Error information (error message, type, details)
        
        Raises:
            SourceNotFoundError: If source file doesn't exist
            UnsupportedSourceError: If source is not valid XML
        """
        # Check if streaming should be used
        if self._should_use_streaming(source):
            logger.info(f"Using streaming mode for XML file: {source}")
            yield from self._ingest_streaming(source)
        else:
            yield from self._ingest_traditional(source)
    
    def _ingest_streaming(self, source: str) -> Iterator[Result[GoldenRecord]]:
        """Ingest XML using streaming parser (memory efficient for large files).
        
        Parameters:
            source: Path to XML file
        
        Yields:
            Result[GoldenRecord]: Processed records
        """
        source_path = Path(source)
        if not source_path.exists():
            raise SourceNotFoundError(
                f"XML source not found: {source}",
                source=source
            )
        
        # Get record tag or XPath from config
        record_tag = self._get_record_tag_from_config()
        record_xpath = self.root_xpath if record_tag is None else None
        
        # For very large files, create parser with adjusted limits
        file_size = source_path.stat().st_size
        huge_tree_threshold = 50 * 1024 * 1024  # 50MB
        
        # Use appropriate parser based on file size
        if file_size > huge_tree_threshold and self._streaming_parser:
            from src.infrastructure.xml_streaming_parser import StreamingXMLParser
            from src.infrastructure.settings import settings
            
            # Estimate events: ~30 events per record, ~900 bytes per record
            estimated_records = file_size // 900
            estimated_events = estimated_records * 30
            # Add 50% buffer and round up to nearest million
            safe_max_events = int((estimated_events * 1.5) // 1_000_000 + 1) * 1_000_000
            # Cap at 10M events for safety
            safe_max_events = min(safe_max_events, 10_000_000)
            
            logger.info(
                f"Large file detected ({file_size / (1024*1024):.2f} MB). "
                f"Using parser with huge_tree=True and max_events={safe_max_events:,}"
            )
            
            large_file_parser = StreamingXMLParser(
                max_events=safe_max_events,
                max_depth=settings.xml_max_depth,
                huge_tree=True  # Enable for large files
            )
            parser_to_use = large_file_parser
        else:
            parser_to_use = self._streaming_parser
        
        # Process records using streaming parser
        record_count = 0
        rejected_count = 0
        
        try:
            with parser_to_use.parse(source, record_tag=record_tag, record_xpath=record_xpath) as records:
                root_element = None
                last_root_cleanup = 0
                
                for xml_record in records:
                    record_count += 1
                    
                    # Get root element reference on first iteration
                    if root_element is None:
                        try:
                            root_element = xml_record.getroottree().getroot()
                        except Exception:
                            root_element = None
                    
                    # Triage: Wrap each record in try/except to prevent DoS
                    try:
                        # Extract data FIRST, then clear element IMMEDIATELY
                        record_data = self._extract_record_data_streaming(xml_record, record_count)
                        
                        # Clear and detach element IMMEDIATELY after extraction
                        xml_record.clear(keep_tail=False)
                        parent = xml_record.getparent()
                        if parent is not None:
                            parent.remove(xml_record)
                        
                        # Root cleanup: Clean root periodically
                        if root_element is not None:
                            cleanup_interval = 50 if file_size > 75 * 1024 * 1024 else (100 if file_size > 50 * 1024 * 1024 else 200)
                            if record_count - last_root_cleanup >= cleanup_interval:
                                try:
                                    max_keep = 1 if file_size > 75 * 1024 * 1024 else 2
                                    while len(root_element) > max_keep:
                                        root_element.remove(root_element[0])
                                    last_root_cleanup = record_count
                                except Exception:
                                    pass
                        
                        # Check record size
                        estimated_size = sum(len(str(v).encode('utf-8')) for v in record_data.values()) + len(record_data) * 10
                        if estimated_size > self.max_record_size:
                            error = TransformationError(
                                f"Record {record_count} exceeds maximum size ({self.max_record_size} bytes)",
                                source=source,
                                raw_data={"estimated_size": estimated_size, "record_index": record_count}
                            )
                            rejected_count += 1
                            self._log_security_rejection(
                                source=source,
                                record_index=record_count,
                                error=error,
                                raw_record={"record_index": record_count, "estimated_size": estimated_size}
                            )
                            yield Result.failure_result(
                                error,
                                error_type="TransformationError",
                                error_details={"record_index": record_count, "reason": "record_too_large"}
                            )
                            continue
                        
                        # Transform and validate
                        # Store original record_data before transformation (for raw vault)
                        original_record_data = record_data.copy()
                        
                        golden_record = self._triage_and_transform(record_data, source, record_count)
                        
                        # Return tuple (GoldenRecord, original_record_data) for raw vault support
                        # main.py will handle converting GoldenRecord to DataFrame and using original_record_data as raw_df
                        yield Result.success_result((golden_record, original_record_data))
                        
                        # Periodic garbage collection for very large files
                        if file_size > 75 * 1024 * 1024 and record_count % 500 == 0:
                            gc.collect()
                        
                    except Exception as e:
                        # Triage: Log and continue (fail-safe)
                        rejected_count += 1
                        
                        # Clear element even on error
                        try:
                            xml_record.clear(keep_tail=False)
                            parent = xml_record.getparent()
                            if parent is not None:
                                parent.remove(xml_record)
                        except Exception:
                            pass
                        
                        error = TransformationError(
                            f"Error processing record {record_count}: {str(e)}",
                            source=source,
                            raw_data={"record_index": record_count, "error": str(e)}
                        )
                        self._log_security_rejection(
                            source=source,
                            record_index=record_count,
                            error=error,
                            raw_record={"record_index": record_count, "error": str(e)}
                        )
                        yield Result.failure_result(
                            error,
                            error_type="TransformationError",
                            error_details={"record_index": record_count, "error": str(e)}
                        )
                        continue
                        
        except Exception as e:
            # Streaming parser errors
            raise TransformationError(
                f"Streaming XML parsing failed: {str(e)}",
                source=source
            )
        
        # Log ingestion summary
        if record_count > 0:
            logger.info(
                f"XML streaming ingestion complete: {source} - "
                f"{record_count - rejected_count} accepted, {rejected_count} rejected"
            )
    
    def _ingest_traditional(self, source: str) -> Iterator[Result[GoldenRecord]]:
        """Ingest XML using traditional parsing (loads entire file into memory).
        
        Parameters:
            source: Path to XML file
        
        Yields:
            Result[GoldenRecord]: Processed records
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
                "Consider enabling streaming mode for better performance."
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
                
                # Store original record_data before transformation (for raw vault)
                original_record_data = record_data.copy()
                
                # Transform and validate record
                golden_record = self._triage_and_transform(
                    record_data,
                    source,
                    record_count
                )
                
                # Yield success result with original data for raw vault
                # Return tuple (GoldenRecord, original_record_data)
                yield Result.success_result((golden_record, original_record_data))
                
            except (ValidationError, TransformationError) as e:
                # Security rejection: Log and return failure result
                rejected_count += 1
                self._log_security_rejection(
                    source=source,
                    record_index=record_count,
                    error=e,
                    raw_record={"record_index": record_count}
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
                f"XML ingestion complete: {source} - "
                f"{record_count - rejected_count} accepted, {rejected_count} rejected"
            )
    
    def _extract_record_data_streaming(self, xml_element: Any, record_index: int) -> dict:
        """Extract data from XML element using pre-compiled XPath expressions.
        
        Parameters:
            xml_element: XML element to extract data from
            record_index: Index of record (for error messages)
        
        Returns:
            dict: Extracted data dictionary
        """
        record_data = {}
        
        if self._has_compiled_xpaths:
            # Use pre-compiled XPaths for better performance
            for field_name, compiled_xpath in self._compiled_xpaths.items():
                try:
                    results = compiled_xpath(xml_element)
                    
                    if not results:
                        continue
                    
                    # Handle different result types
                    if isinstance(results, list):
                        valid_results = []
                        for r in results:
                            if hasattr(r, 'text') and r.text:
                                valid_results.append(r.text.strip())
                            elif isinstance(r, str) and r.strip():
                                valid_results.append(r.strip())
                        
                        if valid_results:
                            record_data[field_name] = valid_results if len(valid_results) > 1 else valid_results[0]
                    else:
                        # Single result
                        if hasattr(results, 'text') and results.text:
                            record_data[field_name] = results.text.strip()
                        elif isinstance(results, str) and results.strip():
                            record_data[field_name] = results.strip()
                            
                except Exception:
                    # Silently skip failed extractions
                    continue
        else:
            # Fallback to streaming parser's XPath method or traditional extraction
            if self._streaming_parser:
                for field_name, xpath_expr in self.field_mappings.items():
                    value = self._streaming_parser.extract_with_xpath(xml_element, xpath_expr)
                    if value:
                        record_data[field_name] = value
            else:
                # Fallback to traditional extraction
                return self._extract_record_data(xml_element, record_index)
        
        return record_data
    
    def _extract_record_data(self, xml_element: Any, record_index: int) -> dict:
        """Extract data from XML element using XPath mappings.
        
        Parameters:
            xml_element: XML element to extract data from
            record_index: Index of record (for error messages)
        
        Returns:
            dict: Extracted data dictionary
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
            encounter_data = self._map_to_encounter_record(record_data, patient.patient_id)
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
        """Map XML record data to PatientRecord fields using FieldMapper.
        
        This method uses FieldMapper to transform extracted field names to
        FHIR R5-compliant field names. FieldMapper uses default mappings
        that can be overridden by configuration.
        """
        # Apply field mapping to transform extracted field names to FHIR R5 names
        mapped_data = self.field_mapper.map_patient_fields(record_data)
        
        # Handle backward compatibility: convert first_name/last_name to given_names/family_name
        # if they weren't already mapped by FieldMapper (shouldn't happen with default config)
        if 'first_name' in mapped_data and 'given_names' not in mapped_data:
            first_name = mapped_data.pop('first_name')
            if first_name:
                mapped_data['given_names'] = [first_name] if isinstance(first_name, str) else first_name
        
        if 'last_name' in mapped_data and 'family_name' not in mapped_data:
            mapped_data['family_name'] = mapped_data.pop('last_name')
        
        # Ensure given_names is a list (FHIR R5 format)
        if 'given_names' in mapped_data and not isinstance(mapped_data['given_names'], list):
            given_name = mapped_data['given_names']
            mapped_data['given_names'] = [given_name] if given_name else []
        
        return mapped_data
    
    def _map_to_encounter_record(self, record_data: dict, patient_id: str) -> Optional[dict]:
        """Map XML record data to EncounterRecord fields using FieldMapper.
        
        This method uses FieldMapper to transform extracted field names to
        FHIR R5-compliant field names, then handles special cases like enum
        conversions and date parsing.
        """
        # Apply field mapping to transform extracted field names to FHIR R5 names
        mapped_data = self.field_mapper.map_encounter_fields(record_data)
        
        encounter_data = {}
        
        # Map patient_id (required) - always set from parameter
        encounter_data['patient_id'] = patient_id
        
        # Map encounter_id (required) - generate if missing
        if 'encounter_id' in mapped_data and mapped_data['encounter_id']:
            encounter_data['encounter_id'] = str(mapped_data['encounter_id'])
        else:
            # Generate encounter_id from patient_id and date if available
            if 'period_start' in mapped_data:
                try:
                    date_value = mapped_data['period_start']
                    if isinstance(date_value, datetime):
                        date_str = date_value.strftime('%Y%m%d')
                    else:
                        date_str = str(date_value).replace('-', '').replace(' ', '').replace(':', '')[:8]
                    encounter_data['encounter_id'] = f"{patient_id}_{date_str}"
                except Exception:
                    return None
            else:
                return None
        
        # Map class_code (required) - handle enum conversion
        if 'class_code' in mapped_data and mapped_data['class_code']:
            from src.domain.enums import EncounterClass
            encounter_type = str(mapped_data['class_code']).lower().strip()
            type_mapping = {
                'inpatient': EncounterClass.INPATIENT,
                'outpatient': EncounterClass.OUTPATIENT,
                'ambulatory': EncounterClass.AMBULATORY,
                'emergency': EncounterClass.EMERGENCY,
                'virtual': EncounterClass.VIRTUAL,
            }
            if encounter_type in type_mapping:
                encounter_data['class_code'] = type_mapping[encounter_type]
            else:
                try:
                    encounter_data['class_code'] = EncounterClass(encounter_type)
                except ValueError:
                    encounter_data['class_code'] = EncounterClass.OUTPATIENT
        else:
            from src.domain.enums import EncounterClass
            encounter_data['class_code'] = EncounterClass.OUTPATIENT
        
        # Map period_start (date parsing)
        if 'period_start' in mapped_data and mapped_data['period_start']:
            try:
                date_value = mapped_data['period_start']
                if isinstance(date_value, datetime):
                    encounter_data['period_start'] = date_value
                elif isinstance(date_value, str):
                    date_str = str(date_value)
                    if 'T' in date_str or len(date_str) > 10:
                        encounter_data['period_start'] = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        from datetime import date as date_type
                        parsed_date = date_type.fromisoformat(date_str)
                        encounter_data['period_start'] = datetime.combine(parsed_date, datetime.min.time())
            except Exception:
                pass
        
        # Map status (enum conversion)
        if 'status' in mapped_data and mapped_data['status']:
            from src.domain.enums import EncounterStatus
            status_str = str(mapped_data['status']).lower().strip()
            status_mapping = {
                'planned': EncounterStatus.PLANNED,
                'in-progress': EncounterStatus.IN_PROGRESS,
                'finished': EncounterStatus.FINISHED,
                'cancelled': EncounterStatus.CANCELLED,
            }
            if status_str in status_mapping:
                encounter_data['status'] = status_mapping[status_str]
            else:
                try:
                    encounter_data['status'] = EncounterStatus(status_str)
                except ValueError:
                    pass
        
        # Map other fields (pass through if already in mapped_data)
        for field in ['service_type', 'priority', 'reason_code', 'location_address', 'participant_name']:
            if field in mapped_data and mapped_data[field]:
                encounter_data[field] = mapped_data[field]
        
        # Map diagnosis_codes (handle list conversion)
        # FieldMapper may have mapped primary_diagnosis_code to diagnosis_codes
        if 'diagnosis_codes' in mapped_data:
            codes = mapped_data['diagnosis_codes']
            encounter_data['diagnosis_codes'] = codes if isinstance(codes, list) else [codes]
        # Also check for primary_diagnosis_code in original record_data (if not mapped by FieldMapper)
        elif 'primary_diagnosis_code' in record_data and record_data['primary_diagnosis_code']:
            encounter_data['diagnosis_codes'] = [record_data['primary_diagnosis_code']]
        else:
            # No diagnosis codes found - set empty list
            encounter_data['diagnosis_codes'] = []
        
        return encounter_data
    
    def _map_to_observation_record(self, record_data: dict) -> Optional[dict]:
        """Map XML record data to ClinicalObservation fields using FieldMapper.
        
        This method uses FieldMapper to transform extracted field names to
        FHIR R5-compliant field names.
        """
        # Apply field mapping to transform extracted field names to FHIR R5 names
        mapped_data = self.field_mapper.map_observation_fields(record_data)
        
        # Return mapped data if it contains at least observation_id (required)
        if 'observation_id' in mapped_data or 'category' in mapped_data:
            return mapped_data
        
        return None
    
    def _generate_hash(self, record_data: dict) -> str:
        """Generate hash for transformation audit trail."""
        data_str = json.dumps(record_data, sort_keys=True)
        return hashlib.sha256(data_str.encode('utf-8')).hexdigest()
    
    def _log_security_rejection(
        self,
        source: str,
        record_index: int,
        error: Exception,
        raw_record: dict
    ) -> None:
        """Log a security rejection for audit trail."""
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
        """Truncate data dictionary for safe logging."""
        data_str = json.dumps(data)
        if len(data_str) <= max_size:
            return data
        
        # Truncate and add indicator
        try:
            truncated = json.loads(data_str[:max_size])
            if isinstance(truncated, dict):
                truncated['_truncated'] = True
                truncated['_original_size'] = len(data_str)
            return truncated
        except json.JSONDecodeError:
            return {"_truncated": True, "_original_size": len(data_str)}
