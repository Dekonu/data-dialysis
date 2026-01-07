"""Domain Ports - Abstract Contracts for Data Ingestion.

This module defines the Port interfaces (abstract contracts) that Adapters must implement.
Following Hexagonal Architecture, the Domain Core defines what it needs, not how it's provided.

Security Impact:
    - Ports enforce that adapters yield validated, PII-redacted GoldenRecord objects
    - Streaming interface prevents memory exhaustion with large datasets
    - Type safety ensures only safe records enter the domain

Architecture:
    - Pure abstract interfaces with zero infrastructure dependencies
    - Adapters (JSON, XML, API, etc.) implement these ports
    - Domain Core is isolated from data source specifics
    - Iterator pattern enables memory-efficient streaming ingestion
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, AsyncIterator, Generic, TypeVar, Union
from pathlib import Path
from dataclasses import dataclass

from src.domain.golden_record import GoldenRecord

# Type variable for Result generic
T = TypeVar('T')

import pandas as pd


# ============================================================================
# Result Type for Success/Failure Communication
# ============================================================================

@dataclass(frozen=True)
class Result(Generic[T]):
    """Result type for communicating success or failure without exceptions.
    
    This type enables CircuitBreaker and other guardrails to monitor
    failure rates without relying on exception handling. It follows the
    functional programming pattern of explicit error handling.
    
    Attributes:
        success: True if the operation succeeded, False otherwise
        value: The successful result value (only present if success=True)
        error: Error information (only present if success=False)
        error_type: Type of error (ValidationError, TransformationError, etc.)
        error_details: Additional error context (source, record_index, etc.)
    
    Example:
        ```python
        # Success case
        result = Result.success(golden_record)
        if result.success:
            process(result.value)
        
        # Failure case
        result = Result.failure(
            ValidationError("Invalid MRN"),
            error_type="ValidationError",
            error_details={"source": "data.json", "record_index": 5}
        )
        if not result.success:
            log_error(result.error, result.error_details)
        ```
    """
    
    success: bool
    value: Optional[T] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[dict] = None
    
    @classmethod
    def success_result(cls, value: T) -> 'Result[T]':
        """Create a successful result.
        
        Parameters:
            value: The successful result value
        
        Returns:
            Result: Success result with the value
        """
        return cls(
            success=True,
            value=value,
            error=None,
            error_type=None,
            error_details=None
        )
    
    @classmethod
    def failure_result(
        cls,
        error: Union[str, Exception],
        error_type: Optional[str] = None,
        error_details: Optional[dict] = None
    ) -> 'Result[T]':
        """Create a failure result.
        
        Parameters:
            error: Error message or exception
            error_type: Type of error (e.g., "ValidationError", "TransformationError")
            error_details: Additional context (source, record_index, etc.)
        
        Returns:
            Result: Failure result with error information
        """
        error_message = str(error) if isinstance(error, Exception) else error
        error_type_name = error_type or (type(error).__name__ if isinstance(error, Exception) else "UnknownError")
        
        return cls(
            success=False,
            value=None,
            error=error_message,
            error_type=error_type_name,
            error_details=error_details or {}
        )
    
    def is_success(self) -> bool:
        """Check if result is successful."""
        return self.success
    
    def is_failure(self) -> bool:
        """Check if result is a failure."""
        return not self.success


# ============================================================================
# Custom Exception Hierarchy
# ============================================================================

class IngestionError(Exception):
    """Base exception for all ingestion-related errors.
    
    This exception should be raised when ingestion fails due to
    adapter-specific issues (file format, network, etc.).
    """
    pass


class ValidationError(IngestionError):
    """Raised when data fails validation or transformation.
    
    This exception indicates that the data cannot be transformed
    into a valid GoldenRecord (schema mismatch, missing fields, etc.).
    
    Attributes:
        source: The source identifier that failed validation
        details: Additional error details or validation messages
    """
    
    def __init__(self, message: str, source: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.source = source
        self.details = details or {}


class TransformationError(IngestionError):
    """Raised when data transformation fails.
    
    This exception indicates that raw data cannot be transformed
    into domain models (type conversion, missing mappings, etc.).
    
    Attributes:
        source: The source identifier that failed transformation
        raw_data: The raw data that failed transformation (may be truncated)
    """
    
    def __init__(self, message: str, source: Optional[str] = None, raw_data: Optional[dict] = None):
        super().__init__(message)
        self.source = source
        self.raw_data = raw_data


class SourceNotFoundError(IngestionError):
    """Raised when the source cannot be found or accessed.
    
    This exception indicates that the source file, URL, or connection
    cannot be located or accessed (file not found, network error, etc.).
    
    Attributes:
        source: The source identifier that was not found
    """
    
    def __init__(self, message: str, source: Optional[str] = None):
        super().__init__(message)
        self.source = source


class UnsupportedSourceError(IngestionError):
    """Raised when the source format is not supported by the adapter.
    
    This exception indicates that the adapter cannot handle the
    given source format or structure.
    
    Attributes:
        source: The source identifier that is unsupported
        adapter: The adapter that cannot handle the source
    """
    
    def __init__(self, message: str, source: Optional[str] = None, adapter: Optional[str] = None):
        super().__init__(message)
        self.source = source
        self.adapter = adapter


class IngestionPort(ABC):
    """Abstract contract for data ingestion adapters.
    
    This port defines how the Domain Core wants to receive data, regardless of
    whether it comes from JSON, XML, CSV, REST API, or any other source.
    
    Key Principles:
        - Streaming: Yields records one-by-one to prevent memory exhaustion
        - Validated: All records must be GoldenRecord instances (PII-redacted, validated)
        - Source-agnostic: Domain doesn't care about file format or transport
        - Fail-fast: Adapters should validate and transform before yielding
    
    Security Impact:
        - Adapters must ensure all PII is redacted before yielding GoldenRecord
        - Records must pass Pydantic validation (Safety Layer)
        - Invalid records should raise ValidationError, not be silently skipped
    
    Example Usage:
        ```python
        class JSONAdapter(IngestionPort):
            def ingest(self, source: str) -> Iterator[GoldenRecord]:
                # Parse JSON, validate, redact PII, yield GoldenRecord
                ...
        
        adapter = JSONAdapter()
        for golden_record in adapter.ingest("data.json"):
            # Process validated, redacted record
            ...
        ```
    """
    
    @abstractmethod
    def ingest(self, source: str) -> Iterator[Result[Union[GoldenRecord, 'pd.DataFrame']]]:
        """Ingest data from a source and yield Result objects containing GoldenRecord.
        
        This method is the primary entry point for data ingestion. Adapters must:
        1. Parse the source (file, URL, stream, etc.)
        2. Transform raw data to domain models (PatientRecord, ClinicalObservation, etc.)
        3. Apply PII redaction via RedactorService
        4. Validate using Pydantic (Safety Layer)
        5. Construct GoldenRecord instances
        6. Yield Result objects one-by-one (streaming)
        
        Parameters:
            source: Source identifier (file path, URL, connection string, etc.)
                   The format is adapter-specific but typically a string path or URI.
        
        Yields:
            Result[Union[GoldenRecord, pd.DataFrame]]: Result object containing either:
                - Success: Validated, PII-redacted data (GoldenRecord for row-by-row, DataFrame for batch)
                - Failure: Error information (error message, type, details)
        
        Raises:
            SourceNotFoundError: If source file/URL doesn't exist or cannot be accessed
            UnsupportedSourceError: If source format is invalid or unsupported
            IOError: If source cannot be read (permissions, network, etc.)
        
        Security Impact:
            - Must redact all PII before yielding successful data
            - Must validate schema before yielding (fail-fast)
            - Should log transformation events for audit trail
            - Failures are communicated via Result, not exceptions (enables CircuitBreaker)
        
        Memory Impact:
            - Uses Iterator pattern to stream records/chunks, preventing memory exhaustion
            - Adapters should process records incrementally, not load entire dataset
            - CSV/JSON ingesters use pandas chunked reading for vectorized processing
        
        Note:
            Individual record validation/transformation errors should be returned
            as Result.failure_result(), not raised as exceptions. This enables
            CircuitBreaker and other guardrails to monitor failure rates.
            CSV/JSON ingesters yield Result[pd.DataFrame] for batch processing.
            XML ingester yields Result[GoldenRecord] for row-by-row processing.
        """
        pass
    
    @abstractmethod
    def can_ingest(self, source: str) -> bool:
        """Check if this adapter can handle the given source.
        
        Allows the system to select the appropriate adapter for a source
        without attempting ingestion. Useful for routing logic.
        
        Parameters:
            source: Source identifier to check
        
        Returns:
            bool: True if this adapter can handle the source, False otherwise
        
        Example:
            ```python
            if json_adapter.can_ingest("data.json"):
                adapter = json_adapter
            elif xml_adapter.can_ingest("data.xml"):
                adapter = xml_adapter
            ```
        """
        pass
    
    def get_source_info(self, source: str) -> Optional[dict]:
        """Get metadata about the source (optional, adapter-specific).
        
        Provides information about the source without ingesting it.
        Useful for logging, validation, or user feedback.
        
        Parameters:
            source: Source identifier
        
        Returns:
            Optional[dict]: Metadata dictionary with keys like:
                - 'format': File format (json, xml, csv, etc.)
                - 'size': Source size in bytes (if applicable)
                - 'record_count': Estimated number of records (if available)
                - 'encoding': Character encoding (if applicable)
                - 'schema_version': Schema version (if applicable)
            Returns None if metadata cannot be determined.
        
        Note:
            This is a default implementation that returns None.
            Adapters can override to provide source-specific metadata.
        """
        return None


# ============================================================================
# Async Ports
# ============================================================================

class AsyncIngestionPort(ABC):
    """Abstract contract for asynchronous data ingestion adapters.
    
    This port defines how the Domain Core wants to receive data asynchronously,
    enabling non-blocking I/O operations for better performance with large
    datasets or network-based sources.
    
    Key Principles:
        - Async streaming: Yields records asynchronously to prevent blocking
        - Validated: All records must be GoldenRecord instances (PII-redacted, validated)
        - Source-agnostic: Domain doesn't care about file format or transport
        - Fail-fast: Adapters should validate and transform before yielding
    
    Security Impact:
        - Adapters must ensure all PII is redacted before yielding GoldenRecord
        - Records must pass Pydantic validation (Safety Layer)
        - Invalid records should raise ValidationError, not be silently skipped
    
    Example Usage:
        ```python
        class AsyncJSONAdapter(AsyncIngestionPort):
            async def ingest(self, source: str) -> AsyncIterator[GoldenRecord]:
                # Parse JSON asynchronously, validate, redact PII, yield GoldenRecord
                ...
        
        adapter = AsyncJSONAdapter()
        async for golden_record in adapter.ingest("data.json"):
            # Process validated, redacted record
            ...
        ```
    """
    
    @abstractmethod
    async def ingest(self, source: str) -> AsyncIterator[Result[Union[GoldenRecord, 'pd.DataFrame']]]:
        """Ingest data from a source asynchronously and yield validated GoldenRecord objects.
        
        This method is the primary entry point for async data ingestion. Adapters must:
        1. Parse the source asynchronously (file, URL, stream, etc.)
        2. Transform raw data to domain models (PatientRecord, ClinicalObservation, etc.)
        3. Apply PII redaction via RedactorService
        4. Validate using Pydantic (Safety Layer)
        5. Construct GoldenRecord instances
        6. Yield records one-by-one asynchronously (streaming)
        
        Parameters:
            source: Source identifier (file path, URL, connection string, etc.)
                   The format is adapter-specific but typically a string path or URI.
        
        Yields:
            Result[Union[GoldenRecord, pd.DataFrame]]: Result object containing either:
                - Success: Validated, PII-redacted data (GoldenRecord for row-by-row, DataFrame for batch)
                - Failure: Error information (error message, type, details)
        
        Raises:
            SourceNotFoundError: If source file/URL doesn't exist or cannot be accessed
            ValidationError: If data cannot be validated or transformed
            TransformationError: If data transformation fails
            UnsupportedSourceError: If source format is invalid or unsupported
            IOError: If source cannot be read (permissions, network, etc.)
        
        Security Impact:
            - Must redact all PII before yielding successful data
            - Must validate schema before yielding (fail-fast)
            - Should log transformation events for audit trail
        
        Memory Impact:
            - Uses AsyncIterator pattern to stream records/chunks, preventing memory exhaustion
            - Adapters should process records incrementally, not load entire dataset
            - Non-blocking I/O enables better resource utilization
            - CSV/JSON ingesters use pandas chunked reading for vectorized processing
        """
        pass
    
    @abstractmethod
    async def can_ingest(self, source: str) -> bool:
        """Check asynchronously if this adapter can handle the given source.
        
        Allows the system to select the appropriate adapter for a source
        without attempting ingestion. Useful for routing logic.
        
        Parameters:
            source: Source identifier to check
        
        Returns:
            bool: True if this adapter can handle the source, False otherwise
        
        Example:
            ```python
            if await json_adapter.can_ingest("data.json"):
                adapter = json_adapter
            elif await xml_adapter.can_ingest("data.xml"):
                adapter = xml_adapter
            ```
        """
        pass
    
    async def get_source_info(self, source: str) -> Optional[dict]:
        """Get metadata about the source asynchronously (optional, adapter-specific).
        
        Provides information about the source without ingesting it.
        Useful for logging, validation, or user feedback.
        
        Parameters:
            source: Source identifier
        
        Returns:
            Optional[dict]: Metadata dictionary with keys like:
                - 'format': File format (json, xml, csv, etc.)
                - 'size': Source size in bytes (if applicable)
                - 'record_count': Estimated number of records (if available)
                - 'encoding': Character encoding (if applicable)
                - 'schema_version': Schema version (if applicable)
            Returns None if metadata cannot be determined.
        
        Note:
            This is a default implementation that returns None.
            Adapters can override to provide source-specific metadata.
        """
        return None


# ============================================================================
# Storage Ports
# ============================================================================

class StoragePort(ABC):
    """Abstract contract for data persistence adapters.
    
    This port defines how the Domain Core wants to persist validated GoldenRecords,
    regardless of whether storage is DuckDB, PostgreSQL, S3, or any other backend.
    
    Key Principles:
        - Immutable: Records are append-only (audit trail requirement)
        - Transactional: Batch operations are atomic
        - Observable: All operations emit audit events
        - Fail-safe: Invalid records are rejected before persistence
    
    Security Impact:
        - Only validated GoldenRecord instances can be persisted
        - All persistence operations are logged for audit trail
        - Connection credentials are managed securely via configuration
    
    Example Usage:
        ```python
        class DuckDBAdapter(StoragePort):
            def persist(self, record: GoldenRecord) -> Result[str]:
                # Store record, return record_id
                ...
        
        adapter = DuckDBAdapter(config)
        result = adapter.persist(golden_record)
        if result.is_success():
            record_id = result.value
        ```
    """
    
    @abstractmethod
    def persist(self, record: GoldenRecord) -> Result[str]:
        """Persist a single GoldenRecord to storage.
        
        This method stores a validated, PII-redacted record and returns
        a unique identifier for the persisted record.
        
        Parameters:
            record: Validated GoldenRecord instance (PII already redacted)
        
        Returns:
            Result[str]: Result object containing either:
                - Success: Unique record identifier (record_id)
                - Failure: Error information (error message, type, details)
        
        Raises:
            ValidationError: If record fails final validation before persistence
            StorageError: If storage operation fails (connection, disk, etc.)
        
        Security Impact:
            - Record must be validated before persistence
            - Operation is logged for audit trail
            - Returns Result type for error handling without exceptions
        """
        pass
    
    @abstractmethod
    def persist_batch(self, records: list[GoldenRecord]) -> Result[list[str]]:
        """Persist multiple GoldenRecords in a single transaction.
        
        This method stores multiple validated records atomically. If any
        record fails, the entire batch is rolled back.
        
        Parameters:
            records: List of validated GoldenRecord instances
        
        Returns:
            Result[list[str]]: Result object containing either:
                - Success: List of unique record identifiers
                - Failure: Error information (error message, type, details)
        
        Raises:
            ValidationError: If any record fails validation
            StorageError: If storage operation fails
        
        Security Impact:
            - All records must be validated before persistence
            - Batch operation is atomic (all-or-nothing)
            - Operation is logged for audit trail
        """
        pass
    
    @abstractmethod
    def persist_dataframe(self, df: 'pd.DataFrame', table_name: str) -> Result[int]:
        """Persist a pandas DataFrame directly to a table.
        
        This method enables efficient bulk loading of validated DataFrames
        (e.g., from CSV/JSON batch ingestion) without row-by-row processing.
        
        Parameters:
            df: Validated pandas DataFrame (PII already redacted)
            table_name: Target table name (e.g., 'patients', 'observations')
        
        Returns:
            Result[int]: Result object containing either:
                - Success: Number of rows persisted
                - Failure: Error information (error message, type, details)
        
        Raises:
            ValidationError: If DataFrame schema doesn't match expected format
            StorageError: If storage operation fails
        
        Security Impact:
            - DataFrame must be validated before persistence
            - Bulk operation is logged for audit trail
        """
        pass
    
    @abstractmethod
    def log_audit_event(
        self,
        event_type: str,
        record_id: Optional[str],
        transformation_hash: Optional[str],
        details: Optional[dict] = None,
        table_name: Optional[str] = None,
        row_count: Optional[int] = None,
        source_adapter: Optional[str] = None
    ) -> Result[str]:
        """Log an audit trail event for compliance and observability.
        
        This method creates an immutable audit log entry for every transformation,
        redaction, or persistence operation. Required for HIPAA/GDPR compliance.
        
        Parameters:
            event_type: Type of event (e.g., 'REDACTION', 'SCHEMA_COERCION', 'PERSISTENCE', 'BULK_PERSISTENCE')
            record_id: Unique identifier of the affected record (if applicable)
            transformation_hash: Hash of original data for traceability
            details: Additional event metadata (source, adapter, field_changes, etc.)
            table_name: Name of the table affected (for persistence events)
            row_count: Number of rows processed (None/NULL for singular records, integer for bulk operations)
            source_adapter: Source adapter identifier (e.g., 'xml_ingester', 'csv_ingester')
        
        Returns:
            Result[str]: Result object containing either:
                - Success: Unique audit event identifier
                - Failure: Error information
        
        Security Impact:
            - Audit logs are immutable and tamper-proof
            - All PII redactions are logged for compliance
            - Enables forensic analysis of data transformations
        """
        pass
    
    @abstractmethod
    def initialize_schema(self) -> Result[None]:
        """Initialize database schema (tables, indexes, constraints).
        
        This method creates the necessary database structure for storing
        GoldenRecords and audit logs. Should be idempotent (safe to call multiple times).
        
        Returns:
            Result[None]: Result object indicating success or failure
        
        Raises:
            StorageError: If schema initialization fails
        
        Security Impact:
            - Schema enforces data integrity constraints
            - Indexes optimize query performance
            - Should be called once at application startup
        """
        pass
    
    def close(self) -> None:
        """Close storage connection and release resources.
        
        This method should be called when the storage adapter is no longer needed.
        Default implementation does nothing; adapters can override if needed.
        """
        pass


class StorageError(IngestionError):
    """Raised when storage operations fail.
    
    This exception indicates that a persistence operation could not be completed
    (connection error, disk full, constraint violation, etc.).
    
    Attributes:
        operation: The storage operation that failed (e.g., 'persist', 'persist_batch')
        details: Additional error context
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[dict] = None):
        super().__init__(message)
        self.operation = operation
        self.details = details or {}


# ============================================================================
# NER Ports
# ============================================================================

class NERPort(ABC):
    """Port interface for Named Entity Recognition.
    
    This port defines how the Domain Core wants to extract person names from
    unstructured text, following Hexagonal Architecture principles. The domain
    defines the interface, and infrastructure adapters (SpaCy, etc.) provide
    the implementation.
    
    Security Impact:
        - Enables accurate PII detection in unstructured clinical notes
        - Allows swapping NER implementations without changing domain logic
        - Supports graceful fallback to regex if NER unavailable
    
    Example Usage:
        ```python
        class SpaCyNERAdapter(NERPort):
            def extract_person_names(self, text: str) -> List[Tuple[str, int, int]]:
                # Use SpaCy to extract names
                ...
        
        adapter = SpaCyNERAdapter()
        names = adapter.extract_person_names("Patient John Smith visited.")
        # Returns: [("John Smith", 8, 18)]
        ```
    """
    
    @abstractmethod
    def extract_person_names(self, text: str) -> list[tuple[str, int, int]]:
        """Extract person names from unstructured text.
        
        Parameters:
            text: Input text to analyze (e.g., clinical notes, narrative text)
        
        Returns:
            List of (name, start_pos, end_pos) tuples where:
                - name: The extracted person name string
                - start_pos: Character position where name starts in text
                - end_pos: Character position where name ends in text
            Returns empty list if no names found or if NER is unavailable.
        
        Security Impact:
            - Identifies person names that may be PII in unstructured text
            - Position information enables precise redaction
            - Should handle errors gracefully (return empty list on failure)
        
        Example:
            ```python
            names = adapter.extract_person_names("Patient John Smith visited Dr. Jane Doe.")
            # Returns: [("John Smith", 8, 18), ("Jane Doe", 30, 38)]
            ```
        """
        pass
    
    def extract_person_names_batch(self, texts: list[str]) -> list[list[tuple[str, int, int]]]:
        """Extract person names from multiple texts in batch (optional optimization).
        
        This method provides batch processing for better performance when processing
        many texts. Adapters can override this for optimized batch processing, or
        fall back to calling extract_person_names() for each text.
        
        Parameters:
            texts: List of input texts to analyze
        
        Returns:
            List of results, where each result is a list of (name, start_pos, end_pos) tuples.
            The order matches the input texts list.
        
        Security Impact:
            - More efficient than calling extract_person_names() individually
            - Enables processing thousands of texts in seconds instead of minutes
            - Same security guarantees as single-text processing
        
        Example:
            ```python
            texts = ["Patient John Smith visited.", "Dr. Jane Doe examined the patient."]
            results = adapter.extract_person_names_batch(texts)
            # Returns: [[("John Smith", 8, 18)], [("Jane Doe", 0, 8)]]
            ```
        """
        # Default implementation: fall back to individual calls
        # Adapters should override this for optimized batch processing
        return [self.extract_person_names(text) for text in texts]
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if NER service is available and ready to use.
        
        Returns:
            True if NER can be used, False otherwise (e.g., model not loaded)
        
        Security Impact:
            - Allows graceful fallback to regex-based redaction
            - Prevents errors when NER model is unavailable
        """
        pass