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

