"""Main entry point for Data-Dialysis data ingestion pipeline.

This module provides the command-line interface for ingesting clinical data
from various sources (CSV, JSON, XML) and persisting it to configured storage.

Security Impact:
    - All data is validated and PII-redacted before persistence
    - Configuration is loaded securely via configuration manager
    - Audit trail is maintained for all operations

Architecture:
    - Follows Hexagonal Architecture principles
    - Adapters are selected automatically based on source format
    - Storage adapter is configured via configuration manager
    - Domain services (RedactorService) are used for PII redaction
"""

import sys
import argparse
import logging
import uuid
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
import threading

from src.infrastructure.settings import settings
from src.infrastructure.config_manager import get_database_config
from src.infrastructure.redaction_logger import get_redaction_logger, reset_redaction_logger
from src.infrastructure.redaction_context import redaction_context
from src.infrastructure.security_report import generate_security_report, print_security_report_summary
from src.adapters.ingesters import get_adapter
from src.adapters.storage import DuckDBAdapter, PostgreSQLAdapter
from src.domain.ports import Result, StoragePort
from src.domain.guardrails import CircuitBreaker, CircuitBreakerConfig

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Task timing context for modular performance tracking
class TaskTimingContext:
    """Context manager for tracking time spent in different ingestion tasks.
    
    This allows benchmarking and performance analysis by module.
    """
    def __init__(self):
        self.timings: Dict[str, float] = {
            'ingestion_time': 0.0,
            'processing_time': 0.0,
            'persistence_time': 0.0,
            'raw_vault_time': 0.0,
            'cdc_time': 0.0
        }
        self.start_times: Dict[str, float] = {}
    
    def start_task(self, task_name: str) -> None:
        """Start timing a task."""
        self.start_times[task_name] = time.time()
    
    def end_task(self, task_name: str) -> None:
        """End timing a task and accumulate the duration."""
        if task_name in self.start_times:
            duration = time.time() - self.start_times[task_name]
            if task_name in self.timings:
                self.timings[task_name] += duration
            del self.start_times[task_name]
    
    def get_timings(self) -> Dict[str, float]:
        """Get accumulated timings for all tasks."""
        return self.timings.copy()
    
    def reset(self) -> None:
        """Reset all timings."""
        for key in self.timings:
            self.timings[key] = 0.0
        self.start_times.clear()


def ingestion_task(
    adapter,
    source: str,
    timing_context: Optional[TaskTimingContext] = None
):
    """Task 1: Ingestion - Read data from source and yield batches.
    
    This task handles reading the file and preparing batches for processing.
    It's the first step in the pipeline and is I/O bound.
    
    Args:
        adapter: Ingestion adapter instance
        source: Source file path
        timing_context: Optional timing context for performance tracking
    
    Yields:
        Result objects containing batches of data
    """
    if timing_context:
        timing_context.start_task('ingestion_time')
    
    try:
        for result in adapter.ingest(source):
            yield result
    finally:
        if timing_context:
            timing_context.end_task('ingestion_time')


def processing_task(
    result: Result,
    adapter_name: str,
    timing_context: Optional[TaskTimingContext] = None
) -> Tuple[Result, Any]:
    """Task 2: Processing - Validate and redact data.
    
    This task handles validation and PII redaction. The actual processing
    happens in the adapter, so this is mainly a pass-through with timing.
    
    Args:
        result: Result from ingestion task
        adapter_name: Name of the adapter (for context)
        timing_context: Optional timing context for performance tracking
    
    Returns:
        Tuple of (result, processed_data)
    """
    if timing_context:
        timing_context.start_task('processing_time')
    
    try:
        # Processing (validation/redaction) happens in adapter.ingest()
        # This is mainly for timing separation
        return result, result.value
    finally:
        if timing_context:
            timing_context.end_task('processing_time')


def persistence_task(
    storage: StoragePort,
    df: Any,
    table_name: str,
    ingestion_id: str,
    source_adapter: str,
    enable_cdc: bool = True,
    timing_context: Optional[TaskTimingContext] = None
) -> Result[int]:
    """Task 3: Persistence - Store redacted data to main tables.
    
    This task persists the validated and redacted data to the main database tables.
    It does NOT include raw vault or CDC operations.
    
    Args:
        storage: Storage adapter instance
        df: DataFrame or GoldenRecord to persist
        table_name: Target table name
        ingestion_id: Ingestion run ID
        source_adapter: Source adapter identifier
        enable_cdc: Whether CDC is enabled (affects which method is used)
        timing_context: Optional timing context for performance tracking
    
    Returns:
        Result with number of rows persisted
    """
    if timing_context:
        timing_context.start_task('persistence_time')
    
    try:
        # Use standard persist_dataframe (no CDC, no raw vault)
        if hasattr(df, 'shape'):  # DataFrame
            if hasattr(storage, 'persist_dataframe'):
                return storage.persist_dataframe(df, table_name)
            else:
                return Result.failure_result(ValueError("Storage adapter does not support DataFrame persistence"))
        else:  # GoldenRecord
            return storage.persist(df)
    finally:
        if timing_context:
            timing_context.end_task('persistence_time')


def raw_vault_task(
    storage: StoragePort,
    raw_df: Any,
    table_name: str,
    ingestion_id: str,
    source_adapter: str,
    timing_context: Optional[TaskTimingContext] = None
) -> Result[int]:
    """Task 4: Raw Vault - Store original unredacted data (encrypted).
    
    This task persists the original unredacted data to the raw vault.
    The data is encrypted before storage.
    
    Args:
        storage: Storage adapter instance
        raw_df: Original unredacted DataFrame
        table_name: Target table name
        ingestion_id: Ingestion run ID
        source_adapter: Source adapter identifier
        timing_context: Optional timing context for performance tracking
    
    Returns:
        Result with number of rows persisted to raw vault
    """
    if timing_context:
        timing_context.start_task('raw_vault_time')
    
    try:
        if hasattr(storage, '_persist_raw_data_vault'):
            return storage._persist_raw_data_vault(
                raw_df,
                table_name,
                ingestion_id=ingestion_id,
                source_adapter=source_adapter
            )
        else:
            # Storage adapter doesn't support raw vault
            return Result.success_result(0)
    finally:
        if timing_context:
            timing_context.end_task('raw_vault_time')


def cdc_task(
    storage: StoragePort,
    df: Any,
    raw_df: Any,
    table_name: str,
    ingestion_id: str,
    source_adapter: str,
    timing_context: Optional[TaskTimingContext] = None,
    enable_cdc: bool = True,
    enable_raw_vault: bool = True,
) -> Result[int]:
    """Task 5: Change Data Capture - Detect changes and log them.
    
    This task performs change detection, updates only changed fields,
    and logs changes to the audit log. It uses persist_dataframe_smart
    when storage supports it and enable_cdc is True.
    
    Args:
        storage: Storage adapter instance
        df: Redacted DataFrame to persist
        raw_df: Original unredacted DataFrame (for change detection / raw vault)
        table_name: Target table name
        ingestion_id: Ingestion run ID
        source_adapter: Source adapter identifier
        timing_context: Optional timing context for performance tracking
        enable_cdc: Whether to use CDC/smart persist when supported
        enable_raw_vault: Whether to store raw data (encrypted) when supported
    
    Returns:
        Result with number of rows processed
    """
    if timing_context:
        timing_context.start_task('cdc_time')
    
    try:
        if enable_cdc and hasattr(storage, 'persist_dataframe_smart'):
            return storage.persist_dataframe_smart(
                df=df,
                table_name=table_name,
                raw_df=raw_df if enable_raw_vault else None,
                enable_cdc=True,
                enable_raw_vault=enable_raw_vault,
                ingestion_id=ingestion_id,
                source_adapter=source_adapter
            )
        if hasattr(storage, 'persist_dataframe'):
            return storage.persist_dataframe(df, table_name)
        return Result.failure_result(ValueError("Storage adapter does not support DataFrame persistence"))
    finally:
        if timing_context:
            timing_context.end_task('cdc_time')


class IngestionPipeline:
    """Sole entry point for the data ingestion pipeline.
    
    Orchestrates ingestion, PII redaction (via adapters/domain), validation,
    persistence, optional raw vault (encryption), CDC, and security reporting.
    All features are controlled by constructor flags; defaults enable safety
    and audit without heavy optional features (e.g. NER off by default).
    
    Architecture:
        - Single entry point: configure via flags, run process()
        - RedactorService is used by adapters/validators; pipeline configures NER
        - Circuit breaker, redaction logging, raw vault, CDC are pipeline options
        - Thread-safe for parallel processing
    
    Security Impact:
        - All data is validated and PII-redacted before persistence
        - Audit trail (redaction logs) when enable_redaction_logging=True
        - Raw vault (encryption) when enable_raw_vault=True and storage supports it
    
    Feature flags (all in __init__):
        - enable_circuit_breaker: Abort batch when failure rate exceeds threshold (default True)
        - enable_redaction_logging: Log each PII redaction for audit (default True)
        - enable_ner: Use NER for names in unstructured text (e.g. notes); default False (runtime)
        - enable_raw_vault: Store encrypted original rows when storage supports it (default True)
        - enable_cdc: Change data capture / smart updates when storage supports it (default True)
        - enable_adaptive_chunking: Adaptive CSV/JSON chunk sizes (default False)
        - generate_security_report: Generate security report after run (default True)
    
    Example Usage:
        ```python
        pipeline = IngestionPipeline(
            source="data.csv",
            storage=storage_adapter,
            xml_config_path=None,
            batch_size=10000,
            enable_ner=False,
        )
        success_count, failure_count, ingestion_id = pipeline.process()
        ```
    """
    
    def __init__(
        self,
        source: str,
        storage: StoragePort,
        xml_config_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        enable_circuit_breaker: bool = True,
        enable_adaptive_chunking: bool = False,
        enable_redaction_logging: bool = True,
        enable_ner: bool = False,
        enable_raw_vault: bool = True,
        enable_cdc: bool = True,
        generate_security_report: bool = True,
    ):
        """Initialize the ingestion pipeline.
        
        Parameters:
            source: Source file path
            storage: Storage adapter instance
            xml_config_path: Optional XML configuration file path (for XML sources)
            batch_size: Optional batch size for processing (overrides settings)
            enable_circuit_breaker: Enable circuit breaker for failure detection (default True)
            enable_adaptive_chunking: Adaptive chunk sizes for CSV/JSON (default False)
            enable_redaction_logging: Log PII redactions for audit (default True)
            enable_ner: Use NER for names in unstructured text/notes (default False; runtime impact)
            enable_raw_vault: Store encrypted raw data when storage supports it (default True)
            enable_cdc: Change data capture when storage supports it (default True)
            generate_security_report: Generate security report after process (default True)
        """
        self.source = source
        self.storage = storage
        self.xml_config_path = xml_config_path
        self.batch_size = batch_size or settings.batch_size
        self.enable_circuit_breaker = enable_circuit_breaker
        self.enable_adaptive_chunking = enable_adaptive_chunking
        self.enable_redaction_logging = enable_redaction_logging
        self.enable_ner = enable_ner
        self.enable_raw_vault = enable_raw_vault
        self.enable_cdc = enable_cdc
        self.generate_security_report = generate_security_report
        
        # Will be initialized in initialize()
        self.adapter: Optional[Any] = None
        self.redaction_logger: Optional[Any] = None
        self.ingestion_id: Optional[str] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        
        # Executors for parallel processing
        self.flush_executor: Optional[ThreadPoolExecutor] = None
        self.chunk_executor: Optional[ThreadPoolExecutor] = None
        self.pending_flush_futures: List[Future] = []
        self.pending_chunk_futures: List[Future] = []
        self.flush_lock = threading.Lock()
        
        # Counters
        self.success_count = 0
        self.failure_count = 0
    
    def initialize(self) -> None:
        """Initialize adapters, schema, and logging infrastructure.
        
        Raises:
            RuntimeError: If initialization fails
        """
        # Configure NER (used by RedactorService for unstructured text/notes)
        self._configure_ner()
        
        # Initialize ingestion adapter
        self._initialize_adapter()
        
        # Initialize storage schema
        self._initialize_schema()
        
        # Initialize redaction logging when enabled
        if self.enable_redaction_logging:
            self._initialize_redaction_logging()
        else:
            self.redaction_logger = None
            self.ingestion_id = str(uuid.uuid4())  # Still set for context/tracing
        
        # Initialize circuit breaker
        if self.enable_circuit_breaker:
            self._initialize_circuit_breaker()
        
        # Initialize executors
        self._initialize_executors()
    
    def _configure_ner(self) -> None:
        """Configure NER adapter for RedactorService (unstructured text/notes only)."""
        from src.domain.services import RedactorService
        if self.enable_ner:
            try:
                from src.infrastructure.ner.spacy_adapter import SpaCyNERAdapter
                model_name = getattr(settings, 'spacy_model', 'en_core_web_sm')
                RedactorService.set_ner_adapter(SpaCyNERAdapter(model_name=model_name))
                logger.info("NER enabled for pipeline (unstructured text/notes)")
            except Exception as e:
                logger.warning(f"NER requested but failed to load: {e}; using regex-only redaction")
                RedactorService.set_ner_adapter(None)
        else:
            RedactorService.set_ner_adapter(None)
            logger.debug("NER disabled for pipeline")
    
    def _initialize_adapter(self) -> None:
        """Initialize the ingestion adapter based on source format."""
        try:
            adapter_kwargs = {}
            if self.xml_config_path:
                adapter_kwargs["config_path"] = self.xml_config_path
            
            # Only pass chunk_size and target_total_rows for CSV/JSON adapters (XMLIngester doesn't accept them)
            source_path = Path(self.source)
            if source_path.suffix.lower() in ['.csv', '.json']:
                if self.batch_size:
                    adapter_kwargs["chunk_size"] = self.batch_size
                # Enable adaptive chunking by setting target_total_rows (0 disables it)
                if self.enable_adaptive_chunking:
                    adapter_kwargs["target_total_rows"] = 50000  # Default target
                else:
                    adapter_kwargs["target_total_rows"] = 0  # Disable adaptive chunking
            
            self.adapter = get_adapter(self.source, **adapter_kwargs)
            logger.info(f"Selected adapter: {self.adapter.__class__.__name__}")
            if source_path.suffix.lower() in ['.csv', '.json']:
                logger.info(f"Adaptive chunking: {'enabled' if self.enable_adaptive_chunking else 'disabled'}")
        except Exception as e:
            logger.error(f"Failed to get adapter for source '{self.source}': {str(e)}")
            raise
    
    def _initialize_schema(self) -> None:
        """Initialize the storage schema."""
        logger.info("Initializing storage schema...")
        schema_result = self.storage.initialize_schema()
        if not schema_result.is_success():
            logger.error(f"Failed to initialize schema: {schema_result.error}")
            raise RuntimeError(f"Schema initialization failed: {schema_result.error}")
        logger.info("Schema initialized successfully")
    
    def _initialize_redaction_logging(self) -> None:
        """Initialize redaction logger for this ingestion run."""
        self.redaction_logger = get_redaction_logger()
        self.ingestion_id = str(uuid.uuid4())
        self.redaction_logger.set_ingestion_id(self.ingestion_id)
        logger.info(f"Ingestion ID: {self.ingestion_id}")
    
    def _initialize_circuit_breaker(self) -> None:
        """Initialize circuit breaker for failure detection."""
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold_percent=settings.circuit_breaker_threshold,
            window_size=settings.circuit_breaker_window_size,
            min_records_before_check=settings.circuit_breaker_min_records,
            abort_on_open=settings.circuit_breaker_abort_on_open
        )
        self.circuit_breaker = CircuitBreaker(circuit_breaker_config)
    
    def _initialize_executors(self) -> None:
        """Initialize thread pool executors for parallel processing."""
        # Redaction log flusher (background thread)
        self.flush_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="redaction-flusher"
        )
        
        # Chunk processor (parallel workers)
        parallel_chunk_workers = getattr(settings, 'parallel_chunk_workers', 2)
        self.chunk_executor = ThreadPoolExecutor(
            max_workers=parallel_chunk_workers,
            thread_name_prefix="chunk-processor"
        )
        
        logger.info(
            f"Initialized executors: "
            f"parallel chunk processing: {parallel_chunk_workers} workers"
        )
    
    def process(self) -> tuple[int, int, str]:
        """Process the complete ingestion pipeline.
        
        Returns:
            tuple[int, int, str]: (success_count, failure_count, ingestion_id)
        
        Raises:
            RuntimeError: If processing fails
        """
        if not self.adapter:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        
        logger.info(
            f"Starting ingestion from: {self.source} "
            f"(batch size: {self.batch_size})"
        )
        
        # Set redaction context for this ingestion run (logger may be None if logging disabled)
        with redaction_context(
            logger=self.redaction_logger,
            source_adapter=self.adapter.adapter_name,
            ingestion_id=self.ingestion_id
        ):
            try:
                # Process records from adapter
                for result in ingestion_task(self.adapter, self.source):
                    # Check circuit breaker
                    if self.circuit_breaker:
                        self._check_circuit_breaker(result)
                    
                    if result.is_success():
                        # Route to appropriate handler based on data type
                        if self._is_dataframe_result(result):
                            self._process_dataframe_batch(result)
                        else:
                            self._process_golden_record(result)
                    else:
                        self.failure_count += 1
                        logger.debug(f"Record processing failed: {result.error}")
                
                # Wait for all parallel operations to complete
                self._wait_for_parallel_operations()
                
            except Exception as e:
                logger.error(f"Error during ingestion: {e}", exc_info=True)
                raise
        
        # Generate security report when enabled and storage supports it
        if self.generate_security_report:
            self._maybe_generate_security_report()
        
        return self.success_count, self.failure_count, self.ingestion_id
    
    def _check_circuit_breaker(self, result: Result) -> None:
        """Check and record result with circuit breaker."""
        try:
            self.circuit_breaker.record_result(result)
            if self.circuit_breaker.is_open():
                stats = self.circuit_breaker.get_statistics()
                logger.error(
                    f"Circuit breaker opened: failure rate {stats.get('failure_rate', 0):.2%} "
                    f"exceeds threshold {self.circuit_breaker.config.failure_threshold_percent:.1f}%"
                )
        except Exception as e:
            # Circuit breaker may raise if abort_on_open=True
            logger.error(f"Circuit breaker error: {e}")
            raise
    
    def _is_dataframe_result(self, result: Result) -> bool:
        """Check if result contains DataFrame (CSV/JSON) or GoldenRecord (XML)."""
        if isinstance(result.value, tuple):
            # Tuple format: (redacted_df, raw_df)
            return hasattr(result.value[0], 'shape')
        return hasattr(result.value, 'shape')
    
    def _process_dataframe_batch(self, result: Result) -> None:
        """Process DataFrame batch (CSV/JSON ingestion).
        
        This method handles chunked DataFrame processing, collecting complete
        chunks before persisting to maintain referential integrity.
        """
        # Extract DataFrame(s) from result
        if isinstance(result.value, tuple):
            df, raw_df = result.value
        else:
            df = result.value
            raw_df = None
        
        # Determine table name
        table_name = self._determine_table_name(df)
        if not table_name:
            logger.warning(f"Unknown DataFrame structure, skipping batch")
            self.failure_count += len(df) if hasattr(df, '__len__') else 1
            return
        
        # For now, persist immediately (simplified version)
        # Full implementation would collect chunks for parallel processing
        raw_df_for_vault = raw_df if self.enable_raw_vault else None
        persist_result = cdc_task(
            self.storage,
            df,
            raw_df_for_vault,
            table_name,
            self.ingestion_id,
            self.adapter.adapter_name,
            enable_cdc=self.enable_cdc,
            enable_raw_vault=self.enable_raw_vault,
        )
        
        if persist_result.is_success():
            self.success_count += len(df) if hasattr(df, '__len__') else 1
        else:
            self.failure_count += len(df) if hasattr(df, '__len__') else 1
            logger.error(f"Failed to persist {table_name}: {persist_result.error}")
    
    def _process_golden_record(self, result: Result) -> None:
        """Process GoldenRecord (XML row-by-row ingestion).
        
        This method handles individual GoldenRecord instances, batching them
        for efficient persistence.
        """
        # Extract GoldenRecord from result
        if isinstance(result.value, tuple):
            golden_record, _ = result.value
        else:
            golden_record = result.value
        
        # For now, persist immediately (simplified version)
        # Full implementation would batch records
        persist_result = persistence_task(
            self.storage,
            golden_record,
            'patients',  # XML records go to patients table
            self.ingestion_id,
            self.adapter.adapter_name,
            enable_cdc=False
        )
        
        if persist_result.is_success():
            self.success_count += 1
        else:
            self.failure_count += 1
            logger.error(f"Failed to persist GoldenRecord: {persist_result.error}")
    
    def _determine_table_name(self, df: Any) -> Optional[str]:
        """Determine table name from DataFrame columns."""
        if not hasattr(df, 'columns'):
            return None
        
        if 'observation_id' in df.columns:
            return 'observations'
        elif 'encounter_id' in df.columns:
            return 'encounters'
        elif 'patient_id' in df.columns:
            return 'patients'
        return None
    
    def _wait_for_parallel_operations(self) -> None:
        """Wait for all parallel operations (chunks, flushes) to complete."""
        # Wait for chunk processing
        for future in self.pending_chunk_futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error in chunk processing: {e}", exc_info=True)
        
        # Wait for redaction log flushes
        for future in self.pending_flush_futures:
            try:
                future.result()
            except Exception as e:
                logger.error(f"Error flushing redaction logs: {e}", exc_info=True)
    
    def _maybe_generate_security_report(self) -> None:
        """Generate security report when storage supports it."""
        if not hasattr(self.storage, 'generate_security_report'):
            return
        try:
            report_result = generate_security_report(
                self.storage,
                ingestion_id=self.ingestion_id,
            )
            if report_result.is_success() and settings.save_security_report:
                report_dir = Path(settings.security_report_dir)
                report_dir.mkdir(parents=True, exist_ok=True)
                output_path = str(report_dir / f"security_report_{self.ingestion_id or 'run'}.json")
                save_result = generate_security_report(
                    self.storage,
                    output_path=output_path,
                    ingestion_id=self.ingestion_id,
                )
                if not save_result.is_success():
                    logger.warning(f"Failed to save security report: {save_result.error}")
        except Exception as e:
            logger.warning(f"Failed to generate security report: {e}")

    def cleanup(self) -> None:
        """Clean up resources (executors, connections, etc.)."""
        if self.flush_executor:
            self.flush_executor.shutdown(wait=True)
        if self.chunk_executor:
            self.chunk_executor.shutdown(wait=True)
        
        # Flush remaining redaction logs when logging is enabled
        if self.enable_redaction_logging and self.redaction_logger and hasattr(self.storage, 'flush_redaction_logs'):
            try:
                redaction_logs = self.redaction_logger.get_logs()
                if redaction_logs:
                    self.storage.flush_redaction_logs(redaction_logs)
                    self.redaction_logger.clear_logs()
            except Exception as e:
                logger.warning(f"Failed to flush final redaction logs: {e}")


def create_storage_adapter() -> StoragePort:
    """Create storage adapter based on configuration.
    
    Returns:
        StoragePort: Configured storage adapter instance
    
    Raises:
        ValueError: If database type is unsupported
    """
    db_config = get_database_config()
    
    if db_config.db_type == "duckdb":
        logger.info(f"Initializing DuckDB adapter with path: {db_config.db_path or ':memory:'}")
        return DuckDBAdapter(db_config=db_config)
    elif db_config.db_type == "postgresql":
        logger.info(f"Initializing PostgreSQL adapter with host: {db_config.host}")
        return PostgreSQLAdapter(db_config=db_config)
    else:
        raise ValueError(f"Unsupported database type: {db_config.db_type}")


def process_ingestion(
    source: str,
    storage: StoragePort,
    xml_config_path: Optional[str] = None,
    batch_size: Optional[int] = None
) -> tuple[int, int, str]:
    """Process data ingestion from source to storage.
    
    This is a convenience function that wraps the IngestionPipeline class
    for backward compatibility.
    
    Parameters:
        source: Source file path
        storage: Storage adapter instance
        xml_config_path: Optional XML configuration file path (for XML sources)
        batch_size: Optional batch size for processing (overrides settings)
    
    Returns:
        tuple[int, int, str]: (success_count, failure_count, ingestion_id)
    
    Security Impact:
        - All records are validated and PII-redacted before persistence
        - Failures are logged for audit trail
    """
    pipeline = IngestionPipeline(
        source=source,
        storage=storage,
        xml_config_path=xml_config_path,
        batch_size=batch_size
    )
    
    try:
        pipeline.initialize()
        return pipeline.process()
    finally:
        pipeline.cleanup()


def main():
    """Main entry point for the Data-Dialysis ingestion pipeline."""
    parser = argparse.ArgumentParser(
        description=f"{settings.app_name} - Self-Securing Clinical Data Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ingest CSV file
  python -m src.main --input data/patients.csv
  
  # Ingest JSON file with custom batch size
  python -m src.main --input data/observations.json --batch-size 5000
  
  # Ingest XML file with configuration
  python -m src.main --input data/encounters.xml --xml-config mappings.json
  
  # Use environment variables for database configuration
  export DD_DB_TYPE=postgresql
  export DD_DB_HOST=localhost
  export DD_DB_NAME=clinical_db
  python -m src.main --input data/patients.csv
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        required=True,
        type=str,
        help="Input file path (CSV, JSON, or XML)"
    )
    
    parser.add_argument(
        "--xml-config",
        type=str,
        help="XML configuration file path (required for XML sources)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size for processing (default: {settings.batch_size})"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    # Print startup information
    logger.info(f"Starting {settings.app_name}...")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Database type: {settings.db_config.db_type}")
    if settings.db_config.db_type == "duckdb":
        logger.info(f"Database path: {settings.get_db_path()}")
    elif settings.db_config.db_type == "postgresql":
        logger.info(f"Database host: {settings.db_config.host}")
    logger.info(f"Batch size: {args.batch_size or settings.batch_size}")
    
    # Create storage adapter
    try:
        storage = create_storage_adapter()
    except Exception as e:
        logger.error(f"Failed to create storage adapter: {str(e)}")
        sys.exit(1)
    
    # Process ingestion
    try:
        success_count, failure_count, ingestion_id = process_ingestion(
            source=args.input,
            storage=storage,
            xml_config_path=args.xml_config,
            batch_size=args.batch_size
        )
        
        # Print summary
        total_count = success_count + failure_count
        logger.info("=" * 60)
        logger.info("Ingestion Summary:")
        logger.info(f"  Total processed: {total_count}")
        logger.info(f"  Successful: {success_count}")
        logger.info(f"  Failed: {failure_count}")
        if total_count > 0:
            success_rate = (success_count / total_count) * 100
            logger.info(f"  Success rate: {success_rate:.2f}%")
        logger.info(f"  Ingestion ID: {ingestion_id}")
        logger.info("=" * 60)
        
        # Generate security report
        logger.info("Generating security report...")
        if hasattr(storage, 'generate_security_report'):
            report_result = generate_security_report(
                storage=storage,
                ingestion_id=ingestion_id
            )
            
            if report_result.is_success():
                report = report_result.value
                print_security_report_summary(report)
                
                # Save report to file if enabled
                if settings.save_security_report:
                    report_file = Path(settings.security_report_dir) / f"security_report_{ingestion_id}.json"
                    report_file.parent.mkdir(exist_ok=True)
                    save_result = generate_security_report(
                        storage=storage,
                        output_path=str(report_file),
                        ingestion_id=ingestion_id
                    )
                    if save_result.is_success():
                        logger.info(f"Security report saved to: {report_file}")
                        logger.info(f"Report absolute path: {report_file.absolute()}")
                else:
                    logger.info("Security report file saving is disabled (DD_SAVE_SECURITY_REPORT=false)")
            else:
                logger.warning(f"Failed to generate security report: {report_result.error}")
        else:
            logger.warning("Storage adapter does not support security report generation")
        
        # Exit with appropriate code
        if failure_count > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        sys.exit(1)
    
    finally:
        # Clean up storage connection
        try:
            storage.close()
        except Exception as e:
            logger.warning(f"Error closing storage connection: {str(e)}")


if __name__ == "__main__":
    main()

