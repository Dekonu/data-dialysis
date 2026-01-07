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
from pathlib import Path
from typing import Optional
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
) -> tuple[int, int]:
    """Process data ingestion from source to storage.
    
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
    # Get appropriate ingestion adapter
    try:
        adapter_kwargs = {}
        if xml_config_path:
            adapter_kwargs["config_path"] = xml_config_path
        if batch_size:
            adapter_kwargs["chunk_size"] = batch_size
        
        adapter = get_adapter(source, **adapter_kwargs)
        logger.info(f"Selected adapter: {adapter.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to get adapter for source '{source}': {str(e)}")
        raise
    
    # Initialize storage schema
    logger.info("Initializing storage schema...")
    schema_result = storage.initialize_schema()
    if not schema_result.is_success():
        logger.error(f"Failed to initialize schema: {schema_result.error}")
        raise RuntimeError(f"Schema initialization failed: {schema_result.error}")
    logger.info("Schema initialized successfully")
    
    # Initialize redaction logger for this ingestion run
    redaction_logger = get_redaction_logger()
    ingestion_id = str(uuid.uuid4())
    redaction_logger.set_ingestion_id(ingestion_id)
    logger.info(f"Ingestion ID: {ingestion_id}")
    
    # Initialize asynchronous redaction log flusher (background thread)
    # This allows batch processing to continue while logs are flushed
    flush_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="redaction-flusher")
    pending_flush_futures = []  # Track pending flushes for final wait
    flush_lock = threading.Lock()  # Thread-safe access to redaction logger
    
    def flush_redaction_logs_async(storage_adapter: StoragePort, logs_to_flush: list[dict]) -> None:
        """Flush redaction logs asynchronously in background thread.
        
        Parameters:
            storage_adapter: Storage adapter instance
            logs_to_flush: List of redaction log dictionaries to flush
        """
        try:
            if hasattr(storage_adapter, 'flush_redaction_logs') and logs_to_flush:
                flush_result = storage_adapter.flush_redaction_logs(logs_to_flush)
                if flush_result.is_success():
                    logger.debug(f"Flushed {flush_result.value} redaction logs to storage (async)")
                else:
                    logger.warning(f"Failed to flush redaction logs (async): {flush_result.error}")
        except Exception as e:
            logger.error(f"Error flushing redaction logs asynchronously: {e}", exc_info=True)
    
    # Initialize circuit breaker if enabled
    circuit_breaker = None
    if settings.circuit_breaker_enabled:
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold_percent=settings.circuit_breaker_threshold * 100,  # Convert to percentage
            min_records_before_check=settings.circuit_breaker_min_requests
        )
        circuit_breaker = CircuitBreaker(circuit_breaker_config)
        logger.info("Circuit breaker enabled")
    
    # Process records
    success_count = 0
    failure_count = 0
    batch_records = []
    
    # Track chunk completion for periodic redaction log flushing
    # For JSON ingestion: chunk = patients + encounters + observations
    chunk_tables_persisted = set()
    
    # Collect DataFrames for parallel persistence (Strategy 1: Parallel Table Persistence)
    chunk_dataframes = {
        'patients': None,
        'encounters': None,
        'observations': None
    }
    
    logger.info(f"Starting ingestion from: {source}")
    
    # Set redaction context for this ingestion run
    # This context will be available to all Pydantic validators during validation
    with redaction_context(
        logger=redaction_logger,
        source_adapter=adapter.adapter_name,
        ingestion_id=ingestion_id
    ):
        try:
            for result in adapter.ingest(source):
                # Record result with circuit breaker
                if circuit_breaker:
                    circuit_breaker.record_result(result)
                    # Check if circuit breaker is open (will raise if abort_on_open=True)
                    if circuit_breaker.is_open():
                        stats = circuit_breaker.get_statistics()
                        logger.error(
                            f"Circuit breaker opened: failure rate {stats.get('failure_rate', 0):.2%} "
                            f"exceeds threshold {circuit_breaker.config.failure_threshold_percent:.1f}%"
                        )
                        # CircuitBreakerOpenError will be raised by record_result if abort_on_open=True
                
                if result.is_success():
                    # Handle DataFrame results (CSV/JSON batch processing)
                    if hasattr(result.value, 'shape'):  # pandas DataFrame
                        df = result.value
                        logger.info(f"Processing DataFrame batch: {len(df)} rows")
                        
                        # Determine table name based on columns
                        # Check for specific IDs first (observations/encounters have patient_id too)
                        if 'observation_id' in df.columns:
                            table_name = 'observations'
                        elif 'encounter_id' in df.columns:
                            table_name = 'encounters'
                        elif 'patient_id' in df.columns:
                            table_name = 'patients'
                        else:
                            logger.warning(f"Unknown DataFrame structure, skipping batch")
                            failure_count += len(df)
                            continue
                        
                        # Store DataFrame for parallel persistence
                        chunk_dataframes[table_name] = df
                        
                        # Check if we have all 3 DataFrames for a complete chunk
                        # For JSON ingestion: chunk = patients + encounters + observations
                        # Check that all values are not None (can't use all() directly on DataFrames)
                        if all(df is not None for df in chunk_dataframes.values()):
                            # Persist tables in order to respect foreign key constraints:
                            # 1. Patients first (no dependencies)
                            # 2. Encounters and observations in parallel (both depend on patients)
                            logger.info(
                                f"Persisting complete chunk: "
                                f"{len(chunk_dataframes['patients'])} patients, "
                                f"{len(chunk_dataframes['encounters'])} encounters, "
                                f"{len(chunk_dataframes['observations'])} observations"
                            )
                            
                            # Step 1: Persist patients first (required for FK constraints)
                            patients_result = storage.persist_dataframe(
                                chunk_dataframes['patients'],
                                'patients'
                            )
                            if patients_result.is_success():
                                success_count += patients_result.value
                                logger.info(
                                    f"Persisted {patients_result.value} rows to patients"
                                )
                                chunk_tables_persisted.add('patients')
                            else:
                                failure_count += len(chunk_dataframes['patients'])
                                logger.error(f"Failed to persist patients: {patients_result.error}")
                                # Skip rest of chunk if patients failed
                                chunk_dataframes = {
                                    'patients': None,
                                    'encounters': None,
                                    'observations': None
                                }
                                chunk_tables_persisted.clear()
                                continue
                            
                            # Step 2: Persist encounters and observations in parallel
                            # (both depend on patients, but are independent of each other)
                            with ThreadPoolExecutor(max_workers=2) as executor:
                                futures = {
                                    executor.submit(
                                        storage.persist_dataframe,
                                        chunk_dataframes[table],
                                        table
                                    ): (chunk_dataframes[table], table)
                                    for table in ['encounters', 'observations']
                                    if chunk_dataframes[table] is not None
                                }
                                
                                # Process results as they complete
                                for future in as_completed(futures):
                                    df_persisted, table_name_persisted = futures[future]
                                    try:
                                        persist_result = future.result()
                                        if persist_result.is_success():
                                            success_count += persist_result.value
                                            logger.info(
                                                f"Persisted {persist_result.value} rows to {table_name_persisted} "
                                                f"(parallel)"
                                            )
                                            chunk_tables_persisted.add(table_name_persisted)
                                        else:
                                            failure_count += len(df_persisted)
                                            logger.error(
                                                f"Failed to persist {table_name_persisted}: {persist_result.error}"
                                            )
                                    except Exception as e:
                                        logger.error(
                                            f"Exception persisting {table_name_persisted}: {e}",
                                            exc_info=True
                                        )
                                        failure_count += len(df_persisted)
                            
                            # Flush redaction logs asynchronously after completing a full chunk
                            # This doesn't block chunk processing - flush happens in background
                            if hasattr(storage, 'flush_redaction_logs'):
                                with flush_lock:
                                    redaction_logs = redaction_logger.get_logs()
                                    if redaction_logs:
                                        # Copy logs for async flush (thread-safe)
                                        logs_copy = redaction_logs.copy()
                                        # Clear logs immediately to avoid duplicates
                                        redaction_logger.clear_logs()
                                        
                                        # Submit async flush task
                                        future = flush_executor.submit(
                                            flush_redaction_logs_async,
                                            storage,
                                            logs_copy
                                        )
                                        pending_flush_futures.append(future)
                            
                            # Reset chunk tracking for next chunk
                            chunk_dataframes = {
                                'patients': None,
                                'encounters': None,
                                'observations': None
                            }
                            chunk_tables_persisted.clear()
                        else:
                            # Not a complete chunk yet, wait for more DataFrames
                            logger.debug(
                                f"Collected {table_name} DataFrame, waiting for complete chunk. "
                                f"Have: {[k for k, v in chunk_dataframes.items() if v is not None]}"
                            )
                    
                    # Handle GoldenRecord results (XML row-by-row processing)
                    else:
                        golden_record = result.value
                        batch_records.append(golden_record)
                        
                        # Persist in batches for efficiency
                        if len(batch_records) >= (batch_size or settings.batch_size):
                            persist_result = storage.persist_batch(batch_records)
                            if persist_result.is_success():
                                success_count += len(batch_records)
                                logger.info(f"Persisted batch of {len(batch_records)} records")
                                
                                # Flush redaction logs asynchronously for XML ingestion
                                # This doesn't block batch processing - flush happens in background
                                if hasattr(storage, 'flush_redaction_logs'):
                                    with flush_lock:
                                        redaction_logs = redaction_logger.get_logs()
                                        if redaction_logs:
                                            # Copy logs for async flush (thread-safe)
                                            logs_copy = redaction_logs.copy()
                                            # Clear logs immediately to avoid duplicates
                                            redaction_logger.clear_logs()
                                            
                                            # Submit async flush task
                                            future = flush_executor.submit(
                                                flush_redaction_logs_async,
                                                storage,
                                                logs_copy
                                            )
                                            pending_flush_futures.append(future)
                            else:
                                failure_count += len(batch_records)
                                logger.error(f"Failed to persist batch: {persist_result.error}")
                            batch_records.clear()
                else:
                    failure_count += 1
                    logger.warning(
                        f"Ingestion failure: {result.error_type} - {result.error}",
                        extra=result.error_details
                    )
                    # Note: Circuit breaker is already updated at line 141 via record_result(result)
            
            # Persist remaining records (XML ingestion)
            if batch_records:
                persist_result = storage.persist_batch(batch_records)
                if persist_result.is_success():
                    success_count += len(batch_records)
                    logger.info(f"Persisted final batch of {len(batch_records)} records")
                else:
                    failure_count += len(batch_records)
                    logger.error(f"Failed to persist final batch: {persist_result.error}")
            
            # Persist any remaining DataFrames (JSON ingestion - incomplete chunk)
            remaining_tables = [table for table, df in chunk_dataframes.items() if df is not None]
            if remaining_tables:
                logger.info(
                    f"Persisting remaining incomplete chunk: {', '.join(remaining_tables)}"
                )
                for table_name in remaining_tables:
                    df = chunk_dataframes[table_name]
                    persist_result = storage.persist_dataframe(df, table_name)
                    if persist_result.is_success():
                        success_count += persist_result.value
                        logger.info(f"Persisted {persist_result.value} rows to {table_name}")
                    else:
                        failure_count += len(df)
                        logger.error(f"Failed to persist {table_name}: {persist_result.error}")
        
        except KeyboardInterrupt:
            logger.warning("Ingestion interrupted by user")

            # Try to persist any remaining records
            if batch_records:
                logger.info(f"Attempting to persist {len(batch_records)} remaining records...")
                persist_result = storage.persist_batch(batch_records)
                if persist_result.is_success():
                    success_count += len(batch_records)
    
    # Wait for all pending async flushes to complete, then flush any remaining logs
    logger.info("Waiting for pending redaction log flushes to complete...")
    if pending_flush_futures:
        # Wait for all pending flushes (with timeout to prevent hanging)
        for future in pending_flush_futures:
            try:
                future.result(timeout=30)  # 30 second timeout per flush
            except Exception as e:
                logger.warning(f"Pending redaction log flush failed: {e}")
    
    # Flush any remaining redaction logs synchronously (final flush)
    # This ensures we don't miss any logs from the final batch
    logger.info("Flushing remaining redaction logs to storage...")
    with flush_lock:
        redaction_logs = redaction_logger.get_logs()
        if redaction_logs and hasattr(storage, 'flush_redaction_logs'):
            flush_result = storage.flush_redaction_logs(redaction_logs)
            if flush_result.is_success():
                logger.info(f"Flushed {flush_result.value} remaining redaction logs to storage")
                redaction_logger.clear_logs()
            else:
                logger.warning(f"Failed to flush remaining redaction logs: {flush_result.error}")
        else:
            if not redaction_logs:
                logger.debug("No remaining redaction logs to flush (already flushed during ingestion)")
            else:
                logger.warning("Storage adapter does not support flush_redaction_logs")
    
    # Shutdown the flush executor
    flush_executor.shutdown(wait=True)
    logger.debug("Redaction log flusher thread pool shut down")
    
    return success_count, failure_count, ingestion_id


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

