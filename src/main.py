"""Main entry point for Clinical-Sieve data ingestion pipeline.

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
                        
                        # Determine table name based on columns (simplified - could be improved)
                        if 'patient_id' in df.columns:
                            table_name = 'patients'
                        elif 'observation_id' in df.columns:
                            table_name = 'observations'
                        elif 'encounter_id' in df.columns:
                            table_name = 'encounters'
                        else:
                            logger.warning(f"Unknown DataFrame structure, skipping batch")
                            failure_count += len(df)
                            continue
                        
                        persist_result = storage.persist_dataframe(df, table_name)
                        if persist_result.is_success():
                            success_count += persist_result.value
                            logger.info(f"Persisted {persist_result.value} rows to {table_name}")
                        else:
                            failure_count += len(df)
                            logger.error(f"Failed to persist DataFrame: {persist_result.error}")
                    
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
                    
                    # Update circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_failure()
            
            # Persist remaining records
            if batch_records:
                persist_result = storage.persist_batch(batch_records)
                if persist_result.is_success():
                    success_count += len(batch_records)
                    logger.info(f"Persisted final batch of {len(batch_records)} records")
                else:
                    failure_count += len(batch_records)
                    logger.error(f"Failed to persist final batch: {persist_result.error}")
        
        except KeyboardInterrupt:
            logger.warning("Ingestion interrupted by user")

            # Try to persist any remaining records
            if batch_records:
                logger.info(f"Attempting to persist {len(batch_records)} remaining records...")
                persist_result = storage.persist_batch(batch_records)
                if persist_result.is_success():
                    success_count += len(batch_records)
    
    # Flush redaction logs to storage
    logger.info("Flushing redaction logs to storage...")
    redaction_logs = redaction_logger.get_logs()
    if redaction_logs and hasattr(storage, 'flush_redaction_logs'):
        flush_result = storage.flush_redaction_logs(redaction_logs)
        if flush_result.is_success():
            logger.info(f"Flushed {flush_result.value} redaction logs to storage")
        else:
            logger.warning(f"Failed to flush redaction logs: {flush_result.error}")
    else:
        if not redaction_logs:
            logger.info("No redaction logs to flush (redactions may have occurred in Pydantic validators)")
        else:
            logger.warning("Storage adapter does not support flush_redaction_logs")
    
    return success_count, failure_count, ingestion_id


def main():
    """Main entry point for the Clinical-Sieve ingestion pipeline."""
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

