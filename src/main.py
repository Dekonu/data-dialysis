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
        'observations': None,
        'patients_raw': None,
        'encounters_raw': None,
        'observations_raw': None
    }
    
    # Phase 3: Parallel Chunk Processing
    # Determine max parallel workers based on connection pool size
    # PostgreSQL default: pool_size=5, max_overflow=10 → max_connections=15
    # Use 2-4 workers to avoid exhausting connection pool
    # Each worker needs connections for: patients (1) + encounters/observations (2) = 3 connections
    # So max_workers = min(4, (max_connections - 2) // 3) to leave headroom
    try:
        # Try to get pool size from storage adapter
        if hasattr(storage, 'pool_size'):
            pool_size_val = storage.pool_size
        elif hasattr(storage, '_connection_pool') and storage._connection_pool:
            # Try to infer from connection pool
            pool_size_val = 5  # Default
        else:
            pool_size_val = 5  # Default
        max_overflow_val = getattr(storage, 'max_overflow', 10) or 10
    except:
        pool_size_val = 5
        max_overflow_val = 10
    
    max_connections = pool_size_val + max_overflow_val
    # Conservative: use 2-3 workers to avoid pool exhaustion
    # Each chunk needs ~3 connections (patients + encounters + observations in parallel)
    parallel_chunk_workers = min(3, max(2, (max_connections - 2) // 3))
    
    # Thread-safe counters for parallel chunk processing
    parallel_success_count = threading.Lock()
    parallel_failure_count = threading.Lock()
    parallel_success = 0
    parallel_failure = 0
    
    # Chunk processing executor for parallel chunk processing (Phase 3)
    chunk_executor = ThreadPoolExecutor(
        max_workers=parallel_chunk_workers,
        thread_name_prefix="chunk-processor"
    )
    pending_chunk_futures = []  # Track pending chunk processing tasks
    
    logger.info(
        f"Starting ingestion from: {source} "
        f"(parallel chunk processing: {parallel_chunk_workers} workers, "
        f"max connections: {max_connections})"
    )
    
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
                    # Check if result is a tuple (redacted_df, raw_df) for raw vault support
                    if isinstance(result.value, tuple) and len(result.value) == 2:
                        # New format: tuple with (redacted_df, raw_df)
                        df, raw_df = result.value
                        logger.info(f"Processing DataFrame batch: {len(df)} rows (with raw vault data)")
                    elif hasattr(result.value, 'shape'):  # pandas DataFrame (backward compatibility)
                        df = result.value
                        raw_df = None  # No raw data available (backward compatibility)
                        logger.info(f"Processing DataFrame batch: {len(df)} rows")
                    else:
                        # Not a DataFrame or tuple, skip
                        continue
                        
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
                        
                        # For encounters/observations, ensure referenced patients exist
                        if table_name in ['encounters', 'observations'] and 'patient_id' in df.columns:
                            referenced_patient_ids = set(df['patient_id'].dropna().unique())
                            if referenced_patient_ids:
                                try:
                                    from src.dashboard.services.connection_helper import get_db_connection
                                    with get_db_connection(storage) as conn:
                                        if conn:
                                            # Query existing patient_ids
                                            placeholders = ','.join(['%s'] * len(referenced_patient_ids))
                                            query = f"SELECT patient_id FROM patients WHERE patient_id IN ({placeholders})"
                                            result = conn.execute(query, list(referenced_patient_ids))
                                            existing_patient_ids = set()
                                            if result:
                                                existing_patient_ids = {row[0] for row in result.fetchall()}
                                            
                                            # Create minimal patient records for missing patient_ids
                                            missing_patient_ids = referenced_patient_ids - existing_patient_ids
                                            if missing_patient_ids:
                                                logger.info(
                                                    f"Creating {len(missing_patient_ids)} minimal patient records "
                                                    f"for missing patient_ids referenced in {table_name}"
                                                )
                                                import pandas as pd
                                                minimal_patients_data = []
                                                for patient_id in missing_patient_ids:
                                                    minimal_patients_data.append({
                                                        'patient_id': patient_id,
                                                        'identifiers': [],
                                                        'family_name': None,
                                                        'given_names': [],
                                                        'name_prefix': [],
                                                        'name_suffix': [],
                                                        'date_of_birth': None,
                                                        'gender': None,
                                                        'deceased': None,
                                                        'marital_status': None,
                                                        'address_line1': None,
                                                        'address_line2': None,
                                                        'city': None,
                                                        'state': None,
                                                        'postal_code': None,
                                                        'country': None,
                                                        'address_use': None,
                                                        'phone': None,
                                                        'email': None,
                                                        'fax': None,
                                                        'emergency_contact_name': None,
                                                        'emergency_contact_relationship': None,
                                                        'emergency_contact_phone': None,
                                                        'language': None,
                                                        'managing_organization': None,
                                                        'source_adapter': 'csv_ingester',
                                                        'transformation_hash': None
                                                    })
                                                
                                                # Persist minimal patients first
                                                minimal_patients_df = pd.DataFrame(minimal_patients_data)
                                                patients_result = storage.persist_dataframe(minimal_patients_df, 'patients')
                                                if patients_result.is_success():
                                                    logger.info(f"Persisted {patients_result.value} minimal patient records")
                                                else:
                                                    logger.warning(
                                                        f"Failed to persist minimal patients: {patients_result.error}. "
                                                        "Proceeding anyway - may cause foreign key constraint errors."
                                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Could not ensure patients exist for {table_name}: {e}. "
                                        "Proceeding anyway - may cause foreign key constraint errors."
                                    )
                        
                        # Store DataFrame for parallel persistence
                        # Also store raw DataFrame if available (for raw vault)
                        chunk_dataframes[table_name] = df
                        if raw_df is not None:
                            chunk_dataframes[f'{table_name}_raw'] = raw_df
                        
                        # Check if we have all 3 DataFrames for a complete chunk
                        # For JSON ingestion: chunk = patients + encounters + observations
                        # For CSV ingestion: may only have one table type (patients, encounters, or observations)
                        # Check that all values are not None (can't use all() directly on DataFrames)
                        has_complete_chunk = all(df is not None for df in chunk_dataframes.values())
                        
                        # For CSV ingestion with single table type, check if we should persist immediately
                        # This happens when:
                        # 1. We have a DataFrame for one table
                        # 2. The other two tables are None (not part of this ingestion)
                        # 3. This is a single-table CSV (only one table type in the file)
                        non_none_count = sum(1 for df in chunk_dataframes.values() if df is not None)
                        is_single_table_csv = (
                            non_none_count == 1 and
                            chunk_dataframes[table_name] is not None
                        )
                        
                        if has_complete_chunk:
                            # Phase 3: Parallel Chunk Processing
                            # Create a copy of the chunk dataframes for parallel processing
                            # Include both redacted and raw DataFrames
                            chunk_to_process = {
                                'patients': chunk_dataframes['patients'].copy() if chunk_dataframes['patients'] is not None else None,
                                'encounters': chunk_dataframes['encounters'].copy() if chunk_dataframes['encounters'] is not None else None,
                                'observations': chunk_dataframes['observations'].copy() if chunk_dataframes['observations'] is not None else None,
                                'patients_raw': chunk_dataframes.get('patients_raw').copy() if chunk_dataframes.get('patients_raw') is not None else None,
                                'encounters_raw': chunk_dataframes.get('encounters_raw').copy() if chunk_dataframes.get('encounters_raw') is not None else None,
                                'observations_raw': chunk_dataframes.get('observations_raw').copy() if chunk_dataframes.get('observations_raw') is not None else None,
                            }
                            
                            chunk_number = len(pending_chunk_futures) + 1
                            logger.info(
                                f"Submitting chunk {chunk_number} for parallel processing: "
                                f"{len(chunk_to_process['patients'])} patients, "
                                f"{len(chunk_to_process['encounters'])} encounters, "
                                f"{len(chunk_to_process['observations'])} observations"
                            )
                            
                            # Submit chunk for parallel processing
                            def process_chunk_parallel(
                                chunk_data: dict,
                                storage_adapter: StoragePort,
                                chunk_num: int,
                                redaction_logger_instance,
                                flush_lock_instance,
                                flush_executor_instance,
                                pending_flush_futures_list
                            ) -> tuple[int, int]:
                                """Process a complete chunk in parallel.
                                
                                Returns:
                                    tuple[int, int]: (success_count, failure_count) for this chunk
                                """
                                chunk_success = 0
                                chunk_failure = 0
                                
                                try:
                                    # Persist tables in order to respect foreign key constraints:
                                    # 1. Patients first (no dependencies)
                                    # 2. Encounters and observations in parallel (both depend on patients)
                                    
                                    # Step 1: Ensure all referenced patients exist
                                    # Collect patient_ids from encounters and observations
                                    referenced_patient_ids = set()
                                    if chunk_data['encounters'] is not None and not chunk_data['encounters'].empty:
                                        if 'patient_id' in chunk_data['encounters'].columns:
                                            referenced_patient_ids.update(
                                                chunk_data['encounters']['patient_id'].dropna().unique()
                                            )
                                    if chunk_data['observations'] is not None and not chunk_data['observations'].empty:
                                        if 'patient_id' in chunk_data['observations'].columns:
                                            referenced_patient_ids.update(
                                                chunk_data['observations']['patient_id'].dropna().unique()
                                            )
                                    
                                    # Check which patient_ids already exist in the database
                                    existing_patient_ids = set()
                                    if referenced_patient_ids:
                                        try:
                                            from src.dashboard.services.connection_helper import get_db_connection
                                            with get_db_connection(storage_adapter) as conn:
                                                if conn:
                                                    # Query existing patient_ids
                                                    placeholders = ','.join(['%s'] * len(referenced_patient_ids))
                                                    query = f"SELECT patient_id FROM patients WHERE patient_id IN ({placeholders})"
                                                    result = conn.execute(query, list(referenced_patient_ids))
                                                    if result:
                                                        existing_patient_ids = {row[0] for row in result.fetchall()}
                                        except Exception as e:
                                            logger.warning(
                                                f"[Chunk {chunk_num}] Could not check existing patients: {e}. "
                                                "Proceeding with assumption that patients may not exist."
                                            )
                                    
                                    # Create minimal patient records for missing patient_ids
                                    missing_patient_ids = referenced_patient_ids - existing_patient_ids
                                    if missing_patient_ids:
                                        logger.info(
                                            f"[Chunk {chunk_num}] Creating {len(missing_patient_ids)} minimal patient records "
                                            f"for missing patient_ids"
                                        )
                                        minimal_patients_data = []
                                        for patient_id in missing_patient_ids:
                                            minimal_patients_data.append({
                                                'patient_id': patient_id,
                                                'identifiers': [],
                                                'family_name': None,
                                                'given_names': [],
                                                'name_prefix': [],
                                                'name_suffix': [],
                                                'date_of_birth': None,
                                                'gender': None,
                                                'deceased': None,
                                                'marital_status': None,
                                                'address_line1': None,
                                                'address_line2': None,
                                                'city': None,
                                                'state': None,
                                                'postal_code': None,
                                                'country': None,
                                                'address_use': None,
                                                'phone': None,
                                                'email': None,
                                                'fax': None,
                                                'emergency_contact_name': None,
                                                'emergency_contact_relationship': None,
                                                'emergency_contact_phone': None,
                                                'language': None,
                                                'managing_organization': None,
                                                'source_adapter': 'csv_ingester',  # Default, will be overridden if present
                                                'transformation_hash': None
                                            })
                                        
                                        # Add minimal patients to patients DataFrame
                                        import pandas as pd
                                        minimal_patients_df = pd.DataFrame(minimal_patients_data)
                                        if chunk_data['patients'] is None or chunk_data['patients'].empty:
                                            chunk_data['patients'] = minimal_patients_df
                                        else:
                                            # Merge with existing patients, avoiding duplicates
                                            chunk_data['patients'] = pd.concat([
                                                chunk_data['patients'],
                                                minimal_patients_df[~minimal_patients_df['patient_id'].isin(chunk_data['patients']['patient_id'])]
                                            ], ignore_index=True)
                                    
                                    # Step 2: Persist patients (including minimal ones)
                                    # Use persist_dataframe_smart with raw vault support if available
                                    patients_raw_df = chunk_data.get('patients_raw')
                                    if hasattr(storage_adapter, 'persist_dataframe_smart'):
                                        patients_result = storage_adapter.persist_dataframe_smart(
                                            chunk_data['patients'],
                                            'patients',
                                            raw_df=patients_raw_df,
                                            ingestion_id=ingestion_id,
                                            source_adapter=adapter.adapter_name
                                        )
                                    else:
                                        patients_result = storage_adapter.persist_dataframe(
                                            chunk_data['patients'],
                                            'patients'
                                        )
                                    if patients_result.is_success():
                                        chunk_success += patients_result.value
                                        logger.info(
                                            f"[Chunk {chunk_num}] Persisted {patients_result.value} rows to patients"
                                        )
                                    else:
                                        chunk_failure += len(chunk_data['patients']) if chunk_data['patients'] is not None else 0
                                        logger.error(
                                            f"[Chunk {chunk_num}] Failed to persist patients: {patients_result.error}"
                                        )
                                        # Skip rest of chunk if patients failed
                                        return (chunk_success, chunk_failure)
                                    
                                    # Step 2: Persist encounters and observations in parallel
                                    # (both depend on patients, but are independent of each other)
                                    with ThreadPoolExecutor(max_workers=2) as executor:
                                        futures = {}
                                        for table in ['encounters', 'observations']:
                                            if chunk_data[table] is not None:
                                                raw_key = f'{table}_raw'
                                                raw_df = chunk_data.get(raw_key)
                                                if hasattr(storage_adapter, 'persist_dataframe_smart'):
                                                    futures[executor.submit(
                                                        storage_adapter.persist_dataframe_smart,
                                                        chunk_data[table],
                                                        table,
                                                        raw_df=raw_df,
                                                        ingestion_id=ingestion_id,
                                                        source_adapter=adapter.adapter_name
                                                    )] = (chunk_data[table], table)
                                                else:
                                                    futures[executor.submit(
                                                        storage_adapter.persist_dataframe,
                                                        chunk_data[table],
                                                        table
                                                    )] = (chunk_data[table], table)
                                        
                                        # Process results as they complete
                                        for future in as_completed(futures):
                                            df_persisted, table_name_persisted = futures[future]
                                            try:
                                                persist_result = future.result()
                                                if persist_result.is_success():
                                                    chunk_success += persist_result.value
                                                    logger.info(
                                                        f"[Chunk {chunk_num}] Persisted {persist_result.value} rows to {table_name_persisted} "
                                                        f"(parallel)"
                                                    )
                                                else:
                                                    chunk_failure += len(df_persisted)
                                                    logger.error(
                                                        f"[Chunk {chunk_num}] Failed to persist {table_name_persisted}: {persist_result.error}"
                                                    )
                                            except Exception as e:
                                                logger.error(
                                                    f"[Chunk {chunk_num}] Exception persisting {table_name_persisted}: {e}",
                                                    exc_info=True
                                                )
                                                chunk_failure += len(df_persisted)
                                    
                                    # Flush redaction logs asynchronously after completing a full chunk
                                    # This doesn't block chunk processing - flush happens in background
                                    if hasattr(storage_adapter, 'flush_redaction_logs'):
                                        with flush_lock_instance:
                                            redaction_logs = redaction_logger_instance.get_logs()
                                            if redaction_logs:
                                                # Copy logs for async flush (thread-safe)
                                                logs_copy = redaction_logs.copy()
                                                # Clear logs immediately to avoid duplicates
                                                redaction_logger_instance.clear_logs()
                                                
                                                # Submit async flush task
                                                future = flush_executor_instance.submit(
                                                    flush_redaction_logs_async,
                                                    storage_adapter,
                                                    logs_copy
                                                )
                                                pending_flush_futures_list.append(future)
                                
                                except Exception as e:
                                    logger.error(
                                        f"[Chunk {chunk_num}] Error processing chunk: {e}",
                                        exc_info=True
                                    )
                                    # Count all rows as failures
                                    chunk_failure += (
                                        len(chunk_data.get('patients', [])) +
                                        len(chunk_data.get('encounters', [])) +
                                        len(chunk_data.get('observations', []))
                                    )
                                
                                return (chunk_success, chunk_failure)
                            
                            # Submit chunk for parallel processing
                            future = chunk_executor.submit(
                                process_chunk_parallel,
                                chunk_to_process,
                                storage,
                                chunk_number,
                                redaction_logger,
                                flush_lock,
                                flush_executor,
                                pending_flush_futures
                            )
                            pending_chunk_futures.append(future)
                            
                            # Process completed chunks to update counters (non-blocking check)
                            # This allows us to process results as they complete
                            completed_futures = [f for f in pending_chunk_futures if f.done()]
                            for completed_future in completed_futures:
                                try:
                                    chunk_success, chunk_failure = completed_future.result()
                                    with parallel_success_count:
                                        parallel_success += chunk_success
                                    with parallel_failure_count:
                                        parallel_failure += chunk_failure
                                except Exception as e:
                                    logger.error(f"Error getting chunk processing result: {e}", exc_info=True)
                                    # Count as failure
                                    with parallel_failure_count:
                                        parallel_failure += 1
                            
                            # Remove completed futures
                            pending_chunk_futures = [f for f in pending_chunk_futures if not f.done()]
                            
                            # Reset chunk tracking for next chunk (immediately, don't wait)
                            chunk_dataframes = {
                                'patients': None,
                                'encounters': None,
                                'observations': None,
                                'patients_raw': None,
                                'encounters_raw': None,
                                'observations_raw': None
                            }
                            chunk_tables_persisted.clear()
                        elif is_single_table_csv:
                            # Single-table CSV (e.g., patients-only): persist immediately
                            # This handles CSV files that only contain one table type
                            logger.info(
                                f"Single-table CSV detected ({table_name}), persisting immediately: "
                                f"{len(df)} rows"
                            )
                            
                            # Persist this DataFrame directly (no need to wait for other tables)
                            # Use persist_dataframe_smart with raw vault support if available
                            raw_df_for_table = chunk_dataframes.get(f'{table_name}_raw')
                            if hasattr(storage, 'persist_dataframe_smart'):
                                persist_result = storage.persist_dataframe_smart(
                                    df, table_name,
                                    raw_df=raw_df_for_table,
                                    ingestion_id=ingestion_id,
                                    source_adapter=adapter.adapter_name
                                )
                            else:
                                persist_result = storage.persist_dataframe(df, table_name)
                            if persist_result.is_success():
                                success_count += persist_result.value
                                logger.info(f"Persisted {persist_result.value} rows to {table_name}")
                                
                                # Flush redaction logs asynchronously after persisting
                                if hasattr(storage, 'flush_redaction_logs'):
                                    with flush_lock:
                                        redaction_logs = redaction_logger.get_logs()
                                        if redaction_logs:
                                            logs_copy = redaction_logs.copy()
                                            redaction_logger.clear_logs()
                                            future = flush_executor.submit(
                                                flush_redaction_logs_async,
                                                storage,
                                                logs_copy
                                            )
                                            pending_flush_futures.append(future)
                            else:
                                failure_count += len(df)
                                logger.error(f"Failed to persist {table_name}: {persist_result.error}")
                            
                            # Clear this table from chunk_dataframes since we've persisted it
                            chunk_dataframes[table_name] = None
                        else:
                            # Not a complete chunk yet, wait for more DataFrames
                            logger.debug(
                                f"Collected {table_name} DataFrame, waiting for complete chunk. "
                                f"Have: {[k for k, v in chunk_dataframes.items() if v is not None]}"
                            )
                    
                    # Handle GoldenRecord results (XML row-by-row processing)
                    else:
                        # Check if result is tuple (GoldenRecord, original_record_data) for raw vault
                        if isinstance(result.value, tuple) and len(result.value) == 2:
                            golden_record, original_record_data = result.value
                        else:
                            # Backward compatibility: result.value is just GoldenRecord
                            golden_record = result.value
                            original_record_data = None
                        
                        batch_records.append(golden_record)
                        
                        # Store original record data for raw vault (if available)
                        if original_record_data is not None:
                            if not hasattr(batch_records, '_raw_data'):
                                batch_records._raw_data = []
                            batch_records._raw_data.append(original_record_data)
                        
                        # Persist in batches for efficiency
                        if len(batch_records) >= (batch_size or settings.batch_size):
                            # For XML, convert GoldenRecords to DataFrames and use persist_dataframe_smart with raw vault
                            if hasattr(storage, 'persist_dataframe_smart') and hasattr(batch_records, '_raw_data'):
                                # Convert GoldenRecords to DataFrames
                                import pandas as pd
                                
                                # Convert patients
                                patients_data = []
                                patients_raw_data = []
                                for i, record in enumerate(batch_records):
                                    patient_dict = record.patient.model_dump(exclude_none=False)
                                    patient_dict['source_adapter'] = record.source_adapter
                                    patient_dict['transformation_hash'] = record.transformation_hash
                                    patients_data.append(patient_dict)
                                    
                                    # Get original patient data from raw_data
                                    if i < len(batch_records._raw_data):
                                        raw_data = batch_records._raw_data[i]
                                        # Extract patient data from raw_data (may need field mapping)
                                        raw_patient_data = raw_data.copy()  # Use original record_data as-is
                                        patients_raw_data.append(raw_patient_data)
                                
                                patients_df = pd.DataFrame(patients_data)
                                patients_raw_df = pd.DataFrame(patients_raw_data) if patients_raw_data else None
                                
                                # Convert encounters
                                encounters_data = []
                                encounters_raw_data = []
                                for i, record in enumerate(batch_records):
                                    for encounter in record.encounters:
                                        enc_dict = encounter.model_dump(exclude_none=False)
                                        enc_dict['ingestion_timestamp'] = record.ingestion_timestamp
                                        enc_dict['source_adapter'] = record.source_adapter
                                        enc_dict['transformation_hash'] = record.transformation_hash
                                        encounters_data.append(enc_dict)
                                        
                                        # Get original encounter data from raw_data
                                        if i < len(batch_records._raw_data):
                                            raw_data = batch_records._raw_data[i]
                                            # Extract encounter data (may need to reconstruct from original)
                                            # For now, use the encounter dict as raw (it's already from original data)
                                            raw_enc_data = enc_dict.copy()  # Fallback: use validated data
                                            encounters_raw_data.append(raw_enc_data)
                                
                                encounters_df = pd.DataFrame(encounters_data) if encounters_data else pd.DataFrame()
                                encounters_raw_df = pd.DataFrame(encounters_raw_data) if encounters_raw_data else None
                                
                                # Convert observations
                                observations_data = []
                                observations_raw_data = []
                                for i, record in enumerate(batch_records):
                                    for observation in record.observations:
                                        obs_dict = observation.model_dump(exclude_none=False)
                                        obs_dict['ingestion_timestamp'] = record.ingestion_timestamp
                                        obs_dict['source_adapter'] = record.source_adapter
                                        obs_dict['transformation_hash'] = record.transformation_hash
                                        observations_data.append(obs_dict)
                                        
                                        # Get original observation data from raw_data
                                        if i < len(batch_records._raw_data):
                                            raw_data = batch_records._raw_data[i]
                                            # Extract observation data (may need to reconstruct from original)
                                            raw_obs_data = obs_dict.copy()  # Fallback: use validated data
                                            observations_raw_data.append(raw_obs_data)
                                
                                observations_df = pd.DataFrame(observations_data) if observations_data else pd.DataFrame()
                                observations_raw_df = pd.DataFrame(observations_raw_data) if observations_raw_data else None
                                
                                # Persist using persist_dataframe_smart with raw vault
                                total_persisted = 0
                                if not patients_df.empty:
                                    result_patients = storage.persist_dataframe_smart(
                                        patients_df, 'patients',
                                        raw_df=patients_raw_df,
                                        ingestion_id=ingestion_id,
                                        source_adapter=adapter.adapter_name
                                    )
                                    if result_patients.is_success():
                                        total_persisted += result_patients.value
                                
                                if not encounters_df.empty:
                                    result_encounters = storage.persist_dataframe_smart(
                                        encounters_df, 'encounters',
                                        raw_df=encounters_raw_df,
                                        ingestion_id=ingestion_id,
                                        source_adapter=adapter.adapter_name
                                    )
                                    if result_encounters.is_success():
                                        total_persisted += result_encounters.value
                                
                                if not observations_df.empty:
                                    result_observations = storage.persist_dataframe_smart(
                                        observations_df, 'observations',
                                        raw_df=observations_raw_df,
                                        ingestion_id=ingestion_id,
                                        source_adapter=adapter.adapter_name
                                    )
                                    if result_observations.is_success():
                                        total_persisted += result_observations.value
                                
                                persist_result = Result.success_result(total_persisted) if total_persisted > 0 else Result.success_result(len(batch_records))
                            else:
                                # Fallback to standard persist_batch
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
            
            # Wait for all pending parallel chunk processing to complete
            logger.info(f"Waiting for {len(pending_chunk_futures)} pending chunks to complete...")
            for future in pending_chunk_futures:
                try:
                    chunk_success, chunk_failure = future.result()
                    with parallel_success_count:
                        parallel_success += chunk_success
                    with parallel_failure_count:
                        parallel_failure += chunk_failure
                except Exception as e:
                    logger.error(f"Error processing chunk: {e}", exc_info=True)
                    with parallel_failure_count:
                        parallel_failure += 1
            
            # Add parallel processing results to main counters
            success_count += parallel_success
            failure_count += parallel_failure
            
            # Shutdown chunk executor
            chunk_executor.shutdown(wait=True)
            logger.info("Parallel chunk processing completed")
            
            # Persist any remaining DataFrames (JSON ingestion - incomplete chunk)
            remaining_tables = [table for table, df in chunk_dataframes.items() if df is not None]
            if remaining_tables:
                logger.info(
                    f"Persisting remaining incomplete chunk: {', '.join(remaining_tables)}"
                )
                for table_name in remaining_tables:
                    df = chunk_dataframes[table_name]
                    raw_df_for_table = chunk_dataframes.get(f'{table_name}_raw')
                    if hasattr(storage, 'persist_dataframe_smart'):
                        persist_result = storage.persist_dataframe_smart(
                            df, table_name,
                            raw_df=raw_df_for_table,
                            ingestion_id=ingestion_id,
                            source_adapter=adapter.adapter_name
                        )
                    else:
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

