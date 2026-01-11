"""Academic Performance Benchmarking Suite for Data-Dialysis.

This script generates test files in specified sizes, runs comprehensive benchmarks,
and outputs results to CSV for academic analysis.

Target file sizes: [1MB, 10MB, 50MB, 100MB, 250MB, 500MB]
Formats: CSV, JSON, XML
Metrics: Throughput, memory, latency, batch statistics

Security Impact:
    - Generates synthetic test data (not real PII)
    - Benchmarks validate system performance under realistic workloads
    - Memory profiling ensures no resource exhaustion vulnerabilities

Academic Value:
    - Demonstrates quantitative evaluation methodology
    - Provides empirical evidence of system performance
    - Enables comparative analysis across formats and sizes
"""

import argparse
import csv
import logging
import os
import sys
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Optional, Tuple
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import generation scripts
from generate_xml_test_files import generate_xml_file
from generate_json_test_file import generate_json_file
from generate_csv_test_file import generate_csv_file
from generate_bad_data import generate_bad_xml_file, generate_bad_json_file, generate_bad_csv_file

# Import ingestion adapters
from src.adapters.ingesters import get_adapter
from src.adapters.storage import DuckDBAdapter, PostgreSQLAdapter
from src.domain.ports import Result
from src.domain.guardrails import CircuitBreaker, CircuitBreakerConfig
from src.infrastructure.config_manager import get_database_config
from src.infrastructure.redaction_logger import get_redaction_logger
from src.infrastructure.redaction_context import redaction_context
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics for a single run."""
    # Test configuration (required fields first)
    adapter_type: str  # csv, json, xml
    file_size_mb: float
    file_path: str
    num_records: int
    
    # Timing metrics (required)
    total_time_seconds: float
    processing_time_seconds: float
    records_per_second: float
    mb_per_second: float
    
    # Memory metrics (required)
    peak_memory_mb: float
    avg_memory_mb: float
    memory_efficiency_mb_per_record: float
    
    # Batch metrics (required)
    num_batches: int
    avg_batch_time_seconds: float
    
    # Success metrics (required)
    records_successful: int
    records_failed: int
    success_rate: float
    
    # Additional metrics (required)
    cpu_usage_percent: float
    timestamp: str
    
    # Optional fields with defaults (must come after all required fields in Python 3.13+)
    test_scenario: str = "happy_path"  # happy_path, bad_data, xml_performance
    batch_size: Optional[int] = None
    
    # Circuit breaker metrics (for bad data scenarios)
    circuit_breaker_opened: bool = False
    circuit_breaker_failure_rate: Optional[float] = None
    circuit_breaker_threshold: Optional[float] = None
    circuit_breaker_total_processed: Optional[int] = None
    circuit_breaker_total_failures: Optional[int] = None
    
    # XML-specific metrics
    xml_streaming_enabled: Optional[bool] = None
    xml_memory_efficiency: Optional[float] = None


def estimate_records_for_size(target_size_mb: float, format_type: str) -> int:
    """Estimate number of records needed to achieve target file size.
    
    Args:
        target_size_mb: Target file size in MB
        format_type: File format (csv, json, xml)
    
    Returns:
        Estimated number of records
    """
    # Rough estimates based on average record sizes
    # These are approximations - actual sizes will vary
    if format_type == "csv":
        # CSV: ~500-800 bytes per patient record (flat structure)
        bytes_per_record = 650
    elif format_type == "json":
        # JSON: ~1200-1800 bytes per record (nested structure with encounters/observations)
        # Based on generate_json_test_file.py, each record includes patient + encounters + observations
        bytes_per_record = 1500
    elif format_type == "xml":
        # XML: ~900-1200 bytes per record (with tags)
        # Based on generate_xml_test_files.py estimate_record_size()
        bytes_per_record = 1050
    else:
        bytes_per_record = 1000
    
    target_bytes = target_size_mb * 1024 * 1024
    records = int(target_bytes / bytes_per_record)
    return max(1, records)


def generate_test_files(
    output_dir: Path,
    sizes_mb: List[float],
    formats: List[str],
    force_regenerate: bool = False
) -> Dict[str, Dict[float, Path]]:
    """Generate test files for all size/format combinations.
    
    Args:
        output_dir: Directory to store test files
        sizes_mb: List of target file sizes in MB
        formats: List of formats (csv, json, xml)
        force_regenerate: If True, regenerate existing files
    
    Returns:
        Dictionary mapping format -> size -> file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    files = defaultdict(dict)
    
    xml_config_path = project_root / "xml_config.json"
    
    for format_type in formats:
        logger.info(f"Generating {format_type.upper()} test files...")
        
        for size_mb in sizes_mb:
            # Generate filename
            if format_type == "csv":
                # CSV generates multiple files, use patients file as primary
                file_path = output_dir / f"test_{format_type}_{int(size_mb)}mb_patients.csv"
            else:
                file_path = output_dir / f"test_{format_type}_{int(size_mb)}mb.{format_type}"
            
            # Skip if file exists and not forcing regeneration
            if file_path.exists() and not force_regenerate:
                logger.info(f"  Skipping {file_path.name} (already exists)")
                files[format_type][size_mb] = file_path
                continue
            
            logger.info(f"  Generating {size_mb}MB {format_type.upper()} file...")
            
            try:
                if format_type == "xml":
                    # XML generation returns (actual_size_mb, record_count)
                    actual_size, record_count = generate_xml_file(
                        target_size_mb=size_mb,
                        output_path=file_path
                    )
                    files[format_type][size_mb] = file_path
                    
                elif format_type == "json":
                    # Estimate records needed - JSON generation may need iteration to hit target size
                    num_records = estimate_records_for_size(size_mb, format_type)
                    generate_json_file(
                        output_path=file_path,
                        num_records=num_records
                    )
                    # Check actual size and adjust if needed
                    actual_size_mb = file_path.stat().st_size / (1024 * 1024)
                    if abs(actual_size_mb - size_mb) / size_mb > 0.2:  # More than 20% off
                        # Adjust and regenerate
                        adjustment_factor = size_mb / actual_size_mb
                        num_records = int(num_records * adjustment_factor)
                        generate_json_file(
                            output_path=file_path,
                            num_records=num_records
                        )
                    files[format_type][size_mb] = file_path
                    
                elif format_type == "csv":
                    # CSV generates multiple files
                    base_path = output_dir / f"test_{format_type}_{int(size_mb)}mb"
                    num_records = estimate_records_for_size(size_mb, format_type)
                    generate_csv_file(
                        output_path=base_path,
                        num_records=num_records,
                        format_type="flat"
                    )
                    # Use patients file as primary
                    files[format_type][size_mb] = base_path.parent / f"{base_path.stem}_patients.csv"
                    
            except Exception as e:
                logger.error(f"  Failed to generate {file_path}: {e}")
                continue
    
    return files


def benchmark_bad_data(
    file_path: Path,
    adapter_type: str,
    failure_rate_percent: float,
    batch_size: Optional[int] = None,
    storage: Optional[Result] = None
) -> BenchmarkMetrics:
    """Run benchmark for bad data with circuit breaker tracking.
    
    Args:
        file_path: Path to test file with bad data
        adapter_type: Type of adapter (csv, json, xml)
        failure_rate_percent: Expected failure rate percentage
        batch_size: Optional batch size for processing
        storage: Optional storage adapter for persistence and log flushing
    
    Returns:
        BenchmarkMetrics with circuit breaker tracking
    """
    logger.info(f"Benchmarking BAD DATA {adapter_type.upper()}: {file_path.name} (expected {failure_rate_percent}% failures)")
    
    # Get file size
    file_size_bytes = file_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Initialize memory tracking
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Memory samples over time
    memory_samples = []
    batch_times = []
    
    # Get adapter (with config_path for XML files)
    adapter_kwargs = {}
    if adapter_type == "xml":
        xml_config_path = project_root / "xml_config.json"
        if xml_config_path.exists():
            adapter_kwargs["config_path"] = str(xml_config_path)
        else:
            logger.warning(f"XML config file not found at {xml_config_path}, XML adapter may fail")
    
    adapter = get_adapter(str(file_path), **adapter_kwargs)
    if adapter is None:
        raise ValueError(f"No adapter found for {file_path}")
    
    # Set up circuit breaker
    circuit_breaker_config = CircuitBreakerConfig(
        failure_threshold_percent=50.0,  # Default threshold
        window_size=100,
        min_records_before_check=10,
        abort_on_open=False  # Don't abort, just track for benchmarking
    )
    circuit_breaker = CircuitBreaker(circuit_breaker_config)
    
    # Set up redaction logging
    redaction_logger = get_redaction_logger()
    ingestion_id = str(uuid.uuid4())
    redaction_logger.set_ingestion_id(ingestion_id)
    
    # Track metrics
    records_processed = 0
    records_successful = 0
    records_failed = 0
    num_batches = 0
    batch_start = None
    
    # Start timing
    start_time = time.time()
    cpu_start = process.cpu_percent(interval=0.1)
    
    try:
        with redaction_context(
            logger=redaction_logger,
            source_adapter=adapter.adapter_name,
            ingestion_id=ingestion_id
        ):
            # Process records with circuit breaker tracking
            for result in adapter.ingest(str(file_path)):
                # Record result in circuit breaker
                try:
                    circuit_breaker.record_result(result)
                except Exception as e:
                    # Circuit breaker may raise if abort_on_open=True, but we set it to False
                    logger.debug(f"Circuit breaker recorded result: {e}")
                
                # Sample memory periodically
                if num_batches % 10 == 0:
                    current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_samples.append(current_memory)
                
                # Track batch start time
                if batch_start is None:
                    batch_start = time.time()
                
                if isinstance(result.value, tuple):
                    if adapter_type in ["csv", "json"]:
                        df, _ = result.value
                        if hasattr(df, '__len__'):
                            batch_records = len(df)
                            records_processed += batch_records
                            if result.is_success():
                                records_successful += batch_records
                            else:
                                records_failed += batch_records
                    elif adapter_type == "xml":
                        records_processed += 1
                        if result.is_success():
                            records_successful += 1
                        else:
                            records_failed += 1
                elif hasattr(result.value, '__len__'):
                    batch_records = len(result.value)
                    records_processed += batch_records
                    if result.is_success():
                        records_successful += batch_records
                    else:
                        records_failed += batch_records
                else:
                    records_processed += 1
                    if result.is_success():
                        records_successful += 1
                    else:
                        records_failed += 1
                
                num_batches += 1
                
                # Track batch processing time
                batch_end = time.time()
                batch_times.append(batch_end - batch_start)
                batch_start = batch_end
            
            # Flush redaction logs if storage is available
            if storage and hasattr(storage, 'flush_redaction_logs'):
                try:
                    redaction_logs = redaction_logger.get_logs()
                    if redaction_logs:
                        flush_result = storage.flush_redaction_logs(redaction_logs)
                        if flush_result.is_success():
                            logger.debug(f"Flushed {flush_result.value} redaction logs to storage")
                        redaction_logger.clear_logs()
                except Exception as e:
                    logger.warning(f"Failed to flush redaction logs: {e}")
    
    except Exception as e:
        logger.error(f"Error during bad data ingestion: {e}", exc_info=True)
        records_failed += records_processed - records_successful
        
        # Try to flush logs even on error
        if storage and hasattr(storage, 'flush_redaction_logs'):
            try:
                redaction_logs = redaction_logger.get_logs()
                if redaction_logs:
                    storage.flush_redaction_logs(redaction_logs)
                    redaction_logger.clear_logs()
            except Exception:
                pass
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    cpu_end = process.cpu_percent()
    
    # Get memory statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_memory_mb = peak / (1024 * 1024)  # Convert to MB
    peak_memory_mb = max(peak_memory_mb, max(memory_samples) if memory_samples else 0)
    avg_memory_mb = mean(memory_samples) if memory_samples else initial_memory
    
    # Get circuit breaker statistics
    cb_stats = circuit_breaker.get_statistics()
    
    # Calculate metrics
    records_per_second = records_processed / total_time if total_time > 0 else 0
    mb_per_second = file_size_mb / total_time if total_time > 0 else 0
    memory_efficiency = avg_memory_mb / records_processed if records_processed > 0 else 0
    success_rate = records_successful / records_processed if records_processed > 0 else 0
    avg_batch_time = mean(batch_times) if batch_times else 0
    
    # Estimate number of records from file (if not already counted)
    if records_processed == 0:
        if adapter_type == "csv":
            records_processed = int(file_size_bytes / 650)
        elif adapter_type == "json":
            records_processed = int(file_size_bytes / 1500)
        elif adapter_type == "xml":
            records_processed = int(file_size_bytes / 1050)
    
    return BenchmarkMetrics(
        adapter_type=adapter_type,
        file_size_mb=file_size_mb,
        file_path=str(file_path),
        num_records=records_processed,
        test_scenario=f"bad_data_{int(failure_rate_percent)}pct",
        total_time_seconds=total_time,
        processing_time_seconds=total_time,
        records_per_second=records_per_second,
        mb_per_second=mb_per_second,
        peak_memory_mb=peak_memory_mb,
        avg_memory_mb=avg_memory_mb,
        memory_efficiency_mb_per_record=memory_efficiency,
        num_batches=num_batches,
        batch_size=batch_size,
        avg_batch_time_seconds=avg_batch_time,
        records_successful=records_successful,
        records_failed=records_failed,
        success_rate=success_rate,
        circuit_breaker_opened=cb_stats['is_open'],
        circuit_breaker_failure_rate=cb_stats['failure_rate'],
        circuit_breaker_threshold=cb_stats['threshold'],
        circuit_breaker_total_processed=cb_stats['total_processed'],
        circuit_breaker_total_failures=cb_stats['total_failures'],
        cpu_usage_percent=(cpu_start + cpu_end) / 2 if cpu_start > 0 else 0,
        timestamp=datetime.now().isoformat()
    )


def benchmark_xml_performance(
    file_path: Path,
    streaming_enabled: bool,
    batch_size: Optional[int] = None,
    storage: Optional[Result] = None
) -> BenchmarkMetrics:
    """Run XML-specific performance benchmark (streaming vs non-streaming).
    
    Args:
        file_path: Path to XML test file
        streaming_enabled: Whether to use streaming mode
        batch_size: Optional batch size for processing
        storage: Optional storage adapter for persistence and log flushing
    
    Returns:
        BenchmarkMetrics with XML-specific metrics
    """
    logger.info(f"Benchmarking XML PERFORMANCE: {file_path.name} (streaming={streaming_enabled})")
    
    # Get file size
    file_size_bytes = file_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Initialize memory tracking
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Memory samples over time
    memory_samples = []
    batch_times = []
    
    # Get XML adapter with streaming configuration
    xml_config_path = project_root / "xml_config.json"
    adapter_kwargs = {"streaming_enabled": streaming_enabled}
    if xml_config_path.exists():
        adapter_kwargs["config_path"] = str(xml_config_path)
    
    adapter = get_adapter(str(file_path), **adapter_kwargs)
    if adapter is None:
        raise ValueError(f"No adapter found for {file_path}")
    
    # Set up redaction logging
    redaction_logger = get_redaction_logger()
    ingestion_id = str(uuid.uuid4())
    redaction_logger.set_ingestion_id(ingestion_id)
    
    # Track metrics
    records_processed = 0
    records_successful = 0
    records_failed = 0
    num_batches = 0
    batch_start = None
    
    # Start timing
    start_time = time.time()
    cpu_start = process.cpu_percent(interval=0.1)
    
    try:
        with redaction_context(
            logger=redaction_logger,
            source_adapter=adapter.adapter_name,
            ingestion_id=ingestion_id
        ):
            # Process records
            for result in adapter.ingest(str(file_path)):
                # Sample memory periodically
                if num_batches % 10 == 0:
                    current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_samples.append(current_memory)
                
                # Track batch start time
                if batch_start is None:
                    batch_start = time.time()
                
                if isinstance(result.value, tuple):
                    golden_record, _ = result.value
                    records_processed += 1
                    if result.is_success():
                        records_successful += 1
                    else:
                        records_failed += 1
                else:
                    records_processed += 1
                    if result.is_success():
                        records_successful += 1
                    else:
                        records_failed += 1
                
                num_batches += 1
                
                # Track batch processing time
                batch_end = time.time()
                batch_times.append(batch_end - batch_start)
                batch_start = batch_end
            
            # Flush redaction logs if storage is available
            if storage and hasattr(storage, 'flush_redaction_logs'):
                try:
                    redaction_logs = redaction_logger.get_logs()
                    if redaction_logs:
                        flush_result = storage.flush_redaction_logs(redaction_logs)
                        if flush_result.is_success():
                            logger.debug(f"Flushed {flush_result.value} redaction logs to storage")
                        redaction_logger.clear_logs()
                except Exception as e:
                    logger.warning(f"Failed to flush redaction logs: {e}")
    
    except Exception as e:
        logger.error(f"Error during XML performance ingestion: {e}", exc_info=True)
        records_failed += records_processed - records_successful
        
        # Try to flush logs even on error
        if storage and hasattr(storage, 'flush_redaction_logs'):
            try:
                redaction_logs = redaction_logger.get_logs()
                if redaction_logs:
                    storage.flush_redaction_logs(redaction_logs)
                    redaction_logger.clear_logs()
            except Exception:
                pass
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    cpu_end = process.cpu_percent()
    
    # Get memory statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_memory_mb = peak / (1024 * 1024)  # Convert to MB
    peak_memory_mb = max(peak_memory_mb, max(memory_samples) if memory_samples else 0)
    avg_memory_mb = mean(memory_samples) if memory_samples else initial_memory
    
    # Calculate metrics
    records_per_second = records_processed / total_time if total_time > 0 else 0
    mb_per_second = file_size_mb / total_time if total_time > 0 else 0
    memory_efficiency = avg_memory_mb / records_processed if records_processed > 0 else 0
    success_rate = records_successful / records_processed if records_processed > 0 else 0
    avg_batch_time = mean(batch_times) if batch_times else 0
    
    # Estimate number of records from file (if not already counted)
    if records_processed == 0:
        records_processed = int(file_size_bytes / 1050)  # ~1050 bytes per XML record
    
    return BenchmarkMetrics(
        adapter_type="xml",
        file_size_mb=file_size_mb,
        file_path=str(file_path),
        num_records=records_processed,
        test_scenario=f"xml_performance_streaming_{streaming_enabled}",
        total_time_seconds=total_time,
        processing_time_seconds=total_time,
        records_per_second=records_per_second,
        mb_per_second=mb_per_second,
        peak_memory_mb=peak_memory_mb,
        avg_memory_mb=avg_memory_mb,
        memory_efficiency_mb_per_record=memory_efficiency,
        num_batches=num_batches,
        batch_size=batch_size,
        avg_batch_time_seconds=avg_batch_time,
        records_successful=records_successful,
        records_failed=records_failed,
        success_rate=success_rate,
        xml_streaming_enabled=streaming_enabled,
        xml_memory_efficiency=memory_efficiency,
        cpu_usage_percent=(cpu_start + cpu_end) / 2 if cpu_start > 0 else 0,
        timestamp=datetime.now().isoformat()
    )


def benchmark_ingestion(
    file_path: Path,
    adapter_type: str,
    batch_size: Optional[int] = None,
    storage: Optional[Result] = None
) -> BenchmarkMetrics:
    """Run benchmark for a single file.
    
    Args:
        file_path: Path to test file
        adapter_type: Type of adapter (csv, json, xml)
        batch_size: Optional batch size for processing
        storage: Optional storage adapter for persistence and log flushing
    
    Returns:
        BenchmarkMetrics with all collected metrics
    """
    logger.info(f"Benchmarking {adapter_type.upper()}: {file_path.name}")
    
    # Get file size
    file_size_bytes = file_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Initialize memory tracking
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    # Memory samples over time
    memory_samples = []
    batch_times = []
    
    # Get adapter (with config_path for XML files)
    adapter_kwargs = {}
    if adapter_type == "xml":
        xml_config_path = project_root / "xml_config.json"
        if xml_config_path.exists():
            adapter_kwargs["config_path"] = str(xml_config_path)
        else:
            logger.warning(f"XML config file not found at {xml_config_path}, XML adapter may fail")
    
    adapter = get_adapter(str(file_path), **adapter_kwargs)
    if adapter is None:
        raise ValueError(f"No adapter found for {file_path}")
    
    # Set up redaction logging (required for redaction logs to be captured)
    redaction_logger = get_redaction_logger()
    ingestion_id = str(uuid.uuid4())
    redaction_logger.set_ingestion_id(ingestion_id)
    
    # Track metrics
    records_processed = 0
    records_successful = 0
    records_failed = 0
    num_batches = 0
    batch_start = None
    
    # Start timing
    start_time = time.time()
    cpu_start = process.cpu_percent(interval=0.1)  # Get initial CPU with small interval
    
    try:
        # Set up redaction context (required for redaction logging to work)
        with redaction_context(
            logger=redaction_logger,
            source_adapter=adapter.adapter_name,
            ingestion_id=ingestion_id
        ):
            # Process records
            for result in adapter.ingest(str(file_path)):
                # Sample memory periodically (every 10 batches to avoid overhead)
                if num_batches % 10 == 0:
                    current_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_samples.append(current_memory)
                
                # Track batch start time
                if batch_start is None:
                    batch_start = time.time()
                
                if isinstance(result.value, tuple):
                    # Handle tuple returns (redacted_df, raw_df) for raw vault
                    if adapter_type in ["csv", "json"]:
                        df, _ = result.value
                        if hasattr(df, '__len__'):
                            batch_records = len(df)
                            records_processed += batch_records
                            if result.is_success():
                                records_successful += batch_records
                            else:
                                records_failed += batch_records
                    elif adapter_type == "xml":
                        # XML returns (GoldenRecord, original_record_data)
                        records_processed += 1
                        if result.is_success():
                            records_successful += 1
                        else:
                            records_failed += 1
                elif hasattr(result.value, '__len__'):
                    # DataFrame result
                    batch_records = len(result.value)
                    records_processed += batch_records
                    if result.is_success():
                        records_successful += batch_records
                    else:
                        records_failed += batch_records
                else:
                    # Single record (GoldenRecord)
                    records_processed += 1
                    if result.is_success():
                        records_successful += 1
                    else:
                        records_failed += 1
                
                num_batches += 1
                
                # Track batch processing time
                batch_end = time.time()
                batch_times.append(batch_end - batch_start)
                batch_start = batch_end  # Next batch starts where this one ended
            
            # Flush redaction logs if storage is available (after all records processed)
            if storage and hasattr(storage, 'flush_redaction_logs'):
                try:
                    redaction_logs = redaction_logger.get_logs()
                    if redaction_logs:
                        flush_result = storage.flush_redaction_logs(redaction_logs)
                        if flush_result.is_success():
                            logger.debug(f"Flushed {flush_result.value} redaction logs to storage")
                        else:
                            logger.warning(f"Failed to flush redaction logs: {flush_result.error}")
                        redaction_logger.clear_logs()
                except Exception as e:
                    logger.warning(f"Failed to flush redaction logs: {e}")
    
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        records_failed += records_processed - records_successful
        
        # Try to flush logs even on error
        if storage and hasattr(storage, 'flush_redaction_logs'):
            try:
                redaction_logs = redaction_logger.get_logs()
                if redaction_logs:
                    storage.flush_redaction_logs(redaction_logs)
                    redaction_logger.clear_logs()
            except Exception:
                pass  # Ignore errors during error handling
    
    # End timing
    end_time = time.time()
    total_time = end_time - start_time
    cpu_end = process.cpu_percent()
    
    # Get memory statistics
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    peak_memory_mb = peak / (1024 * 1024)  # Convert to MB
    peak_memory_mb = max(peak_memory_mb, max(memory_samples) if memory_samples else 0)
    avg_memory_mb = mean(memory_samples) if memory_samples else initial_memory
    
    # Calculate metrics
    records_per_second = records_processed / total_time if total_time > 0 else 0
    mb_per_second = file_size_mb / total_time if total_time > 0 else 0
    memory_efficiency = avg_memory_mb / records_processed if records_processed > 0 else 0
    success_rate = records_successful / records_processed if records_processed > 0 else 0
    avg_batch_time = mean(batch_times) if batch_times else 0
    
    # Estimate number of records from file (if not already counted)
    if records_processed == 0:
        # Fallback: estimate from file size
        if adapter_type == "csv":
            records_processed = int(file_size_bytes / 650)  # ~650 bytes per CSV record
        elif adapter_type == "json":
            records_processed = int(file_size_bytes / 1500)  # ~1500 bytes per JSON record
        elif adapter_type == "xml":
            records_processed = int(file_size_bytes / 1050)  # ~1050 bytes per XML record
    
    return BenchmarkMetrics(
        adapter_type=adapter_type,
        file_size_mb=file_size_mb,
        file_path=str(file_path),
        num_records=records_processed,
        test_scenario="happy_path",
        total_time_seconds=total_time,
        processing_time_seconds=total_time,  # Same as total for now
        records_per_second=records_per_second,
        mb_per_second=mb_per_second,
        peak_memory_mb=peak_memory_mb,
        avg_memory_mb=avg_memory_mb,
        memory_efficiency_mb_per_record=memory_efficiency,
        num_batches=num_batches,
        batch_size=batch_size,
        avg_batch_time_seconds=avg_batch_time,
        records_successful=records_successful,
        records_failed=records_failed,
        success_rate=success_rate,
        cpu_usage_percent=(cpu_start + cpu_end) / 2 if cpu_start > 0 else 0,
        timestamp=datetime.now().isoformat()
    )


def run_benchmark_suite(
    test_files: Dict[str, Dict[float, Path]],
    output_csv: Path,
    batch_sizes: Optional[List[int]] = None,
    use_storage: bool = True,
    include_bad_data: bool = False,
    include_xml_performance: bool = False,
    failure_rates: List[float] = [25, 50, 75, 90]
) -> List[BenchmarkMetrics]:
    """Run complete benchmark suite and save results to CSV.
    
    Args:
        test_files: Dictionary of format -> size -> file path
        output_csv: Path to output CSV file
        batch_sizes: Optional list of batch sizes to test
        use_storage: If True, use storage adapter for persistence and log flushing
    
    Returns:
        List of all benchmark results
    """
    if batch_sizes is None:
        batch_sizes = [None]  # Use default batch size
    
    all_results = []
    
    # Initialize storage adapter if needed (for redaction log flushing)
    storage = None
    if use_storage:
        try:
            db_config = get_database_config()
            if db_config.db_type == "duckdb":
                storage = DuckDBAdapter(db_config=db_config)
            elif db_config.db_type == "postgresql":
                storage = PostgreSQLAdapter(db_config=db_config)
            
            # Initialize schema
            if storage:
                schema_result = storage.initialize_schema()
                if not schema_result.is_success():
                    logger.warning(f"Schema initialization failed: {schema_result.error}, continuing without storage")
                    storage = None
                else:
                    logger.info(f"Storage adapter initialized: {db_config.db_type}")
        except Exception as e:
            logger.warning(f"Failed to initialize storage adapter: {e}, continuing without storage")
            storage = None
    
    logger.info("=" * 80)
    logger.info("Starting Academic Benchmark Suite")
    logger.info("=" * 80)
    
    # Run happy path benchmarks
    for adapter_type, size_files in test_files.items():
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Testing {adapter_type.upper()} Adapter (Happy Path)")
        logger.info(f"{'=' * 80}")
        
        for size_mb, file_path in sorted(size_files.items()):
            if not file_path.exists():
                logger.warning(f"  File not found: {file_path}, skipping...")
                continue
            
            for batch_size in batch_sizes:
                try:
                    metrics = benchmark_ingestion(
                        file_path=file_path,
                        adapter_type=adapter_type,
                        batch_size=batch_size,
                        storage=storage
                    )
                    all_results.append(metrics)
                    
                    logger.info(f"  ✓ {size_mb}MB: {metrics.records_per_second:.0f} rec/s, "
                              f"{metrics.peak_memory_mb:.1f}MB peak, "
                              f"{metrics.total_time_seconds:.2f}s")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to benchmark {file_path}: {e}")
                    continue
    
    # Run bad data benchmarks
    if include_bad_data:
        logger.info(f"\n{'=' * 80}")
        logger.info("Testing Bad Data Scenarios (Circuit Breaker)")
        logger.info(f"{'=' * 80}")
        
        bad_data_dir = project_root / "test_data" / "bad_data"
        bad_data_dir.mkdir(parents=True, exist_ok=True)
        
        for adapter_type in ["csv", "json", "xml"]:
            for failure_rate in failure_rates:
                # Generate bad data file if it doesn't exist
                if adapter_type == "xml":
                    bad_file = bad_data_dir / f"bad_xml_{int(failure_rate)}pct.xml"
                    if not bad_file.exists():
                        logger.info(f"  Generating {bad_file.name}...")
                        generate_bad_xml_file(
                            output_path=bad_file,
                            num_records=1000,  # Use smaller files for bad data tests
                            failure_rate_percent=failure_rate
                        )
                elif adapter_type == "json":
                    bad_file = bad_data_dir / f"bad_json_{int(failure_rate)}pct.json"
                    if not bad_file.exists():
                        logger.info(f"  Generating {bad_file.name}...")
                        generate_bad_json_file(
                            output_path=bad_file,
                            num_records=1000,
                            failure_rate_percent=failure_rate
                        )
                else:  # csv
                    bad_file = bad_data_dir / f"bad_csv_{int(failure_rate)}pct.csv"
                    if not bad_file.exists():
                        logger.info(f"  Generating {bad_file.name}...")
                        generate_bad_csv_file(
                            output_path=bad_file,
                            num_records=1000,
                            failure_rate_percent=failure_rate
                        )
                
                if bad_file.exists():
                    try:
                        metrics = benchmark_bad_data(
                            file_path=bad_file,
                            adapter_type=adapter_type,
                            failure_rate_percent=failure_rate,
                            batch_size=None,
                            storage=storage
                        )
                        all_results.append(metrics)
                        
                        cb_status = "OPEN" if metrics.circuit_breaker_opened else "CLOSED"
                        logger.info(f"  ✓ {adapter_type.upper()} {int(failure_rate)}% failures: "
                                  f"CB={cb_status}, "
                                  f"failure_rate={metrics.circuit_breaker_failure_rate:.1f}%, "
                                  f"{metrics.records_per_second:.0f} rec/s")
                        
                    except Exception as e:
                        logger.error(f"  ✗ Failed to benchmark bad data {bad_file}: {e}")
                        continue
    
    # Run XML performance benchmarks (streaming vs non-streaming)
    if include_xml_performance and "xml" in test_files:
        logger.info(f"\n{'=' * 80}")
        logger.info("Testing XML Performance (Streaming vs Non-Streaming)")
        logger.info(f"{'=' * 80}")
        
        for size_mb, file_path in sorted(test_files["xml"].items()):
            if not file_path.exists():
                logger.warning(f"  File not found: {file_path}, skipping...")
                continue
            
            for streaming_enabled in [True, False]:
                try:
                    metrics = benchmark_xml_performance(
                        file_path=file_path,
                        streaming_enabled=streaming_enabled,
                        batch_size=None,
                        storage=storage
                    )
                    all_results.append(metrics)
                    
                    mode = "streaming" if streaming_enabled else "non-streaming"
                    logger.info(f"  ✓ {size_mb}MB XML ({mode}): "
                              f"{metrics.records_per_second:.0f} rec/s, "
                              f"{metrics.peak_memory_mb:.1f}MB peak, "
                              f"{metrics.total_time_seconds:.2f}s")
                    
                except Exception as e:
                    logger.error(f"  ✗ Failed to benchmark XML performance {file_path}: {e}")
                    continue
    
    # Write results to CSV
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Writing results to {output_csv}")
    logger.info(f"{'=' * 80}")
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        if not all_results:
            logger.warning("No results to write!")
            return all_results
        
        # Get all field names from dataclass
        fieldnames = list(all_results[0].__dict__.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in all_results:
            writer.writerow(asdict(result))
    
    logger.info(f"✓ Wrote {len(all_results)} benchmark results to {output_csv}")
    
    return all_results


def main():
    """Main entry point for academic benchmark suite."""
    parser = argparse.ArgumentParser(
        description="Academic Performance Benchmarking Suite for Data-Dialysis"
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=float,
        default=[1, 10, 50, 100, 250, 500],
        help="File sizes in MB (default: 1 10 50 100 250 500)"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["csv", "json", "xml"],
        default=["csv", "json", "xml"],
        help="File formats to test (default: csv json xml)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "test_data" / "benchmark",
        help="Directory for test files (default: test_data/benchmark)"
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=project_root / "benchmark_results.csv",
        help="Output CSV file for results (default: benchmark_results.csv)"
    )
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Batch sizes to test (default: use adapter defaults)"
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force regeneration of existing test files"
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip file generation, use existing files"
    )
    parser.add_argument(
        "--no-storage",
        action="store_true",
        help="Disable storage adapter (no persistence, no redaction log flushing)"
    )
    parser.add_argument(
        "--include-bad-data",
        action="store_true",
        help="Include bad data scenarios with circuit breaker tracking"
    )
    parser.add_argument(
        "--include-xml-performance",
        action="store_true",
        help="Include XML-specific performance benchmarks (streaming vs non-streaming)"
    )
    parser.add_argument(
        "--failure-rates",
        nargs="+",
        type=float,
        default=[25, 50, 75, 90],
        help="Failure rates for bad data tests (default: 25 50 75 90)"
    )
    
    args = parser.parse_args()
    
    # Generate test files
    if not args.skip_generation:
        logger.info("Generating test files...")
        test_files = generate_test_files(
            output_dir=args.output_dir,
            sizes_mb=args.sizes,
            formats=args.formats,
            force_regenerate=args.force_regenerate
        )
    else:
        logger.info("Skipping file generation, using existing files...")
        test_files = defaultdict(dict)
        for format_type in args.formats:
            for size_mb in args.sizes:
                if format_type == "csv":
                    file_path = args.output_dir / f"test_{format_type}_{int(size_mb)}mb_patients.csv"
                else:
                    file_path = args.output_dir / f"test_{format_type}_{int(size_mb)}mb.{format_type}"
                if file_path.exists():
                    test_files[format_type][size_mb] = file_path
    
    # Run benchmarks
    results = run_benchmark_suite(
        test_files=test_files,
        output_csv=args.output_csv,
        batch_sizes=args.batch_sizes,
        use_storage=not args.no_storage,
        include_bad_data=args.include_bad_data,
        include_xml_performance=args.include_xml_performance,
        failure_rates=args.failure_rates
    )
    
    # Print summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Benchmark Summary")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total runs: {len(results)}")
    logger.info(f"Results saved to: {args.output_csv}")
    logger.info(f"\nSuggested visualizations:")
    logger.info(f"  1. Throughput (records/sec) vs File Size by Format")
    logger.info(f"  2. Memory Efficiency (MB/record) vs File Size")
    logger.info(f"  3. Processing Time vs File Size (log scale)")
    logger.info(f"  4. Success Rate vs File Size")
    logger.info(f"  5. Batch Processing Efficiency Analysis")
    logger.info(f"  6. Format Comparison (CSV vs JSON vs XML)")
    logger.info(f"  7. Scalability Analysis (linearity check)")


if __name__ == "__main__":
    main()
