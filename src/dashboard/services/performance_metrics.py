"""Performance metrics service.

This service provides performance-specific metrics including throughput,
latency, file processing, and memory usage.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.domain.ports import Result, StoragePort
from src.dashboard.models.metrics import (
    PerformanceMetrics,
    ThroughputMetrics,
    LatencyMetrics,
    FileProcessingMetrics,
    MemoryMetrics
)
from src.dashboard.services.connection_helper import get_db_connection

logger = logging.getLogger(__name__)


class PerformanceMetricsService:
    """Service for performance-specific metrics."""
    
    def __init__(self, storage: StoragePort):
        """Initialize performance metrics service.
        
        Parameters:
            storage: Storage adapter instance
        """
        self.storage = storage
    
    def _parse_time_range(self, time_range: str, end_time: datetime) -> datetime:
        """Parse time range string to start time."""
        time_range = time_range.lower()
        
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
            return end_time - timedelta(hours=hours)
        elif time_range.endswith('d'):
            days = int(time_range[:-1])
            return end_time - timedelta(days=days)
        else:
            logger.warning(f"Unknown time range format: {time_range}, defaulting to 24h")
            return end_time - timedelta(hours=24)
    
    def get_performance_metrics(
        self,
        time_range: str = "24h"
    ) -> Result[PerformanceMetrics]:
        """Get performance metrics for the specified time range.
        
        Parameters:
            time_range: Time range string (1h, 24h, 7d, 30d)
            
        Returns:
            Result[PerformanceMetrics]: Performance metrics or error
        """
        try:
            end_time = datetime.now(timezone.utc).replace(tzinfo=None)
            start_time = self._parse_time_range(time_range, end_time)
            
            # Get throughput metrics
            throughput = self._get_throughput_metrics(start_time, end_time)
            
            # Get latency metrics (placeholder - would need timing data)
            latency = self._get_latency_metrics(start_time, end_time)
            
            # Get file processing metrics
            file_processing = self._get_file_processing_metrics(start_time, end_time)
            
            # Get memory metrics (placeholder - would need memory tracking)
            memory = self._get_memory_metrics(start_time, end_time)
            
            metrics = PerformanceMetrics(
                time_range=time_range,
                throughput=throughput,
                latency=latency,
                file_processing=file_processing,
                memory=memory
            )
            
            return Result.success_result(metrics)
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {str(e)}", exc_info=True)
            return Result.failure_result(
                e,
                error_type="MetricsError"
            )
    
    def _get_throughput_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> ThroughputMetrics:
        """Get throughput metrics based on actual processing time from audit logs.
        
        Uses BULK_PERSISTENCE events from audit_log to calculate actual processing time,
        which is more accurate than using ingestion_timestamp from data tables.
        
        Parameters:
            start_time: Start time
            end_time: End time
            
        Returns:
            ThroughputMetrics: Throughput statistics
        """
        try:
            if not hasattr(self.storage, '_get_connection'):
                return ThroughputMetrics(records_per_second=0.0)
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return ThroughputMetrics(records_per_second=0.0)
                
                # Get BULK_PERSISTENCE events with their timestamps and row counts
                # This gives us actual processing events and their timing
                query = """
                    SELECT 
                        event_timestamp,
                        row_count,
                        details
                    FROM audit_log
                    WHERE event_type = 'BULK_PERSISTENCE'
                    AND event_timestamp >= ? AND event_timestamp <= ?
                    ORDER BY event_timestamp ASC
                """
                results = conn.execute(query, [start_time, end_time]).fetchall()
                
                if not results:
                    return ThroughputMetrics(records_per_second=0.0)
                
                # Calculate total records and sum of processing times from audit events
                # Use processing_time_seconds from details if available (more accurate)
                # Otherwise fall back to wall-clock time between batches
                total_records = 0
                total_processing_time = 0.0
                min_timestamp = None
                max_timestamp = None
                batch_count = 0
                peak_records_per_second = 0.0
                
                for row in results:
                    if isinstance(row, (list, tuple)):
                        event_time = row[0]
                        row_count = row[1] if len(row) > 1 else 0
                        details_json = row[2] if len(row) > 2 else None
                    else:
                        event_time = row.get('event_timestamp')
                        row_count = row.get('row_count', 0)
                        details_json = row.get('details')
                    
                    if event_time and row_count:
                        total_records += row_count
                        batch_count += 1
                        
                        # Try to get processing_time_seconds from details
                        processing_time = None
                        if details_json:
                            try:
                                if isinstance(details_json, str):
                                    details = json.loads(details_json)
                                else:
                                    details = details_json
                                
                                # Check for processing_time_seconds in details
                                processing_time = details.get('processing_time_seconds')
                            except (json.JSONDecodeError, AttributeError, TypeError):
                                pass
                        
                        # If we have processing time, use it; otherwise estimate from row count
                        if processing_time and processing_time > 0:
                            total_processing_time += processing_time
                            # Calculate throughput for this batch
                            batch_throughput = row_count / processing_time
                            if batch_throughput > peak_records_per_second:
                                peak_records_per_second = batch_throughput
                        else:
                            # Fallback: estimate processing time (will use wall-clock time later)
                            pass
                        
                        # Track min/max timestamps for fallback calculation
                        if min_timestamp is None or event_time < min_timestamp:
                            min_timestamp = event_time
                        if max_timestamp is None or event_time > max_timestamp:
                            max_timestamp = event_time
                
                # Calculate throughput based on actual processing time (not wall-clock time)
                if total_records > 0:
                    if total_processing_time > 0:
                        # Use sum of actual processing times (excludes delays between batches)
                        records_per_second = total_records / total_processing_time
                    elif min_timestamp and max_timestamp:
                        # Fallback: use wall-clock time between first and last batch
                        actual_time_delta = (max_timestamp - min_timestamp).total_seconds()
                        # Minimum 1 second to avoid division by zero
                        if actual_time_delta <= 0:
                            actual_time_delta = 1.0
                        records_per_second = total_records / actual_time_delta
                    else:
                        # Final fallback: use full time range
                        time_delta = (end_time - start_time).total_seconds()
                        if time_delta > 0:
                            records_per_second = total_records / time_delta
                        else:
                            records_per_second = 0.0
                else:
                    records_per_second = 0.0
                
                return ThroughputMetrics(
                    records_per_second=round(records_per_second, 2),
                    mb_per_second=None,
                    peak_records_per_second=round(peak_records_per_second, 2) if peak_records_per_second > 0 else None
                )
            
        except Exception as e:
            logger.warning(f"Error getting throughput metrics: {str(e)}")
            return ThroughputMetrics(records_per_second=0.0)
    
    def _get_latency_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> LatencyMetrics:
        """Get latency metrics from BULK_PERSISTENCE events.
        
        Calculates latency statistics from processing_time_ms stored in audit log details.
        
        Parameters:
            start_time: Start time
            end_time: End time
            
        Returns:
            LatencyMetrics: Latency statistics
        """
        try:
            if not hasattr(self.storage, '_get_connection'):
                return LatencyMetrics(
                    avg_processing_time_ms=None,
                    p50_ms=None,
                    p95_ms=None,
                    p99_ms=None
                )
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return LatencyMetrics(
                        avg_processing_time_ms=None,
                        p50_ms=None,
                        p95_ms=None,
                        p99_ms=None
                    )
                
                # Get BULK_PERSISTENCE events with processing times
                query = """
                    SELECT details
                    FROM audit_log
                    WHERE event_type = 'BULK_PERSISTENCE'
                    AND event_timestamp >= ? AND event_timestamp <= ?
                    AND details IS NOT NULL
                """
                results = conn.execute(query, [start_time, end_time]).fetchall()
                
                if not results:
                    return LatencyMetrics(
                        avg_processing_time_ms=None,
                        p50_ms=None,
                        p95_ms=None,
                        p99_ms=None
                    )
                
                # Extract processing times from details
                processing_times = []
                
                for row in results:
                    details_json = row[0] if isinstance(row, (list, tuple)) else row.get('details')
                    if details_json:
                        try:
                            if isinstance(details_json, str):
                                details = json.loads(details_json)
                            else:
                                details = details_json
                            
                            processing_time_ms = details.get('processing_time_ms')
                            if processing_time_ms is not None:
                                processing_times.append(float(processing_time_ms))
                        except (json.JSONDecodeError, AttributeError, TypeError, ValueError):
                            continue
                
                if not processing_times:
                    return LatencyMetrics(
                        avg_processing_time_ms=None,
                        p50_ms=None,
                        p95_ms=None,
                        p99_ms=None
                    )
                
                # Calculate statistics
                processing_times.sort()
                n = len(processing_times)
                
                avg_processing_time_ms = sum(processing_times) / n
                p50_ms = processing_times[n // 2] if n > 0 else None
                p95_ms = processing_times[int(n * 0.95)] if n > 1 else processing_times[-1] if n > 0 else None
                p99_ms = processing_times[int(n * 0.99)] if n > 1 else processing_times[-1] if n > 0 else None
                
                return LatencyMetrics(
                    avg_processing_time_ms=round(avg_processing_time_ms, 2),
                    p50_ms=round(p50_ms, 2) if p50_ms is not None else None,
                    p95_ms=round(p95_ms, 2) if p95_ms is not None else None,
                    p99_ms=round(p99_ms, 2) if p99_ms is not None else None
                )
            
        except Exception as e:
            logger.warning(f"Error getting latency metrics: {str(e)}")
            return LatencyMetrics(
                avg_processing_time_ms=None,
                p50_ms=None,
                p95_ms=None,
                p99_ms=None
            )
    
    def _get_file_processing_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> FileProcessingMetrics:
        """Get file processing metrics.
        
        Parameters:
            start_time: Start time
            end_time: End time
            
        Returns:
            FileProcessingMetrics: File processing statistics
        """
        try:
            if not hasattr(self.storage, '_get_connection'):
                return FileProcessingMetrics(total_files=0)
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return FileProcessingMetrics(total_files=0)
                
                # Count BULK_PERSISTENCE events from audit log
                # Each BULK_PERSISTENCE event represents a batch/ingestion operation
                # This is more accurate than counting distinct ingestion_id since
                # multiple batches can share the same ingestion_id
                total_files = 0
                
                try:
                    query = """
                        SELECT COUNT(*) as count
                        FROM audit_log
                        WHERE event_type = 'BULK_PERSISTENCE'
                        AND event_timestamp >= ? AND event_timestamp <= ?
                    """
                    result = conn.execute(query, [start_time, end_time]).fetchone()
                    if result and result[0]:
                        total_files = result[0]
                except Exception as e:
                    logger.debug(f"Could not query audit_log for BULK_PERSISTENCE: {str(e)}")
                    
                    # Fallback: Count distinct ingestion_id from redaction logs
                    try:
                        query = """
                            SELECT COUNT(DISTINCT ingestion_id) as count
                            FROM logs
                            WHERE timestamp >= ? AND timestamp <= ?
                            AND ingestion_id IS NOT NULL
                        """
                        result = conn.execute(query, [start_time, end_time]).fetchone()
                        if result and result[0]:
                            total_files = result[0]
                    except Exception as e2:
                        logger.debug(f"Could not query logs table for ingestion_id: {str(e2)}")
                
                # File size metrics would need to be tracked separately
                return FileProcessingMetrics(
                    total_files=total_files,
                    avg_file_size_mb=None,
                    total_data_processed_mb=None
                )
            
        except Exception as e:
            logger.warning(f"Error getting file processing metrics: {str(e)}")
            return FileProcessingMetrics(total_files=0)
    
    def _get_memory_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> MemoryMetrics:
        """Get memory usage metrics.
        
        Note: Memory metrics would require memory tracking during ingestion.
        This is a placeholder implementation.
        
        Parameters:
            start_time: Start time
            end_time: End time
            
        Returns:
            MemoryMetrics: Memory usage statistics
        """
        # Memory metrics would need to be tracked during ingestion
        # For now, return placeholder values
        return MemoryMetrics(
            avg_peak_memory_mb=None,
            max_peak_memory_mb=None
        )

