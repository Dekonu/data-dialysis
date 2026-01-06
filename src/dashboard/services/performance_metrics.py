"""Performance metrics service.

This service provides performance-specific metrics including throughput,
latency, file processing, and memory usage.
"""

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
                
                # Calculate total records and time span from audit events
                total_records = 0
                min_timestamp = None
                max_timestamp = None
                
                for row in results:
                    if isinstance(row, (list, tuple)):
                        event_time = row[0]
                        row_count = row[1] if len(row) > 1 else 0
                    else:
                        event_time = row.get('event_timestamp')
                        row_count = row.get('row_count', 0)
                    
                    if event_time and row_count:
                        total_records += row_count
                        
                        # Track min/max timestamps from actual processing events
                        if min_timestamp is None or event_time < min_timestamp:
                            min_timestamp = event_time
                        if max_timestamp is None or event_time > max_timestamp:
                            max_timestamp = event_time
                
                # Calculate throughput based on actual processing time span
                if total_records > 0 and min_timestamp and max_timestamp:
                    # Use actual time span between first and last batch
                    actual_time_delta = (max_timestamp - min_timestamp).total_seconds()
                    # Minimum 1 second to avoid division by zero
                    if actual_time_delta <= 0:
                        actual_time_delta = 1.0
                    records_per_second = total_records / actual_time_delta
                elif total_records > 0:
                    # Fallback: if we can't determine time span, use full range
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
                    peak_records_per_second=None
                )
            
        except Exception as e:
            logger.warning(f"Error getting throughput metrics: {str(e)}")
            return ThroughputMetrics(records_per_second=0.0)
    
    def _get_latency_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> LatencyMetrics:
        """Get latency metrics.
        
        Note: Latency metrics would require timing data to be stored.
        This is a placeholder implementation.
        
        Parameters:
            start_time: Start time
            end_time: End time
            
        Returns:
            LatencyMetrics: Latency statistics
        """
        # Latency metrics would need to be tracked during ingestion
        # For now, return placeholder values
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

