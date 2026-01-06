"""Metrics aggregator service.

This service aggregates metrics from multiple data sources including
database tables, audit logs, and redaction logs.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.domain.ports import Result, StoragePort
from src.dashboard.models.metrics import (
    OverviewMetrics,
    IngestionMetrics,
    RecordMetrics,
    RedactionSummary,
    CircuitBreakerStatus
)
from src.dashboard.services.connection_helper import get_db_connection

logger = logging.getLogger(__name__)


class MetricsAggregator:
    """Aggregates metrics from multiple data sources."""
    
    def __init__(self, storage: StoragePort):
        """Initialize metrics aggregator.
        
        Parameters:
            storage: Storage adapter instance
        """
        self.storage = storage
    
    def _parse_time_range(self, time_range: str, end_time: datetime) -> datetime:
        """Parse time range string to start time.
        
        Parameters:
            time_range: Time range string (1h, 24h, 7d, 30d)
            end_time: End time (usually now)
            
        Returns:
            Start time based on time range
        """
        time_range = time_range.lower()
        
        if time_range.endswith('h'):
            hours = int(time_range[:-1])
            return end_time - timedelta(hours=hours)
        elif time_range.endswith('d'):
            days = int(time_range[:-1])
            return end_time - timedelta(days=days)
        else:
            # Default to 24h
            logger.warning(f"Unknown time range format: {time_range}, defaulting to 24h")
            return end_time - timedelta(hours=24)
    
    def get_overview_metrics(
        self,
        time_range: str = "24h"
    ) -> Result[OverviewMetrics]:
        """Get overview metrics for the specified time range.
        
        Parameters:
            time_range: Time range string (1h, 24h, 7d, 30d)
            
        Returns:
            Result[OverviewMetrics]: Overview metrics or error
        """
        try:
            end_time = datetime.now(timezone.utc).replace(tzinfo=None)
            start_time = self._parse_time_range(time_range, end_time)
            
            # Get record counts from database tables
            record_metrics = self._get_record_metrics(start_time, end_time)
            
            # Get redaction summary
            redaction_summary = self._get_redaction_summary(start_time, end_time)
            
            # Get ingestion counts (approximate from unique ingestion_ids)
            ingestion_metrics = self._get_ingestion_metrics(start_time, end_time)
            
            # Circuit breaker status (placeholder - would need to track this separately)
            circuit_breaker = None
            
            metrics = OverviewMetrics(
                time_range=time_range,
                ingestions=ingestion_metrics,
                records=record_metrics,
                redactions=redaction_summary,
                circuit_breaker=circuit_breaker
            )
            
            return Result.success_result(metrics)
            
        except Exception as e:
            logger.error(f"Failed to get overview metrics: {str(e)}", exc_info=True)
            return Result.failure_result(
                e,
                error_type="MetricsError"
            )
    
    def _get_record_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> RecordMetrics:
        """Get record processing metrics.
        
        Parameters:
            start_time: Start time for metrics
            end_time: End time for metrics
            
        Returns:
            RecordMetrics: Record processing statistics
        """
        try:
            # Query all tables to get record counts
            # This is an approximation - we count records in patients, encounters, observations
            if not hasattr(self.storage, '_get_connection'):
                return RecordMetrics(
                    total_processed=0,
                    total_successful=0,
                    total_failed=0
                )
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return RecordMetrics(
                        total_processed=0,
                        total_successful=0,
                        total_failed=0
                    )
                
                # Count records from all tables within time range
                total_successful = 0
                
                for table in ['patients', 'encounters', 'observations']:
                    try:
                        # Use parameterized query to prevent SQL injection
                        query = f"""
                            SELECT COUNT(*) as count
                            FROM {table}
                            WHERE ingestion_timestamp >= ? AND ingestion_timestamp <= ?
                        """
                        result = conn.execute(query, [start_time, end_time]).fetchone()
                        if result:
                            total_successful += result[0] if result[0] else 0
                    except Exception as e:
                        logger.debug(f"Could not query {table}: {str(e)}")
                        continue
                
                # For failed records, we'd need to track them separately
                # For now, estimate based on audit log errors
                total_failed = self._estimate_failed_records(start_time, end_time)
                
                return RecordMetrics(
                    total_processed=total_successful + total_failed,
                    total_successful=total_successful,
                    total_failed=total_failed
                )
            
        except Exception as e:
            logger.warning(f"Error getting record metrics: {str(e)}")
            return RecordMetrics(
                total_processed=0,
                total_successful=0,
                total_failed=0
            )
    
    def _estimate_failed_records(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """Estimate failed records from audit log.
        
        Parameters:
            start_time: Start time
            end_time: End time
            
        Returns:
            Estimated number of failed records
        """
        try:
            if not hasattr(self.storage, '_get_connection'):
                return 0
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return 0
                
                # Count validation errors and transformation errors from audit log
                query = """
                    SELECT COUNT(*) as count
                    FROM audit_log
                    WHERE event_timestamp >= ? AND event_timestamp <= ?
                    AND event_type IN ('VALIDATION_ERROR', 'TRANSFORMATION_ERROR')
                """
                result = conn.execute(query, [start_time, end_time]).fetchone()
                return result[0] if result and result[0] else 0
            
        except Exception as e:
            logger.debug(f"Could not estimate failed records: {str(e)}")
            return 0
    
    def _get_redaction_summary(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> RedactionSummary:
        """Get redaction summary from logs table.
        
        Parameters:
            start_time: Start time
            end_time: End time
            
        Returns:
            RedactionSummary: Redaction statistics
        """
        try:
            if not hasattr(self.storage, '_get_connection'):
                return RedactionSummary(total=0)
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return RedactionSummary(total=0)
                
                # Query logs table - get total count first
                total_query = """
                    SELECT COUNT(*) as total
                    FROM logs
                    WHERE timestamp >= ? AND timestamp <= ?
                """
                total_result = conn.execute(total_query, [start_time, end_time]).fetchone()
                total = total_result[0] if total_result and total_result[0] else 0
                
                # Query grouped by field
                field_query = """
                    SELECT field_name, COUNT(*) as count
                    FROM logs
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY field_name
                """
                field_results = conn.execute(field_query, [start_time, end_time]).fetchall()
                
                # Query grouped by rule
                rule_query = """
                    SELECT rule_triggered, COUNT(*) as count
                    FROM logs
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY rule_triggered
                """
                rule_results = conn.execute(rule_query, [start_time, end_time]).fetchall()
                
                # Query grouped by adapter
                adapter_query = """
                    SELECT source_adapter, COUNT(*) as count
                    FROM logs
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY source_adapter
                """
                adapter_results = conn.execute(adapter_query, [start_time, end_time]).fetchall()
                
                by_field = {}
                by_rule = {}
                by_adapter = {}
                
                # Process field results
                for row in field_results:
                    field = row[0] if row[0] else 'unknown'
                    count = row[1] if row[1] else 0
                    by_field[field] = count
                
                # Process rule results
                for row in rule_results:
                    rule = row[0] if row[0] else 'unknown'
                    count = row[1] if row[1] else 0
                    by_rule[rule] = count
                
                # Process adapter results
                for row in adapter_results:
                    adapter = row[0] if row[0] else 'unknown'
                    count = row[1] if row[1] else 0
                    by_adapter[adapter] = count
                
                return RedactionSummary(
                    total=total,
                    by_field=by_field,
                    by_rule=by_rule,
                    by_adapter=by_adapter
                )
            
        except Exception as e:
            logger.warning(f"Error getting redaction summary: {str(e)}")
            return RedactionSummary(total=0)
    
    def _get_ingestion_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> IngestionMetrics:
        """Get ingestion metrics from BULK_PERSISTENCE events.
        
        Each BULK_PERSISTENCE event represents a batch/ingestion run.
        This gives us the actual number of ingestion operations, not just unique ingestion IDs.
        
        Parameters:
            start_time: Start time
            end_time: End time
            
        Returns:
            IngestionMetrics: Ingestion statistics
        """
        try:
            if not hasattr(self.storage, '_get_connection'):
                return IngestionMetrics(total=0, successful=0, failed=0, success_rate=0.0)
            
            with get_db_connection(self.storage) as conn:
                if conn is None:
                    return IngestionMetrics(total=0, successful=0, failed=0, success_rate=0.0)
                
                # Count BULK_PERSISTENCE events from audit log
                # Each event represents a batch/ingestion operation
                query = """
                    SELECT COUNT(*) as count
                    FROM audit_log
                    WHERE event_type = 'BULK_PERSISTENCE'
                    AND event_timestamp >= ? AND event_timestamp <= ?
                """
                result = conn.execute(query, [start_time, end_time]).fetchone()
                total = result[0] if result and result[0] else 0
                
                # Count failed ingestions from audit log (events with ERROR/CRITICAL severity)
                failed_query = """
                    SELECT COUNT(*) as count
                    FROM audit_log
                    WHERE event_type = 'BULK_PERSISTENCE'
                    AND severity IN ('ERROR', 'CRITICAL')
                    AND event_timestamp >= ? AND event_timestamp <= ?
                """
                failed_result = conn.execute(failed_query, [start_time, end_time]).fetchone()
                failed = failed_result[0] if failed_result and failed_result[0] else 0
                
                successful = total - failed
                success_rate = (successful / total * 100.0) if total > 0 else 0.0
                
                return IngestionMetrics(
                    total=total,
                    successful=successful,
                    failed=failed,
                    success_rate=success_rate
                )
            
        except Exception as e:
            logger.warning(f"Error getting ingestion metrics: {str(e)}")
            return IngestionMetrics(total=0, successful=0, failed=0, success_rate=0.0)

