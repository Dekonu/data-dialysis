"""Circuit breaker service for dashboard API.

This service provides methods to query circuit breaker status.
Note: Circuit breaker state is currently in-memory, so this service
provides a way to expose status information. In a production system,
you might want to persist circuit breaker state or use a distributed
circuit breaker implementation.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from src.domain.guardrails import CircuitBreaker, CircuitBreakerConfig
from src.domain.ports import Result, StoragePort
from src.dashboard.models.circuit_breaker import CircuitBreakerStatus

logger = logging.getLogger(__name__)


class CircuitBreakerService:
    """Service for querying circuit breaker status."""
    
    def __init__(self, storage: StoragePort, circuit_breaker: Optional[CircuitBreaker] = None):
        """Initialize CircuitBreakerService.
        
        Parameters:
            storage: Storage adapter instance
            circuit_breaker: Optional CircuitBreaker instance (for testing or if persisted)
        """
        self.storage = storage
        self._circuit_breaker = circuit_breaker
    
    def get_status(self) -> Result[CircuitBreakerStatus]:
        """Get circuit breaker status.
        
        Note: Since CircuitBreaker is currently in-memory and created per ingestion,
        this method returns a default status. In a production system, you would:
        1. Persist circuit breaker state to database
        2. Use a distributed circuit breaker (e.g., Redis-based)
        3. Query the persisted state
        
        For now, we return a default "closed" status with default configuration.
        
        Returns:
            Result containing CircuitBreakerStatus or error
        """
        try:
            # If we have a circuit breaker instance, use it
            if self._circuit_breaker:
                # Use public method to get statistics (proper encapsulation)
                stats = self._circuit_breaker.get_statistics()
                
                return Result.success_result(CircuitBreakerStatus(
                    is_open=stats['is_open'],
                    failure_rate=stats['failure_rate'],
                    threshold=stats['threshold'],
                    total_processed=stats['total_processed'],
                    total_failures=stats['total_failures'],
                    window_size=stats['window_size'],
                    failures_in_window=stats['failures_in_window'],
                    records_in_window=stats['records_in_window'],
                    min_records_before_check=stats['min_records_before_check']
                ))
            
            # Default status (circuit breaker not available)
            # In a real system, you might query the database for recent failure rates
            # or maintain a global circuit breaker instance
            
            # Try to calculate failure rate from recent audit logs
            init_result = self.storage.initialize_schema()
            if not init_result.is_success():
                # Return default closed status
                default_config = CircuitBreakerConfig()
                return Result.success_result(CircuitBreakerStatus(
                    is_open=False,
                    failure_rate=0.0,
                    threshold=default_config.failure_threshold_percent,
                    total_processed=0,
                    total_failures=0,
                    window_size=default_config.window_size,
                    failures_in_window=0,
                    records_in_window=0,
                    min_records_before_check=default_config.min_records_before_check
                ))
            
            # Get connection
            if not hasattr(self.storage, '_get_connection'):
                # Return default closed status
                default_config = CircuitBreakerConfig()
                return Result.success_result(CircuitBreakerStatus(
                    is_open=False,
                    failure_rate=0.0,
                    threshold=default_config.failure_threshold_percent,
                    total_processed=0,
                    total_failures=0,
                    window_size=default_config.window_size,
                    failures_in_window=0,
                    records_in_window=0,
                    min_records_before_check=default_config.min_records_before_check
                ))
            
            conn = self.storage._get_connection()
            
            # Query recent errors from audit log
            # Use parameterized query - adjust for DuckDB vs PostgreSQL
            one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
            
            try:
                error_query = """
                    SELECT COUNT(*) as error_count
                    FROM audit_log
                    WHERE severity IN ('ERROR', 'CRITICAL')
                    AND event_timestamp >= ?
                """
                
                error_result = conn.execute(error_query, [one_hour_ago]).fetchone()
                error_count = error_result[0] if error_result else 0
                
                # Query total events
                total_query = """
                    SELECT COUNT(*) as total_count
                    FROM audit_log
                    WHERE event_timestamp >= ?
                """
                
                total_result = conn.execute(total_query, [one_hour_ago]).fetchone()
                total_count = total_result[0] if total_result else 0
            except Exception:
                # If query fails, return default closed status
                error_count = 0
                total_count = 0
            
            # Calculate failure rate
            failure_rate = (error_count / total_count * 100.0) if total_count > 0 else 0.0
            
            # Use default config
            default_config = CircuitBreakerConfig()
            is_open = failure_rate >= default_config.failure_threshold_percent
            
            return Result.success_result(CircuitBreakerStatus(
                is_open=is_open,
                failure_rate=failure_rate,
                threshold=default_config.failure_threshold_percent,
                total_processed=total_count,
                total_failures=error_count,
                window_size=default_config.window_size,
                failures_in_window=error_count,
                records_in_window=total_count,
                min_records_before_check=default_config.min_records_before_check
            ))
            
        except Exception as e:
            error_msg = f"Failed to get circuit breaker status: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return Result.failure_result(
                Exception(error_msg),
                error_type="CircuitBreakerServiceError"
            )

