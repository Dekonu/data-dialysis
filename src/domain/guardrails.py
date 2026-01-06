"""Domain Guardrails - Circuit Breaker and Failure Monitoring.

This module provides guardrails to protect the ingestion pipeline from
cascading failures and data quality issues. The CircuitBreaker monitors
failure rates and can abort batches when quality thresholds are exceeded.

Security Impact:
    - Prevents processing of low-quality data sources
    - Reduces log noise from repeated failures
    - Enables early detection of data corruption or attacks
    - Provides configurable thresholds for different use cases

Architecture:
    - Pure domain logic with no infrastructure dependencies
    - Works with Result type from ports to monitor success/failure
    - Configurable thresholds for different data sources
    - Thread-safe design for concurrent ingestion
"""

import logging
from typing import Optional, Union
from dataclasses import dataclass, field
from threading import Lock

import pandas as pd

from src.domain.ports import Result, GoldenRecord

logger = logging.getLogger(__name__)


@dataclass
class CircuitBreakerConfig:
    """Configuration for CircuitBreaker behavior.
    
    Attributes:
        failure_threshold_percent: Percentage of failures that triggers circuit open (0-100)
        window_size: Number of records to evaluate in the sliding window
        min_records_before_check: Minimum records processed before checking threshold
        abort_on_open: If True, raise CircuitBreakerOpenError when threshold exceeded
                      If False, only log warnings and continue
    """
    failure_threshold_percent: float = 50.0
    window_size: int = 100
    min_records_before_check: int = 10
    abort_on_open: bool = True


class CircuitBreakerOpenError(Exception):
    """Raised when CircuitBreaker opens due to excessive failures.
    
    This exception indicates that the failure rate has exceeded the
    configured threshold and the batch should be aborted.
    
    Attributes:
        failure_rate: The calculated failure rate percentage
        threshold: The configured threshold that was exceeded
        records_processed: Number of records processed when circuit opened
        failures: Number of failures when circuit opened
    """
    
    def __init__(
        self,
        message: str,
        failure_rate: float,
        threshold: float,
        records_processed: int,
        failures: int
    ):
        super().__init__(message)
        self.failure_rate = failure_rate
        self.threshold = threshold
        self.records_processed = records_processed
        self.failures = failures


class CircuitBreaker:
    """Circuit Breaker for monitoring ingestion failure rates.
    
    This class monitors the success/failure rate of ingestion operations
    and can abort batches when quality thresholds are exceeded. It uses
    a sliding window approach to track recent failures.
    
    Key Features:
        - Sliding window: Only considers recent N records
        - Configurable thresholds: Adjustable failure percentage and window size
        - Thread-safe: Uses locks for concurrent access
        - Early abort: Can stop processing when threshold exceeded
    
    Security Impact:
        - Prevents processing of corrupted or malicious data sources
        - Reduces log noise from repeated failures
        - Enables fail-fast behavior for low-quality data
    
    Example Usage:
        ```python
        config = CircuitBreakerConfig(
            failure_threshold_percent=50.0,
            window_size=100,
            min_records_before_check=10,
            abort_on_open=True
        )
        breaker = CircuitBreaker(config)
        
        for result in adapter.ingest(source):
            breaker.record_result(result)
            
            if breaker.is_open():
                raise CircuitBreakerOpenError(...)
            
            if result.is_success():
                process(result.value)
        ```
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """Initialize CircuitBreaker.
        
        Parameters:
            config: CircuitBreaker configuration (uses defaults if None)
        """
        self.config = config or CircuitBreakerConfig()
        self._results: list[bool] = []  # True for success, False for failure
        self._lock = Lock()
        self._is_open = False
        self._total_processed = 0
        self._total_failures = 0
    
    def record_result(self, result: Result[Union[GoldenRecord, pd.DataFrame]]) -> None:
        """Record a result and check if circuit should open.
        
        This method:
        1. Records the result (success or failure)
        2. For DataFrames: Counts rows as individual records
        3. Maintains sliding window of recent results
        4. Checks if failure threshold is exceeded
        5. Opens circuit if threshold exceeded
        
        Parameters:
            result: Result from ingestion operation (can contain GoldenRecord or DataFrame)
        
        Raises:
            CircuitBreakerOpenError: If abort_on_open=True and threshold exceeded
        """
        with self._lock:
            # Record result
            is_success = result.is_success()
            
            # Handle DataFrame chunks (count rows as individual records)
            if is_success and isinstance(result.value, pd.DataFrame):
                num_rows = len(result.value)
                # Add one entry per row in the DataFrame
                for _ in range(num_rows):
                    self._results.append(True)
                    self._total_processed += 1
            elif not is_success:
                # Failure: Check if error_details contains chunk info
                chunk_size = result.error_details.get('chunk_size', 1) if result.error_details else 1
                # For chunk failures, count all rows in chunk as failures
                for _ in range(chunk_size):
                    self._results.append(False)
                    self._total_processed += 1
                    self._total_failures += 1
            else:
                # Single GoldenRecord success
                self._results.append(True)
                self._total_processed += 1
            
            # Maintain sliding window (remove oldest entries if window exceeds size)
            while len(self._results) > self.config.window_size:
                removed = self._results.pop(0)
                # Note: We don't adjust _total_failures here because it tracks
                # total failures ever seen, not just failures in the window.
                # failures_in_window is calculated from the window itself.
            
            # Check threshold (only after minimum records processed)
            if self._total_processed >= self.config.min_records_before_check:
                self._check_threshold()
    
    def _check_threshold(self) -> None:
        """Check if failure threshold is exceeded and open circuit if needed.
        
        This method calculates the failure rate in the sliding window
        and opens the circuit if the threshold is exceeded.
        """
        if len(self._results) == 0:
            return
        
        # Calculate failure rate in current window
        failures_in_window = sum(1 for r in self._results if not r)
        total_in_window = len(self._results)
        failure_rate = (failures_in_window / total_in_window) * 100.0
        
        # Check if threshold exceeded
        if failure_rate >= self.config.failure_threshold_percent:
            if not self._is_open:
                self._is_open = True
                
                logger.error(
                    f"CircuitBreaker OPEN: Failure rate {failure_rate:.1f}% "
                    f"exceeds threshold {self.config.failure_threshold_percent}% "
                    f"(failures: {failures_in_window}/{total_in_window} in window, "
                    f"total: {self._total_failures}/{self._total_processed})"
                )
                
                if self.config.abort_on_open:
                    raise CircuitBreakerOpenError(
                        f"CircuitBreaker opened: {failure_rate:.1f}% failure rate "
                        f"exceeds threshold {self.config.failure_threshold_percent}%",
                        failure_rate=failure_rate,
                        threshold=self.config.failure_threshold_percent,
                        records_processed=self._total_processed,
                        failures=self._total_failures
                    )
        else:
            # Reset if we're back below threshold
            if self._is_open:
                self._is_open = False
                logger.info(
                    f"CircuitBreaker CLOSED: Failure rate {failure_rate:.1f}% "
                    f"is below threshold {self.config.failure_threshold_percent}%"
                )
    
    def is_open(self) -> bool:
        """Check if circuit breaker is currently open.
        
        Returns:
            bool: True if circuit is open (threshold exceeded), False otherwise
        """
        with self._lock:
            return self._is_open
    
    def reset(self) -> None:
        """Reset the circuit breaker to initial state.
        
        This clears all recorded results and resets counters.
        Useful for starting a new batch or after resolving data quality issues.
        """
        with self._lock:
            self._results.clear()
            self._is_open = False
            self._total_processed = 0
            self._total_failures = 0
            logger.info("CircuitBreaker reset")
    
    def get_statistics(self) -> dict:
        """Get current statistics about the circuit breaker.
        
        Returns:
            dict: Statistics including:
                - is_open: Whether circuit is currently open
                - total_processed: Total records processed
                - total_failures: Total failures recorded
                - window_size: Current window size (from config)
                - records_in_window: Number of records in current sliding window
                - failures_in_window: Failures in current window
                - failure_rate: Current failure rate percentage
                - threshold: Configured threshold percentage
                - min_records_before_check: Minimum records before checking threshold
        """
        with self._lock:
            failures_in_window = sum(1 for r in self._results if not r)
            total_in_window = len(self._results)
            failure_rate = (failures_in_window / total_in_window * 100.0) if total_in_window > 0 else 0.0
            
            return {
                'is_open': self._is_open,
                'total_processed': self._total_processed,
                'total_failures': self._total_failures,
                'window_size': self.config.window_size,
                'records_in_window': total_in_window,
                'failures_in_window': failures_in_window,
                'failure_rate': failure_rate,
                'threshold': self.config.failure_threshold_percent,
                'min_records_before_check': self.config.min_records_before_check,
            }

