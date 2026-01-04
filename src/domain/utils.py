"""Domain Utilities - Helper functions for performance optimization.

This module provides utility functions for determining optimal processing strategies
based on dataset characteristics and system resources.

Security Impact:
    - No security impact - pure utility functions
"""

import os
from typing import Optional

import pandas as pd

# Optional pandarallel import
try:
    from pandarallel import pandarallel
    PANDARALLEL_AVAILABLE = True
except ImportError:
    PANDARALLEL_AVAILABLE = False
    pandarallel = None


def should_use_parallel(df: Optional['pd.DataFrame'] = None, row_count: Optional[int] = None) -> bool:
    """Determine if parallel processing (pandarallel) should be used.
    
    Decision logic:
    - Use parallel if: row_count > 50,000 AND pandarallel available AND sufficient CPU cores
    - Use vectorized (default) if: row_count < 10,000 OR pandarallel not available
    - Use pandas.apply if: 10,000 <= row_count <= 50,000 (medium size)
    
    Parameters:
        df: DataFrame to analyze (if provided, row_count is derived from it)
        row_count: Number of rows (if df not provided)
    
    Returns:
        bool: True if parallel processing should be used, False otherwise
    
    Note:
        Vectorized operations (pandas string methods) are typically fastest.
        Parallel processing is only beneficial for very large datasets (>50K rows)
        with complex operations that can't be fully vectorized.
    """
    # Pandas is always available
    
    # Get row count
    if df is not None:
        row_count = len(df)
    elif row_count is None:
        return False
    
    # Check if pandarallel is available
    if not PANDARALLEL_AVAILABLE:
        return False
    
    # Check CPU cores (pandarallel needs multiple cores to be beneficial)
    try:
        cpu_count = os.cpu_count() or 1
        if cpu_count < 4:
            return False  # Not enough cores for parallel processing
    except:
        return False
    
    # Decision threshold: > 50,000 rows
    # Below this, vectorized operations are typically faster
    # Above this, parallel processing may help for complex operations
    return row_count > 50000


def initialize_pandarallel_if_needed(progress_bar: bool = False) -> bool:
    """Initialize pandarallel if available and needed.
    
    Parameters:
        progress_bar: Whether to show progress bar during parallel operations
    
    Returns:
        bool: True if pandarallel was initialized, False otherwise
    
    Note:
        This should be called once at application startup if you plan to use
        parallel processing. It has some overhead, so only initialize if needed.
    """
    if not PANDARALLEL_AVAILABLE:
        return False
    
    try:
        pandarallel.initialize(progress_bar=progress_bar, verbose=0)
        return True
    except Exception:
        return False


# TODO: Future enhancement - Add memory-based chunk size calculation
# def calculate_optimal_chunk_size(
#     available_memory_mb: Optional[int] = None,
#     avg_row_size_bytes: Optional[int] = None
# ) -> int:
#     """Calculate optimal chunk size based on available memory.
#     
#     This would help prevent memory exhaustion when processing very large files.
#     Could use psutil to detect available memory dynamically.
#     
#     Parameters:
#         available_memory_mb: Available memory in MB (if None, auto-detect)
#         avg_row_size_bytes: Average size of a row in bytes
#     
#     Returns:
#         int: Optimal chunk size in rows
#     """
#     pass

