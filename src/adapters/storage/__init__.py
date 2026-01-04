"""Storage adapters for Data-Dialysis.

This module contains storage adapters that implement the StoragePort interface
for persisting validated GoldenRecords and maintaining audit trails.
"""

from src.adapters.storage.duckdb_adapter import DuckDBAdapter
from src.adapters.storage.postgresql_adapter import PostgreSQLAdapter

__all__ = ["DuckDBAdapter", "PostgreSQLAdapter"]

