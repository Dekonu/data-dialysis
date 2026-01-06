"""Dependency injection for dashboard API.

This module provides dependency injection functions for FastAPI,
following Hexagonal Architecture principles by using existing storage adapters.
"""

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from src.domain.ports import StoragePort
from src.infrastructure.config_manager import get_database_config
from src.adapters.storage import DuckDBAdapter, PostgreSQLAdapter

logger = logging.getLogger(__name__)


@lru_cache()
def get_storage_adapter() -> StoragePort:
    """Get storage adapter instance (cached).
    
    This function retrieves the configured storage adapter (DuckDB or PostgreSQL)
    based on environment configuration. The result is cached to avoid recreating
    the adapter on every request.
    
    Returns:
        StoragePort: Configured storage adapter instance
        
    Raises:
        ValueError: If database type is unsupported
        
    Security Impact:
        - Uses existing secure configuration manager
        - No new attack surface (reuses existing adapters)
    """
    db_config = get_database_config()
    
    if db_config.db_type == "duckdb":
        logger.debug(f"Creating DuckDB adapter with path: {db_config.db_path or ':memory:'}")
        return DuckDBAdapter(db_config=db_config)
    elif db_config.db_type == "postgresql":
        logger.debug(f"Creating PostgreSQL adapter with host: {db_config.host}")
        return PostgreSQLAdapter(db_config=db_config)
    else:
        raise ValueError(f"Unsupported database type: {db_config.db_type}")


# Type alias for dependency injection
StorageDep = Annotated[StoragePort, Depends(get_storage_adapter)]

