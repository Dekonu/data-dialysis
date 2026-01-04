"""Application Settings and Configuration.

This module provides application-wide settings that combine configuration
from the configuration manager with application-specific defaults.

Security Impact:
    - Settings are loaded from secure configuration sources
    - Sensitive values are never logged
    - Defaults are provided for development convenience
"""

import os
from pathlib import Path
from typing import Optional

from src.infrastructure.config_manager import ConfigManager, get_database_config, DatabaseConfig

# Application metadata
APP_NAME = "Data-Dialysis"
APP_VERSION = "1.0.0"

# Default batch size for processing (can be overridden via environment)
DEFAULT_BATCH_SIZE = 10000

# Default chunk size for CSV/JSON ingestion
DEFAULT_CHUNK_SIZE = 10000

# Default max record size (10MB)
DEFAULT_MAX_RECORD_SIZE = 10 * 1024 * 1024


class Settings:
    """Application settings loaded from configuration manager and environment.
    
    This class provides a unified interface for accessing application settings,
    combining values from the configuration manager with environment variables
    and sensible defaults.
    
    Security Impact:
        - Database credentials are managed securely via DatabaseConfig
        - Settings are validated before use
        - Sensitive values are never exposed in logs
    """
    
    def __init__(self):
        """Initialize settings from configuration manager and environment."""
        # Load database configuration
        self._db_config: Optional[DatabaseConfig] = None
        self._config_manager: Optional[ConfigManager] = None
        
        # Application settings from environment
        self.app_name = os.getenv("DD_APP_NAME", APP_NAME)
        self.batch_size = int(os.getenv("DD_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
        self.chunk_size = int(os.getenv("DD_CHUNK_SIZE", str(DEFAULT_CHUNK_SIZE)))
        self.max_record_size = int(os.getenv("DD_MAX_RECORD_SIZE", str(DEFAULT_MAX_RECORD_SIZE)))
        
        # Logging level
        self.log_level = os.getenv("DD_LOG_LEVEL", "INFO")
        
        # Circuit breaker settings
        self.circuit_breaker_enabled = os.getenv("DD_CIRCUIT_BREAKER_ENABLED", "true").lower() == "true"
        self.circuit_breaker_threshold = float(os.getenv("DD_CIRCUIT_BREAKER_THRESHOLD", "0.1"))
        self.circuit_breaker_min_requests = int(os.getenv("DD_CIRCUIT_BREAKER_MIN_REQUESTS", "10"))
        
        # Security report settings
        self.save_security_report = os.getenv("DD_SAVE_SECURITY_REPORT", "true").lower() == "true"
        self.security_report_dir = os.getenv("DD_SECURITY_REPORT_DIR", "reports")
        
        # XML streaming settings
        self.xml_streaming_enabled = os.getenv("DD_XML_STREAMING_ENABLED", "true").lower() == "true"
        self.xml_streaming_threshold = int(os.getenv("DD_XML_STREAMING_THRESHOLD", str(100 * 1024 * 1024)))  # 100MB
        self.xml_max_events = int(os.getenv("DD_XML_MAX_EVENTS", "1000000"))
        self.xml_max_depth = int(os.getenv("DD_XML_MAX_DEPTH", "100"))
    
    @property
    def db_config(self) -> DatabaseConfig:
        """Get database configuration.
        
        Returns:
            DatabaseConfig instance loaded from configuration manager
        
        Security Impact:
            - Configuration is loaded lazily on first access
            - Credentials are managed securely
        """
        if self._db_config is None:
            self._db_config = get_database_config()
        return self._db_config
    
    @property
    def config_manager(self) -> ConfigManager:
        """Get configuration manager instance.
        
        Returns:
            ConfigManager instance
        """
        if self._config_manager is None:
            self._config_manager = ConfigManager.from_environment()
        return self._config_manager
    
    def get_db_path(self) -> str:
        """Get database path for DuckDB.
        
        Returns:
            Database path or ':memory:' for in-memory database
        """
        if self.db_config.db_type == "duckdb":
            return self.db_config.db_path or ":memory:"
        raise ValueError(f"Database type '{self.db_config.db_type}' does not use db_path")
    
    def get_connection_string(self) -> str:
        """Get database connection string.
        
        Returns:
            Connection string for the configured database type
        
        Security Impact:
            - Password is retrieved from SecretStr but not logged
        """
        return self.db_config.get_connection_string()


# Global settings instance
settings = Settings()

