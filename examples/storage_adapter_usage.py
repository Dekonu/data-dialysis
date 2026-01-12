"""Example usage of storage adapters with configuration manager.

This example demonstrates how to use the DuckDB and PostgreSQL adapters
with the configuration manager for secure credential handling.
"""

import os
from src.infrastructure.config_manager import ConfigManager, get_database_config
from src.adapters.storage.duckdb_adapter import DuckDBAdapter
from src.adapters.storage.postgresql_adapter import PostgreSQLAdapter
from src.domain.golden_record import GoldenRecord, PatientRecord


def example_duckdb_with_config():
    """Example: Using DuckDB adapter with configuration manager."""
    print("=== DuckDB Adapter with Configuration Manager ===")
    
    # Method 1: Load from environment variables
    db_config = get_database_config()
    
    # Create adapter using DatabaseConfig
    adapter = DuckDBAdapter(db_config=db_config)
    
    # Initialize schema
    result = adapter.initialize_schema()
    if result.is_success():
        print("SUCCESS: Schema initialized successfully")
    else:
        print(f"FAILED: Schema initialization failed: {result.error}")
        return
    
    # Example: Create a test record (in real usage, this would come from ingestion)
    patient = PatientRecord(patient_id="TEST001")
    golden_record = GoldenRecord(
        patient=patient,
        source_adapter="example"
    )
    
    # Persist record
    result = adapter.persist(golden_record)
    if result.is_success():
        print(f"SUCCESS: Record persisted with ID: {result.value}")
    else:
        print(f"FAILED: Persistence failed: {result.error}")
    
    adapter.close()


def example_postgresql_with_config():
    """Example: Using PostgreSQL adapter with configuration manager."""
    print("\n=== PostgreSQL Adapter with Configuration Manager ===")
    
    # Set environment variables (in production, these would come from secrets manager)
    os.environ["DD_DB_TYPE"] = "postgresql"
    os.environ["DD_DB_HOST"] = "localhost"
    os.environ["DD_DB_PORT"] = "5432"
    os.environ["DD_DB_NAME"] = "clinical_db"
    os.environ["DD_DB_USER"] = "admin"
    os.environ["DD_DB_PASSWORD"] = "secret_password"
    os.environ["DD_DB_SSL_MODE"] = "prefer"
    
    # Load configuration
    config = ConfigManager.from_environment()
    db_config = config.get_database_config()
    
    # Create adapter using DatabaseConfig
    adapter = PostgreSQLAdapter(db_config=db_config)
    
    # Initialize schema
    result = adapter.initialize_schema()
    if result.is_success():
        print("SUCCESS: Schema initialized successfully")
    else:
        print(f"FAILED: Schema initialization failed: {result.error}")
        return
    
    # Example: Create a test record
    patient = PatientRecord(patient_id="TEST002")
    golden_record = GoldenRecord(
        patient=patient,
        source_adapter="example"
    )
    
    # Persist record
    result = adapter.persist(golden_record)
    if result.is_success():
        print(f"SUCCESS: Record persisted with ID: {result.value}")
    else:
        print(f"FAILED: Persistence failed: {result.error}")
    
    adapter.close()


def example_config_from_file():
    """Example: Loading configuration from a JSON file."""
    print("\n=== Configuration from File ===")
    
    # Create a config file (in real usage, this would exist)
    import json
    config_data = {
        "database": {
            "db_type": "duckdb",
            "db_path": "data/clinical.duckdb"
        }
    }
    
    # Save to file (for demonstration)
    with open("config.example.json", "w") as f:
        json.dump(config_data, f, indent=2)
    
    # Load from file
    config = ConfigManager.from_file("config.example.json")
    db_config = config.get_database_config()
    
    # Create adapter
    adapter = DuckDBAdapter(db_config=db_config)
    print(f"SUCCESS: Adapter created with database path: {db_config.db_path}")
    
    # Clean up
    import os
    if os.path.exists("config.example.json"):
        os.remove("config.example.json")


def example_backward_compatibility():
    """Example: Using adapters with direct parameters (backward compatibility)."""
    print("\n=== Backward Compatibility Mode ===")
    
    # DuckDB: Can still use db_path directly
    adapter = DuckDBAdapter(db_path=":memory:")
    print("SUCCESS: DuckDB adapter created with direct db_path parameter")
    adapter.close()
    
    # PostgreSQL: Can still use connection string or individual parameters
    # (Note: In production, prefer using DatabaseConfig)
    adapter = PostgreSQLAdapter(
        host="localhost",
        database="test_db",
        username="user",
        password="pass"
    )
    print("SUCCESS: PostgreSQL adapter created with direct parameters")
    adapter.close()


if __name__ == "__main__":
    # Run examples
    example_duckdb_with_config()
    example_postgresql_with_config()
    example_config_from_file()
    example_backward_compatibility()
    
    print("\n=== All examples completed ===")

