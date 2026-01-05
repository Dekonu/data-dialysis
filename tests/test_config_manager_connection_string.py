"""Tests for connection string parsing and bidirectional sync in config manager.

These tests verify that:
1. Connection strings can be parsed to populate individual fields
2. Individual fields can be used to construct connection strings
3. Both directions work correctly with environment variables
"""

import pytest
import os
from unittest.mock import patch

from src.infrastructure.config_manager import (
    DatabaseConfig,
    ConfigManager,
    get_database_config,
)


class TestConnectionStringParsing:
    """Test parsing connection strings into individual fields."""
    
    def test_parse_postgresql_connection_string_basic(self):
        """Test parsing a basic PostgreSQL connection string."""
        conn_str = "postgresql://user:pass@localhost:5432/mydb?sslmode=require"
        parsed = DatabaseConfig._parse_postgresql_connection_string(conn_str)
        
        assert parsed['host'] == 'localhost'
        assert parsed['port'] == 5432
        assert parsed['database'] == 'mydb'
        assert parsed['username'] == 'user'
        assert parsed['password'] == 'pass'
        assert parsed['ssl_mode'] == 'require'
    
    def test_parse_postgres_connection_string(self):
        """Test parsing postgres:// scheme (alternative to postgresql://)."""
        conn_str = "postgres://user:pass@localhost:5432/mydb"
        parsed = DatabaseConfig._parse_postgresql_connection_string(conn_str)
        
        assert parsed['host'] == 'localhost'
        assert parsed['database'] == 'mydb'
        assert parsed['username'] == 'user'
    
    def test_parse_connection_string_with_special_chars(self):
        """Test parsing connection string with URL-encoded special characters."""
        conn_str = "postgresql://user%40domain:p%40ssw0rd%21@localhost:5432/mydb"
        parsed = DatabaseConfig._parse_postgresql_connection_string(conn_str)
        
        assert parsed['username'] == 'user@domain'
        assert parsed['password'] == 'p@ssw0rd!'
    
    def test_parse_connection_string_without_port(self):
        """Test parsing connection string without explicit port."""
        conn_str = "postgresql://user:pass@localhost/mydb"
        parsed = DatabaseConfig._parse_postgresql_connection_string(conn_str)
        
        assert parsed['host'] == 'localhost'
        assert parsed['port'] is None
        assert parsed['database'] == 'mydb'
    
    def test_parse_connection_string_without_password(self):
        """Test parsing connection string without password."""
        conn_str = "postgresql://user@localhost:5432/mydb"
        parsed = DatabaseConfig._parse_postgresql_connection_string(conn_str)
        
        assert parsed['username'] == 'user'
        assert parsed['password'] is None
    
    def test_parse_connection_string_invalid_scheme(self):
        """Test that invalid scheme raises error."""
        conn_str = "mysql://user:pass@localhost:3306/mydb"
        
        with pytest.raises(ValueError) as exc_info:
            DatabaseConfig._parse_postgresql_connection_string(conn_str)
        
        assert "Unsupported connection string scheme" in str(exc_info.value)


class TestConnectionStringToFields:
    """Test that connection strings populate individual fields."""
    
    def test_connection_string_populates_fields(self):
        """Test that providing connection_string populates individual fields."""
        from pydantic import SecretStr
        
        config = DatabaseConfig(
            db_type="postgresql",
            connection_string=SecretStr("postgresql://user:pass@localhost:5432/mydb?sslmode=require")
        )
        
        assert config.host == 'localhost'
        assert config.port == 5432
        assert config.database == 'mydb'
        assert config.username == 'user'
        assert config.password is not None
        assert config.password.get_secret_value() == 'pass'
        assert config.ssl_mode == 'require'
    
    def test_connection_string_takes_precedence_over_explicit_fields(self):
        """Test that connection_string takes precedence over explicitly set fields."""
        from pydantic import SecretStr
        
        config = DatabaseConfig(
            db_type="postgresql",
            host="existing-host",
            database="existing-db",
            connection_string=SecretStr("postgresql://user:pass@localhost:5432/mydb")
        )
        
        # Connection string should override explicitly set fields (it's the source of truth)
        assert config.host == 'localhost'  # From connection string
        assert config.database == 'mydb'  # From connection string
        # All fields should be populated from connection string
        assert config.username == 'user'
        assert config.port == 5432
        assert config.password is not None
        assert config.password.get_secret_value() == 'pass'
    
    def test_connection_string_with_supabase_format(self):
        """Test parsing Supabase-style connection string."""
        from pydantic import SecretStr
        
        # Supabase connection strings often have special characters
        conn_str = "postgresql://postgres:my%40password@db.xxx.supabase.co:5432/postgres?sslmode=require"
        config = DatabaseConfig(
            db_type="postgresql",
            connection_string=SecretStr(conn_str)
        )
        
        assert config.host == 'db.xxx.supabase.co'
        assert config.port == 5432
        assert config.database == 'postgres'
        assert config.username == 'postgres'
        assert config.password.get_secret_value() == 'my@password'
        assert config.ssl_mode == 'require'


class TestFieldsToConnectionString:
    """Test that individual fields construct connection strings."""
    
    def test_fields_construct_connection_string(self):
        """Test that individual fields automatically construct connection_string."""
        from pydantic import SecretStr
        
        config = DatabaseConfig(
            db_type="postgresql",
            host="localhost",
            port=5432,
            database="mydb",
            username="user",
            password=SecretStr("pass"),
            ssl_mode="require"
        )
        
        assert config.connection_string is not None
        conn_str = config.connection_string.get_secret_value()
        assert "postgresql://" in conn_str
        assert "user:pass@localhost:5432/mydb" in conn_str
        assert "sslmode=require" in conn_str
    
    def test_fields_without_password_constructs_connection_string(self):
        """Test constructing connection string without password."""
        config = DatabaseConfig(
            db_type="postgresql",
            host="localhost",
            database="mydb",
            username="user"
        )
        
        assert config.connection_string is not None
        conn_str = config.connection_string.get_secret_value()
        assert "postgresql://user@localhost" in conn_str
    
    def test_fields_with_special_chars_url_encoded(self):
        """Test that special characters in password are URL-encoded."""
        from pydantic import SecretStr
        
        config = DatabaseConfig(
            db_type="postgresql",
            host="localhost",
            database="mydb",
            username="user@domain",
            password=SecretStr("p@ssw0rd!")
        )
        
        assert config.connection_string is not None
        conn_str = config.connection_string.get_secret_value()
        # Special characters should be URL-encoded
        assert "%40" in conn_str or "user%40domain" in conn_str
        assert "%21" in conn_str or "p%40ssw0rd%21" in conn_str


class TestEnvironmentVariables:
    """Test that environment variables work with bidirectional sync."""
    
    def test_environment_connection_string_populates_fields(self):
        """Test that DD_DB_CONNECTION_STRING populates individual fields."""
        with patch.dict(os.environ, {
            'DD_DB_TYPE': 'postgresql',
            'DD_DB_CONNECTION_STRING': 'postgresql://user:pass@localhost:5432/mydb?sslmode=require'
        }):
            config = get_database_config()
            
            assert config.host == 'localhost'
            assert config.port == 5432
            assert config.database == 'mydb'
            assert config.username == 'user'
            assert config.password is not None
            assert config.ssl_mode == 'require'
    
    def test_environment_individual_fields_construct_connection_string(self):
        """Test that individual environment variables construct connection_string."""
        # Clear any existing database-related env vars to avoid conflicts
        env_vars_to_clear = [
            'DD_DB_CONNECTION_STRING',  # Clear connection string to force field-based construction
            'DD_DB_DATABASE',  # Clear any alternative database name
            'DD_DB_NAME',  # Clear existing database name
        ]
        cleared_values = {}
        for var in env_vars_to_clear:
            if var in os.environ:
                cleared_values[var] = os.environ.pop(var)
        
        try:
            # Mock load_dotenv to prevent .env file from overriding our test values
            # Since load_dotenv is imported inside the try block, we need to patch it at the dotenv module level
            # This prevents the .env file from being loaded and overriding our test environment variables
            with patch('dotenv.load_dotenv', side_effect=lambda *args, **kwargs: None), \
                 patch.dict(os.environ, {
                    'DD_DB_TYPE': 'postgresql',
                    'DD_DB_HOST': 'localhost',
                    'DD_DB_PORT': '5432',
                    'DD_DB_NAME': 'mydb',
                    'DD_DB_USER': 'user',
                    'DD_DB_PASSWORD': 'pass',
                    'DD_DB_SSL_MODE': 'require'
                }, clear=False):
                # Force reload by creating a new ConfigManager
                # This will call from_environment() which will try to load .env, but our mock prevents it
                config_manager = ConfigManager.from_environment()
                config = config_manager.get_database_config()
                
                assert config.database == 'mydb', f"Expected database='mydb', got {config.database}"
                assert config.connection_string is not None
                conn_str = config.connection_string.get_secret_value()
                assert "postgresql://" in conn_str
                assert "localhost" in conn_str
                assert "mydb" in conn_str, f"Expected 'mydb' in connection string, got: {conn_str}"
        finally:
            # Restore cleared values
            for var, value in cleared_values.items():
                os.environ[var] = value
    
    def test_environment_connection_string_takes_precedence(self):
        """Test that connection_string takes precedence over individual fields."""
        with patch.dict(os.environ, {
            'DD_DB_TYPE': 'postgresql',
            'DD_DB_HOST': 'old-host',
            'DD_DB_NAME': 'old-db',
            'DD_DB_CONNECTION_STRING': 'postgresql://user:pass@new-host:5432/new-db'
        }):
            config = get_database_config()
            
            # Connection string should override individual fields
            assert config.host == 'new-host'
            assert config.database == 'new-db'
            # But if individual fields were set, they might be preserved
            # Actually, the model_validator should populate missing fields from connection_string
            assert config.username == 'user'


class TestBidirectionalSync:
    """Test bidirectional synchronization between connection string and fields."""
    
    def test_connection_string_syncs_to_fields(self):
        """Test that connection_string syncs to fields on creation."""
        from pydantic import SecretStr
        
        config = DatabaseConfig(
            db_type="postgresql",
            connection_string=SecretStr("postgresql://user:pass@localhost:5432/mydb")
        )
        
        # Fields should be populated from connection_string
        assert config.host == 'localhost'
        assert config.database == 'mydb'
        assert config.username == 'user'
    
    def test_fields_sync_to_connection_string(self):
        """Test that fields sync to connection_string on creation."""
        from pydantic import SecretStr
        
        config = DatabaseConfig(
            db_type="postgresql",
            host="localhost",
            database="mydb",
            username="user",
            password=SecretStr("pass")
        )
        
        # Connection string should be constructed from fields
        assert config.connection_string is not None
        conn_str = config.connection_string.get_secret_value()
        assert "postgresql://" in conn_str
        assert "localhost" in conn_str
    
    def test_get_connection_string_returns_existing(self):
        """Test that get_connection_string() returns existing connection_string."""
        from pydantic import SecretStr
        
        config = DatabaseConfig(
            db_type="postgresql",
            connection_string=SecretStr("postgresql://user:pass@localhost:5432/mydb")
        )
        
        conn_str = config.get_connection_string()
        assert conn_str == "postgresql://user:pass@localhost:5432/mydb"
    
    def test_get_connection_string_constructs_from_fields(self):
        """Test that get_connection_string() constructs from fields if needed."""
        from pydantic import SecretStr
        
        config = DatabaseConfig(
            db_type="postgresql",
            host="localhost",
            database="mydb",
            username="user",
            password=SecretStr("pass")
        )
        
        # Even though connection_string was auto-constructed, get_connection_string should work
        conn_str = config.get_connection_string()
        assert "postgresql://" in conn_str
        assert "localhost" in conn_str
        assert "mydb" in conn_str

