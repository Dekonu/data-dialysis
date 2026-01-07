"""Tests for circuit breaker status endpoint."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from src.dashboard.api.main import app
from src.domain.ports import Result


@pytest.fixture
def mock_storage_adapter():
    """Mock storage adapter for testing."""
    adapter = Mock()
    adapter.initialize_schema.return_value = Result.success_result(None)
    
    # Mock connection with cursor for PostgreSQL compatibility
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    adapter._get_connection = Mock(return_value=mock_conn)
    adapter.connection_params = {}  # Indicates PostgreSQL adapter
    
    return adapter


@pytest.fixture
def client(mock_storage_adapter):
    """Test client with mocked dependencies."""
    app.dependency_overrides = {}
    
    def get_storage():
        return mock_storage_adapter
    
    from src.dashboard.api.dependencies import get_storage_adapter
    app.dependency_overrides[get_storage_adapter] = get_storage
    
    yield TestClient(app)
    
    app.dependency_overrides.clear()


class TestCircuitBreakerStatusEndpoint:
    """Tests for GET /api/circuit-breaker/status endpoint."""
    
    def test_get_circuit_breaker_status_success(self, client, mock_storage_adapter):
        """Test successful retrieval of circuit breaker status."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(5,), (100,)]  # Error count, then total count
        
        response = client.get("/api/circuit-breaker/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "is_open" in data
        assert "failure_rate" in data
        assert "threshold" in data
        assert "total_processed" in data
        assert "total_failures" in data
        assert "window_size" in data
        assert isinstance(data["is_open"], bool)
        assert isinstance(data["failure_rate"], (int, float))
        assert data["failure_rate"] == 5.0  # 5 errors / 100 total * 100
    
    def test_get_circuit_breaker_status_open(self, client, mock_storage_adapter):
        """Test circuit breaker status when open (high failure rate)."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(60,), (100,)]  # Error count, then total count
        
        response = client.get("/api/circuit-breaker/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_open"] is True
        assert data["failure_rate"] == 60.0
        assert data["total_failures"] == 60
        assert data["total_processed"] == 100
    
    def test_get_circuit_breaker_status_closed(self, client, mock_storage_adapter):
        """Test circuit breaker status when closed (low failure rate)."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(5,), (100,)]  # Error count, then total count
        
        response = client.get("/api/circuit-breaker/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_open"] is False
        assert data["failure_rate"] == 5.0
    
    def test_get_circuit_breaker_status_no_events(self, client, mock_storage_adapter):
        """Test circuit breaker status when no events exist."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_execute = Mock()
        mock_conn.execute = mock_execute
        
        mock_error_result = Mock()
        mock_error_result.fetchone.return_value = (0,)
        
        mock_total_result = Mock()
        mock_total_result.fetchone.return_value = (0,)
        
        mock_execute.side_effect = [mock_error_result, mock_total_result]
        
        response = client.get("/api/circuit-breaker/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["is_open"] is False
        assert data["failure_rate"] == 0.0
        assert data["total_processed"] == 0
        assert data["total_failures"] == 0
    
    def test_get_circuit_breaker_status_database_error(self, client, mock_storage_adapter):
        """Test circuit breaker status with database error."""
        mock_storage_adapter.initialize_schema.return_value = Result.failure_result(
            Exception("Database connection failed")
        )
        
        response = client.get("/api/circuit-breaker/status")
        
        # Should return default closed status even on init error
        assert response.status_code == 200
        data = response.json()
        assert data["is_open"] is False
        assert data["failure_rate"] == 0.0
    
    def test_get_circuit_breaker_status_query_error(self, client, mock_storage_adapter):
        """Test circuit breaker status when query fails."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.execute.side_effect = Exception("Query failed")
        
        response = client.get("/api/circuit-breaker/status")
        
        # Should return default closed status even on query error
        assert response.status_code == 200
        data = response.json()
        assert data["is_open"] is False

