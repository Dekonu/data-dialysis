"""Tests for audit log endpoints."""

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from src.dashboard.api.main import app
from src.dashboard.models.audit import AuditLogEntry, PaginationMeta, RedactionLogEntry
from src.domain.ports import Result


@pytest.fixture
def mock_storage_adapter():
    """Mock storage adapter for testing."""
    adapter = Mock()
    adapter.initialize_schema.return_value = Result.success_result(None)
    
    # Mock connection with cursor for PostgreSQL compatibility
    # get_db_connection wraps connections in ConnectionWrapper which uses cursors
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


class TestAuditLogsEndpoint:
    """Tests for GET /api/audit-logs endpoint."""
    
    def test_get_audit_logs_success(self, client, mock_storage_adapter):
        """Test successful retrieval of audit logs."""
        # Mock connection and cursor (ConnectionWrapper uses cursor for PostgreSQL)
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        # Set up side effect for cursor.execute calls
        # ConnectionWrapper calls cursor() then executes on cursor
        mock_cursor.fetchone.side_effect = [(5,)]  # Count query
        mock_cursor.fetchall.return_value = [  # Data query
            (
                "audit-1",
                "BULK_PERSISTENCE",
                datetime.now(timezone.utc),
                "record-1",
                "hash-1",
                '{"key": "value"}',
                "csv_adapter",
                "INFO",
                None,  # table_name
                None   # row_count
            ),
            (
                "audit-2",
                "REDACTION",
                datetime.now(timezone.utc),
                "record-2",
                "hash-2",
                None,
                "json_adapter",
                "WARNING",
                None,  # table_name
                None   # row_count
            )
        ]
        
        response = client.get("/api/audit-logs?limit=10&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "pagination" in data
        assert len(data["logs"]) == 2
        assert data["pagination"]["total"] == 5
        assert data["pagination"]["limit"] == 10
        assert data["pagination"]["offset"] == 0
    
    def test_get_audit_logs_with_filters(self, client, mock_storage_adapter):
        """Test audit logs endpoint with filters."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(2,)]
        mock_cursor.fetchall.return_value = [
            (
                "audit-1",
                "REDACTION",
                datetime.now(timezone.utc),
                "record-1",
                "hash-1",
                None,
                "csv_adapter",
                "ERROR",
                None,  # table_name
                None   # row_count
            )
        ]
        
        response = client.get(
            "/api/audit-logs?"
            "limit=10&offset=0&"
            "severity=ERROR&"
            "event_type=REDACTION&"
            "source_adapter=csv_adapter"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["logs"]) == 1
        assert data["logs"][0]["severity"] == "ERROR"
        assert data["logs"][0]["event_type"] == "REDACTION"
    
    def test_get_audit_logs_with_date_filters(self, client, mock_storage_adapter):
        """Test audit logs endpoint with date filters."""
        # Use ISO format with Z (not +00:00) - endpoint will replace Z with +00:00
        # isoformat() with timezone produces +00:00, so we need to replace it with Z
        start_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(1,)]
        mock_cursor.fetchall.return_value = [
            (
                "audit-1",
                "BULK_PERSISTENCE",
                datetime.now(timezone.utc),
                "record-1",
                "hash-1",
                None,
                "csv_adapter",
                "INFO",
                None,  # table_name
                None   # row_count
            )
        ]
        
        response = client.get(
            f"/api/audit-logs?"
            f"start_date={start_date}&"
            f"limit=10"
        )
        
        if response.status_code != 200:
            print(f"Error response: {response.status_code}")
            print(f"Error detail: {response.json()}")
        
        assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.json() if response.status_code != 200 else ''}"
        data = response.json()
        assert len(data["logs"]) == 1
    
    def test_get_audit_logs_invalid_date_format(self, client, mock_storage_adapter):
        """Test audit logs endpoint with invalid date format."""
        response = client.get("/api/audit-logs?start_date=invalid-date")
        
        assert response.status_code == 400
        assert "Invalid start_date format" in response.json()["detail"]
    
    def test_get_audit_logs_pagination(self, client, mock_storage_adapter):
        """Test audit logs pagination."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(100,)]
        mock_cursor.fetchall.return_value = []  # Empty page
        
        response = client.get("/api/audit-logs?limit=10&offset=90")
        
        assert response.status_code == 200
        data = response.json()
        assert data["pagination"]["total"] == 100
        assert data["pagination"]["has_next"] is False
        assert data["pagination"]["has_previous"] is True
    
    def test_get_audit_logs_database_error(self, client, mock_storage_adapter):
        """Test audit logs endpoint with database error."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.execute.side_effect = Exception("Database error")
        
        response = client.get("/api/audit-logs")
        
        assert response.status_code == 500
    
    def test_get_audit_logs_empty_result(self, client, mock_storage_adapter):
        """Test audit logs endpoint with no results."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(0,)]
        mock_cursor.fetchall.return_value = []
        
        response = client.get("/api/audit-logs")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["logs"]) == 0
        assert data["pagination"]["total"] == 0


class TestRedactionLogsEndpoint:
    """Tests for GET /api/redaction-logs endpoint."""
    
    def test_get_redaction_logs_success(self, client, mock_storage_adapter):
        """Test successful retrieval of redaction logs."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        # Set up side effects for multiple queries (count, summary, data)
        mock_cursor.fetchone.side_effect = [(8,)]  # Count query
        mock_cursor.fetchall.side_effect = [
            [  # Summary query
                (5, "ssn", "regex_ssn", "csv_adapter"),
                (3, "dob", "regex_dob", "json_adapter")
            ],
            [  # Data query
            (
                "log-1",
                "ssn",
                "hash-1",
                datetime.now(timezone.utc),
                "regex_ssn",
                "record-1",
                "csv_adapter",
                "ingestion-1",
                "***-**-****",
                11
            ),
            (
                "log-2",
                "dob",
                "hash-2",
                datetime.now(timezone.utc),
                "regex_dob",
                "record-2",
                "json_adapter",
                "ingestion-1",
                "****-**-**",
                10
            )
            ]
        ]
        
        response = client.get("/api/redaction-logs?time_range=24h&limit=10")
        
        assert response.status_code == 200
        data = response.json()
        assert "logs" in data
        assert "pagination" in data
        assert "summary" in data
        assert len(data["logs"]) == 2
        assert data["summary"]["total"] == 8
        assert "ssn" in data["summary"]["by_field"]
        assert "dob" in data["summary"]["by_field"]
    
    def test_get_redaction_logs_with_filters(self, client, mock_storage_adapter):
        """Test redaction logs endpoint with filters."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(2,)]  # Count query
        mock_cursor.fetchall.side_effect = [
            [  # Summary query
                (2, "ssn", "regex_ssn", "csv_adapter")
            ],
            [  # Data query
                (
                    "log-1",
                    "ssn",
                    "hash-1",
                datetime.now(timezone.utc),
                "regex_ssn",
                "record-1",
                "csv_adapter",
                "ingestion-1",
                "***-**-****",
                11
            )
            ]
        ]
        
        response = client.get(
            "/api/redaction-logs?"
            "field_name=ssn&"
            "rule_triggered=regex_ssn&"
            "time_range=7d"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["logs"]) == 1
        assert data["logs"][0]["field_name"] == "ssn"
        assert data["logs"][0]["rule_triggered"] == "regex_ssn"
    
    def test_get_redaction_logs_invalid_time_range(self, client, mock_storage_adapter):
        """Test redaction logs endpoint with invalid time range."""
        response = client.get("/api/redaction-logs?time_range=invalid")
        
        assert response.status_code == 422  # Validation error
    
    def test_get_redaction_logs_database_error(self, client, mock_storage_adapter):
        """Test redaction logs endpoint with database error."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        mock_cursor.execute.side_effect = Exception("Database error")
        
        response = client.get("/api/redaction-logs")
        
        assert response.status_code == 500


class TestExportAuditLogsEndpoint:
    """Tests for GET /api/audit-logs/export endpoint."""
    
    def test_export_audit_logs_json(self, client, mock_storage_adapter):
        """Test export audit logs as JSON."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(2,)]
        mock_cursor.fetchall.return_value = [
            (
                "audit-1",
                "BULK_PERSISTENCE",
                datetime.now(timezone.utc),
                "record-1",
                "hash-1",
                '{"key": "value"}',
                "csv_adapter",
                "INFO",
                None,  # table_name
                None   # row_count
            ),
            (
                "audit-2",
                "REDACTION",
                datetime.now(timezone.utc),
                "record-2",
                "hash-2",
                None,
                "json_adapter",
                "WARNING",
                None,  # table_name
                None   # row_count
            )
        ]
        
        response = client.get("/api/audit-logs/export?format=json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "attachment" in response.headers["content-disposition"]
        
        data = json.loads(response.content)
        assert isinstance(data, list)
        assert len(data) == 2
    
    def test_export_audit_logs_csv(self, client, mock_storage_adapter):
        """Test export audit logs as CSV."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(1,)]
        mock_cursor.fetchall.return_value = [
            (
                "audit-1",
                "BULK_PERSISTENCE",
                datetime.now(timezone.utc),
                "record-1",
                "hash-1",
                None,
                "csv_adapter",
                "INFO",
                None,  # table_name
                None   # row_count
            )
        ]
        
        response = client.get("/api/audit-logs/export?format=csv")
        
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        
        content = response.content.decode("utf-8")
        assert "audit_id" in content
        assert "event_type" in content
        assert "audit-1" in content
    
    def test_export_audit_logs_invalid_format(self, client, mock_storage_adapter):
        """Test export audit logs with invalid format."""
        response = client.get("/api/audit-logs/export?format=xml")
        
        assert response.status_code == 422  # Validation error
    
    def test_export_audit_logs_with_filters(self, client, mock_storage_adapter):
        """Test export audit logs with filters."""
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.fetchone.side_effect = [(1,)]
        mock_cursor.fetchall.return_value = [
            (
                "audit-1",
                "REDACTION",
                datetime.now(timezone.utc),
                "record-1",
                "hash-1",
                None,
                "csv_adapter",
                "ERROR",
                None,  # table_name
                None   # row_count
            )
        ]
        
        response = client.get(
            "/api/audit-logs/export?"
            "format=json&"
            "severity=ERROR&"
            "event_type=REDACTION"
        )
        
        assert response.status_code == 200
        data = json.loads(response.content)
        assert len(data) == 1
        assert data[0]["severity"] == "ERROR"

