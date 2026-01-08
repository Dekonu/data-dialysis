"""Tests for change history API endpoints."""

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from src.dashboard.api.main import app
from src.domain.ports import Result
from src.dashboard.services.connection_helper import ConnectionWrapper


@pytest.fixture
def mock_storage_adapter():
    """Mock storage adapter for testing."""
    adapter = Mock()
    adapter.initialize_schema.return_value = Result.success_result(None)
    
    # Mock connection for PostgreSQL compatibility
    mock_conn = Mock()
    mock_cursor = Mock()
    mock_conn.cursor.return_value = mock_cursor
    adapter._get_connection = Mock(return_value=mock_conn)
    adapter._return_connection = Mock()
    adapter.connection_params = {}  # Indicates PostgreSQL adapter
    
    return adapter


@pytest.fixture
def mock_connection_wrapper(mock_storage_adapter):
    """Mock ConnectionWrapper for testing."""
    mock_conn = mock_storage_adapter._get_connection.return_value
    mock_cursor = mock_conn.cursor.return_value
    
    # Create a mock wrapper that returns the cursor when execute() is called
    wrapper = MagicMock(spec=ConnectionWrapper)
    wrapper.execute = Mock(return_value=mock_cursor)
    wrapper.close = Mock()
    
    return wrapper


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


def create_mock_result(fetchone_value, fetchall_value=None, description=None):
    """Helper to create mock result objects."""
    result = Mock()
    result.fetchone.return_value = fetchone_value
    if fetchall_value is not None:
        result.fetchall.return_value = fetchall_value
    if description:
        result.description = description
    return result


class TestChangeHistoryEndpoint:
    """Tests for GET /api/change-history endpoint."""
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_history_success(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test successful retrieval of change history."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((5,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                ),
                (
                    "change-2", "patients", "P002", "phone", "555-0101", "555-0202",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history?limit=10&offset=0")
        
        assert response.status_code == 200
        data = response.json()
        assert "changes" in data
        assert "total" in data
        assert len(data["changes"]) == 2
        assert data["total"] == 5
        assert data["limit"] == 10
        assert data["offset"] == 0
        assert data["changes"][0]["table_name"] == "patients"
        assert data["changes"][0]["record_id"] == "P001"
        assert data["changes"][0]["field_name"] == "city"
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_history_with_filters(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change history endpoint with filters."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((2,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get(
            "/api/change-history?"
            "limit=10&offset=0&"
            "table_name=patients&"
            "change_type=UPDATE&"
            "source_adapter=csv_ingester"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["changes"]) == 1
        assert data["changes"][0]["table_name"] == "patients"
        assert data["changes"][0]["change_type"] == "UPDATE"
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_history_with_date_filters(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change history endpoint with date filters."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        start_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        end_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((1,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get(
            f"/api/change-history?"
            f"start_date={start_date}&"
            f"end_date={end_date}"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["changes"]) == 1
    
    def test_get_change_history_invalid_date_format(self, client):
        """Test change history endpoint with invalid date format."""
        response = client.get("/api/change-history?start_date=invalid-date")
        
        assert response.status_code == 400
        assert "Invalid start_date format" in response.json()["detail"]
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_history_with_record_id_filter(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change history endpoint with record_id filter."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((3,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                ),
                (
                    "change-2", "patients", "P001", "phone", "555-0101", "555-0202",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history?record_id=P001")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["changes"]) == 2
        assert all(change["record_id"] == "P001" for change in data["changes"])
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_history_with_field_name_filter(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change history endpoint with field_name filter."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((2,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history?field_name=city")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["changes"]) == 1
        assert data["changes"][0]["field_name"] == "city"
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_history_with_sorting(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change history endpoint with sorting."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((2,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history?sort_by=field_name&sort_order=ASC")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["changes"]) == 1


class TestChangeHistorySummaryEndpoint:
    """Tests for GET /api/change-history/summary endpoint."""
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_summary_success(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test successful retrieval of change summary."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        summary_result = create_mock_result(
            (100, 50, 3, 15, 20, 75, 0)  # total_changes, unique_records, tables_affected, fields_changed, inserts, updates, deletes
        )
        
        mock_connection_wrapper.execute.return_value = summary_result
        
        response = client.get("/api/change-history/summary?time_range=24h")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_changes"] == 100
        assert data["unique_records"] == 50
        assert data["tables_affected"] == 3
        assert data["fields_changed"] == 15
        assert data["inserts"] == 20
        assert data["updates"] == 75
        assert data["deletes"] == 0
        assert data["time_range"] == "24h"
        assert "start_time" in data
        assert "end_time" in data
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_summary_with_table_filter(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change summary endpoint with table filter."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        summary_result = create_mock_result(
            (25, 10, 1, 5, 5, 20, 0)  # total_changes, unique_records, tables_affected, fields_changed, inserts, updates, deletes
        )
        
        mock_connection_wrapper.execute.return_value = summary_result
        
        response = client.get("/api/change-history/summary?time_range=24h&table_name=patients")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total_changes"] == 25
        assert data["tables_affected"] == 1
    
    def test_get_change_summary_invalid_time_range(self, client):
        """Test change summary endpoint with invalid time range."""
        response = client.get("/api/change-history/summary?time_range=invalid")
        
        assert response.status_code == 422  # Validation error


class TestRecordChangeHistoryEndpoint:
    """Tests for GET /api/change-history/record/{table_name}/{record_id} endpoint."""
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_record_change_history_success(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test successful retrieval of record change history."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((3,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                ),
                (
                    "change-2", "patients", "P001", "phone", "555-0101", "555-0202",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                ),
                (
                    "change-3", "patients", "P001", "family_name", None, "Smith",
                    "INSERT", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history/record/patients/P001")
        
        assert response.status_code == 200
        data = response.json()
        assert "changes" in data
        assert data["table_name"] == "patients"
        assert data["record_id"] == "P001"
        assert len(data["changes"]) == 3
        assert data["changes"][0]["record_id"] == "P001"
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_record_change_history_with_limit(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test record change history endpoint with limit."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((5,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history/record/patients/P001?limit=1")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["changes"]) == 1


class TestExportChangeHistoryEndpoint:
    """Tests for GET /api/change-history/export endpoint."""
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_export_change_history_csv(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test exporting change history as CSV."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((2,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                ),
                (
                    "change-2", "patients", "P002", "phone", "555-0101", "555-0202",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history/export?format=csv")
        
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        assert "attachment" in response.headers["content-disposition"]
        assert "change_history" in response.headers["content-disposition"]
        
        # Verify CSV content
        content = response.text
        assert "change_id" in content
        assert "table_name" in content
        assert "P001" in content
        assert "P002" in content
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_export_change_history_json(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test exporting change history as JSON."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((1,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history/export?format=json")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        assert "attachment" in response.headers["content-disposition"]
        assert "change_history" in response.headers["content-disposition"]
        
        # Verify JSON content - the endpoint returns a list
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["change_id"] == "change-1"
        assert data[0]["table_name"] == "patients"
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_export_change_history_with_filters(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test exporting change history with filters."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((1,))
        data_result = create_mock_result(
            None,
            [
                (
                    "change-1", "patients", "P001", "city", "Boston", "Cambridge",
                    "UPDATE", datetime.now(timezone.utc), "ing-123", "csv_ingester", "system"
                )
            ],
            mock_cursor.description
        )
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get(
            "/api/change-history/export?"
            "format=csv&"
            "table_name=patients&"
            "change_type=UPDATE"
        )
        
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
    
    def test_export_change_history_invalid_format(self, client):
        """Test export endpoint with invalid format."""
        response = client.get("/api/change-history/export?format=xml")
        
        assert response.status_code == 422  # Validation error
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_export_change_history_with_date_filters(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test export endpoint with date filters."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        start_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        end_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((0,))
        data_result = create_mock_result(None, [], mock_cursor.description)
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get(
            f"/api/change-history/export?"
            f"format=json&"
            f"start_date={start_date}&"
            f"end_date={end_date}"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0


class TestChangeHistoryErrorHandling:
    """Tests for error handling in change history endpoints."""
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_history_database_error(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change history endpoint handles database errors."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_connection_wrapper.execute.side_effect = Exception("Database connection error")
        
        response = client.get("/api/change-history")
        
        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower() or "error" in str(response.json())
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_summary_database_error(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change summary endpoint handles database errors."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_connection_wrapper.execute.side_effect = Exception("Database connection error")
        
        response = client.get("/api/change-history/summary")
        
        assert response.status_code == 500
    
    @patch('src.dashboard.services.change_history_service.get_db_connection')
    def test_get_change_history_empty_result(self, mock_get_conn, client, mock_storage_adapter, mock_connection_wrapper):
        """Test change history endpoint with no results."""
        mock_get_conn.return_value.__enter__ = Mock(return_value=mock_connection_wrapper)
        mock_get_conn.return_value.__exit__ = Mock(return_value=False)
        
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_cursor = mock_conn.cursor.return_value
        
        mock_cursor.description = [
            ('change_id',), ('table_name',), ('record_id',), ('field_name',),
            ('old_value',), ('new_value',), ('change_type',), ('changed_at',),
            ('ingestion_id',), ('source_adapter',), ('changed_by',)
        ]
        
        count_result = create_mock_result((0,))
        data_result = create_mock_result(None, [], mock_cursor.description)
        
        mock_connection_wrapper.execute.side_effect = [count_result, data_result]
        
        response = client.get("/api/change-history")
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["changes"]) == 0
        assert data["total"] == 0
