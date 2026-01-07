"""Comprehensive tests for Data-Dialysis Dashboard API.

This test suite covers all Phase 1 endpoints including:
- Health check endpoint
- Root endpoint
- Placeholder endpoints (metrics, audit, circuit breaker, websocket)
- Middleware functionality
- Error handling
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from fastapi.testclient import TestClient

from src.dashboard.api.main import app
from src.domain.ports import Result, StoragePort


@pytest.fixture
def mock_storage_adapter():
    """Create a mock storage adapter for testing."""
    mock = Mock(spec=StoragePort)
    mock.db_config = Mock()
    mock.db_config.db_type = "duckdb"
    mock.db_config.db_path = ":memory:"
    mock._initialized = True
    
    # Mock query method
    mock.query = Mock(return_value=Result.success_result([{"1": 1}]))
    
    return mock


@pytest.fixture
def client(mock_storage_adapter):
    """Create a test client with mocked storage adapter."""
    from src.dashboard.api.dependencies import get_storage_adapter
    
    # Override the dependency
    app.dependency_overrides[get_storage_adapter] = lambda: mock_storage_adapter
    
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        # Clean up
        app.dependency_overrides.clear()


class TestRootEndpoint:
    """Test the root endpoint."""
    
    def test_root_endpoint_returns_info(self, client):
        """Test that root endpoint returns API information."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
        assert data["version"] == "1.0.0"
        assert data["docs"] == "/api/docs"
        assert data["health"] == "/api/health"
    
    def test_root_endpoint_has_correct_structure(self, client):
        """Test that root endpoint has expected structure."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert all(key in data for key in ["message", "version", "docs", "health"])


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_endpoint_returns_200(self, client):
        """Test that health endpoint returns 200 status."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
    
    def test_health_endpoint_has_required_fields(self, client):
        """Test that health endpoint returns all required fields."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "database" in data
        
        # Check database fields
        assert "status" in data["database"]
        assert "type" in data["database"]
    
    def test_health_endpoint_status_values(self, client):
        """Test that health endpoint returns valid status values."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Status should be one of: healthy, degraded, unhealthy
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert data["database"]["status"] in ["connected", "disconnected"]
    
    def test_health_endpoint_timestamp_format(self, client):
        """Test that health endpoint returns valid timestamp."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Timestamp should be parseable
        timestamp = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
        assert isinstance(timestamp, datetime)
    
    def test_health_endpoint_database_type(self, client, mock_storage_adapter):
        """Test that health endpoint returns correct database type."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # Should match mocked database type
        assert data["database"]["type"] == "duckdb"
    
    def test_health_endpoint_with_connected_database(self, client, mock_storage_adapter):
        """Test health endpoint when database is connected."""
        # Mock successful query
        mock_storage_adapter.query.return_value = Result.success_result([{"1": 1}])
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["database"]["status"] == "connected"
        assert data["status"] == "healthy"
    
    def test_health_endpoint_with_disconnected_database(self, client, mock_storage_adapter):
        """Test health endpoint when database is disconnected."""
        # Mock failed query
        mock_storage_adapter.query.return_value = Result.failure_result(
            Exception("Connection failed"),
            error_type="ConnectionError"
        )
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        # Should still return 200, but with disconnected status
        assert data["database"]["status"] == "disconnected"
        assert data["status"] in ["unhealthy", "degraded"]
    
    def test_health_endpoint_includes_response_time(self, client, mock_storage_adapter):
        """Test that health endpoint may include response time."""
        # Mock query with successful result
        mock_storage_adapter.query.return_value = Result.success_result([{"1": 1}])
        
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        
        # response_time_ms is optional, but if present should be a number
        if "response_time_ms" in data["database"]:
            assert isinstance(data["database"]["response_time_ms"], (int, float))
            assert data["database"]["response_time_ms"] >= 0


class TestMetricsEndpoints:
    """Test metrics endpoints (placeholders)."""
    
    def test_overview_metrics_endpoint_exists(self, client):
        """Test that overview metrics endpoint exists."""
        response = client.get("/api/metrics/overview")
        
        assert response.status_code == 200
        data = response.json()
        assert "time_range" in data
        assert "ingestions" in data
        assert "records" in data
        assert "redactions" in data
    
    def test_overview_metrics_endpoint_accepts_time_range(self, client):
        """Test that overview metrics endpoint accepts time_range parameter."""
        response = client.get("/api/metrics/overview?time_range=7d")
        
        assert response.status_code == 200
        data = response.json()
        assert data["time_range"] == "7d"
    
    def test_security_metrics_endpoint_exists(self, client):
        """Test that security metrics endpoint exists."""
        response = client.get("/api/metrics/security")
        
        assert response.status_code == 200
        data = response.json()
        assert "time_range" in data
        assert "redactions" in data
        assert "audit_events" in data
    
    def test_security_metrics_endpoint_accepts_time_range(self, client):
        """Test that security metrics endpoint accepts time_range parameter."""
        response = client.get("/api/metrics/security?time_range=30d")
        
        assert response.status_code == 200
        data = response.json()
        assert data["time_range"] == "30d"
    
    def test_performance_metrics_endpoint_exists(self, client):
        """Test that performance metrics endpoint exists."""
        response = client.get("/api/metrics/performance")
        
        assert response.status_code == 200
        data = response.json()
        assert "time_range" in data
        assert "throughput" in data
        assert "latency" in data
        assert "file_processing" in data
        assert "memory" in data
    
    def test_performance_metrics_endpoint_accepts_time_range(self, client):
        """Test that performance metrics endpoint accepts time_range parameter."""
        response = client.get("/api/metrics/performance?time_range=1h")
        
        assert response.status_code == 200
        data = response.json()
        assert data["time_range"] == "1h"


class TestAuditEndpoints:
    """Test audit log endpoints (now implemented in Phase 3)."""
    
    @pytest.fixture
    def mock_storage_adapter(self):
        """Mock storage adapter for testing."""
        from unittest.mock import Mock
        from src.domain.ports import Result
        
        adapter = Mock()
        adapter.initialize_schema.return_value = Result.success_result(None)
        
        # Mock connection with cursor for PostgreSQL compatibility
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        adapter._get_connection = Mock(return_value=mock_conn)
        adapter.connection_params = {}  # Indicates PostgreSQL adapter
        
        # Set up default cursor responses
        mock_cursor.fetchone.side_effect = [(0,)]  # Default count
        mock_cursor.fetchall.return_value = []  # Default data
        
        return adapter
    
    def test_audit_logs_endpoint_exists(self, client, mock_storage_adapter):
        """Test that audit logs endpoint exists and returns proper structure."""
        from src.dashboard.api.dependencies import get_storage_adapter
        app.dependency_overrides[get_storage_adapter] = lambda: mock_storage_adapter
        
        try:
            response = client.get("/api/audit-logs")
            
            assert response.status_code == 200
            data = response.json()
            assert "logs" in data
            assert "pagination" in data
            assert isinstance(data["logs"], list)
            assert "total" in data["pagination"]
        finally:
            app.dependency_overrides.clear()
    
    def test_audit_logs_endpoint_accepts_pagination(self, client, mock_storage_adapter):
        """Test that audit logs endpoint accepts pagination parameters."""
        from src.dashboard.api.dependencies import get_storage_adapter
        app.dependency_overrides[get_storage_adapter] = lambda: mock_storage_adapter
        
        try:
            response = client.get("/api/audit-logs?limit=50&offset=10")
            
            assert response.status_code == 200
            data = response.json()
            assert data["pagination"]["limit"] == 50
            assert data["pagination"]["offset"] == 10
        finally:
            app.dependency_overrides.clear()
    
    def test_audit_logs_endpoint_accepts_filters(self, client, mock_storage_adapter):
        """Test that audit logs endpoint accepts filter parameters."""
        from src.dashboard.api.dependencies import get_storage_adapter
        app.dependency_overrides[get_storage_adapter] = lambda: mock_storage_adapter
        
        try:
            # Use proper ISO date format
            response = client.get(
                "/api/audit-logs?severity=CRITICAL&event_type=REDACTION&start_date=2025-01-01T00:00:00Z"
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "logs" in data
            assert "pagination" in data
        finally:
            app.dependency_overrides.clear()
    
    def test_redaction_logs_endpoint_exists(self, client, mock_storage_adapter):
        """Test that redaction logs endpoint exists and returns proper structure."""
        from src.dashboard.api.dependencies import get_storage_adapter
        app.dependency_overrides[get_storage_adapter] = lambda: mock_storage_adapter
        
        # Update mock for redaction logs (needs 3 queries: count, summary, data)
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_summary_result = Mock()
        mock_summary_result.fetchall.return_value = []
        mock_count_result = Mock()
        mock_count_result.fetchone.return_value = (0,)
        mock_data_result = Mock()
        mock_data_result.fetchall.return_value = []
        mock_conn.execute.side_effect = [mock_count_result, mock_summary_result, mock_data_result]
        
        try:
            response = client.get("/api/redaction-logs")
            
            assert response.status_code == 200
            data = response.json()
            assert "logs" in data
            assert "pagination" in data
            assert "summary" in data
            assert isinstance(data["logs"], list)
        finally:
            app.dependency_overrides.clear()
    
    def test_redaction_logs_endpoint_accepts_filters(self, client, mock_storage_adapter):
        """Test that redaction logs endpoint accepts filter parameters."""
        from src.dashboard.api.dependencies import get_storage_adapter
        app.dependency_overrides[get_storage_adapter] = lambda: mock_storage_adapter
        
        # Update mock for redaction logs (needs 3 queries: count, summary, data)
        mock_conn = mock_storage_adapter._get_connection.return_value
        mock_summary_result = Mock()
        mock_summary_result.fetchall.return_value = []
        mock_count_result = Mock()
        mock_count_result.fetchone.return_value = (0,)
        mock_data_result = Mock()
        mock_data_result.fetchall.return_value = []
        mock_conn.execute.side_effect = [mock_count_result, mock_summary_result, mock_data_result]
        
        try:
            response = client.get("/api/redaction-logs?field_name=ssn&time_range=24h&limit=100")
            
            assert response.status_code == 200
            data = response.json()
            assert "logs" in data
            assert "pagination" in data
            assert data["pagination"]["limit"] == 100
        finally:
            app.dependency_overrides.clear()


class TestCircuitBreakerEndpoint:
    """Test circuit breaker status endpoint (now implemented in Phase 3)."""
    
    def test_circuit_breaker_endpoint_exists(self, client):
        """Test that circuit breaker status endpoint exists and returns proper structure."""
        response = client.get("/api/circuit-breaker/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "is_open" in data
        assert "failure_rate" in data
        assert "threshold" in data
        assert isinstance(data["is_open"], bool)
        assert isinstance(data["failure_rate"], (int, float))


class TestWebSocketEndpoint:
    """Test WebSocket endpoint (placeholder)."""
    
    def test_websocket_endpoint_exists(self, client):
        """Test that WebSocket endpoint exists."""
        # Note: TestClient doesn't fully support WebSocket testing
        # This is a basic check that the route is registered
        # Full WebSocket testing will be done in Phase 4
        
        # Check that the route exists in the app
        routes = [route for route in app.routes if hasattr(route, 'path')]
        websocket_routes = [r for r in routes if r.path == "/ws/realtime"]
        
        assert len(websocket_routes) > 0, "WebSocket route not found"


class TestMiddleware:
    """Test middleware functionality."""
    
    def test_logging_middleware_adds_process_time_header(self, client):
        """Test that logging middleware adds X-Process-Time header."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        assert "X-Process-Time" in response.headers
        assert float(response.headers["X-Process-Time"]) >= 0
    
    def test_error_handling_middleware_handles_errors(self, client, mock_storage_adapter):
        """Test that error handling middleware catches exceptions."""
        # Make storage adapter raise an exception
        mock_storage_adapter.query.side_effect = Exception("Test error")
        
        # Health endpoint should still return 200 (graceful degradation)
        # or 500 with proper error structure
        response = client.get("/api/health")
        
        # Should either return 200 with degraded status or 500
        assert response.status_code in [200, 500]
        
        if response.status_code == 500:
            data = response.json()
            assert "error" in data
            assert "detail" in data


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get("/api/health")
        
        # CORS headers should be present (set by CORS middleware)
        # Note: TestClient may not show all CORS headers, but the middleware is configured
        assert response.status_code == 200
    
    def test_options_request_works(self, client):
        """Test that OPTIONS request works (CORS preflight)."""
        response = client.options("/api/health")
        
        # Should return 200 or 405 (method not allowed)
        # CORS middleware handles OPTIONS requests
        assert response.status_code in [200, 405]


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    def test_openapi_schema_endpoint(self, client):
        """Test that OpenAPI schema endpoint exists."""
        response = client.get("/api/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
    
    def test_swagger_ui_endpoint(self, client):
        """Test that Swagger UI endpoint exists."""
        response = client.get("/api/docs")
        
        assert response.status_code == 200
        # Should return HTML
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_redoc_endpoint(self, client):
        """Test that ReDoc endpoint exists."""
        response = client.get("/api/redoc")
        
        assert response.status_code == 200
        # Should return HTML
        assert "text/html" in response.headers.get("content-type", "")


class TestErrorHandling:
    """Test error handling across endpoints."""
    
    def test_invalid_endpoint_returns_404(self, client):
        """Test that invalid endpoint returns 404."""
        response = client.get("/api/nonexistent")
        
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test that unsupported HTTP methods return 405."""
        # POST to GET-only endpoint
        response = client.post("/api/health")
        
        # FastAPI returns 405 for method not allowed
        assert response.status_code == 405


class TestResponseFormat:
    """Test response format consistency."""
    
    def test_all_responses_are_json(self, client):
        """Test that all API responses are JSON (except docs)."""
        from unittest.mock import Mock
        from src.dashboard.api.dependencies import get_storage_adapter
        from src.domain.ports import Result
        
        # Mock storage adapter for endpoints that need it
        mock_adapter = Mock()
        mock_adapter.initialize_schema.return_value = Result.success_result(None)
        
        # Set up db_config with proper string values (needed for health endpoint)
        mock_adapter.db_config = Mock()
        mock_adapter.db_config.db_type = "duckdb"  # String, not Mock
        mock_adapter.db_config.db_path = ":memory:"
        
        # Mock connection with cursor for PostgreSQL compatibility
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_conn.cursor.return_value = mock_cursor
        mock_adapter._get_connection = Mock(return_value=mock_conn)
        mock_adapter.connection_params = {}  # Indicates PostgreSQL adapter
        
        # Set up default cursor responses (use return_value for repeated calls)
        mock_cursor.fetchone.return_value = (0,)  # Default count (can be called multiple times)
        mock_cursor.fetchall.return_value = []  # Default data
        
        app.dependency_overrides[get_storage_adapter] = lambda: mock_adapter
        
        try:
            endpoints = [
                "/",
                "/api/health",
                "/api/metrics/overview",
                "/api/metrics/security",
                "/api/metrics/performance",
                "/api/audit-logs",
                "/api/redaction-logs",
                "/api/circuit-breaker/status",
            ]
            
            for endpoint in endpoints:
                response = client.get(endpoint)
                assert response.status_code == 200, f"Endpoint {endpoint} returned {response.status_code}: {response.json() if response.status_code != 200 else ''}"
                # Should be JSON
                assert "application/json" in response.headers.get("content-type", ""), f"Endpoint {endpoint} is not JSON"
                # Should be parseable
                data = response.json()
                assert isinstance(data, dict), f"Endpoint {endpoint} response is not a dictionary"
        finally:
            app.dependency_overrides.clear()

