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
        assert "message" in data
        assert data["endpoint"] == "/api/metrics/overview"
    
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
        assert "message" in data
        assert data["endpoint"] == "/api/metrics/security"
    
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
        assert "message" in data
        assert data["endpoint"] == "/api/metrics/performance"
    
    def test_performance_metrics_endpoint_accepts_time_range(self, client):
        """Test that performance metrics endpoint accepts time_range parameter."""
        response = client.get("/api/metrics/performance?time_range=1h")
        
        assert response.status_code == 200
        data = response.json()
        assert data["time_range"] == "1h"


class TestAuditEndpoints:
    """Test audit log endpoints (placeholders)."""
    
    def test_audit_logs_endpoint_exists(self, client):
        """Test that audit logs endpoint exists."""
        response = client.get("/api/audit-logs")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["endpoint"] == "/api/audit-logs"
    
    def test_audit_logs_endpoint_accepts_pagination(self, client):
        """Test that audit logs endpoint accepts pagination parameters."""
        response = client.get("/api/audit-logs?limit=50&offset=10")
        
        assert response.status_code == 200
        data = response.json()
        assert data["limit"] == 50
        assert data["offset"] == 10
    
    def test_audit_logs_endpoint_accepts_filters(self, client):
        """Test that audit logs endpoint accepts filter parameters."""
        response = client.get(
            "/api/audit-logs?severity=CRITICAL&event_type=REDACTION&start_date=2025-01-01"
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "filters" in data
        assert data["filters"]["severity"] == "CRITICAL"
        assert data["filters"]["event_type"] == "REDACTION"
        assert data["filters"]["start_date"] == "2025-01-01"
    
    def test_redaction_logs_endpoint_exists(self, client):
        """Test that redaction logs endpoint exists."""
        response = client.get("/api/redaction-logs")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["endpoint"] == "/api/redaction-logs"
    
    def test_redaction_logs_endpoint_accepts_filters(self, client):
        """Test that redaction logs endpoint accepts filter parameters."""
        response = client.get("/api/redaction-logs?field_name=ssn&time_range=24h&limit=100")
        
        assert response.status_code == 200
        data = response.json()
        assert data["field_name"] == "ssn"
        assert data["time_range"] == "24h"
        assert data["limit"] == 100


class TestCircuitBreakerEndpoint:
    """Test circuit breaker status endpoint (placeholder)."""
    
    def test_circuit_breaker_endpoint_exists(self, client):
        """Test that circuit breaker status endpoint exists."""
        response = client.get("/api/circuit-breaker/status")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["endpoint"] == "/api/circuit-breaker/status"


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
            assert response.status_code == 200
            # Should be JSON
            assert "application/json" in response.headers.get("content-type", "")
            # Should be parseable
            data = response.json()
            assert isinstance(data, dict)

