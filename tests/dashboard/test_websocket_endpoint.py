"""Tests for WebSocket endpoint.

This module contains tests for the WebSocket real-time updates endpoint.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import WebSocket

from src.dashboard.api.main import app
from src.dashboard.api.dependencies import get_storage_adapter
from src.dashboard.services.websocket_manager import ConnectionManager, get_connection_manager
from src.domain.ports import Result


@pytest.fixture
def mock_storage_adapter():
    """Create a mock storage adapter for testing."""
    mock = Mock()
    mock.db_config = Mock()
    mock.db_config.db_type = "duckdb"
    mock.db_config.db_path = ":memory:"
    mock._initialized = True
    mock.query = Mock(return_value=Result.success_result([{"1": 1}]))
    mock.initialize_schema = Mock(return_value=Result.success_result(None))
    
    # Mock _get_connection for services that need it
    mock_conn = Mock()
    mock_conn.execute = Mock(return_value=Mock(fetchone=Mock(return_value=(0,)), fetchall=Mock(return_value=[])))
    mock._get_connection = Mock(return_value=mock_conn)
    
    return mock


@pytest.fixture
def client(mock_storage_adapter):
    """Create a test client with mocked dependencies."""
    app.dependency_overrides[get_storage_adapter] = lambda: mock_storage_adapter
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestWebSocketManager:
    """Test WebSocket connection manager."""
    
    def test_connection_manager_initialization(self):
        """Test that connection manager initializes correctly."""
        manager = ConnectionManager()
        assert manager.active_connections == set()
    
    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        """Test connecting and disconnecting a WebSocket."""
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        
        await manager.connect(mock_websocket)
        assert mock_websocket in manager.active_connections
        assert len(manager.active_connections) == 1
        
        await manager.disconnect(mock_websocket)
        assert mock_websocket not in manager.active_connections
        assert len(manager.active_connections) == 0
    
    @pytest.mark.asyncio
    async def test_get_connection_count(self):
        """Test getting connection count."""
        manager = ConnectionManager()
        assert await manager.get_connection_count() == 0
        
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        await manager.connect(mock_websocket)
        
        assert await manager.get_connection_count() == 1
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self):
        """Test sending a personal message."""
        from src.dashboard.models.websocket import ConnectionMessage
        
        manager = ConnectionManager()
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_json = AsyncMock()
        
        await manager.connect(mock_websocket)
        
        message = ConnectionMessage(message="Test message")
        await manager.send_personal_message(message, mock_websocket)
        
        mock_websocket.send_json.assert_called_once()
        call_args = mock_websocket.send_json.call_args[0][0]
        assert call_args["type"] == "connection"
        assert call_args["message"] == "Test message"
    
    @pytest.mark.asyncio
    async def test_broadcast(self):
        """Test broadcasting messages to all connections."""
        from src.dashboard.models.websocket import HeartbeatMessage
        
        manager = ConnectionManager()
        
        # Create two mock connections
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_json = AsyncMock()
        
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_json = AsyncMock()
        
        await manager.connect(mock_ws1)
        await manager.connect(mock_ws2)
        
        message = HeartbeatMessage()
        sent_count = await manager.broadcast(message)
        
        assert sent_count == 2
        mock_ws1.send_json.assert_called_once()
        mock_ws2.send_json.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_broadcast_removes_disconnected_clients(self):
        """Test that broadcast removes disconnected clients."""
        from src.dashboard.models.websocket import HeartbeatMessage
        
        manager = ConnectionManager()
        
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_json = AsyncMock()
        
        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_json = AsyncMock(side_effect=Exception("Connection closed"))
        
        await manager.connect(mock_ws1)
        await manager.connect(mock_ws2)
        
        message = HeartbeatMessage()
        sent_count = await manager.broadcast(message)
        
        # Only one message should be sent successfully
        assert sent_count == 1
        # Disconnected client should be removed
        assert await manager.get_connection_count() == 1
        assert mock_ws1 in manager.active_connections
        assert mock_ws2 not in manager.active_connections


class TestWebSocketEndpoint:
    """Test WebSocket endpoint."""
    
    def test_websocket_route_exists(self):
        """Test that WebSocket route is registered."""
        websocket_routes = [
            route for route in app.routes
            if hasattr(route, 'path') and '/ws/realtime' in route.path
        ]
        assert len(websocket_routes) > 0, "WebSocket route not found"
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, mock_storage_adapter):
        """Test WebSocket connection establishment."""
        from src.dashboard.services.metrics_aggregator import MetricsAggregator
        from src.dashboard.services.security_metrics import SecurityMetricsService
        from src.dashboard.services.performance_metrics import PerformanceMetricsService
        from src.dashboard.services.circuit_breaker_service import CircuitBreakerService
        from src.domain.ports import Result
        from src.dashboard.models.metrics import OverviewMetrics, IngestionMetrics, RecordMetrics, RedactionSummary
        from src.dashboard.models.circuit_breaker import CircuitBreakerStatus
        
        # Mock metrics results
        overview_metrics = OverviewMetrics(
            time_range="24h",
            ingestions=IngestionMetrics(total=10, successful=9, failed=1, success_rate=0.9),
            records=RecordMetrics(total_processed=100, total_successful=95, total_failed=5),
            redactions=RedactionSummary(total=50, by_field={}, by_rule=None, by_adapter=None)
        )
        
        security_metrics = Mock()
        performance_metrics = Mock()
        circuit_breaker_status = CircuitBreakerStatus(
            is_open=False,
            failure_rate=5.0,
            threshold=10.0,
            total_processed=100,
            total_failures=5,
            window_size=100,
            failures_in_window=5,
            records_in_window=100,
            min_records_before_check=10
        )
        
        # Mock services
        with patch('src.dashboard.api.routes.websocket.MetricsAggregator') as mock_agg, \
             patch('src.dashboard.api.routes.websocket.SecurityMetricsService') as mock_sec, \
             patch('src.dashboard.api.routes.websocket.PerformanceMetricsService') as mock_perf, \
             patch('src.dashboard.api.routes.websocket.CircuitBreakerService') as mock_cb:
            
            mock_agg_instance = Mock()
            mock_agg_instance.get_overview_metrics = Mock(return_value=Result.success_result(overview_metrics))
            mock_agg.return_value = mock_agg_instance
            
            mock_sec_instance = Mock()
            mock_sec_instance.get_security_metrics = Mock(return_value=Result.success_result(security_metrics))
            mock_sec.return_value = mock_sec_instance
            
            mock_perf_instance = Mock()
            mock_perf_instance.get_performance_metrics = Mock(return_value=Result.success_result(performance_metrics))
            mock_perf.return_value = mock_perf_instance
            
            mock_cb_instance = Mock()
            mock_cb_instance.get_status = Mock(return_value=Result.success_result(circuit_breaker_status))
            mock_cb.return_value = mock_cb_instance
            
            # Create a test client
            client = TestClient(app)
            app.dependency_overrides[get_storage_adapter] = lambda: mock_storage_adapter
            
            try:
                # Note: TestClient doesn't fully support WebSocket testing
                # This test verifies the route exists and can be imported
                # Full WebSocket testing would require a different approach
                with client.websocket_connect("/ws/realtime?time_range=24h") as websocket:
                    # Receive connection message
                    data = websocket.receive_json()
                    assert data["type"] == "connection"
                    assert "message" in data
                    
                    # Receive at least one metrics update (with timeout)
                    # Note: In a real test, you'd want to wait for updates
                    # For now, we just verify the connection works
                    pass
            finally:
                app.dependency_overrides.clear()
    
    def test_get_connection_manager_singleton(self):
        """Test that get_connection_manager returns a singleton."""
        manager1 = get_connection_manager()
        manager2 = get_connection_manager()
        assert manager1 is manager2

