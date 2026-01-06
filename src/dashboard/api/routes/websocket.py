"""WebSocket endpoint for real-time updates.

This module provides WebSocket support for real-time dashboard updates.
"""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.dashboard.api.dependencies import get_storage_adapter
from src.dashboard.models.websocket import (
    ConnectionMessage,
    ErrorMessage,
    MetricsUpdateMessage,
    SecurityMetricsUpdateMessage,
    PerformanceMetricsUpdateMessage,
    CircuitBreakerUpdateMessage,
)
from src.dashboard.services.circuit_breaker_service import CircuitBreakerService
from src.dashboard.services.metrics_aggregator import MetricsAggregator
from src.dashboard.services.performance_metrics import PerformanceMetricsService
from src.dashboard.services.security_metrics import SecurityMetricsService
from src.dashboard.services.websocket_manager import get_connection_manager

router = APIRouter(tags=["websocket"])

logger = logging.getLogger(__name__)

# Update interval in seconds
UPDATE_INTERVAL = 5.0


async def fetch_latest_metrics(storage, time_range: str = "24h"):
    """Fetch latest metrics from all services.
    
    Parameters:
        storage: Storage adapter instance
        time_range: Time range for metrics (default: 24h)
        
    Returns:
        Dictionary containing all metrics or None if error
    """
    try:
        # Initialize services
        metrics_aggregator = MetricsAggregator(storage)
        security_service = SecurityMetricsService(storage)
        performance_service = PerformanceMetricsService(storage)
        circuit_breaker_service = CircuitBreakerService(storage)
        
        # Fetch metrics (these are synchronous, but we're in async context)
        overview_result = metrics_aggregator.get_overview_metrics(time_range)
        security_result = security_service.get_security_metrics(time_range)
        performance_result = performance_service.get_performance_metrics(time_range)
        circuit_breaker_result = circuit_breaker_service.get_status()
        
        metrics = {}
        
        if overview_result.is_success():
            metrics["overview"] = overview_result.value
        else:
            logger.warning(f"Failed to fetch overview metrics: {overview_result.error}")
        
        if security_result.is_success():
            metrics["security"] = security_result.value
        else:
            logger.warning(f"Failed to fetch security metrics: {security_result.error}")
        
        if performance_result.is_success():
            metrics["performance"] = performance_result.value
        else:
            logger.warning(f"Failed to fetch performance metrics: {performance_result.error}")
        
        if circuit_breaker_result.is_success():
            metrics["circuit_breaker"] = circuit_breaker_result.value
        else:
            logger.warning(f"Failed to fetch circuit breaker status: {circuit_breaker_result.error}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}", exc_info=True)
        return None


@router.websocket("/ws/realtime")
async def websocket_realtime(
    websocket: WebSocket,
    time_range: str = "24h"
):
    """WebSocket endpoint for real-time metrics updates.
    
    This endpoint maintains a persistent connection and sends periodic updates
    including:
    - Overview metrics (every UPDATE_INTERVAL seconds)
    - Security metrics (every UPDATE_INTERVAL seconds)
    - Performance metrics (every UPDATE_INTERVAL seconds)
    - Circuit breaker status (every UPDATE_INTERVAL seconds)
    
    Parameters:
        websocket: WebSocket connection
        time_range: Time range for metrics queries (default: "24h")
        
    Message Types:
        - connection: Sent when connection is established
        - metrics_update: Overview metrics update
        - security_metrics_update: Security metrics update
        - performance_metrics_update: Performance metrics update
        - circuit_breaker_update: Circuit breaker status update
        - error: Error message
        
    Security Impact:
        - Validates WebSocket connection
        - Handles disconnections gracefully
        - Limits update frequency to prevent resource exhaustion
    """
    manager = get_connection_manager()
    await manager.connect(websocket)
    
    # Get storage adapter (we need it for fetching metrics)
    storage = get_storage_adapter()
    
    try:
        # Send connection confirmation
        connection_msg = ConnectionMessage(
            message="WebSocket connected. Real-time updates enabled.",
            server_time=datetime.now(timezone.utc)
        )
        await manager.send_personal_message(connection_msg, websocket)
        
        # Main update loop
        while True:
            # Fetch latest metrics
            metrics = await fetch_latest_metrics(storage, time_range)
            
            if metrics:
                # Send overview metrics update
                if "overview" in metrics:
                    overview_msg = MetricsUpdateMessage(data=metrics["overview"])
                    await manager.send_personal_message(overview_msg, websocket)
                
                # Send security metrics update
                if "security" in metrics:
                    security_msg = SecurityMetricsUpdateMessage(data=metrics["security"])
                    await manager.send_personal_message(security_msg, websocket)
                
                # Send performance metrics update
                if "performance" in metrics:
                    performance_msg = PerformanceMetricsUpdateMessage(data=metrics["performance"])
                    await manager.send_personal_message(performance_msg, websocket)
                
                # Send circuit breaker update
                if "circuit_breaker" in metrics:
                    cb_msg = CircuitBreakerUpdateMessage(data=metrics["circuit_breaker"])
                    await manager.send_personal_message(cb_msg, websocket)
            else:
                # Send error message if metrics fetch failed
                error_msg = ErrorMessage(
                    error="Failed to fetch metrics",
                    error_type="MetricsFetchError"
                )
                await manager.send_personal_message(error_msg, websocket)
            
            # Wait before next update
            await asyncio.sleep(UPDATE_INTERVAL)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            error_msg = ErrorMessage(
                error=f"Internal server error: {str(e)}",
                error_type="WebSocketError"
            )
            await manager.send_personal_message(error_msg, websocket)
        except Exception:
            # Connection might be closed, ignore
            pass
    finally:
        await manager.disconnect(websocket)
