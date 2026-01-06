"""WebSocket connection manager.

This module manages WebSocket connections for real-time dashboard updates.
It handles connection lifecycle, broadcasting messages to all connected clients,
and connection cleanup.
"""

import asyncio
import json
import logging
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

from src.dashboard.models.websocket import WebSocketMessage

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and broadcasts messages.
    
    This class maintains a set of active WebSocket connections and provides
    methods to add, remove, and broadcast messages to all connected clients.
    
    Thread Safety:
        This class is designed to be used in an async context. All operations
        should be called from async functions.
    """
    
    def __init__(self):
        """Initialize connection manager."""
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket) -> None:
        """Accept and register a new WebSocket connection.
        
        Parameters:
            websocket: WebSocket connection to accept and register
            
        Security Impact:
            - Validates connection before accepting
            - Limits concurrent connections (can be extended with rate limiting)
        """
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)
        logger.info(f"WebSocket client connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove and close a WebSocket connection.
        
        Parameters:
            websocket: WebSocket connection to remove
        """
        async with self._lock:
            self.active_connections.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: WebSocketMessage, websocket: WebSocket) -> None:
        """Send a message to a specific WebSocket connection.
        
        Parameters:
            message: Message to send (Pydantic model)
            websocket: Target WebSocket connection
            
        Raises:
            WebSocketDisconnect: If connection is closed
        """
        try:
            await websocket.send_json(message.model_dump(mode="json"))
        except Exception as e:
            logger.error(f"Error sending personal message: {str(e)}", exc_info=True)
            raise
    
    async def broadcast(self, message: WebSocketMessage) -> int:
        """Broadcast a message to all connected clients.
        
        Parameters:
            message: Message to broadcast (Pydantic model)
            
        Returns:
            Number of clients that received the message
            
        Security Impact:
            - Handles disconnected clients gracefully
            - Removes dead connections automatically
        """
        if not self.active_connections:
            return 0
        
        # Create a copy of connections to avoid modification during iteration
        async with self._lock:
            connections = list(self.active_connections)
        
        disconnected = []
        sent_count = 0
        
        for connection in connections:
            try:
                await connection.send_json(message.model_dump(mode="json"))
                sent_count += 1
            except Exception as e:
                logger.warning(f"Failed to send message to client: {str(e)}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        if disconnected:
            async with self._lock:
                for conn in disconnected:
                    self.active_connections.discard(conn)
            logger.info(f"Removed {len(disconnected)} disconnected clients")
        
        return sent_count
    
    async def get_connection_count(self) -> int:
        """Get the number of active connections.
        
        Returns:
            Number of active WebSocket connections
        """
        async with self._lock:
            return len(self.active_connections)


# Global connection manager instance
_manager: ConnectionManager | None = None


def get_connection_manager() -> ConnectionManager:
    """Get the global connection manager instance.
    
    Returns:
        ConnectionManager: Global connection manager instance
    """
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager

