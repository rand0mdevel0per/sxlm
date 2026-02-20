"""WebSocket handler for streaming inference"""

from typing import AsyncIterator

class WebSocketHandler:
    def __init__(self):
        self.active_connections = []

    async def connect(self, websocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)

    async def stream_response(self, prompt: str) -> AsyncIterator[str]:
        """Stream response tokens"""
        for token in prompt.split():
            yield token
