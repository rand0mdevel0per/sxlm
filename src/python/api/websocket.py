"""WebSocket handler for streaming inference"""

from typing import AsyncIterator, List
from fastapi import WebSocket
import asyncio

class WebSocketHandler:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.remove(websocket)

    async def stream_response(self, prompt: str, model=None) -> AsyncIterator[str]:
        """Stream response tokens"""
        if model:
            response = model.inference(prompt)
        else:
            response = f"Streaming response to: {prompt}"

        for token in response.split():
            yield token
            await asyncio.sleep(0.05)

    async def broadcast(self, message: str):
        """Broadcast message to all connections"""
        for connection in self.active_connections:
            await connection.send_text(message)
