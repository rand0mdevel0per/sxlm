"""Official Qualia Python API Client"""

import asyncio
import websockets
import json
from typing import AsyncIterator, List, Dict, Optional

class QualiaClient:
    def __init__(self, api_key: str, base_url: str = "ws://localhost:8000"):
        self.api_key = api_key
        self.base_url = base_url

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "defrateio/qualia-v1-20260221",
        effort: str = "adaptive",
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> AsyncIterator[Dict]:
        """Stream chat responses"""

        request = {
            "model": model,
            "max_tokens": max_tokens,
            "effort": effort,
            "temperature": temperature,
            "messages": messages,
            "mcp": True
        }

        uri = f"{self.base_url}/v1/chat/stream"

        async with websockets.connect(
            uri,
            extra_headers={"Authorization": f"Bearer {self.api_key}"}
        ) as websocket:
            # Send request
            await websocket.send(json.dumps(request))

            # Stream responses
            async for message in websocket:
                yield json.loads(message)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """Non-streaming chat (collects full response)"""

        full_response = ""
        async for chunk in self.chat_stream(messages, **kwargs):
            if chunk.get("msg_type") == "text" and chunk.get("text"):
                full_response += chunk["text"]

        return full_response
