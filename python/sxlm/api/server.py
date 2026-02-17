"""Qualia API Server - WebSocket streaming handler"""

from fastapi import FastAPI, WebSocket, Header, HTTPException
from fastapi.responses import JSONResponse
import json
import asyncio
from typing import Optional

from .models import QualiaRequest, QualiaResponse, MessageType, EffortLevel

app = FastAPI(title="Qualia API", version="v0.1.0")

# Global model instance (will be loaded on startup)
model_instance = None

@app.post("/v1/chat/completions")
async def chat_completions(
    request: QualiaRequest,
    authorization: str = Header(...),
    x_protocol_version: str = Header(default="v0.1.0")
):
    """Main chat endpoint - returns WebSocket upgrade"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization")

    # Return WebSocket connection info
    return JSONResponse({
        "ws_url": "/v1/chat/stream",
        "protocol_version": x_protocol_version
    })

@app.websocket("/v1/chat/stream")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket streaming endpoint"""
    await websocket.accept()

    try:
        # Receive request
        data = await websocket.receive_text()
        request = QualiaRequest.parse_raw(data)

        # Stream responses
        async for response in generate_stream(request):
            await websocket.send_text(response.json())

    except Exception as e:
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()

async def generate_stream(request: QualiaRequest):
    """Generate streaming responses with PTE flow"""

    # Phase 1: Plan
    yield QualiaResponse(
        msg_type=MessageType.PLAN,
        text="Analyzing request and planning approach...",
        continue_=True,
        model=request.model
    )

    await asyncio.sleep(0.1)  # Simulate planning

    # Phase 2: Think (effort-based)
    if request.effort == EffortLevel.HIGH:
        # Full Think phase with deep reasoning
        yield QualiaResponse(
            msg_type=MessageType.THINK,
            text="Processing with deep reasoning...",
            continue_=True,
            model=request.model
        )
        await asyncio.sleep(0.1)
    elif request.effort == EffortLevel.MEDIUM:
        # Lightweight Think phase
        yield QualiaResponse(
            msg_type=MessageType.THINK,
            text="Processing...",
            continue_=True,
            model=request.model
        )
        await asyncio.sleep(0.05)
    elif request.effort == EffortLevel.ADAPTIVE:
        # Planner-Port dynamically controls effort based on task complexity
        # TODO: Integrate with actual Planner-Port effort signal
        planner_effort = 0.7  # Will come from Planner-Port.forward()
        if planner_effort > 0.6:
            yield QualiaResponse(
                msg_type=MessageType.THINK,
                text="Processing with adaptive reasoning...",
                continue_=True,
                model=request.model
            )
            await asyncio.sleep(0.1 * planner_effort)
    # LOW: Skip Think phase entirely

    # Phase 3: Execute - stream text response
    response_text = "I am Qualia, an AI assistant developed by Defrate.IO. "

    for i, char in enumerate(response_text):
        yield QualiaResponse(
            msg_type=MessageType.TEXT,
            text=char,
            continue_=True,
            model=request.model
        )
        await asyncio.sleep(0.01)

    # Final response
    yield QualiaResponse(
        msg_type=MessageType.TEXT,
        text="",
        continue_=False,
        stop_reason="end_turn",
        stop_seq=None,
        model=request.model,
        usage_billing=3.4,
        use_tier="standard"
    )
