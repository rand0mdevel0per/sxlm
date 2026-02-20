"""REST API server for Quila model"""

from fastapi import FastAPI, WebSocket
from typing import Dict, List

app = FastAPI(title="Quila API")

@app.post("/inference")
async def inference(prompt: str) -> Dict:
    """Run inference on prompt"""
    return {"response": f"Response to: {prompt}"}

@app.get("/status")
async def status() -> Dict:
    """Get model status"""
    return {"status": "ready", "neurons": 32768}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming"""
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Echo: {data}")
