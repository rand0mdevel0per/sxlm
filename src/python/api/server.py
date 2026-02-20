"""REST API server for Quila model"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import sys
sys.path.append('../../bindings')

try:
    import quila_core
    model = quila_core.QuilaModel(num_neurons=32768, hidden_dim=256)
except ImportError:
    model = None
    print("Warning: quila_core module not found, using mock mode")

app = FastAPI(title="Quila API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

class InferenceResponse(BaseModel):
    response: str
    tokens: int

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest) -> InferenceResponse:
    """Run inference on prompt"""
    if model:
        response = model.inference(request.prompt)
    else:
        response = f"Mock response to: {request.prompt}"
    return InferenceResponse(response=response, tokens=len(response.split()))

@app.get("/status")
async def status() -> Dict:
    """Get model status"""
    if model:
        return model.get_status()
    return {"status": "mock", "neurons": 32768}

@app.get("/health")
async def health() -> Dict:
    """Health check"""
    return {"status": "healthy"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming inference"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            if model:
                response = model.inference(data)
            else:
                response = f"Mock streaming: {data}"

            for token in response.split():
                await websocket.send_text(token + " ")
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
