# WebSocket API

## Endpoint

```
ws://localhost:8000/ws
```

## Usage

Connect to the WebSocket endpoint for streaming inference.

### JavaScript Example

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => {
  ws.send('Your prompt here');
};

ws.onmessage = (event) => {
  console.log('Token:', event.data);
};
```

### Python Example

```python
import asyncio
import websockets

async def stream_inference():
    async with websockets.connect('ws://localhost:8000/ws') as ws:
        await ws.send('Your prompt here')
        async for message in ws:
            print(message, end=' ')

asyncio.run(stream_inference())
```

## Features

- Real-time token streaming
- Low latency response
- Connection management
