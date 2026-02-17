# API Reference

## Authentication

All requests require an API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

## WebSocket Streaming

Connect to: `ws://api.defrate.io/v1/chat/stream`

### Request Format

```json
{
  "model": "defrateio/qualia-v1-20260221",
  "max_tokens": 4096,
  "effort": "adaptive",
  "temperature": 0.3,
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "mcp": true
}
```

### Effort Levels

- `adaptive`: Planner-Port dynamically controls effort based on task complexity
- `high`: Full Think phase with deep reasoning
- `medium`: Lightweight Think phase
- `low`: Skip Think phase, direct execution

### Response Format

```json
{
  "msg_type": "text",
  "text": "Hello!",
  "continue": true,
  "model": "defrateio/qualia-v1-20260221"
}
```

### Message Types

- `plan`: Planning phase output
- `think`: Thinking phase output
- `text`: Final text response
- `mcp`: Tool execution results

## Python Client

```python
from sxlm.client import QualiaClient

client = QualiaClient(api_key="your-key")

# Streaming
async for chunk in client.chat_stream([
    {"role": "user", "content": "Explain quantum computing"}
]):
    if chunk["msg_type"] == "text":
        print(chunk["text"], end="")

# Non-streaming
response = await client.chat([
    {"role": "user", "content": "Hello"}
])
print(response)
```
