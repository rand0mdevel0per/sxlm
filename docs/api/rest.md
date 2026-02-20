# REST API

## Base URL

```
http://localhost:8000
```

## Endpoints

### POST /inference

Run inference on a prompt.

**Request:**
```json
{
  "prompt": "Your prompt here",
  "max_tokens": 512,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "response": "Generated response",
  "tokens": 42
}
```

### GET /status

Get model status.

**Response:**
```json
{
  "neurons": 32768,
  "hidden_dim": 256,
  "phase": 0
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

## Example Usage

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing"}'
```
