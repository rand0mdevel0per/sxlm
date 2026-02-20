# Getting Started

## Prerequisites

- CUDA 12.0+
- CMake 3.20+
- Python 3.10+
- GPU with 80GB+ VRAM (A100/H100 recommended)

## Installation

```bash
# Clone repository
git clone https://github.com/rand0mdevel0per/sxlm.git
cd sxlm

# Install Python dependencies
pip install -r requirements.txt

# Build CUDA components
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j8
```

## Running the API Server

```bash
# Start server with uvicorn
python src/python/run_server.py

# Or use uvicorn directly
uvicorn src.python.api.server:app --host 0.0.0.0 --port 8000
```

API will be available at `http://localhost:8000`

## Quick Test

```bash
# Run integration tests
./build/Release/integration_test

# Test API endpoint
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello Quila"}'
```

## Next Steps

- [Architecture Overview](/guide/architecture)
- [API Reference](/api/rest)
- [GCP Deployment](/deployment/gcp)
