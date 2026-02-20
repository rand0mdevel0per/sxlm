# Quila (QualiaTrace Language Model)

Novel dynamic graph reasoning system with 161B effective parameters.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-vitepress-green.svg)](https://sxlm.pages.dev)

## Overview

Quila is a fundamentally different approach to language modeling that uses a neuron-based dynamic graph architecture instead of traditional Transformers. With 32,768 neurons operating in parallel and a 6-phase inference pipeline, Quila achieves sophisticated reasoning through dynamic topology evolution and multi-tier memory management.

## Key Features

- **161B Effective Parameters**: 13B explicit + 148B implicit KFE library
- **Dynamic Graph Architecture**: 32,768 neurons with ~19% sparse activation
- **6-Phase Inference Pipeline**: Replay → LastInput → Context → Plan → Think → Output
- **Hybrid Attention System**: NSA, SSM, Linear Attention, DRC streams
- **5-Tier Memory Hierarchy**: GPU SRAM → NVMe disk with intelligent eviction
- **Multimodal Support**: Text, image, audio, video via VQ-GAN encoding
- **Multi-GPU Distributed**: CUDA Unified Memory with NCCL for scaling

## Architecture

### Neuron Structure

Each neuron maintains 8 channels of state (S1-S8):
- **S1**: Semantic (long-term knowledge)
- **S2**: Episodic (session memory)
- **S3**: Working (active computation)
- **S4**: Plan (goal representation)
- **S5**: Tool (external interaction)
- **S6**: Output (generation buffer)
- **S7**: Conflict (contradiction detection)
- **S8**: Meta (self-monitoring)

### Computation Streams

4 parallel streams process each neuron:
- **Stream A**: Micro-NSA (Native Sparse Attention)
- **Stream B**: SSM (Mamba-style State Space Model)
- **Stream C**: Linear Attention (RWKV time-mix)
- **Stream D**: DRC (Dynamic Residual Correction)

### Memory Hierarchy

- **L1 (GPU SRAM)**: Neuron states
- **L2 (GPU HBM)**: WMQ, active KFE
- **L3 (System RAM)**: SCT, cold KFE
- **L4 (NVRAM)**: Persona vector
- **L5 (NVMe)**: NMDB, Engram

## Installation

### Prerequisites

- CUDA 12.0+
- CMake 3.20+
- Python 3.10+
- GPU with 80GB+ VRAM (A100/H100 recommended)

### Build

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

## Usage

### API Server

```bash
# Start server
python src/python/run_server.py

# Or use uvicorn directly
uvicorn src.python.api.server:app --host 0.0.0.0 --port 8000
```

### REST API

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing", "max_tokens": 512}'
```

### WebSocket Streaming

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.onopen = () => ws.send('Your prompt here');
ws.onmessage = (event) => console.log(event.data);
```

### Python Bindings

```python
import quila_core

model = quila_core.QuilaModel(num_neurons=32768, hidden_dim=256)
response = model.inference("Your prompt here")
print(response)
```

## Deployment

### GCP with Terraform

```bash
cd terraform/gcp
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your project ID
terraform init
terraform apply
```

See [deployment documentation](docs/deployment/gcp.md) for details.

## Documentation

Full documentation available at: https://sxlm.pages.dev

- [Getting Started](docs/guide/getting-started.md)
- [Architecture](docs/guide/architecture.md)
- [API Reference](docs/api/rest.md)
- [GCP Deployment](docs/deployment/gcp.md)

## Project Structure

```
sxlm/
├── src/
│   ├── cuda/          # CUDA kernels and core logic
│   ├── cpp/           # C++ utilities and storage
│   ├── python/        # Python API and bindings
│   └── tests/         # Integration tests
├── terraform/         # Infrastructure as code
├── docs/              # VitePress documentation
├── specs/             # Technical specifications
└── CMakeLists.txt     # Build configuration
```

## Testing

```bash
# Run integration tests
./build/Release/integration_test

# Test API
curl http://localhost:8000/health
```

## Performance

- **Inference Latency**: <1ms per neuron forward pass
- **Memory Efficiency**: ~19% sparse activation reduces compute
- **Scalability**: Linear scaling across multiple GPUs with NCCL

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{quila2024,
  title={Quila: QualiaTrace Language Model},
  author={Sintellix Research},
  year={2024},
  url={https://github.com/rand0mdevel0per/sxlm}
}
```

## Acknowledgments

Based on SX-SPEC-QUILA-001 Rev 0.3 specification.
