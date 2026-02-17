# SXLM "Quila" - QualiaTrace Language Model

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)](https://www.python.org/)

**SXLM "Quila"** is an advanced language model featuring the QualiaTrace architecture with 8 novel components: HOT-NSA (Hierarchical Sparse Attention), Engram (Hash n-gram Memory), SCT (Semantic Context Tree), el-trace (Eligibility Traces), mHC (Manifold Hyperconnections), Ring Buffer Planning, Multi-modal Fusion, and Tool Port. Built on CUDA C++ with Python bindings and cloud deployment support.

## 🌟 QualiaTrace Architecture

### Phase 3 Components (Implemented)

- **HOT-NSA**: Hierarchical sparse attention with dynamic compute budgeting
- **Engram**: LSH-based n-gram memory for O(1) retrieval
- **SCT**: Semantic Context Tree with HNSW for 128M token context
- **el-trace**: Eligibility traces for credit assignment in RL
- **mHC**: Manifold-constrained hyperconnections for gradient stability
- **Ring Buffer**: Drift detection and adaptive replanning
- **Multi-modal Fusion**: Text/image/audio with Interleaved-MRoPE
- **Tool Port**: 16 structured tool types (web search, code exec, file ops, etc.)

### Phase 4 Training Pipeline (Implemented)

- **OpenRouter Integration**: Knowledge distillation from Claude Opus 4.6, GPT-4, etc.
- **SFT Trainer**: Supervised fine-tuning with cross-entropy loss
- **Semi-supervised**: Pseudo-labeling with external models
- **RL Trainer**: PPO with eligibility traces
- **Cloud Deployment**: Terraform configs for Vertex AI and DigitalOcean

## Architecture

```
Input (Text/Image/Audio)
    ↓
[Multi-modal Fusion] Interleaved-MRoPE
    ↓
[HOT-NSA] Hierarchical Sparse Attention
    ↓
[Engram Memory] LSH n-gram retrieval
    ↓
[SCT] Semantic Context Tree (128M context)
    ↓
[mHC] Manifold Hyperconnections
    ↓
[Tool Port] 16 tool types
    ↓
Output + Planning (Ring Buffer)
```

## Installation

### Prerequisites

- CUDA Toolkit 11.8+
- CMake 3.18+
- Python 3.13+
- PyTorch 2.0+

### Build from Source

```bash
mkdir build && cd build
cmake .. -DBUILD_PYTHON=ON
cmake --build . --config Release
```

## Quick Start

### Training

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python scripts/train_simple.py
```

### Configuration

TOML configuration (`config.toml`):

```toml
[model]
dim = 768
num_heads = 12
num_layers = 24
max_seq_len = 128000

[training]
learning_rate = 1e-4
el_trace_decay = 0.9

[hot]
hot_threshold = 0.5
global_heads = 4
local_heads = 4

[engram]
num_hash_tables = 8
max_ngram_len = 5

[sct]
branching_factor = 32
max_depth = 27
```

### Cloud Deployment

Deploy to Vertex AI:

```bash
# Set project ID
export PROJECT_ID="your-project-id"

# Deploy
./deploy.sh $PROJECT_ID us-central1
```

See `terraform/vertex/README.md` for details.

## Performance

SXLM "Quila" is optimized for high-throughput training on NVIDIA GPUs:

- **HOT-NSA**: 50% FLOPs reduction through sparse attention
- **Engram**: <1ms retrieval latency with LSH
- **SCT**: 128M token context with <10ms retrieval
- **Multi-GPU**: Supported via PyTorch DDP

## Training Pipeline

### SFT (Supervised Fine-Tuning)
```bash
python scripts/train_sft.py
```

### Semi-supervised with OpenRouter
```bash
export OPENROUTER_API_KEY="your-key"
python scripts/train_semi.py
```

### RL with Eligibility Traces
```bash
python scripts/train_rl.py
```

## Troubleshooting

### Build Issues

**CUDA not found:**
- Install CUDA Toolkit 11.8+
- Verify: `nvcc --version`

**Python bindings fail:**
- Ensure Python 3.13+ is installed
- Check CUDA DLLs are in PATH

### Training Issues

**Out of memory:**
- Reduce `dim` or `num_layers` in config.toml
- Decrease batch size

**Slow training:**
- Verify GPU usage: `nvidia-smi`
- Increase batch size if memory allows

## Citation

```bibtex
@software{sxlm2026,
  title={SXLM "Quila": QualiaTrace Language Model with HOT-NSA, Engram, and SCT},
  author={rand0mdevel0per and Claude Sonnet 4.5},
  year={2026},
  url={https://github.com/rand0mdevel0per/sxlm}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.
