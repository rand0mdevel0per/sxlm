---
layout: home

hero:
  name: Quila
  text: QualiaTrace Language Model
  tagline: Novel dynamic graph reasoning system with 161B effective parameters
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: View on GitHub
      link: https://github.com/rand0mdevel0per/sxlm

features:
  - title: Dynamic Graph Architecture
    details: 32,768 neurons with dynamic topology evolution, ~19% sparse activation
  - title: 6-Phase Inference Pipeline
    details: Replay, LastInput, Context Re-read, Plan, Think, Output with Higher-Order Thinking
  - title: Multimodal Support
    details: Text, image, audio, video via VQ-GAN encoding boundary
  - title: Multi-GPU Distributed
    details: CUDA Unified Memory with NCCL for efficient multi-GPU training/inference
---

## Quick Start

```bash
# Clone repository
git clone https://github.com/rand0mdevel0per/sxlm.git
cd sxlm

# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run API server
python src/python/run_server.py
```

## Architecture Highlights

- **161B effective parameters** (13B explicit + 148B implicit KFE)
- **Hybrid attention system** with NSA, SSM, Linear Attention, DRC
- **5-tier memory hierarchy** (GPU SRAM → NVMe disk)
- **Consistency Model** for 1-step denoising
