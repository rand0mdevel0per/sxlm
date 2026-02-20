# What is Quila?

Quila (QualiaTrace Language Model) is a novel dynamic graph reasoning system that differs fundamentally from traditional Transformer architectures.

## Key Features

- **161B effective parameters**: 13B explicit + 148B implicit KFE library
- **32,768 neurons**: Dynamic graph topology with ~19% sparse activation
- **6-phase inference pipeline**: Replay → LastInput → Context → Plan → Think → Output
- **Multimodal**: Text, image, audio, video via VQ-GAN encoding
- **Multi-GPU**: CUDA Unified Memory with NCCL for distributed training/inference

## Architecture Overview

Quila uses a neuron-based architecture where each neuron maintains 8 channels of state (S1-S8) and processes information through 4 parallel computation streams:

- **Stream A**: Micro-NSA (Native Sparse Attention)
- **Stream B**: SSM (Mamba-style State Space Model)
- **Stream C**: Linear Attention (RWKV time-mix)
- **Stream D**: DRC (Dynamic Residual Correction)

## Memory Hierarchy

5-tier storage system:
- **L1**: GPU SRAM (neuron states)
- **L2**: GPU HBM (WMQ, active KFE)
- **L3**: System RAM (SCT, cold KFE)
- **L4**: NVRAM (Persona vector)
- **L5**: NVMe SSD (NMDB, Engram)
