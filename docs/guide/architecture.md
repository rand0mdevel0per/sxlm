# Architecture

## System Overview

Quila uses a neuron-based dynamic graph architecture with 32,768 neurons operating in parallel.

## Neuron Structure

Each neuron maintains 8 channels of state:
- **S1**: Semantic (long-term knowledge)
- **S2**: Episodic (session memory)
- **S3**: Working (active computation)
- **S4**: Plan (goal representation)
- **S5**: Tool (external interaction)
- **S6**: Output (generation buffer)
- **S7**: Conflict (contradiction detection)
- **S8**: Meta (self-monitoring)

## Computation Streams

4 parallel streams process each neuron:
- **Stream A**: Micro-NSA (sparse attention)
- **Stream B**: SSM (state space model)
- **Stream C**: Linear Attention (RWKV-style)
- **Stream D**: DRC (residual correction)

## Inference Pipeline

6 phases execute sequentially:
1. **Phase 0**: VQ-GAN encoding
2. **Phase 1**: Replay (adaptive skip)
3. **Phase 2**: LastInput (demand identification)
4. **Phase 3**: Context re-read (HOT-NSA grading)
5. **Phase 4**: Plan generation
6. **Phase 5**: Think + Tool + Generate
7. **Phase 6**: Output (Attention Merge)

## Memory Hierarchy

- **L1 (GPU SRAM)**: Neuron states
- **L2 (GPU HBM)**: WMQ, active KFE
- **L3 (System RAM)**: SCT, cold KFE
- **L4 (NVRAM)**: Persona vector
- **L5 (NVMe)**: NMDB, Engram
