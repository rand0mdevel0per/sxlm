# Architecture Overview

## QualiaTrace System

Qualia is built on the QualiaTrace architecture, featuring Plan-Think-Execute (PTE) flow with dynamic replanning.

## Core Components

### Plan-Think-Execute (PTE)

```
1. Plan: Planner-Port generates implicit instruction sequence
2. Think: Second read with biased attention
3. Execute: Generate final answer or tool calls
```

### Planner-Port

Generates implicit instruction sequences (hidden variables z) and effort signals for dynamic threshold adjustment.

### Ring Buffer

Monitors attention entropy and cosine similarity to detect logical drift. Triggers automatic replanning when threshold is exceeded.

### el-trace (Eligibility Trace)

Parameter-level credit assignment for reinforcement learning:
- Tracks contributions of each parameter
- Distributes multi-dimensional rewards (usefulness, conciseness, anti-hallucination)

### KFE (Key-Feature Encoding)

Tiered storage system:
- **GPU**: Hot cache (~10GB, <1ms access)
- **RAM**: Warm cache (~100GB, ~5ms access)
- **Disk**: Cold storage (~TB scale, ~10ms access)

Total effective parameters: 7B explicit + up to 143B implicit (KFE)

## Training Pipeline

1. **Stage 1**: SFT + Engram solidification (25h)
2. **Stage 2**: Long CoT training (15h)
3. **Stage 3**: AZR self-play (30h)
4. **Stage 4**: el-trace optimization (10h)

Total: 80 hours on single A100
