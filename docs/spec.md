# SINTELLIX QUILA (QUALIATRACE LM)
## Engineering Specification
### Document Number: SX-SPEC-QUILA-001
### Revision: 0.3 DRAFT

---

| Field | Value |
|---|---|
| Title | Sintellix Quila (QualiaTrace Language Model) — Engineering Specification |
| Document ID | SX-SPEC-QUILA-001 |
| Revision | 0.3 DRAFT |
| Status | Draft for Engineering Implementation |
| Classification | Internal / Confidential |

---

## Table of Contents

1. [Scope](#1-scope)
2. [Normative References](#2-normative-references)
3. [Terms, Definitions, and Abbreviations](#3-terms-definitions-and-abbreviations)
4. [System Overview](#4-system-overview)
5. [Neuron Internal Architecture](#5-neuron-internal-architecture)
6. [Hybrid Attention System](#6-hybrid-attention-system)
7. [Multimodal Positional Encoding](#7-multimodal-positional-encoding)
8. [Inter-Neuron Communication](#8-inter-neuron-communication)
9. [Memory and Storage Subsystem](#9-memory-and-storage-subsystem)
10. [Inference Pipeline](#10-inference-pipeline)
11. [Higher-Order Thinking System (HOT)](#11-higher-order-thinking-system-hot)
12. [Graph Topology Evolution](#12-graph-topology-evolution)
13. [Conflict Detection and Resolution](#13-conflict-detection-and-resolution)
14. [Credit Assignment (el-trace)](#14-credit-assignment-el-trace)
15. [Global Aggregation Layer](#15-global-aggregation-layer)
16. [Training Procedure](#16-training-procedure)
17. [Detailed Phase Data Flows](#17-detailed-phase-data-flows)
18. [VQ-GAN Encoding Boundary](#18-vq-gan-encoding-boundary)
19. [Persona Vector](#19-persona-vector)
20. [KFE Eviction Mechanism](#20-kfe-eviction-mechanism)
21. [.sxlm Model Format](#21-sxlm-model-format)
22. [Multi-GPU Distributed Inference](#22-multi-gpu-distributed-inference)
23. [Conformance Requirements](#23-conformance-requirements)

---

## 1. Scope

This document specifies the engineering requirements for the Sintellix Quila language model system, internally designated QualiaTrace LM. It covers all components necessary for correct implementation, including neuron computation, attention mechanisms, memory management, inference pipeline, training procedure, and deployment format.

### 1.1 System Profile

| Parameter | Value |
|---|---|
| Explicit parameters | ~13B |
| Implicit KFE equivalent | ~148B |
| Total effective parameters | ~161B |
| Neuron count | 32,768 (dynamic graph topology) |
| Sparse activation rate | ~19% at any given time |
| Maximum context length | 128M tokens (theoretical) |
| Numerical formats | fp8 (messages), fp16 (kernels), fp32 (accumulation) |

### 1.2 Architectural Classification

Quila is NOT a Transformer and NOT a classical Spiking Neural Network. It is a **dynamic graph reasoning system** in which:

- Each neuron is a self-contained hybrid computation unit
- Network topology evolves dynamically during training (edges may grow, be pruned, or be rewired)
- Sparse activation: approximately 19% of neurons are active at any moment
- Decentralised evolution: no global coordinator; a global aggregation layer using attention-based picking aligns local neuron behaviour with global objectives

---

## 2. Normative References

The following documents are referenced in this specification. Implementers SHALL comply with the referenced versions unless otherwise noted.

| Reference | Title |
|---|---|
| CUDA Toolkit ≥ 12.0 | NVIDIA CUDA C++ Programming Guide |
| cuDNN ≥ 8.9 | NVIDIA cuDNN Developer Guide |
| NCCL ≥ 2.18 | NVIDIA Collective Communications Library |
| pybind11 ≥ 2.11 | pybind11 — Seamless operability between C++11 and Python |
| protobuf ≥ 3.21 | Protocol Buffers Language Guide |
| zstd ≥ 1.5 | Zstandard Real-time Data Compression Algorithm |
| HNSW | Malkov & Yashunin (2018), Efficient and Robust ANN Search |
| B-link-tree | Lehman & Yao (1981), Efficient Locking for Concurrent Operations on B-Trees |
| mHC | DeepSeek (2025), Manifold-Constrained Hyper-Connections, arXiv:2512.24880 |
| Mamba/SSM | Gu & Dao (2023), Mamba: Linear-Time Sequence Modelling |
| RWKV | Peng et al. (2023), RWKV: Reinventing RNNs for the Transformer Era |
| NSA | Native Sparse Attention, as referenced in DeepSeek-V3 |
| Consistency Model | Song et al. (2023), Consistency Models |
| RoPE | Su et al. (2021), RoFormer: Enhanced Transformer with Rotary Position Embedding |
| VQ-GAN | Esser et al. (2021), Taming Transformers for High-Resolution Image Synthesis |
| AZR | Absolute Zero Reinforcement (Sintellix internal reference) |
| ZeRO-3 | Rajbhandari et al. (2020), ZeRO: Memory Optimization Towards Training Trillion Parameter Models |

---

## 3. Terms, Definitions, and Abbreviations

### 3.1 Terms and Definitions

**Neuron**: The fundamental computation unit of Quila. Each neuron encapsulates an eight-channel internal state (S1–S8), local parameters (W_in, W_out, four computation streams), and a Key Feature Encoding (KFE) library.

**Port**: A macro-level attention head group operating over the entire 32K neuron grid. Ports include Planner-Port, Thinker-Port×N, and Analysis-Port×N.

**KFE (Key Feature Encoding)**: A compressed snapshot of a neuron's activated state and attention parameters, stored in a per-neuron library and retrieved via HNSW approximate nearest-neighbour search.

**SCT (Session Context Tree)**: A tree-structured index of the current API call's context, built from VQ-GAN codebook embeddings. Lifetime: one API call.

**WMQ (Working Memory Queue)**: A hierarchical ring buffer serving as the scratchpad for the active reasoning session. Lifecycle bound to the producing stage.

**el-trace (eligibility trace)**: A per-parameter scalar that accumulates activation history for credit assignment. Used during training to propagate rewards to parameters that contributed to the outcome.

**stage**: A unit of Thinker reasoning bounded by a sharp drop in WMQ relevance. Tool call + return = one stage.

**turn**: An eventloop unit. A replan constitutes a new turn. A tool return does NOT constitute a turn boundary.

**session_el_trace**: A per-API-call copy of the el-trace, initialised from `persistent_el_trace` at call start and discarded at call end.

**persistent_el_trace**: The el-trace frozen after training, stored in S7. Read-only during inference.

**suspicious layer**: A conflict isolation mechanism in which a neuron whose output conflicts with expected semantics has its confidence reduced, propagating uncertainty downstream without halting computation.

**aggr**: A fixed-size aggregation vector produced by a Port's linear attention scan over the neuron grid. Size is O(1) regardless of neuron count.

**mHC (Manifold-Constrained Hyper-Connections)**: A residual mixing mechanism that constrains the mixing matrix to the Birkhoff polytope (doubly-stochastic matrices), ensuring bounded signal gain (≤ 1.6×) and closed-under-composition stability.

**NMDB (Neuron Model Data Bus)**: The internal data bus connecting neuron state storage, KFE libraries, and model parameters to peripheral modality stores (text, image, audio).

**CIC (Channel-Interleaved Container)**: A Protobuf multi-channel data container for multimodal data, supporting per-channel extraction and repacking.

### 3.2 Abbreviations

| Abbreviation | Expansion |
|---|---|
| AZR | Absolute Zero Reinforcement |
| CM | Consistency Model |
| CoT | Chain-of-Thought |
| DRC | Dynamic Residual Correction (predict-correct loop) |
| EMA | Exponential Moving Average |
| FXAA | Fast Approximate Anti-Aliasing (temporal smoothing, adapted from GPU rendering) |
| HBM | High-Bandwidth Memory |
| HOT | Higher-Order Thinking (system) |
| HNSW | Hierarchical Navigable Small World |
| KFE | Key Feature Encoding |
| MCP | Model Context Protocol |
| mHC | Manifold-Constrained Hyper-Connections |
| NSA | Native Sparse Attention |
| NVRAM | Non-Volatile RAM (e.g. Intel Optane, Samsung Z-NAND) |
| RoPE | Rotary Position Embedding |
| SCT | Session Context Tree |
| SFT | Supervised Fine-Tuning |
| SSM | State Space Model |
| UM | NVIDIA Unified Memory (`cudaMallocManaged`) |
| VQ-GAN | Vector-Quantised Generative Adversarial Network |
| WMQ | Working Memory Queue |

---

## 4. System Overview

### 4.1 Top-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer (REST + WebSocket + MCP)    │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                VQ-GAN Encoding Boundary                  │
│   Input: text / image / audio / video → codebook idx    │
│   Output: neuron grid output → VQ-GAN decoder → tokens  │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│              Macro Port Layer (HOT System)               │
│   Planner-Port | Thinker-Port×N | Analysis-Port×N       │
│   Linear Attention scan → utility → NSA → Intercept     │
└───────────────────────────┬─────────────────────────────┘
                            │ h_final (all neurons)
┌───────────────────────────▼─────────────────────────────┐
│            Neuron Grid  (32,768 neurons, CUDA)           │
│   Each neuron: W_in → [NSA|SSM|LinAttn|DRC] → mHC       │
│                → KFE recall → CM → FXAA → h_final       │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│              Five-Tier Memory & Storage                  │
│  L1 GPU SRAM | L2 GPU HBM | L3 RAM-hot | L4 RAM/Disk   │
│  L5 Disk (NMDB, zstd)                                   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Implementation Languages

| Component | Language / Runtime |
|---|---|
| Neuron computation core | CUDA C++ (`.cu`) |
| VQ-GAN encoder/decoder | CUDA C++ (`.cu`) |
| Tiered storage manager | CUDA C++ (`.cu`) |
| Python bindings | pybind11 |
| Training orchestration | Python + DeepSpeed + ZeRO-3 |
| API server | Python (REST + WebSocket) |
| MCP integration | Python (native MCP protocol) |

### 4.3 Key Design Invariants

The following invariants SHALL be maintained by all implementations:

1. **Confidence–el-trace coupling**: The pathway `suspicious_state → confidence_drop → el-trace_drop → gradient_scaling` is a single unified mechanism. No additional mechanism SHALL be introduced for gradient scaling of suspicious neurons.
2. **Stage-lifecycle WMQ**: WMQ entries are bound to the stage that produced them. No explicit GC mechanism is required; lifecycle management is implicit.
3. **Codebook fixity**: The VQ-GAN codebook is frozen post-training. The embedding space is stable. SCT construction is entirely codebook-based and orthogonal to neuron suspicious state.
4. **CodeBook output-only**: The CodeBook and VQ-GAN decoder are used ONLY at the output boundary. Internal neuron-to-neuron communication and all inter-Port aggregation operate entirely in continuous latent space.
5. **NSA utility signal unity**: At both micro level (within-neuron) and macro level (Port over neuron grid), the NSA sparse mask is driven by the same utility signal family (el-trace + linear attention weights + activation magnitude). No separate gating mechanism is introduced.


---

## 5. Neuron Internal Architecture

### 5.1 State Channel Specification (S1–S8)

Each neuron maintains eight internal state channels. All channels MUST be preserved across forward passes within a session.

| Channel | Name | Contents | Dtype | Persistence |
|---|---|---|---|---|
| S1 | Activation | current activation scalar, activation history ring (16 entries), refractory counter | fp32 | session |
| S2 | Computational | hidden_state R^D, ssm_state, drc_accumulator, residual_stream | fp32 | step |
| S3 | Temporal | 8-frame hierarchical memory, momentum vector | fp16 | session |
| S4 | Identity | attention_type enum, receptive_field mask, specialization_vector R^D, graph_position | fp32 | persistent |
| S5 | Confidence | output_confidence [0,1], input_reliability [0,1], uncertainty_vector R^D, calibration | fp32 | step |
| S6 | Connectivity | active_connections list, routing_table, neighbour_summary embedding | u32+fp16 | session |
| S7 | Plasticity | Adam moment m, Adam moment v, persistent_el_trace, learnable_lambda | fp32 | persistent |
| S8 | Utility | utility_ema scalar, contribution_history ring (32 entries), eviction_priority | fp32 | session |

**Req 5.1.1**: S4 and S7 MUST be saved to `.sxlm` and restored on deployment. They encode trained knowledge and SHALL NOT be reinitialised at runtime.

**Req 5.1.2**: S1, S3, S6, S8 are session-scoped; initialised from `.sxlm` baseline values at call start; discarded at call end.

**Req 5.1.3**: S2 and S5 are step-scoped transient buffers; persistence is NOT required.

### 5.2 Neuron Forward Pass (Ordered Steps)

```
INPUT
  Messages from predecessor neurons.
  Each message: fp8 256x256 matrix with embedded position tag (pos_i).

STEP 1 — INPUT AGGREGATION
  For each incoming message m_i with pos_i:
    Place m_i into base matrix B at pos_i.
    If pos_i already occupied: B[pos_i] = mean(B[pos_i], m_i)
  B_expanded = ConvTranspose(B, W_in_conv_8x8)  // fp16 kernel
  B_expanded += S2.residual_stream
  h = W_in * B_expanded                          // GEMM, fp16 weights, fp32 accum
  // h in R^D

STEP 2 — REFRACTORY CHECK
  If S1.refractory_counter > 0:
    S1.refractory_counter -= 1
    RETURN previous h_final (skip all remaining steps)

STEP 3 — RECEPTIVE FIELD FILTER
  Apply S4.receptive_field mask; zero out non-matching dimensions.

STEP 4 — FOUR PARALLEL COMPUTATION STREAMS (concurrent CUDA streams)

  Stream A — Micro-NSA:
    h_filtered = top-k dims of h by S8.utility_score
    Q, K, V = Linear_A(h_filtered)
    out_A = SparseAttention(Q, K, V)

  Stream B — SSM (Mamba-style):
    input_b = concat(h, S3.frame_memory)
    Delta = softplus(W_Delta * input_b)
    A = -exp(W_A)
    B_ssm = W_B * input_b
    C = W_C * input_b
    out_B = selective_scan(input_b, Delta, A, B_ssm, C)
    S3.frame_memory = shift_and_prepend(S3.frame_memory, h)

  Stream C — Linear Attention (RWKV time-mix):
    r = sigmoid(W_r * h)
    k = W_k * h
    v = W_v * h
    wkv_new = exp(w + k) * v + S2.ssm_state
    S2.ssm_state = exp(w) * S2.ssm_state + exp(k) * v  // O(1)
    out_C = r * sigmoid(wkv_new)
    wkv_weights = softmax(wkv_new)  // recorded as per-dim utility signal

  Stream D — DRC (predict-correct):
    n = adaptive_iters(S5.output_confidence)
      // confidence > 0.8 => n = 1..2
      // confidence > 0.5 => n = 3..6
      // confidence <= 0.5 => n = 8..16
    h_drc = h; h_prev = h
    for i in 1..n:
      h_pred = W_pred * h_drc
      h_drc  = h_drc + W_corr * (h - h_pred)
      if cosine_sim(h_drc, h_prev) > 0.999: break
      h_prev = h_drc
    out_D = h_drc

STEP 5 — mHC MIXING (micro-level, baseline+opt)
  gate = softmax(W_gate * S4.specialization_vector)  // R^4
  fused = gate[0]*out_A + gate[1]*out_B + gate[2]*out_C + gate[3]*out_D
  W_raw = W_baseline + delta_W_opt   // delta_W_opt=0 for non-specialised neurons
  H_res = SinkhornKnopp(W_raw, n=20) // doubly-stochastic, gain <= 1.6x
  h_mid = H_res * concat(h, fused)

STEP 6 — KFE RECALL
  candidates = HNSW_search(neuron.kfe_lib, h_mid, top_k=32)
  weights = softmax([dot(h_mid, c.embedding) for c in candidates])
  h_kfe = h_mid + sum(weights[i] * candidates[i].state_snapshot)

STEP 7 — CONSISTENCY MODEL DENOISING
  t = 1.0 - S5.output_confidence
  h_clean = CM_forward(h_kfe, t)          // 1-step
  if S5.output_confidence < 0.4:          // optional 2-step refinement
    eps = N(0, 0.01)
    h_clean = CM_forward(h_clean + eps, t * 0.5)

STEP 8 — FXAA TEMPORAL SMOOTHING
  h_smooth = lerp(h_clean, S3.frame_memory[0], alpha_fxaa)
  // alpha_fxaa: learnable scalar in [0, 0.5]

STEP 9 — STATE UPDATES
  S1.activation_history.push(mean(|h_smooth|))
  S1.refractory_counter = floor(ref_base - S5.output_confidence * ref_range)
  S2.hidden_state = h_smooth
  S5.output_confidence = sigmoid(W_conf * h_smooth)
  S6: update routing weights
  S8.utility_ema = beta*S8.utility_ema + (1-beta)*utility(h_smooth)
  session_el_trace[neuron_id] += mean(|h_smooth|)

STEP 10 — KFE GENERALISATION CHECK (Phase 3+ only)
  if (S1.activation_history.mean() > alpha_kfe)
  AND (S8.utility_ema > beta_kfe)
  AND (S5.output_confidence > gamma_kfe):
    entry.embedding = h_mid
    entry.state_snapshot = h_smooth
    entry.utility = S8.utility_ema
    entry.confidence = S5.output_confidence
    HNSW_insert(neuron.kfe_lib, entry)
    // Merge: if cosine_sim(entry, existing) > 0.97 => merge, keep higher utility

STEP 11 — THERMAL NOISE
  sigma = temperature * (1.0 - S5.output_confidence)
  h_final = h_smooth + N(0, sigma^2)

STEP 12 — OUTPUT ROUTING
  for each target t in S6.routing_table:
    msg = conv(W_out * h_final, edge_kernel_8x8[t])  // -> fp8 256x256 + pos
    enqueue_message(target=t, payload=msg)
  global_aggregation_buffer[neuron_id] = h_final
```

### 5.3 Numerical Precision

| Operation | Compute | Accumulation |
|---|---|---|
| W_in GEMM | fp16 | fp32 |
| SSM scan | fp16 | fp32 |
| RWKV wkv | fp16 | fp32 |
| NSA (micro) | fp16 | fp32 |
| mHC SinkhornKnopp | fp32 | fp32 |
| CM inference | fp16 | fp32 |
| Inter-neuron messages | fp8 E4M3 | — |
| Convolutional kernels | fp16 | fp32 |
| el-trace | fp32 | fp32 |

**Req 5.3.1**: All reduction operations SHALL use fp32 accumulation.

**Req 5.3.2**: Inter-neuron messages SHALL be quantised to fp8 E4M3 before transmission; dequantised to fp16 at receiver before GEMM.

### 5.4 Refractory Period Parameters

```
refractory_counter = floor(ref_base - S5.output_confidence * ref_range)
  ref_base  : default = 4
  ref_range : default = 3
  Effective range: [1, 4] steps
```

During refractory, the neuron emits its previous `h_final` unchanged. This prevents runaway oscillation in densely connected sub-graphs.


---

## 6. Hybrid Attention System

The attention system operates at two independent levels. Implementors SHALL NOT conflate them.

### 6.1 Level 1 — Micro-Level (within neuron, Section 5.2 Stream A)

Described in Section 5.2, Stream A. The NSA inside each neuron operates on the neuron's own hidden state `h`, selecting the top-k high-utility dimensions. It does NOT involve any other neuron's state.

### 6.2 Level 2 — Macro-Level Port Attention (over 32K neuron grid)

Each Port (Planner, Thinker_i, Analysis_i) is a group of attention heads operating on the collection of all neurons' `h_final` outputs.

#### 6.2.1 Port Attention Steps

```
STEP 1 — LINEAR ATTENTION SCAN (full neuron grid, O(1) aggr)
  Inputs: h_final[0..32767]  (reported by neurons in Step 12)
  Mask:   zero out contributions from this Port's own previous activation
          (prevents self-loop; forces cross-Port information flow)

  RWKV-style scan over neuron grid:
    For each neuron i (in graph-community order):
      r_i = sigmoid(W_r_port * h_final[i])
      k_i = W_k_port * h_final[i]
      v_i = W_v_port * h_final[i]
      wkv_state = exp(w_port) * wkv_state + exp(k_i) * v_i
    aggr = r_last * sigmoid(wkv_state)
    // aggr: fixed-size vector, O(1) regardless of neuron count
    // wkv per-neuron weights are ALSO the utility contribution for Step 2

  aggr is written to global_aggregation_buffer[port_id]
  Other Ports read this buffer in their own linear scan.

STEP 2 — UTILITY COMPUTATION AND NSA MASK
  For each neuron i:
    utility_i = alpha * wkv_weight_i            // reused from Step 1 linear scan
              + beta  * session_el_trace[i]     // current context activation
              + gamma * persistent_el_trace[i]  // training-time importance

    alpha, beta, gamma: learnable scalars per Port
    alpha schedule (varies by inference phase):
      Phase 1: alpha=0.2   (rely mainly on prior knowledge)
      Phase 2: alpha=0.5
      Phase 3: alpha=0.7
      Phase 5: alpha=0.9   (rely mainly on current context)

  NSA_mask = top-k indices of utility_i  // k is a hyperparameter per Port

STEP 3 — NSA SPARSE ATTENTION (over masked neurons)
  Q = W_Q_port * port_current_state
  K = W_K_port * stack(h_final[NSA_mask])
  V = W_V_port * stack(h_final[NSA_mask])
  out_NSA = softmax(Q*K^T / sqrt(D)) * V

STEP 4 — INTERCEPT LAYER
  inputs = concat(aggr, out_NSA, h_final[NSA_mask])
  W_raw_port = W_baseline_port + delta_W_port
  H_res_port = SinkhornKnopp(W_raw_port, n=20)
  merged = H_res_port * inputs

  // Two output paths:
  Path A (output boundary only):
    merged -> CodeBook lookup -> VQ-GAN decoder -> multimodal tokens
    THIS PATH IS ONLY TAKEN by Generator-Port for final output
    or by Tool-Port for MCP serialisation

  Path B (internal, all Ports):
    new_aggr = Linear_compress(merged)
    Apply backward alignment layer:
      if ||new_aggr - other_port_aggr|| > tau:
        new_aggr = W_back * new_aggr + (1-W_back) * other_port_aggr
    global_aggregation_buffer[port_id] = new_aggr
```

#### 6.2.2 Cross-Port Perception Relationships

| Port | Linear scan perceives | NSA attends |
|---|---|---|
| Planner | all Thinker aggr (excl. self) | highest utility Thinker-related neurons |
| Thinker_i | planner_aggr + other Thinker aggr (excl. self) + analysis_i_aggr | highest utility context neurons |
| Analysis_i | thinker_i_aggr only (bound pair) | highest utility SCT/Engram neurons |

#### 6.2.3 mHC Parameter Structure (macro-level)

The macro-level mHC parameters are INDEPENDENT of the micro-level mHC in Section 5.2.

```
W_raw_port = W_baseline_port + delta_W_port
  W_baseline_port: globally shared across all Ports of same type
  delta_W_port:    per-Port optional fine-tuning (may be zero)

SinkhornKnopp(W, n):
  for iter in 1..n:
    W = W / row_sums(W)   // normalise rows
    W = W / col_sums(W)   // normalise cols
  return W
  // Convergence guaranteed; n=20 is sufficient for fp32
```

### 6.3 Consistency Model Specification

The Consistency Model (CM) replaces DDPM for latent-space denoising within neurons.

```
Self-consistency property:
  f(x_t1, t1) = f(x_t2, t2)  for all t1, t2 on the same PF-ODE trajectory

Training objective (Consistency Distillation from DDPM teacher):
  L_CD = E[ d(f_theta(x_t_{n+1}, t_{n+1}), f_theta_ema(x_t_n, t_n)) ]
  where d is a distance metric (e.g. L2 or LPIPS)
  f_theta_ema: exponential moving average of f_theta (student -> teacher)

Inference:
  Standard 1-step:  h_clean = f(h_noisy, t_max)
  2-step refinement (low confidence path):
    h_mid = f(h_noisy, t_max)
    h_clean = f(h_mid + N(0, 0.01), t_mid)

Noise schedule mapping:
  t = 1.0 - S5.output_confidence
  t_max = 1.0  (fully uncertain)
  t_min = 0.0  (fully confident, CM is identity)
```

**Req 6.3.1**: CM MUST be trained using Consistency Distillation (CD) from a pre-trained DDPM teacher. Consistency Training (CT) from scratch is permitted as an alternative if no DDPM teacher is available, but is expected to converge more slowly.

---

## 7. Multimodal Positional Encoding

All modalities are encoded with Rotary Position Embedding (RoPE). The rotation matrices differ per modality to reflect the native spatial structure.

### 7.1 Per-Modality RoPE Formulas

```
Text (1D sequence):
  For token at position i, dimension pair (2k, 2k+1):
    theta_i_k = i * base_text^(-2k/d)
    Rotation: [cos(theta), -sin(theta); sin(theta), cos(theta)]

Image (H x W patch grid):
  For patch at (row r, col c):
    theta_row_k = r * base_row^(-2k/d)
    theta_col_k = c * base_col^(-2k/d)
    R_patch = R_row (x) R_col    // Kronecker product

Audio (T frames x F frequency bins):
  For frame at (time t, freq f):
    theta_t_k = t * base_time^(-2k/d)
    theta_f_k = f * base_freq^(-2k/d)
    R_audio = R_time (x) R_freq

Video (T x H x W):
  For voxel at (time t, row r, col c):
    theta_t_k, theta_r_k, theta_c_k computed independently
    R_video = R_time (x) R_row (x) R_col
    // base_time is INDEPENDENT from base_row, base_col
    // Allows different temporal and spatial receptive field scales
```

### 7.2 Base Parameters

| Modality | Base parameters |
|---|---|
| Text | `base_text` (default 10000, learnable) |
| Image | `base_row`, `base_col` (independently learnable) |
| Audio | `base_time_audio`, `base_freq` (independently learnable) |
| Video | `base_time_video`, `base_row`, `base_col` (base_time_video independent) |

**Req 7.2.1**: Temporal and spatial base parameters for video MUST be trained independently to allow the model to adapt to varying frame rates and resolutions without retraining.

### 7.3 Cross-Modal Alignment

- VQ-GAN training ensures semantically equivalent concepts across modalities are mapped to nearby codebook vectors.
- RoPE is applied AFTER codebook lookup; it affects relative position awareness but does NOT disturb semantic alignment.
- Neuron specialisation (S4.attention_type) determines which positional structure a neuron is sensitive to.

---

## 8. Inter-Neuron Communication

### 8.1 Message Format

```
Message structure (per edge):
  payload: fp8 E4M3, shape [256, 256]   // quantised activation map
  pos:     u16 (2D position tag)         // where to place in receiver's base matrix
  sender:  u32 (neuron ID)
  flags:   u8  (partial_send flag, etc.)
```

**Total message size**: 256*256*1 (payload) + 2+4+1 (header) ≈ 65,543 bytes per message.

### 8.2 Edge Parameter Layout

Each directed edge (sender → receiver) carries:

| Parameter | Size | Description |
|---|---|---|
| W_out projection | ~256–1024 params (fp16) | Sender-side linear compression |
| Convolutional kernel | 8×8 fp16 = 64×2 = 128 bytes | Sender-side learnable conv; selectively focuses on subregions |
| **Total per edge** | ~1–2 KB | — |
| **Total (32K × 100 avg edges)** | ~6.4 GB | Stored in neuron graph in `.sxlm` |

### 8.3 Partial Send Mechanism

When a neuron computes a strong attention focus on a sub-region of its activation map, it MAY send only that sub-region with the `partial_send` flag set.

```
If max_attention_weight(subregion) > partial_send_threshold:
  msg.payload = conv(W_out * h_final[subregion], kernel)
  msg.pos = subregion_offset
  msg.flags |= PARTIAL_SEND_FLAG
  // Receiver appends to base matrix at msg.pos, does NOT overwrite other regions
```

### 8.4 Receiver Processing

```
On receiving message m:
  If m.flags & PARTIAL_SEND_FLAG:
    B[m.pos] = mean(B[m.pos], m.payload) if occupied else m.payload
  Else:
    Scatter m.payload across B at positions derived from m.pos
    Average with existing values where overlap occurs
Apply residual: B += residual_stream
Proceed to W_in GEMM
```

### 8.5 Edge Parameter Scale

**Req 8.5.1**: Edge parameter tensors (W_out, conv kernels) MUST be stored per directed edge. They are NOT shared between edges. Total edge parameter storage MUST be budgeted at ~6.4 GB for the default configuration (32K neurons, 100 edges average per neuron).


---

## 9. Memory and Storage Subsystem

### 9.1 Five-Tier Storage Architecture

| Tier | Location | Contents | Index / Access | Latency |
|---|---|---|---|---|
| L1 | GPU SRAM (shared mem) | Active neuron S1–S5, S8 (hot) | Direct register | <1 μs |
| L2 | GPU HBM (VRAM) | Explicit parameters (13B), W_in/W_out/stream params | CUDA UM pages | <1 μs |
| L3 | CPU RAM (hot zone) | KFE hot library (~148B equiv), WMQ hot | HNSW in-memory | ~10 ms |
| L4 | CPU RAM / SSD | SCT, Engram, VQ-GAN codebook B-link-tree | HNSW / B-link | ~50 ms |
| L5 | NVMe / HDD | NMDB cold storage (zstd compressed), KFE cold | B-tree file index | ~1–50 ms |

**Req 9.1.1**: All GPU memory (L1 + L2) for neuron states and parameters MUST be allocated with `cudaMallocManaged` (Unified Memory). This enables automatic demand-paged migration across GPU devices and CPU RAM without explicit `cudaMemcpy` calls.

**Req 9.1.2**: NVRAM (e.g. Intel Optane, Samsung Z-NAND, if available) SHALL be used to store the Persona vector (Section 19) due to its combination of low access latency and persistence.

### 9.2 KFE (Key Feature Encoding) Library

#### 9.2.1 Entry Structure

```protobuf
message KFEEntry {
  uint64  kfe_id            = 1;
  bytes   embedding         = 2;  // fp16 R^D, used as HNSW key
  bytes   state_snapshot    = 3;  // fp16 R^D
  float   utility           = 4;
  float   confidence        = 5;
  uint32  access_count      = 6;
  int64   last_access_epoch = 7;
  bool    soft_delete_flag  = 8;
}
```

#### 9.2.2 KFE Lifecycle

```
CREATE:
  Conditions: S1.mean_activation > alpha_kfe
           AND S8.utility_ema > beta_kfe
           AND S5.output_confidence > gamma_kfe
           AND Phase >= 3
  Action: insert entry into HNSW; check for merge (cosine_sim > 0.97 => merge)

RECALL:
  HNSW_search(query=h_mid, top_k=32)
  Result weighted by dot(h_mid, entry.embedding) softmax
  entry.access_count += 1; entry.last_access_epoch = current_epoch

MERGE:
  When inserting: if cosine_sim(new.embedding, existing.embedding) > 0.97:
    merged.utility = max(new.utility, existing.utility)
    merged.embedding = normalise(new.embedding + existing.embedding)
    Remove existing; insert merged

EVICT:
  See Section 20 for full eviction specification.

SOFT DELETE:
  entry.soft_delete_flag = True
  Entry retained for K turns; if no positive activation in K turns: HARD DELETE
  K: default = 8 turns

HARD DELETE (permanent):
  Conditions:
    (A) el-trace receives extreme negative reward (AZR hallucination penalty)
     OR (B) introspection detects irreconcilable logical contradiction
    AND soft_delete_flag = True AND no positive activation in K turns
```

#### 9.2.3 HNSW Index Parameters

```
M         = 16        // number of bidirectional links per node
ef_construction = 200 // search width during construction
ef_search   = 64      // search width during query
distance    = cosine  // L2-normalised dot product
layers      = floor(ln(N)) + 1
```

### 9.3 SCT (Session Context Tree)

```
Lifetime: one API call; destroyed on call completion.
Builder:  VQ-GAN codebook side (orthogonal to neuron state).
Index:    HNSW (in-memory, CPU RAM).
Structure: tree nodes = codebook embedding vectors with position tags.
           Edges = temporal/spatial adjacency derived from context order.

Construction phases:
  Phase 1 (Replay): build skeleton (coarse nodes, ~1 per context block)
  Phase 3 (Re-read): refine (add detail nodes for high-utility regions)
  Phase 4 (Plan):   backward query starts; relevant subtrees prefetched

API cache:
  If cache hit on context hash: reuse previous SCT partially.
  On provider load-balancing miss: rebuild fully.
```

### 9.4 Working Memory Queue (WMQ)

```
Structure: hierarchical ring buffer
  Hot zone: GPU HBM (CUDA UM), capacity = wm_hot_size (default 512 entries)
  Cold zone: CPU RAM, capacity = wm_cold_size (default 4096 entries)

Entry structure:
  content:      fp16 R^D
  relation:     fp32 scalar (relevance score)
  stage_id:     u32 (producing stage)
  thinker_id:   u32 (producing thinker)
  trajectory_id: u32

Lifecycle:
  ENQUEUE: on high-relevance Thinker output (relation > wm_enqueue_threshold)
  EVICT to cold: ring buffer overflow; lowest relation entry demoted
  STAGE END: entries with matching stage_id moved to deletion_buffer
  DELETION: turn end; entries in deletion_buffer are freed
  NOTE: entries with utility > utility_dealloc_threshold are deprioritised
        for deletion (may survive one extra turn in deletion_buffer)

Retrieval:
  Attention-based: q = current Thinker state
                   scores = dot(q, entry.content) for all entries
                   return top-k by score, filtered by trajectory compatibility
```

### 9.5 NMDB (Neuron Model Data Bus)

```
Main channel:   neuron states, KFE storage, model parameters
Peripheral channels (isolated):
  text_channel:   text token sequences, LRU cache
  image_channel:  image patch embeddings, LRU cache
  audio_channel:  audio frame embeddings, LRU cache

CIC (Channel-Interleaved Container) format: protobuf
  Supports per-channel extraction and repacking.
  Used for composing multimodal inputs from multiple peripheral channels.

Storage backend: L5 NVMe, zstd compressed (level 3 for hot path, level 19 for archival)
```

### 9.6 Engram

```
Purpose: compressed long-term factual knowledge store.
          SFT-phase training compresses static knowledge into Engram.
Access:   O(1) hash-based lookup.
Training: "closed-book" Engram training: force model to reconstruct facts
          purely from Engram without KFE or SCT.
Persistence: stored in .sxlm Block 6; loaded at deployment.
```


---

## 10. Inference Pipeline

The inference pipeline consists of six sequential phases (Phase 0–5) plus an asynchronous Replan mechanism. Each API call executes all phases.

### 10.1 Phase 0 — Encoding

```
Input: raw multimodal data (text tokens, image patches, audio frames, video voxels)

For each modality:
  1. VQ-GAN encoder forward pass -> codebook indices
  2. Look up codebook vectors -> continuous embeddings
  3. Apply per-modality RoPE (Section 7)
  4. Quantise to fp8 for transmission

Output: sequence of (codebook_idx, embedding, position_tag) tuples
        fed into neuron grid as initial messages
```

### 10.2 Phase 1 — Replay

```
Goal: warm all neurons with full context; build SCT skeleton; init Engram.

Input: full context token sequence (Phase 0 encoded)
Active Ports: Planner-Port only; all Thinker-Ports remain in offload state.

Processing (adaptive skip per context block):
  Compute: cos_sim = cosine_similarity(block_embedding, last_msg_embedding)

  cos_sim < theta_low   [LOW RELEVANCE]:
    Execute: update S3.frame_memory only + SCT skeleton node insertion
    Skip:    NSA Stream A, DRC Stream D, CM denoising
    session_el_trace weight: x 0.1

  theta_low <= cos_sim < theta_high   [MEDIUM RELEVANCE]:
    Execute: full 4-stream forward pass
    Skip:    KFE generalisation write
    session_el_trace weight: x 1.0

  cos_sim >= theta_high   [HIGH RELEVANCE]:
    Execute: full 4-stream forward pass + KFE micro-update (lr_kfe_replay << lr_default)
    session_el_trace weight: x 1.2

Parallel (independent of neuron computation):
  SCT construction (codebook side)
  Engram initialisation

Phase 1 end state:
  h_0: all neurons warmed
  SCT skeleton established
  Engram initialised
  session_el_trace reflects activation distribution across context
```

**Threshold defaults**: `theta_low = 0.3`, `theta_high = 0.7`

### 10.3 Phase 2 — Last-Input Read-in

```
Goal: precise demand identification; seed KFE candidate set.

Input: last user message ONLY (re-encoded from Phase 0 output; no new VQ-GAN call)
Active Ports: Planner-Port.

Processing:
  Full 4-stream forward pass with NO adaptive skip.
  NSA (micro + macro): masks fully open; all dimensions / positions attended.
  DRC: force maximum iterations (ignore confidence adaptive threshold).
  KFE recall: ENABLED; seed initial candidate set.

Output:
  need_vector = Linear_need(mean_pool(h_final[all neurons]))
  KFE candidate queue initialised with Phase 2 hits.
  session_el_trace: x 2.0 weight for neurons activated by last message.
  No token output generated.
```

### 10.4 Phase 3 — Context Re-read

```
Goal: re-read full context biased by need_vector; finalise SCT; open KFE writes.

Input: full context (Phase 0 encodings reused; no re-encode)
       need_vector injected as attention bias into Port-layer NSA
Active Ports: Planner-Port.

HOT-NSA graded processing:
  sim = cosine_similarity(block_embedding, need_vector)

  sim >= theta_hi_reread   [HIGH]:
    Macro Port: full NSA attend this block
    Micro neurons: full 4-stream
    KFE write: ENABLED (confidence gate still applies)
    session_el_trace: x 1.5

  theta_lo_reread <= sim < theta_hi_reread   [MEDIUM]:
    Macro Port: linear attention only (no NSA for this block)
    Micro neurons: 3-stream (skip DRC)
    KFE write: DISABLED

  sim < theta_lo_reread   [LOW]:
    Skip neuron computation entirely; update SCT index only.

Phase 3 end state:
  SCT refined (detail nodes added for high-utility regions)
  N-Gram table constructed
  KFE candidate set fully populated
  session_el_trace captures context x need cross-activation
  alpha schedule advances to 0.7 for Phase 4+
```

### 10.5 Phase 4 — Plan

```
Goal: generate latent plan vector z; initialise monitoring; activate Thinkers.

Input: h_0 (Phase 3 neuron states) + SCT + N-Gram + KFE candidates
Active Port: Planner-Port.

Planner execution:
  Linear scan over all neurons (all Thinker Ports still idle)
  NSA: attend highest utility neurons
  Intercept layer: mHC merge -> latent plan z (continuous, not decoded)

z initialises:
  HOT-NSA dynamic sparse mask (initial bias for Thinker-Port NSA)
  SCT retrieval path (HNSW query direction hints)
  Effort-Adaptor budget estimate (expected n_stages, max_tokens)
  WMQ scratchpad

Parallel initialisations:
  Ring Buffer: init snapshot for Replan drift detection
  Replan Monitor: start async monitoring loop
  SCT backward query: prefetch relevant subtrees based on z

Thinker activation (based on z complexity estimate):
  simple task:  activate 1–2 Thinkers; remainder in offload
  complex task: activate all N Thinkers
  Each activated Thinker_i receives z_shard_i (partition of z) as initial bias
```

### 10.6 Phase 5 — Think + Tool + Generate

#### 10.6.1 Thinker Step Loop

```
Each active Thinker_i executes independently on its CUDA stream.

Per-step:
  1. Linear scan (self-filtered):
     Perceive: planner_aggr + other_thinker_aggr (self excluded) + analysis_i_aggr
  2. Compute utility (alpha=0.9 in Phase 5)
  3. NSA attend high-utility neurons
  4. Intercept layer mHC -> thinker_i merged state
  5. WMQ read: retrieve top-k relevant entries by attention
  6. Generate output token distribution or tool call instruction

  Stage detection (Thinker-side, NO Planner involvement):
    relevance_moving_avg = EMA(dot(current_output, WMQ_entries))
    If relevance_moving_avg < stage_start_relevance * stage_drop_threshold:
      stage_end = True
    If step_count > max_steps_per_stage OR wall_time > max_time_per_stage:
      stage_end = True (forced)
    On stage end:
      Move bound WMQ entries to deletion_buffer
      Increment stage_counter

  WMQ write:
    If dot(output, WMQ_context) > wm_enqueue_threshold:
      Enqueue output as new WMQ entry (bound to current stage_id)
```

#### 10.6.2 Hypothesis Verification Fork

```
Optional; triggered by Thinker when uncertainty is high.

  fork = copy(thinker_i.state)
  fork.hypothesis = H
  fork executes N steps under hypothesis H
  Analysis_i validates fork output against SCT / Engram

  CONSISTENT:  merge fork into main stream
               el-trace: positive reward for contributing neurons
  CONTRADICTED: discard fork
               el-trace: negative reward
  UNCERTAIN:   retain fork with low confidence flag
               may trigger Tool-Port call for external verification
```

#### 10.6.3 Replan Monitor (Async)

```
Runs on dedicated CUDA stream, polls every poll_interval steps.

Replan trigger conditions (ANY of):
  attention_entropy(current_port_state) > theta_entropy_dynamic
  cosine_sim(current_hidden, ring_buffer_snapshot) < theta_drift
  contradiction_head_signal > theta_contradiction

On trigger (= new turn):
  Async interrupt token generation pipeline
  HNSW query SCT for new evidence matching current hidden state
  Inject new evidence as KV pairs (NO neuron state reset)
  Planner-Port incremental update of z
  Resume generation

Note: tool return does NOT trigger Replan and does NOT constitute a turn boundary.
```

#### 10.6.4 Tool-Port (Native MCP)

```
On Thinker tool call instruction:
  Intercept layer -> CodeBook -> MCP protocol serialisation -> external tool
  Tool execution (async)
  Tool return -> VQ-GAN encode + RoPE -> inject into neuron grid
  tool_call + tool_return = one stage (increments Thinker stage_counter)
```

#### 10.6.5 Generator-Port (Streaming Output)

```
Input: Attention Merge result (see Phase 6)
Intercept layer -> mHC -> CodeBook -> VQ-GAN decoder
Output: streaming multimodal tokens over WebSocket
        (text / image / audio)
```

### 10.7 Phase 6 — Output

```
Attention Merge:
  For each Thinker_i output o_i:
    weight_i = S5.output_confidence_i * S8.utility_ema_i
  merged = softmax(weights) @ stack(o_i)
  // low-confidence forks are naturally suppressed

Self-consistency verification:
  Key factual claims in merged checked against SCT
  Inconsistencies: trigger suspicious layer flag on source neurons

Persona vector application:
  Planner-Port evaluates whether to apply Persona (RL-trained decision)
  If apply:
    merged = merged + W_persona * persona_vector
    // persona_vector loaded from NVRAM

Final output:
  merged -> CodeBook lookup -> VQ-GAN decoder -> multimodal tokens
  Stream to user via Generator-Port WebSocket
```


---

## 11. Higher-Order Thinking System (HOT)

### 11.1 Planner-Port

| Property | Specification |
|---|---|
| Always active | YES; never frozen or offloaded during inference |
| Primary outputs | latent plan z, SCT retrieval path, Effort-Adaptor config |
| Stage detection | NOT involved; Thinker-side only |
| N-Gram injection | Implicit attention flow (lightweight); influences Thinker attention bias |
| WMQ management | Initialises and monitors WMQ; does NOT write per-step entries |

### 11.2 Thinker-Port and Analysis-Port

```
Count N: fixed in .sxlm meta header at training time.
         Cannot be changed at runtime without retraining.

Binding: Thinker_i <-> Analysis_i (1:1 pair, fixed)

Planner dynamic control:
  simple task:  offload (N - k) Thinkers; k = 1..2
  complex task: activate all N Thinkers
  Offloaded Thinkers: excluded from Attention Merge; CUDA UM pages demoted
  Reactivation: Planner wakes offloaded Thinkers by issuing UM page hint
                to bring state back into HBM

Stage independence: each Thinker advances its own stage_counter independently.
```

### 11.3 Contradiction Detection Head

```
Location: intercept layer, runs on every Port step
Metric: ||aggr_i - aggr_j|| for all active Thinker pairs (i, j)
Threshold: tau_contradiction (learnable scalar)

On detection:
  Mild (||diff|| between tau_mild and tau_contradiction):
    Flag in global_aggregation_buffer; Planner may choose to ignore
  Severe (||diff|| > tau_contradiction):
    Emit signal to Replan Monitor
    Do NOT terminate conflicting Thinker paths
    Allow Attention Merge to arbitrate
```

### 11.4 Uncertainty Channel

```
Dedicated dimensions within fp8 256x256 inter-neuron messages.
Carries: S5.uncertainty_vector components
Propagation: along inference chain; accumulates at each neuron
At Thinker level: aggregated uncertainty triggers:
  - additional verification (Analysis fork)
  - Tool-Port call for external grounding
  - Replan if uncertainty exceeds threshold
```

---

## 12. Graph Topology Evolution

### 12.1 Edge Growth

```
Trigger: Thinker_i's output correlates strongly with an unconnected neighbour j
         (cross-activation > growth_threshold during training)

New edge parameters:
  W_out_new: initialised from source neuron's W_out + attention-weighted perturbation
  el_trace_new: mean(el_trace[source], el_trace[target]) * attention_weight_at_creation
  W_in_kernel_new: initialised from target neuron's mean incoming kernel

Frequency: checked every training batch; limited to max_new_edges_per_batch
```

### 12.2 Edge Pruning

```
Trigger: el_trace[edge] * |gradient[edge]| < prune_threshold
         for prune_window consecutive batches

Action:
  Accumulate W_out weight of pruned edge into target neuron's W_in bias
  Remove edge from routing table (S6)
  HNSW index update: remove edge from graph neighbourhood

Frequency: executed once per training batch end (after gradient update)
```

### 12.3 Cluster Emergence

```
Mechanism: fully passive; emergent from connectivity density.
           Graph community detection (Louvain or label propagation) runs
           periodically to identify clusters for:
             - CUDA device sharding (Section 22)
             - Utility aggregation (cluster-level utility includes bridge node value)

Bridge edge protection:
  Bridge edges carry high clustering utility (removal would disconnect components).
  S8.utility_ema for bridge-adjacent neurons is boosted by clustering_utility_bonus.
  This creates natural pressure against pruning bridge edges.
  NO explicit pruning whitelist is required.
```

### 12.4 Neuron Specialisation (S4 Evolution)

```
S4.specialization_vector evolves via gradient updates during training.
Direction emerges from which computation stream (A/B/C/D) dominates gate[i].

Diversity constraint (market-driven, not hard):
  Neurons in same cluster with similar specialization_vector:
    Apply soft utility penalty if cluster already has sufficient coverage
    Penalty = max(0, (cluster_coverage - target_coverage) * lambda_diversity)
    -> encourages complementary specialisation within clusters
```

---

## 13. Conflict Detection and Resolution

### 13.1 Two-Level Architecture

```
Level 1 — Light disagreement:
  Handled via edge communication; neurons self-negotiate via message passing.
  No explicit mechanism required; emergent from graph dynamics.

Level 2 — Severe conflict -> suspicious layer:
  Trigger: CDCL soft-solver detects conflict signature in neuron features.
```

### 13.2 CDCL Soft Solver

```
Conflict signature: continuous soft data (mean differences, variance ratios)
                    NOT discrete booleans.

Solver operation:
  1. Scanner:    scan neuron feature vectors for conflict patterns
  2. Matcher:    match against learned pattern library (heuristic)
  3. On match:   tag neuron as SUSPICIOUS; increase inter-neuron communication
                 bandwidth (routing weight boost to conflicting pair)
  4. Small divergence: neurons self-negotiate; solver monitors
  5. Severe:     attention-head notices -> fast response (Replan or fork)

Solver self-bootstrap:
  High-confidence solutions + corresponding features enter solver library.
  Cold-start: low-confidence solutions do NOT enter library.
  Solution invalidation:
    Parameter analysis reveals no actual conflict -> rule confidence drops
    Rule confidence < confidence_dead_threshold -> soft-delete rule
```

### 13.3 Suspicious Layer Protocol

```
On entering suspicious state:
  S5.output_confidence *= suspicious_confidence_penalty  // e.g. * 0.5
  Downstream receivers: observe S5.input_reliability of incoming messages
                        and compute own confidence accordingly
  Uncertainty channel: propagate uncertainty vector to Thinker-Port level
  Monitoring: track attention proximity rate, state similarity rate

Exit condition:
  Output entropy returns to normal range (allow minor residual divergence)
  S5.output_confidence gradually recovers via EMA update

Training:
  Gradient scaling for suspicious neurons: REUSES confidence -> el-trace
  pathway. No additional mechanism required.
```

---

## 14. Credit Assignment (el-trace)

### 14.1 Formula

```
persistent_el_trace[t] = lambda * persistent_el_trace[t-1] + |activation[t]|

lambda: per-parameter learnable scalar
        adjusted by: activation_magnitude * S8.utility_ema
        Range: [lambda_min, lambda_max] (default [0.8, 0.999])

Decoupled from context length:
  el-trace does NOT require full backpropagation through 128M context
  It accumulates locally per-neuron; reward is applied at episode end
```

### 14.2 Dual el-trace at Inference

```
persistent_el_trace:
  Frozen after training (stored in S7).
  Read-only during inference.
  Represents: "how important is this neuron to the model's knowledge"

session_el_trace:
  Initialised from persistent_el_trace at API call start.
  Updated during forward passes of all phases.
  Discarded at API call end (same lifecycle as SCT).
  Represents: "how much has this neuron been activated in this context"

Combined utility at phase P:
  utility_i = alpha_P * wkv_weight_i
            + beta_P  * session_el_trace[i]
            + gamma_P * persistent_el_trace[i]

Phase alpha schedule:
  Phase 1: alpha=0.2, Phase 2: alpha=0.5, Phase 3: alpha=0.7, Phase 5: alpha=0.9
```

### 14.3 Confidence-el-trace Coupling

```
Single unified pathway:
  suspicious_state -> S5.output_confidence * penalty
  -> el-trace magnitude decreases (trace weighted by confidence)
  -> gradient scaling decreases during backward pass
  -> KFE generalisation blocked (confidence gate at KFE entry)

No additional scaling mechanism SHALL be introduced for suspicious neurons.
```

### 14.4 Oscillation Damping

```
Detection:
  Compute el-trace integral over sliding window W_osc
  Classify via temporal and spatial coherence:
    Normal negotiation:  ordered, coherent activation patterns
    Pathological osc.:  high entropy, spatially incoherent, repeating cycles

Damping:
  Identify oscillating sub-graph via graph subtree matching
  Apply lambda_damp < lambda to el-trace in oscillating region
  lambda_damp: decays toward lambda_min over damping_window steps
```

### 14.5 Reward Distribution (AZR Training)

```
On reward receipt (end of AZR episode):
  For each neuron i with non-zero el-trace:
    delta_W[i] += learning_rate * reward * session_el_trace[i]

Reward decomposition:
  utility_reward:    +1.0 * base_reward      (correct output)
  refinement_penalty: -penalty_coeff * excess_tokens (over budget)
  hallucination_penalty: -halluc_coeff * strong_penalty (factual error)
  replan_penalty:   -replan_coeff  (applied to Planner-Port el-trace)
```

---

## 15. Global Aggregation Layer

### 15.1 Architecture

The global aggregation layer is a passive staging buffer. It has no complex learned parameters of its own. Aggregation is performed by the REQUESTING party (collector) using attention-based picking.

```
Structure: fixed-size buffer, one slot per Port ID
  Buffer[port_id] = partial_state (fp32, R^D_aggr)

Write (each Port, at intercept layer):
  partial_state = Linear_compress(merged)    // merged from intercept layer
  Buffer[port_id] = partial_state

Read (by collector, e.g. Planner, contradiction head, Attention Merge):
  q = Linear_q(collector_current_state)
  k = Linear_k(stack(Buffer[all ports]))
  v = Linear_v(stack(Buffer[all ports]))
  collected = softmax(q * k^T / sqrt(D_aggr)) * v
  // Ports whose partial_state is more relevant to collector contribute more

Training gradient:
  Collector's attention weights backpropagate through el-trace:
  useful partial_state sources receive positive el-trace signal

Lifecycle: buffer cleared at end of each turn.
```

---

## 16. Training Procedure

### 16.1 Phase 1 — SFT (Supervised Fine-Tuning)

```
Data:       Distillation from external SOTA (e.g. Claude Opus, GPT-4) via OpenRouter
Loss:       cross-entropy(model_output_tokens, teacher_tokens)
KFE:        enabled; accumulates from start
Graph:      S4 slow specialisation; S6 pruning/growth active
Engram:     closed-book training enforced:
              mask KFE and SCT; force reconstruction from Engram only
              Engram loss = cross-entropy(engram_reconstruction, target)

Infrastructure: ZeRO-3 + DeepSpeed offload; S7 offloaded to RAM/NVMe
                Terraform + GCP Vertex AI
```

**Req 16.1.1**: During the first 10% of SFT steps, KFE growth and graph topology evolution SHALL be disabled. Only after basic forward pass and el-trace convergence is observed SHALL these be re-enabled. Premature topology evolution solidifies noisy representations.

### 16.2 Phase 2 — Long CoT

```
Data:       math/code datasets + distilled plan-execution chains
Loss:       cross-entropy
          + KL(plan_z, distilled_plan_z)    // Planner plan quality
          + plan_execution_alignment_loss   // plan -> output coherence
el-trace:   long-chain credit assignment (lambda decay across hundreds of steps)
WMQ:        management behaviour included in training
Ring Buffer: drift detection included in training
```

### 16.3 Phase 3 — AZR (Absolute Zero Reinforcement)

```
Model fork: Challenger vs Solver (same base weights, independent fine-tuning)

Challenger generates problems via three tracks:

Track A — Self-generation (default):
  Challenger-Port generates problems autonomously
  Difficulty target: Solver success rate ~50%
  Losses:
    difficulty_calibration_loss: MSE(solver_success_rate, 0.5)
    diversity_loss: -entropy(problem_type_distribution)
    // prevents collapse to single problem type

Track B — External SOTA intervention:
  External models (Claude Opus, GPT-4) generate problems + reference answers
  Purpose: cold-start quality bootstrap and sustained high-quality ceiling
  Challenger learns from external style and difficulty distribution:
    style_alignment_loss: KL(challenger_distribution, external_distribution)
  External SOTA also acts as JUDGE for open-ended problems:
    judge_score = external_model.evaluate(solver_answer,
                                           rubric=[logical_coherence,
                                                   step_completeness,
                                                   conclusion_correctness])
    Multiple judges (>1 external model): take mean score to reduce judge bias
    Cross-validate with programmatic verification where available
  Cost control: mix ratio starts at 70% external (cold start), decays to 20%

Track C — Benchmark injection (MMLU, MATH, HumanEval, GPQA, etc.):
  Used for:
    (1) Training: mixed into Solver training to prevent capability regression
    (2) Evaluation: fixed holdout set for periodic benchmark monitoring
  NOT used for Challenger gradient (benchmark problems do not train Challenger)

Mix ratio auto-schedule:
  solver_benchmark_score < regression_threshold  -> increase Track C ratio
  solver_success_rate > 0.6                      -> increase Track A/B ratio
  after stabilisation                            -> decay Track B ratio (cost)

Solver training:
  Full Phase 0-5 inference + el-trace recording
  Verification: code execution engine + SCT fact-check
  Reward signals:
    R_utility:    +1.0 * base for correct answer
    R_refine:     -token_excess_penalty for over-budget reasoning
    R_halluc:     -halluc_penalty (strong) for factual errors
    R_replan:     -replan_penalty applied to Planner-Port el-trace
  All rewards distributed via el-trace (Section 14.5)
```

---

## 17. Detailed Phase Data Flows

### 17.1 Backward Pass (Training Only)

```
STEP 1 — Loss gradient entry point
  Gradient flows from Generator-Port output layer / global aggregation layer

STEP 2 — W_out gradient
  Propagate through deconv path to predecessor neurons
  Edge parameters (W_out, conv_kernel) receive gradient

STEP 3 — Micro-layer backward
  h_final -> CM backward (straight-through for noise step)
  -> CM output -> DRC backward (only actually-executed iterations unrolled)
  -> h_kfe -> KFE entry gradients (* S5.output_confidence scaling)
  -> h_mid -> four streams backward:
       NSA:          gradient only through utility-selected dimensions
       SSM:          standard selective scan backward
       Linear Attn:  RWKV BPTT (O(1) state backward recursion)
       DRC:          already handled above
  -> W_in gradient

STEP 4 — Macro-layer backward
  Intercept layer mHC backward (Sinkhorn straight-through approximation:
    d/dW_raw ≈ d/dH_res; ignore Sinkhorn iteration Jacobian)
  NSA sparse mask backward: gradient only through selected neuron positions
  Linear attention wkv backward: O(1) state recursion

STEP 5 — KFE gradients
  Retrieved KFE entries receive residual-stream contribution gradient
  Gradient magnitude * S5.output_confidence
  suspicious neuron additional scaling: REUSES confidence pathway

STEP 6 — S7 update
  persistent_el_trace[t] = lambda * persistent_el_trace[t-1] + |activation[t]|
  lambda updated via: lambda_grad = activation_magnitude * S8.utility_ema
  Adam update for all parameters (W_in, W_out, stream params, mHC, Port params)

STEP 7 — Graph topology update (once per batch end)
  Prune: edges where el_trace * |gradient| < prune_threshold for prune_window batches
         -> accumulate W_out weight into target W_in bias -> remove edge
  Grow: neurons where cross-activation to non-neighbour > growth_threshold
         -> create new edge with initialised parameters (Section 12.1)
```

### 17.2 Forward Pass, Replay, Last-Input, Re-read

Fully specified in Sections 10.2, 10.3, 10.4, and Section 5.2 respectively.

### 17.3 Phase 4 and Phase 5 Data Flows

Fully specified in Sections 10.5 and 10.6 respectively.

### 17.4 Training Phase Backward Differences

| Training phase | Loss sources | Special backward behaviour |
|---|---|---|
| SFT | Cross-entropy; Engram closed-book loss | Standard; no reward signal |
| Long CoT | Cross-entropy + plan KL + alignment | Planner-Port receives KL gradient from plan z vs distilled plan |
| AZR | el-trace distributed reward | Challenger: difficulty calibration + diversity + style losses. Solver: no explicit loss; reward via el-trace. |


---

## 18. VQ-GAN Encoding Boundary

### 18.1 Role

```
Input boundary:  text / image / audio / video
                 -> VQ-GAN encoder -> discrete codebook indices
                 -> codebook vector lookup -> continuous embeddings
                 -> + RoPE positional encoding
                 -> neuron grid (as initial messages)

Output boundary: neuron grid output (merged, continuous)
                 -> CodeBook lookup (nearest codebook vector)
                 -> VQ-GAN decoder
                 -> multimodal output tokens (text / image / audio / MCP calls)

INVARIANT: The CodeBook and VQ-GAN decoder are ONLY used at the output
           boundary. All internal Port aggregation, Thinker merge, and
           inter-neuron communication operate in continuous latent space.
           No intermediate discretisation occurs.
```

### 18.2 Codebook Design

```
Size:     user-configurable (default: 8192 per modality)
Index:    B-link-tree (Lehman-Yao concurrent B-tree variant)
          Supports O(log n) lookup; concurrent read during inference
Fixity:   codebook is FROZEN after training; embedding space is stable
          This ensures SCT constructed across sessions is consistent

Uniformity:
  During training: non-uniform distribution (frequency-weighted)
  Post-training:   apply codebook uniformisation pass to reduce dead entries
  Alternative:     each model binary binds its own VQ-GAN; no cross-model reuse

Semantic alignment:
  VQ-GAN trained with semantic alignment objective:
  cross-modal pairs (text description, image) -> nearby codebook vectors
  Ensures modality-agnostic operation of neuron grid

Collapse handling:
  Commitment loss + exponential moving average codebook update
  Dead entry detection: entries not selected in N batches -> reinitialised
                        from random encoder outputs
```

---

## 19. Persona Vector

### 19.1 Specification

```
Storage:    NVRAM (e.g. Intel Optane) for persistence + low latency
            Fallback: CPU RAM if NVRAM unavailable (lower persistence guarantee)
Dimension:  persona_dim; user-defined; stored in .sxlm meta header
Count:      ONE global Persona vector per model instance
Persistence: survives across API calls and server restarts (NVRAM)
             NOT affected by session_el_trace lifecycle
```

### 19.2 Application

```
Applied at Phase 6 output, before CodeBook:
  Planner-Port decides whether to apply Persona (RL-trained binary/soft gate)
  If apply:
    output_biased = output_merged + W_persona * persona_vector
    W_persona: learnable scalar or matrix (trained)
  If not apply (e.g. pure tool call outputs):
    output_biased = output_merged

Planner Persona gate:
  gate_logit = W_gate_persona * planner_state
  gate = sigmoid(gate_logit)   // soft gate; hard threshold optional
  Trained via RLHF to maximise human preference for personality consistency
```

### 19.3 Training

```
Gradient backward:
  persona_vector participates in Adam update as a standard trainable parameter
  Gradient source: Phase 6 output loss (cross-entropy during SFT/CoT;
                   reward via el-trace during AZR)

RLHF signal:
  Human annotators rate persona consistency (tone, style, values)
  Reward flows through el-trace to:
    (1) persona_vector directly
    (2) Planner-Port gate parameters (W_gate_persona)
  Result: model learns WHEN applying Persona improves user preference
```

---

## 20. KFE Eviction Mechanism

### 20.1 Eviction Score

Each KFE entry in L2/L3 is assigned an eviction score. Lower score = evicted first.

```
eviction_score = w1 * S8.utility_ema
               + w2 * wkv_weight_at_last_scan   // Port linear scan weight
               + w3 * access_recency_decay       // exp(-lambda_recency * t)
               + w4 * persistent_el_trace_norm

w1, w2, w3, w4: learnable scalars (initialised to [0.4, 0.2, 0.2, 0.2])

access_recency_decay: exp(-lambda_recency * (current_epoch - last_access_epoch))
  lambda_recency: default = 0.001 (slow decay; recent access is important but not dominant)
```

### 20.2 Eviction Trigger and Cascade

```
L2 (GPU HBM) capacity exceeded:
  Sort L2 KFE entries by eviction_score ascending
  Demote lowest-score entries to L3 (CPU RAM) via Unified Memory demotion hint
  cudaMemAdvise(ptr, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId)

L3 (CPU RAM hot zone) capacity exceeded:
  Sort L3 KFE entries by eviction_score ascending
  Demote lowest-score entries to L5 (disk cold storage)
  Write as zstd-compressed blocks

L5 (disk) capacity exceeded:
  Enter soft-delete evaluation:
    Flag entries with eviction_score < soft_delete_threshold as CANDIDATE
    Do NOT delete immediately

Soft-delete to hard-delete:
  For each CANDIDATE entry:
    If no positive activation in K turns (default K=8):
      AND (el-trace extreme penalty received OR contradiction detected):
        Permanently delete entry + HNSW index node
    Else:
      Clear CANDIDATE flag; restore to normal eviction queue
```

### 20.3 Promotion

```
On KFE cache miss (entry is on disk, L5):
  Load entry from L5 to L3 on access
  Update access_count and last_access_epoch
  If access_count exceeds promotion_threshold within window:
    Promote to L2 (GPU HBM) via cudaMemPrefetchAsync
```

---

## 21. .sxlm Model Format

`.sxlm` (Sintellix eXtended Language Model) is the binary packaging format for a Quila model instance.

### 21.1 File Layout

```
[.sxlm file]
+---------------------------+
| HEADER (uncompressed)     |  magic="SXLM", version:u32, meta struct
+---------------------------+
| BLOCK 1 (zstd level 19)   |  Model parameters
| BLOCK 2 (zstd level 19)   |  Neuron states
| BLOCK 3 (zstd level 19)   |  Graph topology
| BLOCK 4 (zstd level 19)   |  VQ-GAN + Codebook + B-link-tree
| BLOCK 5 (zstd level 3)    |  KFE library + HNSW index  [hot path; fast decompress]
| BLOCK 6 (zstd level 19)   |  Engram
| BLOCK 7 (zstd level 19)   |  Persona vector
+---------------------------+
```

### 21.2 Header Protobuf Schema

```protobuf
message SXLMHeader {
  bytes   magic         = 1;   // "SXLM" (4 bytes)
  uint32  version       = 2;   // format version
  SXLMMeta meta         = 3;
  repeated BlockDescriptor blocks = 4;
}

message SXLMMeta {
  uint32  neuron_count         = 1;   // e.g. 32768
  uint32  thinker_n            = 2;   // N Thinkers (fixed at train time)
  uint32  hidden_dim           = 3;   // D
  uint32  persona_dim          = 4;
  uint32  vqgan_codebook_size  = 5;
  uint64  kfe_count            = 6;
  uint64  graph_edge_count     = 7;
  int64   created_at           = 8;   // Unix timestamp
  bytes   checkpoint_sha256    = 9;   // 32 bytes; covers all block content
  string  base_model_id        = 10;  // for lineage tracking
}

message BlockDescriptor {
  uint32  block_id     = 1;
  uint64  offset       = 2;   // byte offset from file start
  uint64  size_bytes   = 3;   // compressed size
  uint64  raw_size     = 4;   // uncompressed size
  uint32  zstd_level   = 5;
  bytes   block_sha256 = 6;   // 32 bytes; per-block integrity
}
```

### 21.3 Block Contents

| Block | Contents |
|---|---|
| 1 — Parameters | W_in, W_out, stream params (NSA, SSM, RWKV, DRC), mHC params (baseline + opt), Port params (Planner, Thinker×N, Analysis×N), global aggregation layer params, Persona W_persona |
| 2 — Neuron states | S1–S8 all channels, per-neuron, indexed by neuron_id |
| 3 — Graph topology | Edge list: [(src_id:u32, dst_id:u32, W_out_params:bytes, conv_kernel:bytes)], cluster membership table |
| 4 — VQ-GAN | Encoder weights, Decoder weights, Codebook vectors (per-modality sections), B-link-tree index serialised |
| 5 — KFE library | Per-neuron KFE entry lists (protobuf, KFEEntry[]), HNSW index serialised (hnswlib format) |
| 6 — Engram | Compressed knowledge table, O(1) hash index |
| 7 — Persona | Persona vector (fp32, R^persona_dim) |

### 21.4 Deployment Procedure

```
1. Validate header:
   SHA-256(all blocks concatenated) == meta.checkpoint_sha256

2. Parallel block extraction to /tmp/sxlm/:
   Block 1 -> /tmp/sxlm/params/
   Block 2 -> /tmp/sxlm/states/
   Block 3 -> /tmp/sxlm/graph/
   Block 4 -> /tmp/sxlm/vqgan/
   Block 5 -> /tmp/sxlm/kfe/
   Block 6 -> /tmp/sxlm/engram/
   Block 7 -> /tmp/sxlm/persona/

3. Initialise tiered storage:
   L1/L2 (GPU, Unified Memory):
     cudaMallocManaged for all neuron states (S1-S8)
     cudaMallocManaged for all model parameters (Block 1)
     Load high-utility KFE hot zone from Block 5
     cudaMemAdvise: PreferredLocation=GPU for hot parameters

   L3 (CPU RAM hot zone):
     Load remaining KFE entries (Block 5) into in-memory HNSW index

   L4 (CPU RAM / SSD):
     Mount SCT index structures (empty at deployment; built per-session)
     Load Engram (Block 6) into hash-indexed memory structure
     Load VQ-GAN (Block 4): encoder/decoder to GPU; B-link-tree to CPU RAM

   L5 (disk):
     KFE cold storage: point to /tmp/sxlm/kfe/ directly (on-demand load)

4. Persona vector:
   Write Block 7 content to NVRAM device file or memory-mapped NVRAM region
   If NVRAM unavailable: load into CPU RAM with persistence warning

5. Sanity checks:
   neuron_count matches loaded parameter tensor dimensions
   graph_edge_count matches loaded edge list length
   kfe_count matches loaded HNSW entry count
   All block SHA-256 hashes verified

6. Mark model READY; begin accepting API requests.
```

**Req 21.4.1**: Blocks MUST be decompressed in parallel using independent threads. Block 5 (KFE) MUST use a lower zstd level (3) to minimise decompression latency, as it is on the hot inference path.

**Req 21.4.2**: The checkpoint SHA-256 MUST be verified before any block is loaded. Deployment MUST abort on checksum mismatch.

---

## 22. Multi-GPU Distributed Inference

### 22.1 Memory Model

**Req 22.1.1**: All neuron states (S1–S8), KFE hot zone, and model parameters SHALL be allocated with `cudaMallocManaged`. This is the Unified Memory (UM) model; the CUDA driver automatically migrates pages between GPU HBM and CPU RAM across multiple devices.

**Req 22.1.2**: `cudaMemAdvise` hints SHALL be set for all hot parameter pages:
```
cudaMemAdvise(param_ptr, cudaMemAdviseSetPreferredLocation, device_id)
cudaMemAdvise(param_ptr, cudaMemAdviseSetAccessedBy, device_id)
```

**Req 22.1.3**: For cross-GPU page access, NVLink P2P DMA is used automatically by the UM driver when available. No explicit `cudaMemcpy` between devices SHALL be coded by the application layer.

### 22.2 Neuron Sharding

```
Sharding strategy: graph community-based
  Run Louvain community detection on neuron graph
  Assign each community to a GPU such that:
    (1) GPU memory load is balanced (within 10%)
    (2) Number of cross-GPU edges is minimised

Rationale:
  Intra-community edges (high frequency) remain local -> no cross-GPU traffic
  Inter-community edges (bridge edges, low frequency) cross GPUs -> NVLink P2P
  -> NVLink bandwidth pressure scales with bridge edge frequency, not total edges
```

### 22.3 CUDA Stream Layout

```
Per GPU, 8 CUDA streams:
  Stream 0:    Neuron batch 0 (forward computation)
  Stream 1:    Neuron batch 1
  ...
  Stream 5:    Neuron batch 5
  Stream 6:    Cross-GPU edge communication (NCCL P2P send/recv)
  Stream 7:    Intercept layer AllReduce (NCCL; partial_state aggregation)

Scheduling:
  Neuron batches assigned round-robin within GPU's community
  Cross-GPU messages overlap with local computation (streams 0-5 independent of stream 6)
  Stream 7 AllReduce fires after all stream 0-5 are complete for the current step
```

### 22.4 KFE Distribution

```
Each GPU caches its own community's KFE hot zone in local HBM (UM preferred location)
Cross-GPU KFE hits:
  Low frequency (bridge neurons accessed from wrong GPU)
  UM driver handles migration transparently
  OR: direct NVLink P2P access to remote page without migration
      cudaMemAdvise(ptr, cudaMemAdviseSetAccessedBy, requesting_device)
```

### 22.5 Minimum GPU Requirements

| Configuration | GPUs | VRAM per GPU | NVLink |
|---|---|---|---|
| Minimum (development) | 1 | 80 GB | N/A |
| Recommended (inference) | 4 | 80 GB | NVLink 4.0+ |
| Full training | 8+ | 80 GB | NVLink 4.0+ |

---

## 23. Conformance Requirements

### 23.1 Mandatory Requirements Summary

The following requirements are MANDATORY for a conformant implementation. All use the keyword SHALL or MUST.

| Req ID | Section | Summary |
|---|---|---|
| 5.1.1 | 5.1 | S4 and S7 channels saved/restored from .sxlm |
| 5.3.1 | 5.3 | fp32 accumulation for all reductions |
| 5.3.2 | 5.3 | fp8 E4M3 quantisation for inter-neuron messages |
| 6.3.1 | 6.3 | CM trained via Consistency Distillation |
| 7.2.1 | 7.2 | Video temporal and spatial RoPE bases trained independently |
| 8.5.1 | 8.5 | Edge parameters NOT shared between edges |
| 9.1.1 | 9.1 | All GPU memory allocated with cudaMallocManaged |
| 9.1.2 | 9.1 | Persona vector stored on NVRAM if available |
| 16.1.1 | 16.1 | KFE growth and topology evolution disabled for first 10% of SFT |
| 21.4.1 | 21.4 | Block 5 uses zstd level 3; all blocks decompressed in parallel |
| 21.4.2 | 21.4 | Checkpoint SHA-256 verified before any block loaded |
| 22.1.1 | 22.1 | All neuron states and parameters use cudaMallocManaged |
| 22.1.2 | 22.1 | cudaMemAdvise hints set for all hot parameter pages |
| 22.1.3 | 22.1 | No explicit cudaMemcpy between GPUs in application layer |

### 23.2 Design Invariants (Non-Negotiable)

| Invariant | Implication |
|---|---|
| CodeBook output-boundary-only | No intermediate discretisation in any internal path |
| Confidence-el-trace single pathway | No separate gradient scaling mechanism for suspicious neurons |
| Stage-lifecycle WMQ | No explicit GC; lifecycle implicit in stage boundaries |
| Codebook frozen post-training | SCT construction is cross-session consistent |
| NSA utility signal unity | Micro and macro NSA share the same utility signal family |
| Thinker N fixed at training | N cannot be changed at runtime; requires retraining |
| session_el_trace discarded at call end | No cross-call contamination of context-specific utility |

### 23.3 Recommended (SHOULD) Behaviours

- Persona vector SHOULD be loaded from NVRAM to ensure cross-restart persistence.
- The Louvain community detection for GPU sharding SHOULD be re-run whenever the graph topology changes by more than 5% of total edges.
- External SOTA judges in AZR Track B SHOULD use ≥ 2 distinct models and take the mean score to reduce single-model bias.
- The AZR mix ratio SHOULD be automatically scheduled (Section 16.3) rather than set manually.
- Challenger and Solver in AZR SHOULD be fine-tuned from the same base checkpoint to ensure capability parity at the start of RL.

---

*End of Document — SX-SPEC-QUILA-001 Rev 0.3 DRAFT*

*For updates or corrections, increment revision and note changes in a revision history table appended to this document.*
