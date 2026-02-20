#pragma once
#include <cuda_runtime.h>

namespace quila {

enum Modality {
    TEXT = 0,
    IMAGE = 1,
    AUDIO = 2,
    VIDEO = 3
};

// Apply RoPE for specific modality
__device__ void apply_rope(float* vec, int pos, int dim, Modality modality);

// RoPE kernel
__global__ void rope_kernel(float* embeddings, int seq_len, int hidden_dim, Modality modality);

// Host wrapper
__host__ void apply_rope_host(float* embeddings, int seq_len, int hidden_dim, Modality modality);

} // namespace quila
