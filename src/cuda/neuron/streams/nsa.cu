#pragma once
#include <cuda_runtime.h>
#include "../utils/error.cuh"

namespace quila {

// Minimal NSA implementation
__device__ void stream_a_nsa(
    const float* h_input,
    const float* utility_scores,
    float* output,
    int hidden_dim,
    int top_k
) {
    // Top-k selection by utility
    int tid = threadIdx.x;
    if (tid < top_k) {
        // Simplified: just copy top-k dims
        output[tid] = h_input[tid];
    }
}

} // namespace quila
