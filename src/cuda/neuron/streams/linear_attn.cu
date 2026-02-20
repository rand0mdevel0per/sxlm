#pragma once
#include <cuda_runtime.h>
#include "../utils/error.cuh"

namespace quila {

// Minimal Linear Attention (RWKV-style) implementation
__device__ void stream_c_linear_attn(
    const float* h_input,
    float* wkv_state,
    float* output,
    int hidden_dim
) {
    int tid = threadIdx.x;
    if (tid < hidden_dim) {
        float r = 1.0f / (1.0f + expf(-h_input[tid]));  // sigmoid
        float k = h_input[tid];
        float v = h_input[tid];
        float w = 0.9f;  // decay factor

        // O(1) state update
        wkv_state[tid] = expf(w) * wkv_state[tid] + expf(k) * v;
        output[tid] = r * tanhf(wkv_state[tid]);
    }
}

} // namespace quila
