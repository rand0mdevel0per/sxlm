#pragma once
#include <cuda_runtime.h>
#include "../utils/error.cuh"

namespace quila {

// Minimal SSM (Mamba-style) implementation
__device__ void stream_b_ssm(
    const float* h_input,
    const float* frame_memory,
    float* ssm_state,
    float* output,
    int hidden_dim
) {
    int tid = threadIdx.x;
    if (tid < hidden_dim) {
        // Simplified selective scan
        float delta = 1.0f + tanhf(h_input[tid]);
        float a = -expf(h_input[tid] * 0.1f);
        ssm_state[tid] = expf(a) * ssm_state[tid] + delta * h_input[tid];
        output[tid] = ssm_state[tid];
    }
}

} // namespace quila
