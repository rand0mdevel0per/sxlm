#pragma once
#include <cuda_runtime.h>
#include "port.cuh"

namespace quila {

// Backward alignment for cross-Port perception
__device__ void backward_alignment(
    const float* global_buffer,
    const float* port_state,
    float* aligned_output,
    int port_id,
    int num_ports,
    int hidden_dim
) {
    int tid = threadIdx.x;

    if (tid < hidden_dim) {
        float sum = 0.0f;
        float attn_sum = 0.0f;

        // Attend to other ports
        for (int p = 0; p < num_ports; p++) {
            if (p == port_id) continue;

            const float* other_port = &global_buffer[p * hidden_dim];

            // Compute attention weight
            float attn = expf(port_state[tid] * other_port[tid] / sqrtf((float)hidden_dim));
            attn_sum += attn;
            sum += attn * other_port[tid];
        }

        aligned_output[tid] = (attn_sum > 1e-8f) ? (sum / attn_sum) : 0.0f;
    }
}

} // namespace quila
