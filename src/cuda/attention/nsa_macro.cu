#pragma once
#include <cuda_runtime.h>
#include "port.cuh"
#include "../utils/error.cuh"

namespace quila {

// Macro-level NSA sparse attention over selected neurons
__device__ void nsa_macro_attention(
    const float* port_state,
    const float* neuron_outputs,
    const int* nsa_mask,
    float* output,
    int hidden_dim,
    int k
) {
    int tid = threadIdx.x;

    if (tid < hidden_dim) {
        // Compute Q from port state
        float q = port_state[tid];

        // Compute attention over masked neurons
        float attn_sum = 0.0f;
        float value_sum = 0.0f;

        for (int i = 0; i < k; i++) {
            int neuron_idx = nsa_mask[i];
            const float* h_i = &neuron_outputs[neuron_idx * hidden_dim];

            // Simplified attention: dot product
            float k_i = h_i[tid];
            float v_i = h_i[tid];

            float attn_weight = expf(q * k_i / sqrtf((float)hidden_dim));
            attn_sum += attn_weight;
            value_sum += attn_weight * v_i;
        }

        // Normalize and output
        output[tid] = (attn_sum > 1e-8f) ? (value_sum / attn_sum) : 0.0f;
    }
}

} // namespace quila
