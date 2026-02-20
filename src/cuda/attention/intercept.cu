#pragma once
#include <cuda_runtime.h>
#include "port.cuh"
#include "../utils/error.cuh"

namespace quila {

// Macro-level mHC intercept layer
__device__ void intercept_layer(
    const float* aggr,
    const float* out_nsa,
    const float* neuron_outputs,
    const int* nsa_mask,
    float* output,
    int hidden_dim,
    int k
) {
    int tid = threadIdx.x;

    if (tid < hidden_dim) {
        // Compute average of selected neurons
        float selected_avg = 0.0f;
        for (int i = 0; i < k; i++) {
            int neuron_idx = nsa_mask[i];
            selected_avg += neuron_outputs[neuron_idx * hidden_dim + tid];
        }
        selected_avg /= (float)k;

        // Simple mHC mixing: weighted combination
        // W = [w1, w2, w3] normalized to sum to 1
        float w1 = 0.4f;  // aggr weight
        float w2 = 0.4f;  // out_nsa weight
        float w3 = 0.2f;  // selected neurons weight

        output[tid] = w1 * aggr[tid] + w2 * out_nsa[tid] + w3 * selected_avg;
    }
}

} // namespace quila
