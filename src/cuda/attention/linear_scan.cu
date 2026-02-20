#pragma once
#include <cuda_runtime.h>
#include "port.cuh"
#include "../utils/error.cuh"

namespace quila {

// RWKV-style linear attention scan over neuron grid
__device__ void linear_attention_scan(
    const float* neuron_outputs,  // h_final from all neurons
    PortState* port,
    int num_neurons,
    int hidden_dim
) {
    int tid = threadIdx.x;

    // Initialize aggregation
    if (tid < hidden_dim) {
        port->aggr[tid] = 0.0f;
    }
    __syncthreads();

    // Scan over neurons
    for (int i = 0; i < num_neurons; i++) {
        const float* h_i = &neuron_outputs[i * hidden_dim];

        if (tid < hidden_dim) {
            // Compute r, k, v
            float r = 1.0f / (1.0f + expf(-h_i[tid]));
            float k = h_i[tid];
            float v = h_i[tid];
            float w = 0.9f;

            // Update wkv state (O(1))
            port->wkv_state[tid] = expf(w) * port->wkv_state[tid] + expf(k) * v;
        }
        __syncthreads();
    }

    // Final aggregation
    if (tid < hidden_dim) {
        float r_last = 1.0f / (1.0f + expf(-port->current_state[tid]));
        port->aggr[tid] = r_last * tanhf(port->wkv_state[tid]);
    }
}

} // namespace quila
