#pragma once
#include <cuda_runtime.h>
#include "port.cuh"

namespace quila {

// Global aggregation buffer for cross-Port communication
__device__ void write_to_global_buffer(
    const float* port_output,
    float* global_buffer,
    int port_id,
    int num_ports,
    int hidden_dim
) {
    int tid = threadIdx.x;

    if (tid < hidden_dim) {
        int offset = port_id * hidden_dim;
        global_buffer[offset + tid] = port_output[tid];
    }
}

__device__ void read_from_global_buffer(
    const float* global_buffer,
    float* output,
    int num_ports,
    int hidden_dim
) {
    int tid = threadIdx.x;

    if (tid < hidden_dim) {
        float sum = 0.0f;
        for (int p = 0; p < num_ports; p++) {
            sum += global_buffer[p * hidden_dim + tid];
        }
        output[tid] = sum / (float)num_ports;
    }
}

} // namespace quila
