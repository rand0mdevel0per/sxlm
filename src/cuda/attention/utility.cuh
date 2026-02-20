#pragma once
#include <cuda_runtime.h>
#include "port.cuh"

namespace quila {

// Compute utility scores for NSA masking
static __device__ void compute_utility(
    const float* wkv_weights,
    const float* session_el_trace,
    const float* persistent_el_trace,
    float* utility_scores,
    const PortConfig& config,
    int num_neurons
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < num_neurons) {
        utility_scores[tid] =
            config.alpha * wkv_weights[tid] +
            config.beta * session_el_trace[tid] +
            config.gamma * persistent_el_trace[tid];
    }
}

// Select top-k neurons by utility
static __device__ void select_top_k(
    const float* utility_scores,
    int* nsa_mask,
    int num_neurons,
    int k
) {
    // Simplified: just take first k (full version would sort)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < k) {
        nsa_mask[tid] = tid;
    }
}

} // namespace quila
