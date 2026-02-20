#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Phase 1: Replay with adaptive skip
__global__ void phase1_replay_kernel(
    const float* sct_skeleton,
    float* neuron_states,
    int num_neurons,
    int hidden_dim,
    bool adaptive_skip
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_neurons * hidden_dim) return;

    if (adaptive_skip) {
        // Skip replay if context is fresh
        return;
    }

    // Load from SCT skeleton
    neuron_states[tid] = sct_skeleton[tid];
}

__host__ void run_phase1_replay(
    const float* sct_skeleton,
    float* neuron_states,
    int num_neurons,
    int hidden_dim,
    bool adaptive_skip
) {
    int total_size = num_neurons * hidden_dim;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    phase1_replay_kernel<<<blocks, threads>>>(
        sct_skeleton, neuron_states, num_neurons, hidden_dim, adaptive_skip
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
