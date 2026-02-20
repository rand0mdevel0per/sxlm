#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Async replan monitor
__global__ void replan_monitor_kernel(
    const float* neuron_states,
    bool* replan_triggered,
    int num_neurons,
    int hidden_dim,
    float threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_neurons) return;

    // Check if neuron state exceeds threshold
    float state_norm = 0.0f;
    for (int i = 0; i < hidden_dim; i++) {
        float val = neuron_states[tid * hidden_dim + i];
        state_norm += val * val;
    }

    if (state_norm > threshold * threshold) {
        *replan_triggered = true;
    }
}

__host__ void check_replan_trigger(
    const float* neuron_states,
    bool* replan_triggered,
    int num_neurons,
    int hidden_dim,
    float threshold
) {
    int threads = 256;
    int blocks = (num_neurons + threads - 1) / threads;

    replan_monitor_kernel<<<blocks, threads>>>(
        neuron_states, replan_triggered, num_neurons, hidden_dim, threshold
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
