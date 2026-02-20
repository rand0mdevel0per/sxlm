#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Phase 2: Last-Input read-in (demand identification)
__global__ void phase2_lastinput_kernel(
    const float* encoded_input,
    float* demand_vector,
    int hidden_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_dim) return;

    // Simplified: extract demand from last input
    demand_vector[tid] = encoded_input[tid];
}

__host__ void run_phase2_lastinput(
    const float* encoded_input,
    float* demand_vector,
    int hidden_dim
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;

    phase2_lastinput_kernel<<<blocks, threads>>>(
        encoded_input, demand_vector, hidden_dim
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
