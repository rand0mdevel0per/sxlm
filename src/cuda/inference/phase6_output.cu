#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Phase 6: Output (Attention Merge, Persona application)
__global__ void phase6_output_kernel(
    const float* thinker_output,
    const float* persona_vector,
    float* final_output,
    int hidden_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_dim) return;

    // Apply Persona vector
    final_output[tid] = thinker_output[tid] + persona_vector[tid];
}

__host__ void run_phase6_output(
    const float* thinker_output,
    const float* persona_vector,
    float* final_output,
    int hidden_dim
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;

    phase6_output_kernel<<<blocks, threads>>>(
        thinker_output, persona_vector, final_output, hidden_dim
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
