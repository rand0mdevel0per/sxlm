#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Phase 5: Think + Tool + Generate
__global__ void phase5_think_kernel(
    const float* latent_z,
    float* thinker_output,
    int hidden_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_dim) return;

    // Activate Thinker-Ports
    thinker_output[tid] = latent_z[tid];
}

__host__ void run_phase5_think(
    const float* latent_z,
    float* thinker_output,
    int hidden_dim
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;

    phase5_think_kernel<<<blocks, threads>>>(
        latent_z, thinker_output, hidden_dim
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
