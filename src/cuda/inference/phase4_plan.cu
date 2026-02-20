#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Phase 4: Plan generation (latent z, Thinker activation)
__global__ void phase4_plan_kernel(
    const float* need_vector,
    float* latent_z,
    int hidden_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_dim) return;

    // Generate plan latent vector
    latent_z[tid] = need_vector[tid];
}

__host__ void run_phase4_plan(
    const float* need_vector,
    float* latent_z,
    int hidden_dim
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;

    phase4_plan_kernel<<<blocks, threads>>>(
        need_vector, latent_z, hidden_dim
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
