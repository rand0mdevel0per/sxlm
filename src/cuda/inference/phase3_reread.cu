#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Phase 3: Context re-read with need_vector bias
__global__ void phase3_reread_kernel(
    const float* demand_vector,
    const float* kfe_embeddings,
    float* need_vector,
    int hidden_dim,
    int num_kfe
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_dim) return;

    // Compute need_vector from demand and KFE library
    float sum = 0.0f;
    for (int i = 0; i < num_kfe; i++) {
        sum += demand_vector[tid] * kfe_embeddings[i * hidden_dim + tid];
    }
    need_vector[tid] = sum / (num_kfe + 1e-8f);
}

__host__ void run_phase3_reread(
    const float* demand_vector,
    const float* kfe_embeddings,
    float* need_vector,
    int hidden_dim,
    int num_kfe
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;

    phase3_reread_kernel<<<blocks, threads>>>(
        demand_vector, kfe_embeddings, need_vector, hidden_dim, num_kfe
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
