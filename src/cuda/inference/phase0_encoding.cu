#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Phase 0: VQ-GAN encoding boundary
__global__ void phase0_vqgan_encode(
    const float* input_tokens,
    float* encoded_output,
    int seq_len,
    int hidden_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seq_len * hidden_dim) return;

    // Simplified: just copy (real implementation would use VQ-GAN encoder)
    encoded_output[tid] = input_tokens[tid];
}

__host__ void run_phase0_encoding(
    const float* input_tokens,
    float* encoded_output,
    int seq_len,
    int hidden_dim
) {
    int total_size = seq_len * hidden_dim;
    int threads = 256;
    int blocks = (total_size + threads - 1) / threads;

    phase0_vqgan_encode<<<blocks, threads>>>(
        input_tokens, encoded_output, seq_len, hidden_dim
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
