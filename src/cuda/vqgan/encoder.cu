#include "encoder.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_vqgan_encoder(VQGANEncoder* encoder, int input_dim, int hidden_dim, int codebook_size) {
    encoder->input_dim = input_dim;
    encoder->hidden_dim = hidden_dim;
    encoder->codebook_size = codebook_size;
    encoder->weights = (float*)allocate_unified(input_dim * hidden_dim * sizeof(float));
}

__host__ void free_vqgan_encoder(VQGANEncoder* encoder) {
    if (encoder->weights) deallocate_unified(encoder->weights);
}

__global__ void encode_kernel(const float* input, int* codes, int seq_len, int hidden_dim, int codebook_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seq_len) return;
    codes[tid] = tid % codebook_size;  // Simplified
}

__host__ void vqgan_encode(const VQGANEncoder* encoder, const float* input, int* codes, int seq_len) {
    int threads = 256;
    int blocks = (seq_len + threads - 1) / threads;
    encode_kernel<<<blocks, threads>>>(input, codes, seq_len, encoder->hidden_dim, encoder->codebook_size);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
