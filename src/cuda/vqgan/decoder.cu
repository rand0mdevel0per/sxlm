#include "decoder.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_vqgan_decoder(VQGANDecoder* decoder, int hidden_dim, int output_dim, int codebook_size) {
    decoder->hidden_dim = hidden_dim;
    decoder->output_dim = output_dim;
    decoder->codebook_size = codebook_size;
    decoder->weights = (float*)allocate_unified(hidden_dim * output_dim * sizeof(float));
}

__host__ void free_vqgan_decoder(VQGANDecoder* decoder) {
    if (decoder->weights) deallocate_unified(decoder->weights);
}

__global__ void decode_kernel(const int* codes, float* output, int seq_len, int output_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seq_len * output_dim) return;
    output[tid] = (float)(codes[tid / output_dim]);  // Simplified
}

__host__ void vqgan_decode(const VQGANDecoder* decoder, const int* codes, float* output, int seq_len) {
    int threads = 256;
    int total = seq_len * decoder->output_dim;
    int blocks = (total + threads - 1) / threads;
    decode_kernel<<<blocks, threads>>>(codes, output, seq_len, decoder->output_dim);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
