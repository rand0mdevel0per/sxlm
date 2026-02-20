#pragma once
#include <cuda_runtime.h>

namespace quila {

// VQ-GAN decoder for multimodal output
struct VQGANDecoder {
    float* weights;
    int hidden_dim;
    int output_dim;
    int codebook_size;
};

// Initialize decoder
__host__ void init_vqgan_decoder(VQGANDecoder* decoder, int hidden_dim, int output_dim, int codebook_size);

// Free decoder
__host__ void free_vqgan_decoder(VQGANDecoder* decoder);

// Decode latent codes to output
__host__ void vqgan_decode(const VQGANDecoder* decoder, const int* codes, float* output, int seq_len);

} // namespace quila
