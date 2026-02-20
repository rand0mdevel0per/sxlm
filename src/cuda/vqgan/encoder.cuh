#pragma once
#include <cuda_runtime.h>

namespace quila {

// VQ-GAN encoder for multimodal input
struct VQGANEncoder {
    float* weights;
    int input_dim;
    int hidden_dim;
    int codebook_size;
};

// Initialize encoder
__host__ void init_vqgan_encoder(VQGANEncoder* encoder, int input_dim, int hidden_dim, int codebook_size);

// Free encoder
__host__ void free_vqgan_encoder(VQGANEncoder* encoder);

// Encode input to latent codes
__host__ void vqgan_encode(const VQGANEncoder* encoder, const float* input, int* codes, int seq_len);

} // namespace quila
