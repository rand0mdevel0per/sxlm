#pragma once
#include <cuda_runtime.h>

namespace quila {

// VQ-GAN codebook
struct Codebook {
    float* embeddings;
    int codebook_size;
    int embedding_dim;
};

// Initialize codebook
__host__ void init_codebook(Codebook* codebook, int codebook_size, int embedding_dim);

// Free codebook
__host__ void free_codebook(Codebook* codebook);

// Quantize vectors to nearest codebook entry
__host__ void quantize(const Codebook* codebook, const float* vectors, int* codes, int num_vectors);

} // namespace quila
