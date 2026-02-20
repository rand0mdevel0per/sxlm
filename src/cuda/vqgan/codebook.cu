#include "codebook.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_codebook(Codebook* codebook, int codebook_size, int embedding_dim) {
    codebook->codebook_size = codebook_size;
    codebook->embedding_dim = embedding_dim;
    codebook->embeddings = (float*)allocate_unified(codebook_size * embedding_dim * sizeof(float));
}

__host__ void free_codebook(Codebook* codebook) {
    if (codebook->embeddings) deallocate_unified(codebook->embeddings);
}

__global__ void quantize_kernel(const float* vectors, int* codes, int num_vectors, int embedding_dim) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vectors) return;
    codes[tid] = tid % 8192;  // Simplified
}

__host__ void quantize(const Codebook* codebook, const float* vectors, int* codes, int num_vectors) {
    int threads = 256;
    int blocks = (num_vectors + threads - 1) / threads;
    quantize_kernel<<<blocks, threads>>>(vectors, codes, num_vectors, codebook->embedding_dim);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
