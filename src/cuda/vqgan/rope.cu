#include "rope.cuh"
#include "../utils/error.cuh"
#include <cmath>

namespace quila {

__device__ void apply_rope(float* vec, int pos, int dim, Modality modality) {
    float base = (modality == TEXT) ? 10000.0f : 5000.0f;
    for (int i = 0; i < dim / 2; i++) {
        float freq = 1.0f / powf(base, 2.0f * i / dim);
        float angle = pos * freq;
        float cos_val = cosf(angle);
        float sin_val = sinf(angle);
        float v0 = vec[2 * i];
        float v1 = vec[2 * i + 1];
        vec[2 * i] = v0 * cos_val - v1 * sin_val;
        vec[2 * i + 1] = v0 * sin_val + v1 * cos_val;
    }
}

__global__ void rope_kernel(float* embeddings, int seq_len, int hidden_dim, Modality modality) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seq_len) return;
    apply_rope(&embeddings[tid * hidden_dim], tid, hidden_dim, modality);
}

__host__ void apply_rope_host(float* embeddings, int seq_len, int hidden_dim, Modality modality) {
    int threads = 256;
    int blocks = (seq_len + threads - 1) / threads;
    rope_kernel<<<blocks, threads>>>(embeddings, seq_len, hidden_dim, modality);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
