#include "video_rope.cuh"
#include "../utils/error.cuh"
#include <cmath>

namespace quila {

__host__ void init_video_rope(VideoRoPE* rope, int dim) {
    rope->dim = dim;

    // Allocate independent frequency bases (Req 7.2.1)
    CHECK_CUDA_ERROR(cudaMalloc(&rope->temporal_freqs, dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&rope->spatial_freqs_h, dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&rope->spatial_freqs_w, dim * sizeof(float)));

    // Initialize with different frequency ranges
    float* temp_freqs = new float[dim];
    for (int i = 0; i < dim; i++) {
        temp_freqs[i] = 1.0f / powf(10000.0f, 2.0f * i / dim);
    }
    CHECK_CUDA_ERROR(cudaMemcpy(rope->temporal_freqs, temp_freqs, dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(rope->spatial_freqs_h, temp_freqs, dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(rope->spatial_freqs_w, temp_freqs, dim * sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp_freqs;
}

__global__ void apply_temporal_rope(
    float* embeddings,
    const float* freqs,
    int seq_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= seq_len * dim) return;

    int t = idx / dim;
    int d = idx % dim;

    float angle = t * freqs[d];
    embeddings[idx] = embeddings[idx] * cosf(angle);
}

__global__ void apply_spatial_rope(
    float* embeddings,
    const float* freqs_h,
    const float* freqs_w,
    int height,
    int width,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height * width * dim) return;

    int hw = idx / dim;
    int h = hw / width;
    int w = hw % width;
    int d = idx % dim;

    float angle_h = h * freqs_h[d];
    float angle_w = w * freqs_w[d];
    embeddings[idx] = embeddings[idx] * cosf(angle_h + angle_w);
}

} // namespace quila
