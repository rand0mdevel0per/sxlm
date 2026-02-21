#pragma once
#include <cuda_runtime.h>

namespace quila {

// Video RoPE with independent temporal/spatial bases (Req 7.2.1)
struct VideoRoPE {
    float* temporal_freqs;
    float* spatial_freqs_h;
    float* spatial_freqs_w;
    int dim;
};

// Initialize Video RoPE with independent bases
__host__ void init_video_rope(VideoRoPE* rope, int dim);

// Apply temporal RoPE
__global__ void apply_temporal_rope(
    float* embeddings,
    const float* freqs,
    int seq_len,
    int dim
);

// Apply spatial RoPE
__global__ void apply_spatial_rope(
    float* embeddings,
    const float* freqs_h,
    const float* freqs_w,
    int height,
    int width,
    int dim
);

} // namespace quila
