#pragma once
#include <cuda_runtime.h>
#include "../utils/error.cuh"

namespace quila {

// Minimal Consistency Model denoising
__device__ void consistency_model_denoise(
    const float* h_noisy,
    float confidence,
    float* output,
    int hidden_dim
) {
    int tid = threadIdx.x;
    if (tid < hidden_dim) {
        float t = 1.0f - confidence;  // noise level

        // 1-step denoising (simplified)
        float h_clean = h_noisy[tid] * (1.0f - t * 0.5f);

        // Optional 2-step refinement for low confidence
        if (confidence < 0.4f) {
            float eps = 0.01f;
            h_clean = h_clean + eps;
            h_clean = h_clean * (1.0f - t * 0.25f);
        }

        output[tid] = h_clean;
    }
}

} // namespace quila
