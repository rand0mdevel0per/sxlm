#pragma once
#include <cuda_runtime.h>
#include "../utils/error.cuh"

namespace quila {

// Minimal Sinkhorn-Knopp normalization
__device__ void sinkhorn_knopp(float* W, int n, int iters = 20) {
    for (int iter = 0; iter < iters; iter++) {
        // Row normalization
        for (int i = 0; i < n; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < n; j++) {
                row_sum += W[i * n + j];
            }
            if (row_sum > 1e-8f) {
                for (int j = 0; j < n; j++) {
                    W[i * n + j] /= row_sum;
                }
            }
        }

        // Column normalization
        for (int j = 0; j < n; j++) {
            float col_sum = 0.0f;
            for (int i = 0; i < n; i++) {
                col_sum += W[i * n + j];
            }
            if (col_sum > 1e-8f) {
                for (int i = 0; i < n; i++) {
                    W[i * n + j] /= col_sum;
                }
            }
        }
    }
}

// mHC mixing
__device__ void mhc_mix(
    const float* out_a,
    const float* out_b,
    const float* out_c,
    const float* out_d,
    const float* spec_vector,
    float* output,
    int hidden_dim
) {
    int tid = threadIdx.x;
    if (tid < hidden_dim) {
        // Simplified gate computation
        float gate[4] = {0.25f, 0.25f, 0.25f, 0.25f};

        // Fuse outputs
        output[tid] = gate[0] * out_a[tid] + gate[1] * out_b[tid] +
                      gate[2] * out_c[tid] + gate[3] * out_d[tid];
    }
}

} // namespace quila
