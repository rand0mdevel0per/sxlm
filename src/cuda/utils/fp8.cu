#include "fp8.cuh"

namespace quila {

__global__ void quantize_fp32_to_fp8_kernel(
    const float* input,
    __nv_fp8_e4m3* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fp32_to_fp8_e4m3(input[idx]);
    }
}

__global__ void dequantize_fp8_to_fp32_kernel(
    const __nv_fp8_e4m3* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fp8_e4m3_to_fp32(input[idx]);
    }
}

} // namespace quila
