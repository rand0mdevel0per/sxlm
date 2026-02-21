#pragma once
#include <cuda_runtime.h>
#include <cuda_fp8.h>

namespace quila {

// fp8 E4M3 quantization for inter-neuron messages (Req 5.3.2)
__device__ __forceinline__ __nv_fp8_e4m3 fp32_to_fp8_e4m3(float val) {
    return __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float fp8_e4m3_to_fp32(__nv_fp8_e4m3 val) {
    return float(val);
}

// Batch conversion kernels
__global__ void quantize_fp32_to_fp8_kernel(
    const float* input,
    __nv_fp8_e4m3* output,
    int size
);

__global__ void dequantize_fp8_to_fp32_kernel(
    const __nv_fp8_e4m3* input,
    float* output,
    int size
);

} // namespace quila
