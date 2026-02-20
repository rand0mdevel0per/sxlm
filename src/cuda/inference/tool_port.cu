#include "pipeline.cuh"
#include "../utils/error.cuh"

namespace quila {

// Tool-Port for MCP integration
__global__ void tool_port_kernel(
    const float* tool_request,
    float* tool_response,
    int hidden_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= hidden_dim) return;

    // Simplified: echo request as response
    tool_response[tid] = tool_request[tid];
}

__host__ void invoke_tool_port(
    const float* tool_request,
    float* tool_response,
    int hidden_dim
) {
    int threads = 256;
    int blocks = (hidden_dim + threads - 1) / threads;

    tool_port_kernel<<<blocks, threads>>>(
        tool_request, tool_response, hidden_dim
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
