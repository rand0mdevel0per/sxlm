#include "suspicious_layer.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_suspicious_layer(SuspiciousLayer* layer, int num_neurons) {
    layer->num_neurons = num_neurons;
    layer->suspicion_scores = (float*)allocate_unified(num_neurons * sizeof(float));
    cudaMemset(layer->suspicion_scores, 0, num_neurons * sizeof(float));
}

__host__ void free_suspicious_layer(SuspiciousLayer* layer) {
    if (layer->suspicion_scores) deallocate_unified(layer->suspicion_scores);
}

__global__ void mark_suspicious_kernel(float* suspicion_scores, const bool* conflict_flags, int num_neurons) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_neurons) return;
    if (conflict_flags[tid]) suspicion_scores[tid] = 1.0f;
}

__host__ void mark_suspicious(SuspiciousLayer* layer, const bool* conflict_flags) {
    int threads = 256;
    int blocks = (layer->num_neurons + threads - 1) / threads;
    mark_suspicious_kernel<<<blocks, threads>>>(layer->suspicion_scores, conflict_flags, layer->num_neurons);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
