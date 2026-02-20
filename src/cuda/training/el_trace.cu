#include "el_trace.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_el_trace(ElTrace* trace, int num_neurons, int hidden_dim, float decay_rate) {
    trace->num_neurons = num_neurons;
    trace->hidden_dim = hidden_dim;
    trace->decay_rate = decay_rate;

    trace->session_trace = (float*)allocate_unified(num_neurons * sizeof(float));
    trace->persistent_trace = (float*)allocate_unified(num_neurons * sizeof(float));

    cudaMemset(trace->session_trace, 0, num_neurons * sizeof(float));
    cudaMemset(trace->persistent_trace, 0, num_neurons * sizeof(float));
}

__host__ void free_el_trace(ElTrace* trace) {
    if (trace->session_trace) deallocate_unified(trace->session_trace);
    if (trace->persistent_trace) deallocate_unified(trace->persistent_trace);
}

__device__ void update_el_trace(ElTrace* trace, int neuron_id, float reward) {
    trace->session_trace[neuron_id] += reward;
    trace->persistent_trace[neuron_id] += reward * 0.1f;
}

__global__ void decay_el_trace_kernel(float* trace, int size, float decay_rate) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    trace[tid] *= decay_rate;
}

__host__ void decay_el_trace(ElTrace* trace) {
    int threads = 256;
    int blocks = (trace->num_neurons + threads - 1) / threads;

    decay_el_trace_kernel<<<blocks, threads>>>(trace->session_trace, trace->num_neurons, trace->decay_rate);
    decay_el_trace_kernel<<<blocks, threads>>>(trace->persistent_trace, trace->num_neurons, trace->decay_rate * 0.99f);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
