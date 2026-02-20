#pragma once
#include <cuda_runtime.h>

namespace quila {

// Eligibility trace for credit assignment
struct ElTrace {
    float* session_trace;    // Session-level trace
    float* persistent_trace; // Persistent trace
    int num_neurons;
    int hidden_dim;
    float decay_rate;
};

// Initialize el-trace
__host__ void init_el_trace(ElTrace* trace, int num_neurons, int hidden_dim, float decay_rate);

// Free el-trace
__host__ void free_el_trace(ElTrace* trace);

// Update el-trace
__device__ void update_el_trace(ElTrace* trace, int neuron_id, float reward);

// Decay el-trace
__host__ void decay_el_trace(ElTrace* trace);

} // namespace quila
