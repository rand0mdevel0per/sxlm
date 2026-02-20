#pragma once
#include <cuda_runtime.h>
#include "el_trace.cuh"

namespace quila {

// Credit assignment system
struct CreditAssignment {
    float* neuron_credits;
    int num_neurons;
};

// Initialize credit assignment
__host__ void init_credit_assignment(CreditAssignment* ca, int num_neurons);

// Free credit assignment
__host__ void free_credit_assignment(CreditAssignment* ca);

// Assign credit based on el-trace and reward
__host__ void assign_credit(CreditAssignment* ca, const ElTrace* trace, float reward);

} // namespace quila
