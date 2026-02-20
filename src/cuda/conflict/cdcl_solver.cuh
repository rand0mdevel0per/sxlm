#pragma once
#include <cuda_runtime.h>

namespace quila {

// CDCL soft solver for conflict detection
struct CDCLSolver {
    bool* conflict_flags;
    int num_neurons;
};

// Initialize CDCL solver
__host__ void init_cdcl_solver(CDCLSolver* solver, int num_neurons);

// Free CDCL solver
__host__ void free_cdcl_solver(CDCLSolver* solver);

// Detect conflicts in neuron states
__host__ bool detect_conflicts(CDCLSolver* solver, const float* neuron_states, int hidden_dim);

} // namespace quila
