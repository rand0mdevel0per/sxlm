#pragma once
#include <cuda_runtime.h>

namespace quila {

// Hypothesis fork for verification
struct HypothesisFork {
    float* fork_states;
    int num_forks;
    int hidden_dim;
};

// Initialize hypothesis fork
__host__ void init_hypothesis_fork(HypothesisFork* fork, int num_forks, int hidden_dim);

// Free hypothesis fork
__host__ void free_hypothesis_fork(HypothesisFork* fork);

// Create fork from current state
__host__ void create_fork(HypothesisFork* fork, const float* current_state, int fork_id);

} // namespace quila
