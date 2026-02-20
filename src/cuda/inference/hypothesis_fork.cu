#include "hypothesis_fork.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_hypothesis_fork(HypothesisFork* fork, int num_forks, int hidden_dim) {
    fork->num_forks = num_forks;
    fork->hidden_dim = hidden_dim;
    fork->fork_states = (float*)allocate_unified(num_forks * hidden_dim * sizeof(float));
}

__host__ void free_hypothesis_fork(HypothesisFork* fork) {
    if (fork->fork_states) deallocate_unified(fork->fork_states);
}

__host__ void create_fork(HypothesisFork* fork, const float* current_state, int fork_id) {
    cudaMemcpy(&fork->fork_states[fork_id * fork->hidden_dim], current_state,
               fork->hidden_dim * sizeof(float), cudaMemcpyDeviceToDevice);
}

} // namespace quila
