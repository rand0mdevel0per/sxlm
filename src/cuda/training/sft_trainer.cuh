#pragma once
#include <cuda_runtime.h>

namespace quila {

// SFT trainer configuration (Req 16.1.1)
struct SFTConfig {
    int total_steps;
    int current_step;
    bool kfe_enabled;
    bool topology_evolution_enabled;
};

// Check if in first 10% of training (Req 16.1.1)
__host__ __device__ inline bool is_warmup_phase(const SFTConfig* config) {
    return config->current_step < (config->total_steps / 10);
}

// Update SFT config for current step
__host__ void update_sft_config(SFTConfig* config, int step);

} // namespace quila
