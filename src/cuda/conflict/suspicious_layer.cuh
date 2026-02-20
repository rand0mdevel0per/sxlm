#pragma once
#include <cuda_runtime.h>

namespace quila {

// Suspicious layer for conflict resolution
struct SuspiciousLayer {
    float* suspicion_scores;
    int num_neurons;
};

// Initialize suspicious layer
__host__ void init_suspicious_layer(SuspiciousLayer* layer, int num_neurons);

// Free suspicious layer
__host__ void free_suspicious_layer(SuspiciousLayer* layer);

// Mark neurons as suspicious
__host__ void mark_suspicious(SuspiciousLayer* layer, const bool* conflict_flags);

} // namespace quila
