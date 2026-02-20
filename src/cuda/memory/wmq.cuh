#pragma once
#include <cuda_runtime.h>

namespace quila {

// Working Memory Queue - hierarchical ring buffer
struct WMQStage {
    float* data;           // R^D buffer
    int capacity;          // Max entries
    int head;              // Write position
    int tail;              // Read position
    int size;              // Current entries
    bool is_active;
};

struct WMQ {
    WMQStage* stages;      // Multiple stages (Replay, LastInput, Context, Plan, Think, Output)
    int num_stages;
    int hidden_dim;
};

// Initialize WMQ
__host__ void init_wmq(WMQ* wmq, int num_stages, int hidden_dim, int stage_capacity);

// Free WMQ
__host__ void free_wmq(WMQ* wmq);

// Push to stage
__device__ void wmq_push(WMQStage* stage, const float* data, int hidden_dim);

// Pop from stage
__device__ void wmq_pop(WMQStage* stage, float* data, int hidden_dim);

} // namespace quila
