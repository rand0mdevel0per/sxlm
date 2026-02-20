#pragma once
#include <cuda_runtime.h>

namespace quila {

enum PortType {
    PLANNER,
    THINKER,
    ANALYSIS
};

// Port state structure
struct PortState {
    PortType type;
    int port_id;
    float* current_state;      // R^D
    float* aggr;               // Fixed-size aggregation vector
    float* wkv_state;          // RWKV state for linear scan
    bool is_active;
    int hidden_dim;
};

// Port configuration
struct PortConfig {
    int hidden_dim;
    int num_neurons;
    float alpha;  // Utility weight for wkv
    float beta;   // Utility weight for session_el_trace
    float gamma;  // Utility weight for persistent_el_trace
    int nsa_top_k;
};

} // namespace quila
