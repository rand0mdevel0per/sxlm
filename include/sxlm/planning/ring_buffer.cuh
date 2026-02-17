#pragma once

#include <cuda_runtime.h>
#include <vector>

namespace sxlm {

// Ring Buffer Planner: Stores (state, plan, confidence) tuples
// Detects drift and triggers re-planning

struct PlanEntry {
    double* state;           // State embedding
    double* plan;            // Plan embedding
    float confidence;        // Confidence score
    int timestamp;
};

struct RingBufferConfig {
    int buffer_size;         // Ring buffer size (e.g., 128)
    int state_dim;           // State dimension
    int plan_dim;            // Plan dimension
    float drift_threshold;   // Drift threshold (e.g., 0.3)
};

class RingBufferPlanner {
public:
    RingBufferPlanner(const RingBufferConfig& config);
    ~RingBufferPlanner();

    // Add entry to ring buffer
    void add_entry(
        const double* state,
        const double* plan,
        float confidence
    );

    // Detect drift between current state and recent plans
    float detect_drift(const double* current_state);

    // Clear buffer (trigger re-plan)
    void clear();

private:
    RingBufferConfig config_;
    std::vector<PlanEntry> buffer_;
    int head_;
    int tail_;
    int size_;
};

} // namespace sxlm
