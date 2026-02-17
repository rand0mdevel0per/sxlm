#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <string>

namespace sxlm {

// el-trace: Eligibility Trace for parameter-level reinforcement learning
// Records precise contribution of each parameter to outputs

struct ElTraceConfig {
    float decay_rate;         // Trace decay (e.g., 0.9)
    int trace_buffer_size;    // Buffer size for traces
    bool enable_negative;     // Enable negative feedback for hallucinations
};

class EligibilityTrace {
public:
    EligibilityTrace(const ElTraceConfig& config);
    ~EligibilityTrace();

    // Update traces with current gradients
    void update_traces(
        const double* gradients,
        int num_params
    );

    // Apply reward to parameters based on traces
    void apply_reward(
        double reward,
        double* param_updates,
        int num_params
    );

    // Reset traces (e.g., at episode boundary)
    void reset();

private:
    ElTraceConfig config_;
    double* d_traces_;        // Device traces
    int num_params_;
};

} // namespace sxlm
