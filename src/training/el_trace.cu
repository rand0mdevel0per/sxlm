#include "sxlm/training/el_trace.cuh"

namespace sxlm {

// Kernel: Update eligibility traces with decay
__global__ void update_trace_kernel(
    double* traces,
    const double* gradients,
    float decay_rate,
    int num_params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_params) {
        traces[idx] = decay_rate * traces[idx] + gradients[idx];
    }
}

// Kernel: Apply reward to parameter updates
__global__ void apply_reward_kernel(
    const double* traces,
    double reward,
    double* param_updates,
    int num_params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_params) {
        param_updates[idx] += reward * traces[idx];
    }
}

EligibilityTrace::EligibilityTrace(const ElTraceConfig& config)
    : config_(config), num_params_(0) {
    d_traces_ = nullptr;
}

EligibilityTrace::~EligibilityTrace() {
    if (d_traces_) cudaFree(d_traces_);
}

void EligibilityTrace::update_traces(
    const double* gradients,
    int num_params
) {
    if (num_params_ != num_params) {
        if (d_traces_) cudaFree(d_traces_);
        cudaMalloc(&d_traces_, num_params * sizeof(double));
        cudaMemset(d_traces_, 0, num_params * sizeof(double));
        num_params_ = num_params;
    }

    int threads = 256;
    int blocks = (num_params + threads - 1) / threads;
    update_trace_kernel<<<blocks, threads>>>(
        d_traces_, gradients, config_.decay_rate, num_params
    );
    cudaDeviceSynchronize();
}

void EligibilityTrace::apply_reward(
    double reward,
    double* param_updates,
    int num_params
) {
    int threads = 256;
    int blocks = (num_params + threads - 1) / threads;
    apply_reward_kernel<<<blocks, threads>>>(
        d_traces_, reward, param_updates, num_params
    );
    cudaDeviceSynchronize();
}

void EligibilityTrace::reset() {
    if (d_traces_) {
        cudaMemset(d_traces_, 0, num_params_ * sizeof(double));
    }
}

} // namespace sxlm
