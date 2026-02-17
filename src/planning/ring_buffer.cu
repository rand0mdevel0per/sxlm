#include "sxlm/planning/ring_buffer.cuh"
#include <cmath>

namespace sxlm {

RingBufferPlanner::RingBufferPlanner(const RingBufferConfig& config)
    : config_(config), head_(0), tail_(0), size_(0) {
    buffer_.resize(config.buffer_size);

    // Allocate GPU memory for each entry
    for (int i = 0; i < config.buffer_size; i++) {
        cudaMalloc(&buffer_[i].state, config.state_dim * sizeof(double));
        cudaMalloc(&buffer_[i].plan, config.plan_dim * sizeof(double));
    }
}

RingBufferPlanner::~RingBufferPlanner() {
    for (auto& entry : buffer_) {
        cudaFree(entry.state);
        cudaFree(entry.plan);
    }
}

void RingBufferPlanner::add_entry(
    const double* state,
    const double* plan,
    float confidence
) {
    PlanEntry& entry = buffer_[head_];

    cudaMemcpy(entry.state, state, config_.state_dim * sizeof(double),
               cudaMemcpyDeviceToDevice);
    cudaMemcpy(entry.plan, plan, config_.plan_dim * sizeof(double),
               cudaMemcpyDeviceToDevice);
    entry.confidence = confidence;
    entry.timestamp = size_;

    head_ = (head_ + 1) % config_.buffer_size;
    if (size_ < config_.buffer_size) {
        size_++;
    } else {
        tail_ = (tail_ + 1) % config_.buffer_size;
    }
}

float RingBufferPlanner::detect_drift(const double* current_state) {
    if (size_ == 0) return 0.0f;

    // Compute average drift from recent states
    float total_drift = 0.0f;
    double* h_current = new double[config_.state_dim];
    double* h_state = new double[config_.state_dim];

    cudaMemcpy(h_current, current_state, config_.state_dim * sizeof(double),
               cudaMemcpyDeviceToHost);

    for (int i = 0; i < size_; i++) {
        int idx = (tail_ + i) % config_.buffer_size;
        cudaMemcpy(h_state, buffer_[idx].state, config_.state_dim * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // Compute L2 distance
        float dist = 0.0f;
        for (int d = 0; d < config_.state_dim; d++) {
            float diff = h_current[d] - h_state[d];
            dist += diff * diff;
        }
        total_drift += sqrt(dist);
    }

    delete[] h_current;
    delete[] h_state;

    return total_drift / size_;
}

void RingBufferPlanner::clear() {
    head_ = 0;
    tail_ = 0;
    size_ = 0;
}

} // namespace sxlm
