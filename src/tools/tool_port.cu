#include "sxlm/tools/tool_port.cuh"
#include <cmath>

namespace sxlm {

__global__ void classify_tool_kernel(
    const double* hidden_state,
    const double* classifier_weights,
    double* logits,
    int seq_len,
    int dim,
    int num_tools
) {
    int tool_idx = blockIdx.x;
    if (tool_idx < num_tools) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            sum += hidden_state[i] * classifier_weights[tool_idx * dim + i];
        }
        logits[tool_idx] = sum;
    }
}

__global__ void generate_params_kernel(
    const double* hidden_state,
    const double* param_weights,
    double* params,
    int dim,
    int max_param_len
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < max_param_len) {
        double sum = 0.0;
        for (int i = 0; i < dim; i++) {
            sum += hidden_state[i] * param_weights[idx * dim + i];
        }
        params[idx] = tanh(sum);
    }
}

ToolPort::ToolPort(const ToolPortConfig& config)
    : config_(config), classifier_weights_(nullptr), param_weights_(nullptr) {
    cudaMalloc(&classifier_weights_, config.num_tools * config.dim * sizeof(double));
    cudaMalloc(&param_weights_, config.max_param_len * config.dim * sizeof(double));
}

ToolPort::~ToolPort() {
    if (classifier_weights_) cudaFree(classifier_weights_);
    if (param_weights_) cudaFree(param_weights_);
}

ToolType ToolPort::classify(
    const double* hidden_state,
    int seq_len
) {
    double* d_logits;
    cudaMalloc(&d_logits, config_.num_tools * sizeof(double));

    classify_tool_kernel<<<config_.num_tools, 1>>>(
        hidden_state, classifier_weights_, d_logits,
        seq_len, config_.dim, config_.num_tools
    );

    std::vector<double> logits(config_.num_tools);
    cudaMemcpy(logits.data(), d_logits, config_.num_tools * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaFree(d_logits);

    int max_idx = 0;
    for (int i = 1; i < config_.num_tools; i++) {
        if (logits[i] > logits[max_idx]) max_idx = i;
    }

    return static_cast<ToolType>(max_idx);
}

void ToolPort::generate_params(
    const double* hidden_state,
    ToolType tool_type,
    double* params,
    int seq_len
) {
    int threads = 256;
    int blocks = (config_.max_param_len + threads - 1) / threads;

    generate_params_kernel<<<blocks, threads>>>(
        hidden_state, param_weights_, params,
        config_.dim, config_.max_param_len
    );
}

} // namespace sxlm
