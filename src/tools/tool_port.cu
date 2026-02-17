#include "sxlm/tools/tool_port.cuh"
#include <cmath>
#include <algorithm>

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
    cudaMalloc(&param_weights_, config.max_param_len * config.dim * sizeof(double));
}

ToolPort::~ToolPort() {
    if (classifier_weights_) cudaFree(classifier_weights_);
    if (param_weights_) cudaFree(param_weights_);
}

void ToolPort::discover_tools(const std::string& mcp_server_url) {
    // TODO: Implement MCP server discovery via HTTP
    // For now, use placeholder tools
    tools_.clear();
    tools_.push_back({"web_search", "Search the web", {}});
    tools_.push_back({"code_exec", "Execute code", {}});
    tools_.push_back({"file_ops", "File operations", {}});

    // Allocate classifier weights for discovered tools
    if (classifier_weights_) cudaFree(classifier_weights_);
    cudaMalloc(&classifier_weights_, tools_.size() * config_.dim * sizeof(double));
}

std::string ToolPort::classify(const double* hidden_state, int seq_len) {
    if (tools_.empty()) return "";

    double* d_logits;
    cudaMalloc(&d_logits, tools_.size() * sizeof(double));

    classify_tool_kernel<<<tools_.size(), 1>>>(
        hidden_state, classifier_weights_, d_logits,
        seq_len, config_.dim, tools_.size()
    );

    std::vector<double> logits(tools_.size());
    cudaMemcpy(logits.data(), d_logits, tools_.size() * sizeof(double),
               cudaMemcpyDeviceToHost);
    cudaFree(d_logits);

    int max_idx = std::max_element(logits.begin(), logits.end()) - logits.begin();
    return tools_[max_idx].name;
}

void ToolPort::generate_params(
    const double* hidden_state,
    const std::string& tool_name,
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
