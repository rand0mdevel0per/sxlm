#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace sxlm {

// Tool Port: Structured symbolic output for tool calls

enum class ToolType {
    WEB_SEARCH,
    CODE_EXEC,
    FILE_OPS,
    DB_QUERY,
    API_CALL,
    MATH,
    VIZ,
    TEXT_GEN,
    IMAGE_GEN,
    AUDIO_PROC,
    VIDEO_PROC,
    TRANSLATE,
    SUMMARIZE,
    QA,
    REASONING,
    PLANNING
};

struct ToolPortConfig {
    int dim;                    // Model dimension
    int num_tools;              // Number of tool types (16)
    int max_param_len;          // Max parameter length
};

class ToolPort {
public:
    ToolPort(const ToolPortConfig& config);
    ~ToolPort();

    // Classify tool type from hidden state
    ToolType classify(
        const double* hidden_state,
        int seq_len
    );

    // Generate tool parameters
    void generate_params(
        const double* hidden_state,
        ToolType tool_type,
        double* params,
        int seq_len
    );

private:
    ToolPortConfig config_;
    double* classifier_weights_;    // Tool classifier weights
    double* param_weights_;         // Parameter generator weights
};

} // namespace sxlm
