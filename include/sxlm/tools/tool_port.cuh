#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <unordered_map>

namespace sxlm {

// MCP Tool descriptor
struct MCPTool {
    std::string name;
    std::string description;
    std::unordered_map<std::string, std::string> schema;
};

struct ToolPortConfig {
    int dim;
    int max_param_len;
};

// MCP-based Tool Port
class ToolPort {
public:
    ToolPort(const ToolPortConfig& config);
    ~ToolPort();

    // Discover tools from MCP server
    void discover_tools(const std::string& mcp_server_url);

    // Classify which tool to use
    std::string classify(const double* hidden_state, int seq_len);

    // Generate parameters for tool
    void generate_params(const double* hidden_state, const std::string& tool_name,
                        double* params, int seq_len);

    // Get available tools
    std::vector<MCPTool> get_tools() const { return tools_; }

private:
    ToolPortConfig config_;
    std::vector<MCPTool> tools_;
    double* classifier_weights_;
    double* param_weights_;
};

} // namespace sxlm
