#pragma once
#include <cstdint>
#include <vector>
#include <memory>

namespace quila {

// Session Context Tree node
struct SCTNode {
    uint64_t node_id;
    std::vector<float> context_vector;  // R^D
    std::vector<uint64_t> children;
    uint64_t parent;
    uint64_t timestamp;
    bool is_leaf;

    SCTNode(uint64_t id, const std::vector<float>& ctx)
        : node_id(id), context_vector(ctx), parent(0),
          timestamp(0), is_leaf(true) {}
};

// Session Context Tree - B-link-tree structure
class SCT {
private:
    std::vector<std::shared_ptr<SCTNode>> nodes;
    uint64_t root_id;
    uint64_t next_id;
    int hidden_dim;

public:
    SCT(int dim) : root_id(0), next_id(0), hidden_dim(dim) {}

    // Insert context node
    uint64_t insert(const std::vector<float>& context, uint64_t parent_id = 0);

    // Query context path
    std::vector<uint64_t> query_path(uint64_t node_id);

    // Get node
    SCTNode* get_node(uint64_t node_id);
};

} // namespace quila
