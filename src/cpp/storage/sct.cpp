#include "sct.h"

namespace quila {

uint64_t SCT::insert(const std::vector<float>& context, uint64_t parent_id) {
    uint64_t id = next_id++;
    auto node = std::make_shared<SCTNode>(id, context);
    node->parent = parent_id;

    if (parent_id < nodes.size() && nodes[parent_id]) {
        nodes[parent_id]->children.push_back(id);
        nodes[parent_id]->is_leaf = false;
    }

    if (id >= nodes.size()) {
        nodes.resize(id + 1);
    }
    nodes[id] = node;

    if (root_id == 0 && id == 0) {
        root_id = id;
    }

    return id;
}

std::vector<uint64_t> SCT::query_path(uint64_t node_id) {
    std::vector<uint64_t> path;
    uint64_t current = node_id;

    while (current < nodes.size() && nodes[current]) {
        path.push_back(current);
        if (current == root_id) break;
        current = nodes[current]->parent;
    }

    std::reverse(path.begin(), path.end());
    return path;
}

SCTNode* SCT::get_node(uint64_t node_id) {
    if (node_id < nodes.size()) {
        return nodes[node_id].get();
    }
    return nullptr;
}

} // namespace quila
