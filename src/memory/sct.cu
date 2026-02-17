#include "sxlm/memory/sct.cuh"
#include <cstring>

namespace sxlm {

SemanticContextTree::SemanticContextTree(const SCTConfig& config)
    : config_(config), next_node_id_(1), hnsw_index_(nullptr) {
}

SemanticContextTree::~SemanticContextTree() {
    for (auto& pair : nodes_) {
        if (pair.second->embedding) {
            cudaFree(pair.second->embedding);
        }
        delete pair.second;
    }
}

uint64_t SemanticContextTree::insert(
    const double* embedding,
    const std::string& data,
    uint64_t parent_id
) {
    SCTNode* node = new SCTNode();
    node->node_id = next_node_id_++;
    node->parent_id = parent_id;
    node->data = data;
    node->depth = (parent_id == 0) ? 0 : nodes_[parent_id]->depth + 1;

    // Allocate and copy embedding
    cudaMalloc(&node->embedding, config_.embedding_dim * sizeof(double));
    cudaMemcpy(node->embedding, embedding,
               config_.embedding_dim * sizeof(double),
               cudaMemcpyHostToDevice);

    nodes_[node->node_id] = node;

    // Update parent's children
    if (parent_id != 0 && nodes_.find(parent_id) != nodes_.end()) {
        nodes_[parent_id]->children.push_back(node->node_id);
    }

    return node->node_id;
}

std::vector<SCTNode*> SemanticContextTree::retrieve(
    const double* query_embedding,
    int top_k
) {
    // Simplified retrieval (full implementation would use HNSW)
    std::vector<SCTNode*> results;
    return results;
}

void SemanticContextTree::build_hnsw_index() {
    // HNSW index building (requires hnswlib integration)
}

} // namespace sxlm
