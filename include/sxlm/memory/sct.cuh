#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

namespace sxlm {

// SCT: Semantic Context Tree with HNSW indexing
// Supports 128M logical context through hierarchical abstraction

struct SCTConfig {
    int embedding_dim;        // Embedding dimension
    int branching_factor;     // Tree branching factor (e.g., 32)
    int max_depth;            // Maximum tree depth (e.g., 27 for 128M)
    size_t gpu_nodes;         // GPU cache nodes (e.g., 10K)
    size_t ram_nodes;         // RAM cache nodes (e.g., 1M)
    size_t disk_nodes;        // Disk storage nodes (e.g., 127M)
};

struct SCTNode {
    uint64_t node_id;
    uint64_t parent_id;
    std::vector<uint64_t> children;
    double* embedding;        // Summary embedding
    std::string data;         // Original text/data
    int depth;
};

class SemanticContextTree {
public:
    SemanticContextTree(const SCTConfig& config);
    ~SemanticContextTree();

    // Insert node into tree
    uint64_t insert(
        const double* embedding,
        const std::string& data,
        uint64_t parent_id = 0
    );

    // Retrieve top-k nodes by similarity
    std::vector<SCTNode*> retrieve(
        const double* query_embedding,
        int top_k
    );

    // Build HNSW index for fast retrieval
    void build_hnsw_index();

private:
    SCTConfig config_;
    std::unordered_map<uint64_t, SCTNode*> nodes_;
    uint64_t next_node_id_;
    void* hnsw_index_;        // HNSW index handle
};

} // namespace sxlm
