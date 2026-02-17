#pragma once

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>
#include <string>

namespace sxlm {

// Engram: O(1) Hash N-gram Memory Routing System
// Offloads static knowledge from main network

struct EngramConfig {
    int embedding_dim;        // Embedding dimension
    int max_ngram_length;     // Maximum N-gram length (e.g., 5)
    int num_hash_tables;      // Number of LSH tables (e.g., 8)
    size_t gpu_cache_size;    // GPU cache size in bytes
    size_t ram_cache_size;    // RAM cache size in bytes
};

class EngramMemory {
public:
    EngramMemory(const EngramConfig& config);
    ~EngramMemory();

    // Store N-gram with its embedding
    void store(
        const std::vector<int>& ngram_ids,
        const double* embedding
    );

    // Retrieve top-k N-grams by cosine similarity
    std::vector<std::pair<std::vector<int>, double*>> retrieve(
        const double* query_embedding,
        int top_k
    );

    // Prefetch next layer N-grams asynchronously
    void prefetch_async(
        const std::vector<std::vector<int>>& ngram_batch
    );

private:
    EngramConfig config_;

    // Hash tables for LSH
    std::vector<std::unordered_map<uint64_t, std::vector<int>>> hash_tables_;

    // Storage: ngram_id -> embedding
    std::unordered_map<uint64_t, double*> gpu_storage_;
    std::unordered_map<uint64_t, double*> ram_storage_;

    // Compute hash for N-gram
    uint64_t compute_hash(const std::vector<int>& ngram_ids, int table_idx);
};

} // namespace sxlm
