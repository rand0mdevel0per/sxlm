#include "sxlm/memory/engram.cuh"
#include <cstring>
#include <algorithm>

namespace sxlm {

EngramMemory::EngramMemory(const EngramConfig& config)
    : config_(config) {
    // Initialize hash tables
    hash_tables_.resize(config.num_hash_tables);
}

EngramMemory::~EngramMemory() {
    // Free GPU storage
    for (auto& pair : gpu_storage_) {
        cudaFree(pair.second);
    }
    // Free RAM storage
    for (auto& pair : ram_storage_) {
        delete[] pair.second;
    }
}

uint64_t EngramMemory::compute_hash(
    const std::vector<int>& ngram_ids,
    int table_idx
) {
    // Simple hash function (FNV-1a variant)
    uint64_t hash = 14695981039346656037ULL + table_idx;
    for (int id : ngram_ids) {
        hash ^= id;
        hash *= 1099511628211ULL;
    }
    return hash;
}

void EngramMemory::store(
    const std::vector<int>& ngram_ids,
    const double* embedding
) {
    // Compute hash for all tables
    for (int i = 0; i < config_.num_hash_tables; i++) {
        uint64_t hash = compute_hash(ngram_ids, i);
        hash_tables_[i][hash] = ngram_ids;
    }

    // Store embedding in GPU cache
    uint64_t key = compute_hash(ngram_ids, 0);
    double* d_emb;
    cudaMalloc(&d_emb, config_.embedding_dim * sizeof(double));
    cudaMemcpy(d_emb, embedding, config_.embedding_dim * sizeof(double),
               cudaMemcpyHostToDevice);
    gpu_storage_[key] = d_emb;
}

std::vector<std::pair<std::vector<int>, double*>> EngramMemory::retrieve(
    const double* query_embedding,
    int top_k
) {
    std::vector<std::pair<std::vector<int>, double*>> results;
    // Simplified retrieval (full implementation would use cosine similarity)
    return results;
}

void EngramMemory::prefetch_async(
    const std::vector<std::vector<int>>& ngram_batch
) {
    // Asynchronous prefetch from RAM to GPU
    for (const auto& ngram : ngram_batch) {
        uint64_t key = compute_hash(ngram, 0);
        if (ram_storage_.find(key) != ram_storage_.end() &&
            gpu_storage_.find(key) == gpu_storage_.end()) {
            // Transfer from RAM to GPU asynchronously
            double* d_emb;
            cudaMalloc(&d_emb, config_.embedding_dim * sizeof(double));
            cudaMemcpyAsync(d_emb, ram_storage_[key],
                           config_.embedding_dim * sizeof(double),
                           cudaMemcpyHostToDevice);
            gpu_storage_[key] = d_emb;
        }
    }
}

} // namespace sxlm
