#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace sxlm {

// HOT-NSA: High-Order Thought Natively Sparse Attention
// Dynamic sparse masking based on attention & utility

struct HOTConfig {
    int dim;              // Model dimension
    int num_heads;        // Total attention heads
    int global_heads;     // Heads for long-range (SCT)
    int local_heads;      // Heads for local window
    int selector_heads;   // Heads for HNSW retrieval
    int window_size;      // Local attention window
    float hot_threshold;  // Threshold for HOT filtering
};

class HOTNSAAttention {
public:
    HOTNSAAttention(const HOTConfig& config);
    ~HOTNSAAttention();

    // Compute HOT scores for each token (thinking complexity)
    void compute_hot_scores(
        const double* input,      // [batch, seq_len, dim]
        double* hot_scores,       // [batch, seq_len]
        int batch, int seq_len
    );

    // Generate dynamic sparse mask based on HOT scores
    void generate_sparse_mask(
        const double* hot_scores, // [batch, seq_len]
        bool* mask,               // [batch, seq_len, seq_len]
        int batch, int seq_len
    );

    // Forward pass with HOT-NSA
    void forward(
        const double* query,      // [batch, seq_len, dim]
        const double* key,        // [batch, seq_len, dim]
        const double* value,      // [batch, seq_len, dim]
        const bool* mask,         // [batch, seq_len, seq_len]
        double* output,           // [batch, seq_len, dim]
        int batch, int seq_len
    );

private:
    HOTConfig config_;
    double* d_qkv_weights_;       // QKV projection weights
    double* d_output_weights_;    // Output projection weights
    cublasHandle_t cublas_handle_;
};

} // namespace sxlm
