#pragma once

#include <string>

namespace sxlm {

// Minimal TOML-based configuration
struct QuilaConfig {
    // Model dimensions
    int dim = 768;
    int num_heads = 12;
    int num_layers = 24;
    int max_seq_len = 128000;

    // Grid size (for spatial neurons)
    int grid_x = 32;
    int grid_y = 32;
    int grid_z = 32;

    // HOT-NSA config
    int global_heads = 4;
    int local_heads = 4;
    int selector_heads = 4;
    int local_window = 512;
    float hot_threshold = 0.5f;

    // Engram config
    int engram_tables = 8;
    int max_ngram_len = 5;

    // SCT config
    int sct_branching = 32;
    int sct_max_depth = 27;

    // Training config
    float learning_rate = 1e-4f;
    float el_trace_decay = 0.9f;

    // Storage config
    int gpu_cache_mb = 8192;
    int ram_cache_mb = 32768;

    // Load from TOML file
    static QuilaConfig load(const std::string& path);

    // Save to TOML file
    bool save(const std::string& path) const;
};

} // namespace sxlm
