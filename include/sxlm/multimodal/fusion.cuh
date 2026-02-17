#pragma once

#include <cuda_runtime.h>

namespace sxlm {

// Multi-modal Fusion with Interleaved-MRoPE

struct FusionConfig {
    int dim;                    // Model dimension (768)
    int num_heads;              // Number of attention heads
    int max_seq_len;            // Maximum sequence length
    float rope_theta;           // RoPE base frequency (10000.0)
};

class MultiModalFusion {
public:
    MultiModalFusion(const FusionConfig& config);
    ~MultiModalFusion();

    // Fuse text, image, and audio embeddings
    void fuse(
        const double* text_emb,
        const double* image_emb,
        const double* audio_emb,
        double* output,
        int text_len,
        int image_len,
        int audio_len,
        int batch
    );

    // Apply Interleaved-MRoPE
    void apply_rope(
        double* embeddings,
        int seq_len,
        int batch
    );

private:
    FusionConfig config_;
    double* rope_cache_;        // Cached RoPE frequencies
};

} // namespace sxlm
