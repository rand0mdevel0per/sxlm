#include "sxlm/multimodal/fusion.cuh"
#include <cmath>
#include <vector>

namespace sxlm {

__global__ void rope_kernel(
    double* embeddings,
    const double* rope_cache,
    int seq_len,
    int dim,
    int num_heads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int head_dim = dim / num_heads;

    if (idx < seq_len * dim) {
        int pos = idx / dim;
        int d = idx % dim;
        int head = d / head_dim;
        int d_in_head = d % head_dim;

        if (d_in_head % 2 == 0) {
            int freq_idx = d_in_head / 2;
            double freq = rope_cache[freq_idx];
            double angle = pos * freq;

            double cos_val = cos(angle);
            double sin_val = sin(angle);

            double x = embeddings[idx];
            double y = embeddings[idx + 1];

            embeddings[idx] = x * cos_val - y * sin_val;
            embeddings[idx + 1] = x * sin_val + y * cos_val;
        }
    }
}

__global__ void fusion_kernel(
    const double* text_emb,
    const double* image_emb,
    const double* audio_emb,
    double* output,
    int text_len,
    int image_len,
    int audio_len,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_len = text_len + image_len + audio_len;

    if (idx < total_len * dim) {
        int pos = idx / dim;
        int d = idx % dim;

        if (pos < text_len) {
            output[idx] = text_emb[pos * dim + d];
        } else if (pos < text_len + image_len) {
            int img_pos = pos - text_len;
            output[idx] = image_emb[img_pos * dim + d];
        } else {
            int aud_pos = pos - text_len - image_len;
            output[idx] = audio_emb[aud_pos * dim + d];
        }
    }
}

MultiModalFusion::MultiModalFusion(const FusionConfig& config)
    : config_(config), rope_cache_(nullptr) {
    int head_dim = config.dim / config.num_heads;
    cudaMalloc(&rope_cache_, (head_dim / 2) * sizeof(double));

    // Initialize RoPE frequencies
    std::vector<double> freqs(head_dim / 2);
    for (int i = 0; i < head_dim / 2; i++) {
        freqs[i] = 1.0 / pow(config.rope_theta, 2.0 * i / head_dim);
    }
    cudaMemcpy(rope_cache_, freqs.data(), freqs.size() * sizeof(double),
               cudaMemcpyHostToDevice);
}

MultiModalFusion::~MultiModalFusion() {
    if (rope_cache_) {
        cudaFree(rope_cache_);
    }
}

void MultiModalFusion::fuse(
    const double* text_emb,
    const double* image_emb,
    const double* audio_emb,
    double* output,
    int text_len,
    int image_len,
    int audio_len,
    int batch
) {
    int total_len = text_len + image_len + audio_len;
    int total = total_len * config_.dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    fusion_kernel<<<blocks, threads>>>(
        text_emb, image_emb, audio_emb, output,
        text_len, image_len, audio_len, config_.dim
    );
}

void MultiModalFusion::apply_rope(
    double* embeddings,
    int seq_len,
    int batch
) {
    int total = seq_len * config_.dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    rope_kernel<<<blocks, threads>>>(
        embeddings, rope_cache_,
        seq_len, config_.dim, config_.num_heads
    );
}

} // namespace sxlm
