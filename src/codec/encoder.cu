#include "sintellix/codec/encoder.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace sintellix {

SemanticEncoder::SemanticEncoder(const std::string& model_path, std::shared_ptr<VQCodebook> codebook)
    : model_path_(model_path)
    , codebook_(codebook)
    , clip_session_(nullptr)
    , clip_output_gpu_(nullptr)
    , codes_gpu_(nullptr)
{
}

SemanticEncoder::~SemanticEncoder() {
    if (clip_output_gpu_) cudaFree(clip_output_gpu_);
    if (codes_gpu_) cudaFree(codes_gpu_);

#ifdef USE_ONNXRUNTIME
    if (clip_session_) {
        delete static_cast<Ort::Session*>(clip_session_);
        clip_session_ = nullptr;
    }
#endif
}

bool SemanticEncoder::initialize() {
    // Create VQ-GAN encoder (512-dim CLIP output -> codebook embedding dim)
    size_t clip_dim = 512;
    size_t hidden_dim = codebook_->get_embedding_dim();

    vqgan_encoder_ = std::make_unique<VQGANEncoder>(clip_dim, hidden_dim, codebook_);

    if (!vqgan_encoder_->initialize()) {
        return false;
    }

    // Allocate GPU buffers
    cudaMalloc(&clip_output_gpu_, 256 * clip_dim * sizeof(double));  // Max batch size 256
    cudaMalloc(&codes_gpu_, 256 * 256 * sizeof(int));  // Max 256 codes per input

    // Load CLIP model
    return load_clip_model();
}

bool SemanticEncoder::load_clip_model() {
#ifdef USE_ONNXRUNTIME
    try {
        std::cout << "Loading CLIP model from: " << model_path_ << std::endl;

        // Check if model file exists
        std::ifstream file(model_path_);
        if (!file.good()) {
            std::cerr << "CLIP model file not found: " << model_path_ << std::endl;
            return false;
        }
        file.close();

        // Create ONNX Runtime environment and session
        Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CLIP");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);

        // Enable CUDA if available
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);

        // Create session
        Ort::Session* session = new Ort::Session(*env, model_path_.c_str(), session_options);
        clip_session_ = session;

        std::cout << "CLIP model loaded successfully with ONNX Runtime" << std::endl;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return false;
    }
#else
    // Placeholder implementation when ONNX Runtime is not available
    std::cout << "Loading CLIP model from: " << model_path_ << std::endl;

    std::ifstream file(model_path_);
    if (!file.good()) {
        std::cerr << "CLIP model file not found: " << model_path_ << std::endl;
        return false;
    }

    std::cout << "CLIP model loaded successfully (placeholder - ONNX Runtime not available)" << std::endl;
    return true;
#endif
}

// Helper function: Simplified CLIP tokenizer (word-level)
static std::vector<int> tokenize_text(const std::string& text, int max_length = 77) {
    std::vector<int> tokens;

    // Special tokens
    const int SOT_TOKEN = 49406;  // Start of text
    const int EOT_TOKEN = 49407;  // End of text
    const int PAD_TOKEN = 0;      // Padding

    tokens.push_back(SOT_TOKEN);

    // Simple word-level tokenization
    std::string word;
    for (char c : text) {
        if (std::isspace(c) || std::ispunct(c)) {
            if (!word.empty()) {
                // Hash word to token ID (simplified)
                int token_id = std::hash<std::string>{}(word) % 49405 + 1;
                tokens.push_back(token_id);
                word.clear();
            }
        } else {
            word += std::tolower(c);
        }
    }

    if (!word.empty()) {
        int token_id = std::hash<std::string>{}(word) % 49405 + 1;
        tokens.push_back(token_id);
    }

    tokens.push_back(EOT_TOKEN);

    // Pad or truncate to max_length
    if (tokens.size() < max_length) {
        tokens.resize(max_length, PAD_TOKEN);
    } else if (tokens.size() > max_length) {
        tokens.resize(max_length);
        tokens[max_length - 1] = EOT_TOKEN;
    }

    return tokens;
}

bool SemanticEncoder::run_clip_text_inference(const std::vector<std::string>& texts, double* output, size_t batch_size) {
#ifdef USE_ONNXRUNTIME
    try {
        // CLIP text tokenization
        std::vector<std::vector<int>> tokenized_batch;
        for (const auto& text : texts) {
            tokenized_batch.push_back(tokenize_text(text, 77));
        }

        std::cout << "Tokenized " << texts.size() << " texts (simplified tokenizer)" << std::endl;

        // Note: Full implementation would run ONNX inference here
        std::cerr << "Warning: CLIP model inference not yet connected, using placeholder embeddings" << std::endl;

        // Generate placeholder embeddings
        std::vector<double> embeddings(batch_size * 512);
        for (size_t i = 0; i < batch_size * 512; i++) {
            embeddings[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }

        cudaMemcpy(output, embeddings.data(), batch_size * 512 * sizeof(double), cudaMemcpyHostToDevice);
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
        return false;
    }
#else
    // Placeholder implementation without ONNX Runtime
    std::vector<double> embeddings(batch_size * 512);
    for (size_t i = 0; i < batch_size * 512; i++) {
        embeddings[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    cudaMemcpy(output, embeddings.data(), batch_size * 512 * sizeof(double), cudaMemcpyHostToDevice);
    return true;
#endif
}

// Helper function: Bilinear interpolation for image resizing
static void resize_image(const uint8_t* src, int src_w, int src_h,
                        std::vector<float>& dst, int dst_w, int dst_h) {
    dst.resize(dst_w * dst_h * 3);

    float x_ratio = (float)src_w / dst_w;
    float y_ratio = (float)src_h / dst_h;

    for (int y = 0; y < dst_h; y++) {
        for (int x = 0; x < dst_w; x++) {
            float src_x = x * x_ratio;
            float src_y = y * y_ratio;

            int x1 = (int)src_x;
            int y1 = (int)src_y;
            int x2 = std::min(x1 + 1, src_w - 1);
            int y2 = std::min(y1 + 1, src_h - 1);

            float dx = src_x - x1;
            float dy = src_y - y1;

            for (int c = 0; c < 3; c++) {
                float v1 = src[(y1 * src_w + x1) * 3 + c];
                float v2 = src[(y1 * src_w + x2) * 3 + c];
                float v3 = src[(y2 * src_w + x1) * 3 + c];
                float v4 = src[(y2 * src_w + x2) * 3 + c];

                float val = v1 * (1 - dx) * (1 - dy) +
                           v2 * dx * (1 - dy) +
                           v3 * (1 - dx) * dy +
                           v4 * dx * dy;

                dst[(y * dst_w + x) * 3 + c] = val;
            }
        }
    }
}

bool SemanticEncoder::run_clip_image_inference(const uint8_t* image_data, int width, int height, double* output) {
#ifdef USE_ONNXRUNTIME
    try {
        // CLIP image preprocessing
        const int target_size = 224;

        // Step 1: Resize to 224x224
        std::vector<float> resized;
        resize_image(image_data, width, height, resized, target_size, target_size);

        // Step 2: Normalize with CLIP's mean and std
        const float mean[3] = {0.48145466f, 0.45782750f, 0.40821073f};
        const float std[3] = {0.26862954f, 0.26130258f, 0.27577711f};

        std::vector<float> normalized(target_size * target_size * 3);
        for (int i = 0; i < target_size * target_size; i++) {
            for (int c = 0; c < 3; c++) {
                float pixel = resized[i * 3 + c] / 255.0f;  // Convert to [0, 1]
                normalized[i * 3 + c] = (pixel - mean[c]) / std[c];
            }
        }

        // Step 3: Run CLIP inference (placeholder - requires ONNX model)
        std::cerr << "Warning: CLIP model inference not yet connected, using placeholder" << std::endl;

        // Generate placeholder embeddings
        std::vector<double> embeddings(512);
        for (size_t i = 0; i < 512; i++) {
            embeddings[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
        }

        cudaMemcpy(output, embeddings.data(), 512 * sizeof(double), cudaMemcpyHostToDevice);
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime inference error: " << e.what() << std::endl;
        return false;
    }
#else
    // Placeholder implementation without ONNX Runtime
    std::vector<double> embeddings(512);
    for (size_t i = 0; i < 512; i++) {
        embeddings[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    cudaMemcpy(output, embeddings.data(), 512 * sizeof(double), cudaMemcpyHostToDevice);
    return true;
#endif
}

size_t SemanticEncoder::encode_text(const std::string& text, std::vector<int>& codes, size_t max_codes) {
    std::vector<std::string> texts = {text};
    std::vector<std::vector<int>> batch_codes;

    auto counts = encode_text_batch(texts, batch_codes, max_codes);

    if (!batch_codes.empty()) {
        codes = batch_codes[0];
        return counts[0];
    }

    return 0;
}

std::vector<size_t> SemanticEncoder::encode_text_batch(const std::vector<std::string>& texts,
                                                        std::vector<std::vector<int>>& codes,
                                                        size_t max_codes) {
    size_t batch_size = texts.size();
    std::vector<size_t> code_counts(batch_size);

    // Run CLIP inference to get embeddings
    if (!run_clip_text_inference(texts, clip_output_gpu_, batch_size)) {
        return code_counts;
    }

    // Encode embeddings to VQ codes
    vqgan_encoder_->encode(clip_output_gpu_, codes_gpu_, batch_size);

    // Copy codes back to host
    std::vector<int> codes_host(batch_size);
    cudaMemcpy(codes_host.data(), codes_gpu_, batch_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Organize codes by batch
    codes.resize(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        codes[i] = {codes_host[i]};
        code_counts[i] = 1;
    }

    return code_counts;
}

// ============================================================================
// CIC-based interface (simplified)
// ============================================================================

size_t SemanticEncoder::encode(const CICData& cic_data, std::vector<int>& codes, size_t max_codes) {
    // Extract embedding from CIC data
    std::vector<double> embedding;
    if (!extract_embedding(cic_data, embedding)) {
        return 0;
    }

    // Encode embedding to VQ codes
    return encode_from_emb(embedding, codes, max_codes);
}

size_t SemanticEncoder::encode_from_emb(const std::vector<double>& embedding, std::vector<int>& codes, size_t max_codes) {
    if (embedding.size() != 512) {
        std::cerr << "Invalid embedding dimension: " << embedding.size() << " (expected 512)" << std::endl;
        return 0;
    }

    // Copy embedding to GPU
    cudaMemcpy(clip_output_gpu_, embedding.data(), 512 * sizeof(double), cudaMemcpyHostToDevice);

    // Encode to VQ codes
    vqgan_encoder_->encode(clip_output_gpu_, codes_gpu_, 1);

    // Copy codes back to host
    std::vector<int> codes_host(1);
    cudaMemcpy(codes_host.data(), codes_gpu_, sizeof(int), cudaMemcpyDeviceToHost);

    codes = {codes_host[0]};
    return 1;
}

// ============================================================================
// CIC data processing helpers
// ============================================================================

bool SemanticEncoder::extract_embedding(const CICData& cic_data, std::vector<double>& embedding) {
    // If CIC has embedding, use it directly
    if (cic_data.has_emb()) {
        embedding = cic_data.emb;

        // Verify embedding dimension
        if (embedding.size() != 512) {
            std::cerr << "Invalid embedding dimension: " << embedding.size() << " (expected 512)" << std::endl;
            return false;
        }

        return true;
    }

    // If CIC has nested data, fuse all embeddings
    if (cic_data.has_nested()) {
        std::vector<std::vector<double>> embeddings;
        embeddings.reserve(cic_data.nested.size());

        for (const auto& nested : cic_data.nested) {
            std::vector<double> nested_emb;
            if (extract_embedding(*nested, nested_emb)) {
                embeddings.push_back(nested_emb);
            }
        }

        if (embeddings.empty()) {
            std::cerr << "Failed to extract any embeddings from nested CIC data" << std::endl;
            return false;
        }

        return fuse_embeddings(embeddings, embedding);
    }

    // CIC has neither emb nor nested data
    std::cerr << "CIC data has no embedding or nested data" << std::endl;
    return false;
}

bool SemanticEncoder::fuse_embeddings(const std::vector<std::vector<double>>& embeddings,
                                       std::vector<double>& fused) {
    if (embeddings.empty()) {
        return false;
    }

    // Simple fusion strategy: average all embeddings
    fused.resize(512, 0.0);

    for (const auto& emb : embeddings) {
        if (emb.size() != 512) {
            std::cerr << "Invalid embedding dimension in fusion: " << emb.size() << std::endl;
            continue;
        }

        for (size_t i = 0; i < 512; i++) {
            fused[i] += emb[i];
        }
    }

    // Normalize by count
    double count = static_cast<double>(embeddings.size());
    for (size_t i = 0; i < 512; i++) {
        fused[i] /= count;
    }

    return true;
}

} // namespace sintellix
