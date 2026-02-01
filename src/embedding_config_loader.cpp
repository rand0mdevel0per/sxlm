#include "sintellix/embedding_config_loader.h"
#include <fstream>
#include <stdexcept>

namespace sintellix {

EmbeddingConfig EmbeddingConfigLoader::load_from_file(const std::string& config_path) {
    std::ifstream ifs(config_path, std::ios::binary);
    if (!ifs) {
        throw std::runtime_error("Failed to open config file: " + config_path);
    }

    std::string serialized(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>()
    );

    EmbeddingConfig config;
    if (!config.ParseFromString(serialized)) {
        throw std::runtime_error("Failed to parse config file: " + config_path);
    }

    if (!validate_config(config)) {
        throw std::runtime_error("Invalid config file: " + config_path);
    }

    return config;
}

bool EmbeddingConfigLoader::save_to_file(
    const EmbeddingConfig& config,
    const std::string& config_path
) {
    if (!validate_config(config)) {
        return false;
    }

    std::string serialized;
    if (!config.SerializeToString(&serialized)) {
        return false;
    }

    std::ofstream ofs(config_path, std::ios::binary);
    if (!ofs) {
        return false;
    }

    ofs.write(serialized.data(), serialized.size());
    return ofs.good();
}

EmbeddingConfig EmbeddingConfigLoader::create_default_config() {
    EmbeddingConfig config;

    // Default text embedding: E5-Large
    auto* text_emb = config.mutable_text_embedding();
    text_emb->set_model_type(EmbeddingConfig::TextEmbedding::E5_LARGE);
    text_emb->set_embedding_dim(1024);
    text_emb->set_max_sequence_length(512);

    // Default image embedding: CLIP ViT-L/14
    auto* image_emb = config.mutable_image_embedding();
    image_emb->set_model_type(EmbeddingConfig::ImageEmbedding::CLIP_VIT_L_14);
    image_emb->set_embedding_dim(768);
    image_emb->set_image_size(224);

    // Default audio embedding: Wav2Vec2-Large
    auto* audio_emb = config.mutable_audio_embedding();
    audio_emb->set_model_type(EmbeddingConfig::AudioEmbedding::WAV2VEC2_LARGE);
    audio_emb->set_embedding_dim(1024);
    audio_emb->set_sample_rate(16000);

    return config;
}

bool EmbeddingConfigLoader::validate_config(const EmbeddingConfig& config) {
    // Validate text embedding
    if (config.has_text_embedding()) {
        const auto& text_emb = config.text_embedding();
        if (text_emb.embedding_dim() <= 0) {
            return false;
        }
        if (text_emb.model_type() == EmbeddingConfig::TextEmbedding::CUSTOM) {
            if (text_emb.model_path().empty()) {
                return false;
            }
        }
    }

    // Validate image embedding
    if (config.has_image_embedding()) {
        const auto& image_emb = config.image_embedding();
        if (image_emb.embedding_dim() <= 0) {
            return false;
        }
        if (image_emb.image_size() <= 0) {
            return false;
        }
        if (image_emb.model_type() == EmbeddingConfig::ImageEmbedding::CUSTOM) {
            if (image_emb.model_path().empty()) {
                return false;
            }
        }
    }

    // Validate audio embedding
    if (config.has_audio_embedding()) {
        const auto& audio_emb = config.audio_embedding();
        if (audio_emb.embedding_dim() <= 0) {
            return false;
        }
        if (audio_emb.sample_rate() <= 0) {
            return false;
        }
        if (audio_emb.model_type() == EmbeddingConfig::AudioEmbedding::CUSTOM) {
            if (audio_emb.model_path().empty()) {
                return false;
            }
        }
    }

    return true;
}

} // namespace sintellix
