#pragma once

#include <string>
#include <memory>
#include "embedding_config.pb.h"

namespace sintellix {

/**
 * @brief Embedding Configuration Loader
 *
 * Loads and manages configurable embedding models as documented in Section 6.2
 * of the technical specification.
 */
class EmbeddingConfigLoader {
public:
    EmbeddingConfigLoader() = default;
    ~EmbeddingConfigLoader() = default;

    /**
     * @brief Load embedding configuration from file
     * @param config_path Path to configuration file
     * @return Loaded configuration
     */
    static EmbeddingConfig load_from_file(const std::string& config_path);

    /**
     * @brief Save embedding configuration to file
     * @param config Configuration to save
     * @param config_path Path to save configuration
     * @return true if successful, false otherwise
     */
    static bool save_to_file(
        const EmbeddingConfig& config,
        const std::string& config_path
    );

    /**
     * @brief Create default configuration
     * @return Default embedding configuration
     */
    static EmbeddingConfig create_default_config();

    /**
     * @brief Validate configuration
     * @param config Configuration to validate
     * @return true if valid, false otherwise
     */
    static bool validate_config(const EmbeddingConfig& config);
};

} // namespace sintellix
