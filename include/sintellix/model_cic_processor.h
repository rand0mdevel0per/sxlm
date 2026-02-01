#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "cic_data.pb.h"
#include "embedding_config.pb.h"

namespace sintellix {

// Import CICData from nmdb namespace
using nmdb::CICData;

/**
 * @brief Model CIC Input Processor
 *
 * Processes CIC data as model input as documented in Section 6.5
 * of the technical specification.
 */
class ModelCICInputProcessor {
public:
    explicit ModelCICInputProcessor(const EmbeddingConfig& config);
    ~ModelCICInputProcessor() = default;

    /**
     * @brief Process CIC data as model input
     * @param cic_input CIC data to process
     * @param model_input_buffer Output buffer for model input
     * @param buffer_size Size of output buffer
     * @return true if successful, false otherwise
     */
    bool process_cic_input(
        const CICData& cic_input,
        double* model_input_buffer,
        size_t buffer_size
    );

    /**
     * @brief Process a single channel
     * @param channel Channel to process
     * @param buffer Output buffer
     * @param buffer_size Size of output buffer
     * @return Number of elements written to buffer
     */
    size_t process_channel(
        const CICData::Channel& channel,
        double* buffer,
        size_t buffer_size
    );

private:
    /**
     * @brief Process text channel
     */
    void process_text_channel(
        const CICData::Channel& channel,
        double* buffer,
        size_t& offset
    );

    /**
     * @brief Process image channel
     */
    void process_image_channel(
        const CICData::Channel& channel,
        double* buffer,
        size_t& offset
    );

    /**
     * @brief Process audio channel
     */
    void process_audio_channel(
        const CICData::Channel& channel,
        double* buffer,
        size_t& offset
    );

    /**
     * @brief Process composite channel
     */
    void process_composite_channel(
        const CICData::Channel& channel,
        double* buffer,
        size_t& offset
    );

    EmbeddingConfig config_;
};

/**
 * @brief Model CIC Output Generator
 *
 * Generates CIC data from model output as documented in Section 6.5
 * of the technical specification.
 */
class ModelCICOutputGenerator {
public:
    ModelCICOutputGenerator() = default;
    ~ModelCICOutputGenerator() = default;

    /**
     * @brief Generate CIC data from model output
     * @param model_output_buffer Model output buffer
     * @param output_dim Output dimension
     * @param output_type Output type (e.g., "text", "image", "audio")
     * @return Generated CIC data
     */
    CICData generate_cic_output(
        const double* model_output_buffer,
        size_t output_dim,
        const std::string& output_type
    );

    /**
     * @brief Generate multi-modal CIC output
     * @param outputs Map of output type to (buffer, dimension) pairs
     * @return Generated CIC data with multiple channels
     */
    CICData generate_multimodal_output(
        const std::map<std::string, std::pair<const double*, size_t>>& outputs
    );

private:
    /**
     * @brief Create text channel from buffer
     */
    CICData::Channel create_text_channel(
        const double* buffer,
        size_t dim,
        const std::string& name
    );

    /**
     * @brief Create image channel from buffer
     */
    CICData::Channel create_image_channel(
        const double* buffer,
        size_t dim,
        const std::string& name
    );

    /**
     * @brief Create audio channel from buffer
     */
    CICData::Channel create_audio_channel(
        const double* buffer,
        size_t dim,
        const std::string& name
    );
};

} // namespace sintellix
