#include "sintellix/model_cic_processor.h"
#include "sintellix/cic_channel_ops.h"
#include <stdexcept>
#include <cstring>

namespace sintellix {

// ============================================================================
// ModelCICInputProcessor Implementation
// ============================================================================

ModelCICInputProcessor::ModelCICInputProcessor(const EmbeddingConfig& config)
    : config_(config) {
}

bool ModelCICInputProcessor::process_cic_input(
    const CICData& cic_input,
    double* model_input_buffer,
    size_t buffer_size
) {
    if (!model_input_buffer || buffer_size == 0) {
        return false;
    }

    size_t offset = 0;

    // Process each channel in the CIC
    for (const auto& channel : cic_input.channels()) {
        if (offset >= buffer_size) {
            return false; // Buffer overflow
        }

        try {
            switch (channel.type()) {
                case CICData::CHANNEL_TYPE_TEXT:
                    process_text_channel(channel, model_input_buffer, offset);
                    break;
                case CICData::CHANNEL_TYPE_IMAGE:
                    process_image_channel(channel, model_input_buffer, offset);
                    break;
                case CICData::CHANNEL_TYPE_AUDIO:
                    process_audio_channel(channel, model_input_buffer, offset);
                    break;
                case CICData::CHANNEL_TYPE_COMPOSITE:
                    process_composite_channel(channel, model_input_buffer, offset);
                    break;
                default:
                    // Skip unknown channel types
                    break;
            }
        } catch (const std::exception& e) {
            return false;
        }
    }

    return true;
}

size_t ModelCICInputProcessor::process_channel(
    const CICData::Channel& channel,
    double* buffer,
    size_t buffer_size
) {
    size_t offset = 0;

    switch (channel.type()) {
        case CICData::CHANNEL_TYPE_TEXT:
            process_text_channel(channel, buffer, offset);
            break;
        case CICData::CHANNEL_TYPE_IMAGE:
            process_image_channel(channel, buffer, offset);
            break;
        case CICData::CHANNEL_TYPE_AUDIO:
            process_audio_channel(channel, buffer, offset);
            break;
        case CICData::CHANNEL_TYPE_COMPOSITE:
            process_composite_channel(channel, buffer, offset);
            break;
        default:
            break;
    }

    return offset;
}

void ModelCICInputProcessor::process_text_channel(
    const CICData::Channel& channel,
    double* buffer,
    size_t& offset
) {
    // Extract embedding data from channel
    const std::string& data = channel.data();
    size_t num_elements = data.size() / sizeof(double);

    // Copy embedding data to buffer
    const double* embedding_data = reinterpret_cast<const double*>(data.data());
    std::memcpy(buffer + offset, embedding_data, num_elements * sizeof(double));

    offset += num_elements;
}

void ModelCICInputProcessor::process_image_channel(
    const CICData::Channel& channel,
    double* buffer,
    size_t& offset
) {
    // Extract embedding data from channel
    const std::string& data = channel.data();
    size_t num_elements = data.size() / sizeof(double);

    // Copy embedding data to buffer
    const double* embedding_data = reinterpret_cast<const double*>(data.data());
    std::memcpy(buffer + offset, embedding_data, num_elements * sizeof(double));

    offset += num_elements;
}

void ModelCICInputProcessor::process_audio_channel(
    const CICData::Channel& channel,
    double* buffer,
    size_t& offset
) {
    // Extract embedding data from channel
    const std::string& data = channel.data();
    size_t num_elements = data.size() / sizeof(double);

    // Copy embedding data to buffer
    const double* embedding_data = reinterpret_cast<const double*>(data.data());
    std::memcpy(buffer + offset, embedding_data, num_elements * sizeof(double));

    offset += num_elements;
}

void ModelCICInputProcessor::process_composite_channel(
    const CICData::Channel& channel,
    double* buffer,
    size_t& offset
) {
    // Unpack composite channel
    CICChannelRepackager repackager;
    auto channels = repackager.unpack_composite(channel);

    // Process each sub-channel
    for (const auto& sub_channel : channels) {
        switch (sub_channel.type()) {
            case CICData::CHANNEL_TYPE_TEXT:
                process_text_channel(sub_channel, buffer, offset);
                break;
            case CICData::CHANNEL_TYPE_IMAGE:
                process_image_channel(sub_channel, buffer, offset);
                break;
            case CICData::CHANNEL_TYPE_AUDIO:
                process_audio_channel(sub_channel, buffer, offset);
                break;
            default:
                break;
        }
    }
}

// ============================================================================
// ModelCICOutputGenerator Implementation
// ============================================================================

CICData ModelCICOutputGenerator::generate_cic_output(
    const double* model_output_buffer,
    size_t output_dim,
    const std::string& output_type
) {
    CICData cic_output;
    cic_output.set_cic_id("model_output_" + std::to_string(std::time(nullptr)));
    cic_output.set_timestamp(std::time(nullptr));

    // Create channel based on output type
    CICData::Channel* channel = cic_output.add_channels();

    if (output_type == "text") {
        *channel = create_text_channel(model_output_buffer, output_dim, "text_output");
    } else if (output_type == "image") {
        *channel = create_image_channel(model_output_buffer, output_dim, "image_output");
    } else if (output_type == "audio") {
        *channel = create_audio_channel(model_output_buffer, output_dim, "audio_output");
    } else {
        // Default to text channel
        *channel = create_text_channel(model_output_buffer, output_dim, output_type);
    }

    return cic_output;
}

CICData ModelCICOutputGenerator::generate_multimodal_output(
    const std::map<std::string, std::pair<const double*, size_t>>& outputs
) {
    CICData cic_output;
    cic_output.set_cic_id("multimodal_output_" + std::to_string(std::time(nullptr)));
    cic_output.set_timestamp(std::time(nullptr));

    // Create a channel for each output
    for (const auto& [output_type, buffer_info] : outputs) {
        const double* buffer = buffer_info.first;
        size_t dim = buffer_info.second;

        CICData::Channel* channel = cic_output.add_channels();

        if (output_type == "text") {
            *channel = create_text_channel(buffer, dim, "text_output");
        } else if (output_type == "image") {
            *channel = create_image_channel(buffer, dim, "image_output");
        } else if (output_type == "audio") {
            *channel = create_audio_channel(buffer, dim, "audio_output");
        } else {
            *channel = create_text_channel(buffer, dim, output_type);
        }
    }

    return cic_output;
}

CICData::Channel ModelCICOutputGenerator::create_text_channel(
    const double* buffer,
    size_t dim,
    const std::string& name
) {
    CICData::Channel channel;
    channel.set_name(name);
    channel.set_type(CICData::CHANNEL_TYPE_TEXT);
    channel.set_dimension(static_cast<uint32_t>(dim));

    // Serialize embedding data
    std::string data(reinterpret_cast<const char*>(buffer), dim * sizeof(double));
    channel.set_data(data);

    return channel;
}

CICData::Channel ModelCICOutputGenerator::create_image_channel(
    const double* buffer,
    size_t dim,
    const std::string& name
) {
    CICData::Channel channel;
    channel.set_name(name);
    channel.set_type(CICData::CHANNEL_TYPE_IMAGE);
    channel.set_dimension(static_cast<uint32_t>(dim));

    // Serialize embedding data
    std::string data(reinterpret_cast<const char*>(buffer), dim * sizeof(double));
    channel.set_data(data);

    return channel;
}

CICData::Channel ModelCICOutputGenerator::create_audio_channel(
    const double* buffer,
    size_t dim,
    const std::string& name
) {
    CICData::Channel channel;
    channel.set_name(name);
    channel.set_type(CICData::CHANNEL_TYPE_AUDIO);
    channel.set_dimension(static_cast<uint32_t>(dim));

    // Serialize embedding data
    std::string data(reinterpret_cast<const char*>(buffer), dim * sizeof(double));
    channel.set_data(data);

    return channel;
}

} // namespace sintellix
