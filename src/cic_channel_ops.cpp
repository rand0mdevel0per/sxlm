#include "sintellix/cic_channel_ops.h"
#include <stdexcept>
#include <sstream>

namespace sintellix {

// ============================================================================
// CICChannelExtractor Implementation
// ============================================================================

std::unique_ptr<CICData::Channel> CICChannelExtractor::extract_channel(
    const CICData& cic_data,
    const std::string& channel_name
) {
    for (const auto& channel : cic_data.channels()) {
        if (channel.name() == channel_name) {
            auto extracted = std::make_unique<CICData::Channel>();
            extracted->CopyFrom(channel);
            return extracted;
        }
    }
    return nullptr;
}

std::vector<CICData::Channel> CICChannelExtractor::extract_channels(
    const CICData& cic_data,
    const std::vector<std::string>& channel_names
) {
    std::vector<CICData::Channel> extracted_channels;
    extracted_channels.reserve(channel_names.size());

    for (const auto& name : channel_names) {
        auto channel = extract_channel(cic_data, name);
        if (channel) {
            extracted_channels.push_back(*channel);
        }
    }

    return extracted_channels;
}

std::vector<CICData::Channel> CICChannelExtractor::extract_by_type(
    const CICData& cic_data,
    CICData::ChannelType channel_type
) {
    std::vector<CICData::Channel> extracted_channels;

    for (const auto& channel : cic_data.channels()) {
        if (channel.type() == channel_type) {
            extracted_channels.push_back(channel);
        }
    }

    return extracted_channels;
}

// ============================================================================
// CICChannelRepackager Implementation
// ============================================================================

CICData::Channel CICChannelRepackager::repackage_channels(
    const std::vector<CICData::Channel>& channels,
    const std::string& composite_name
) {
    CICData::Channel composite;
    composite.set_name(composite_name);
    composite.set_type(CICData::CHANNEL_TYPE_COMPOSITE);

    // Serialize all channels into composite data
    std::string serialized = serialize_channels(channels);
    composite.set_data(serialized);

    // Set dimension as total of all channel dimensions
    uint32_t total_dim = 0;
    for (const auto& channel : channels) {
        total_dim += channel.dimension();
    }
    composite.set_dimension(total_dim);

    // Add metadata about contained channels
    for (size_t i = 0; i < channels.size(); ++i) {
        std::string key = "channel_" + std::to_string(i) + "_name";
        composite.mutable_metadata()->insert({key, channels[i].name()});
    }
    composite.mutable_metadata()->insert(
        {"num_channels", std::to_string(channels.size())}
    );

    return composite;
}

CICData CICChannelRepackager::create_composite_cic(
    const CICData& original_cic,
    const std::vector<std::string>& channel_names,
    const std::string& composite_name
) {
    CICChannelExtractor extractor;
    auto channels = extractor.extract_channels(original_cic, channel_names);

    CICData new_cic;
    new_cic.set_cic_id(original_cic.cic_id() + "_composite");
    new_cic.set_timestamp(original_cic.timestamp());

    // Add the composite channel
    auto* composite_channel = new_cic.add_channels();
    *composite_channel = repackage_channels(channels, composite_name);

    return new_cic;
}

std::vector<CICData::Channel> CICChannelRepackager::unpack_composite(
    const CICData::Channel& composite_channel
) {
    if (composite_channel.type() != CICData::CHANNEL_TYPE_COMPOSITE) {
        throw std::invalid_argument("Channel is not a composite channel");
    }

    return deserialize_channels(composite_channel.data());
}

std::string CICChannelRepackager::serialize_channels(
    const std::vector<CICData::Channel>& channels
) {
    std::ostringstream oss;

    // Write number of channels
    uint32_t num_channels = static_cast<uint32_t>(channels.size());
    oss.write(reinterpret_cast<const char*>(&num_channels), sizeof(num_channels));

    // Serialize each channel
    for (const auto& channel : channels) {
        std::string serialized;
        if (!channel.SerializeToString(&serialized)) {
            throw std::runtime_error("Failed to serialize channel: " + channel.name());
        }

        // Write size then data
        uint32_t size = static_cast<uint32_t>(serialized.size());
        oss.write(reinterpret_cast<const char*>(&size), sizeof(size));
        oss.write(serialized.data(), serialized.size());
    }

    return oss.str();
}

std::vector<CICData::Channel> CICChannelRepackager::deserialize_channels(
    const std::string& data
) {
    std::vector<CICData::Channel> channels;
    std::istringstream iss(data);

    // Read number of channels
    uint32_t num_channels;
    iss.read(reinterpret_cast<char*>(&num_channels), sizeof(num_channels));

    channels.reserve(num_channels);

    // Deserialize each channel
    for (uint32_t i = 0; i < num_channels; ++i) {
        // Read size
        uint32_t size;
        iss.read(reinterpret_cast<char*>(&size), sizeof(size));

        // Read data
        std::string serialized(size, '\0');
        iss.read(&serialized[0], size);

        // Deserialize channel
        CICData::Channel channel;
        if (!channel.ParseFromString(serialized)) {
            throw std::runtime_error("Failed to deserialize channel " + std::to_string(i));
        }

        channels.push_back(std::move(channel));
    }

    return channels;
}

} // namespace sintellix
