#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include "cic_data.pb.h"

namespace sintellix {

// Import CICData from nmdb namespace
using nmdb::CICData;

/**
 * @brief CIC Channel Extractor
 *
 * Extracts specific channels from CIC data as documented in Section 6.4
 * of the technical specification.
 */
class CICChannelExtractor {
public:
    CICChannelExtractor() = default;
    ~CICChannelExtractor() = default;

    /**
     * @brief Extract a single channel by name
     * @param cic_data Source CIC data
     * @param channel_name Name of channel to extract
     * @return Extracted channel, or nullptr if not found
     */
    std::unique_ptr<CICData::Channel> extract_channel(
        const CICData& cic_data,
        const std::string& channel_name
    );

    /**
     * @brief Extract multiple channels by names
     * @param cic_data Source CIC data
     * @param channel_names Names of channels to extract
     * @return Vector of extracted channels
     */
    std::vector<CICData::Channel> extract_channels(
        const CICData& cic_data,
        const std::vector<std::string>& channel_names
    );

    /**
     * @brief Extract all channels of a specific type
     * @param cic_data Source CIC data
     * @param channel_type Type of channels to extract
     * @return Vector of extracted channels
     */
    std::vector<CICData::Channel> extract_by_type(
        const CICData& cic_data,
        CICData::ChannelType channel_type
    );
};

/**
 * @brief CIC Channel Repackager
 *
 * Repackages multiple channels into composite channels as documented
 * in Section 6.4 of the technical specification.
 */
class CICChannelRepackager {
public:
    CICChannelRepackager() = default;
    ~CICChannelRepackager() = default;

    /**
     * @brief Repackage multiple channels into a composite channel
     * @param channels Channels to repackage
     * @param composite_name Name for the composite channel
     * @return Composite channel containing all input channels
     */
    CICData::Channel repackage_channels(
        const std::vector<CICData::Channel>& channels,
        const std::string& composite_name
    );

    /**
     * @brief Create a new CIC with composite channel from original CIC
     * @param original_cic Original CIC data
     * @param channel_names Names of channels to extract and repackage
     * @param composite_name Name for the composite channel
     * @return New CIC data with composite channel
     */
    CICData create_composite_cic(
        const CICData& original_cic,
        const std::vector<std::string>& channel_names,
        const std::string& composite_name
    );

    /**
     * @brief Unpack a composite channel into individual channels
     * @param composite_channel Composite channel to unpack
     * @return Vector of individual channels
     */
    std::vector<CICData::Channel> unpack_composite(
        const CICData::Channel& composite_channel
    );

private:
    /**
     * @brief Serialize channels into composite data
     * @param channels Channels to serialize
     * @return Serialized data
     */
    std::string serialize_channels(
        const std::vector<CICData::Channel>& channels
    );

    /**
     * @brief Deserialize composite data into channels
     * @param data Serialized data
     * @return Vector of channels
     */
    std::vector<CICData::Channel> deserialize_channels(
        const std::string& data
    );
};

} // namespace sintellix
