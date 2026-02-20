#include "nmdb.h"
#include <cstring>

namespace quila {

void NMDB::store(uint64_t neuron_id, const void* state, size_t size) {
    auto compressed = compress(state, size);
    entries.emplace_back(neuron_id, 0, compressed, size);
}

bool NMDB::load(uint64_t neuron_id, void* state, size_t size) {
    for (const auto& entry : entries) {
        if (entry.neuron_id == neuron_id) {
            decompress(entry.compressed_state, state, size);
            return true;
        }
    }
    return false;
}

std::vector<uint8_t> NMDB::compress(const void* data, size_t size) {
    // Simplified: just copy (real implementation would use zstd)
    std::vector<uint8_t> result(size);
    std::memcpy(result.data(), data, size);
    return result;
}

void NMDB::decompress(const std::vector<uint8_t>& compressed, void* data, size_t size) {
    // Simplified: just copy
    std::memcpy(data, compressed.data(), std::min(size, compressed.size()));
}

} // namespace quila
