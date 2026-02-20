#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace quila {

// NMDB Entry - stores neuron state snapshots
struct NMDBEntry {
    uint64_t neuron_id;
    uint64_t timestamp;
    std::vector<uint8_t> compressed_state;  // CIC format
    size_t original_size;

    NMDBEntry(uint64_t id, uint64_t ts, const std::vector<uint8_t>& data, size_t orig_size)
        : neuron_id(id), timestamp(ts), compressed_state(data), original_size(orig_size) {}
};

// Neuron Model Data Bus
class NMDB {
private:
    std::vector<NMDBEntry> entries;
    std::string storage_path;

public:
    NMDB(const std::string& path) : storage_path(path) {}

    // Store neuron state
    void store(uint64_t neuron_id, const void* state, size_t size);

    // Load neuron state
    bool load(uint64_t neuron_id, void* state, size_t size);

    // Compress state (CIC format)
    std::vector<uint8_t> compress(const void* data, size_t size);

    // Decompress state
    void decompress(const std::vector<uint8_t>& compressed, void* data, size_t size);
};

} // namespace quila
