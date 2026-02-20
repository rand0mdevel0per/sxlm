#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace quila {

// Engram - hash-based long-term memory storage
struct EngramEntry {
    uint64_t hash;
    std::vector<float> data;
    uint64_t access_count;
    uint64_t last_access;

    EngramEntry(uint64_t h, const std::vector<float>& d)
        : hash(h), data(d), access_count(0), last_access(0) {}
};

class Engram {
private:
    std::unordered_map<uint64_t, EngramEntry> storage;

public:
    // Store data with hash
    void store(uint64_t hash, const std::vector<float>& data);

    // Retrieve data by hash
    const std::vector<float>* retrieve(uint64_t hash);

    // Compute hash
    static uint64_t compute_hash(const float* data, int size);

    // Get size
    size_t size() const { return storage.size(); }
};

} // namespace quila
