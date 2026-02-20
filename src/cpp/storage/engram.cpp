#include "engram.h"
#include <functional>

namespace quila {

void Engram::store(uint64_t hash, const std::vector<float>& data) {
    storage.emplace(hash, EngramEntry(hash, data));
}

const std::vector<float>* Engram::retrieve(uint64_t hash) {
    auto it = storage.find(hash);
    if (it != storage.end()) {
        it->second.access_count++;
        return &it->second.data;
    }
    return nullptr;
}

uint64_t Engram::compute_hash(const float* data, int size) {
    std::hash<float> hasher;
    uint64_t hash = 0;
    for (int i = 0; i < size; i++) {
        hash ^= hasher(data[i]) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }
    return hash;
}

} // namespace quila
