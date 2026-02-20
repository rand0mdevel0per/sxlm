#pragma once
#include <cstdint>
#include <vector>

namespace quila {

// Knowledge Fragment Entry
struct KFEEntry {
    uint64_t kfe_id;
    std::vector<float> embedding;  // R^D
    float utility;
    uint32_t access_count;
    uint64_t last_access_time;

    KFEEntry(uint64_t id, const std::vector<float>& emb, float util = 0.0f)
        : kfe_id(id), embedding(emb), utility(util),
          access_count(0), last_access_time(0) {}
};

} // namespace quila
