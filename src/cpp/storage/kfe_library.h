#pragma once
#include "kfe_entry.h"
#include <unordered_map>
#include <vector>

namespace quila {

// KFE Library - manages knowledge fragments
class KFELibrary {
private:
    std::unordered_map<uint64_t, KFEEntry> entries;
    uint64_t next_id;
    int hidden_dim;

public:
    KFELibrary(int dim) : next_id(0), hidden_dim(dim) {}

    // Create new KFE
    uint64_t create(const std::vector<float>& embedding, float utility = 0.0f);

    // Recall KFE by ID
    KFEEntry* recall(uint64_t kfe_id);

    // Update utility score
    void update_utility(uint64_t kfe_id, float utility);

    // Evict low-utility KFEs
    void evict(int count);

    // Get total count
    size_t size() const { return entries.size(); }
};

} // namespace quila
