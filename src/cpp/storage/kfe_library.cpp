#include "kfe_library.h"
#include <algorithm>

namespace quila {

uint64_t KFELibrary::create(const std::vector<float>& embedding, float utility) {
    uint64_t id = next_id++;
    entries.emplace(id, KFEEntry(id, embedding, utility));
    return id;
}

KFEEntry* KFELibrary::recall(uint64_t kfe_id) {
    auto it = entries.find(kfe_id);
    if (it != entries.end()) {
        it->second.access_count++;
        return &it->second;
    }
    return nullptr;
}

void KFELibrary::update_utility(uint64_t kfe_id, float utility) {
    auto it = entries.find(kfe_id);
    if (it != entries.end()) {
        it->second.utility = utility;
    }
}

void KFELibrary::evict(int count) {
    std::vector<std::pair<uint64_t, float>> utility_list;
    for (const auto& pair : entries) {
        utility_list.push_back({pair.first, pair.second.utility});
    }

    std::sort(utility_list.begin(), utility_list.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    for (int i = 0; i < count && i < (int)utility_list.size(); i++) {
        entries.erase(utility_list[i].first);
    }
}

} // namespace quila
