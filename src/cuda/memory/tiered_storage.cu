#include "tiered_storage.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_tiered_eviction(TieredEviction* eviction, int hidden_dim,
                                    size_t l1_size, size_t l2_size, size_t l3_size) {
    eviction->hidden_dim = hidden_dim;

    eviction->tier_sizes[TIER_L1_SRAM] = l1_size;
    eviction->tier_sizes[TIER_L2_HBM] = l2_size;
    eviction->tier_sizes[TIER_L3_NVRAM] = l3_size;
    eviction->tier_sizes[TIER_L4_SSD] = 0;
    eviction->tier_sizes[TIER_L5_NVME] = 0;

    for (int i = 0; i < 3; i++) {
        eviction->tier_buffers[i] = (float*)allocate_unified(eviction->tier_sizes[i] * hidden_dim * sizeof(float));
        eviction->tier_used[i] = 0;
    }
}

__host__ void free_tiered_eviction(TieredEviction* eviction) {
    for (int i = 0; i < 3; i++) {
        if (eviction->tier_buffers[i]) {
            deallocate_unified(eviction->tier_buffers[i]);
        }
    }
}

__host__ void evict_to_lower_tier(TieredEviction* eviction, MemoryTier from_tier) {
    if (from_tier >= TIER_L5_NVME) return;

    MemoryTier to_tier = (MemoryTier)(from_tier + 1);
    // Simplified: just mark as evicted
    if (eviction->tier_used[from_tier] > 0) {
        eviction->tier_used[from_tier]--;
    }
}

__host__ void promote_to_higher_tier(TieredEviction* eviction, MemoryTier to_tier) {
    if (to_tier <= TIER_L1_SRAM) return;

    // Simplified: just mark as promoted
    if (eviction->tier_used[to_tier] < eviction->tier_sizes[to_tier]) {
        eviction->tier_used[to_tier]++;
    }
}

} // namespace quila
