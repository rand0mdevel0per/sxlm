#pragma once
#include <cuda_runtime.h>

namespace quila {

// Memory tier levels
enum MemoryTier {
    TIER_L1_SRAM = 0,    // GPU SRAM (fastest)
    TIER_L2_HBM = 1,     // GPU HBM
    TIER_L3_NVRAM = 2,   // NVRAM (Optane)
    TIER_L4_SSD = 3,     // SSD
    TIER_L5_NVME = 4     // NVMe (slowest)
};

// Tiered eviction manager
struct TieredEviction {
    float* tier_buffers[5];
    size_t tier_sizes[5];
    size_t tier_used[5];
    int hidden_dim;
};

// Initialize tiered storage
__host__ void init_tiered_eviction(TieredEviction* eviction, int hidden_dim,
                                    size_t l1_size, size_t l2_size, size_t l3_size);

// Free tiered storage
__host__ void free_tiered_eviction(TieredEviction* eviction);

// Evict from tier to lower tier
__host__ void evict_to_lower_tier(TieredEviction* eviction, MemoryTier from_tier);

// Promote from lower tier to higher tier
__host__ void promote_to_higher_tier(TieredEviction* eviction, MemoryTier to_tier);

} // namespace quila
