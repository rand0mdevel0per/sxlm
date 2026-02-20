#pragma once
#include <cuda_runtime.h>

namespace quila {

// Unified memory manager for multi-GPU
struct UnifiedMemoryManager {
    void** device_ptrs;
    size_t* sizes;
    int num_devices;
    int num_allocations;
};

// Initialize unified memory manager
__host__ void init_unified_memory_manager(UnifiedMemoryManager* manager, int num_devices);

// Free unified memory manager
__host__ void free_unified_memory_manager(UnifiedMemoryManager* manager);

// Allocate unified memory across devices
__host__ void* allocate_unified_multi_gpu(UnifiedMemoryManager* manager, size_t size);

// Prefetch to specific device
__host__ void prefetch_to_gpu(void* ptr, size_t size, int device_id);

// Set memory advice for hot parameters
__host__ void set_read_mostly_advice(void* ptr, size_t size, int device_id);

} // namespace quila
