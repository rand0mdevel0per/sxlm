#include "unified_memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_unified_memory_manager(UnifiedMemoryManager* manager, int num_devices) {
    manager->num_devices = num_devices;
    manager->num_allocations = 0;
    manager->device_ptrs = nullptr;
    manager->sizes = nullptr;
}

__host__ void free_unified_memory_manager(UnifiedMemoryManager* manager) {
    if (manager->device_ptrs) free(manager->device_ptrs);
    if (manager->sizes) free(manager->sizes);
}

__host__ void* allocate_unified_multi_gpu(UnifiedMemoryManager* manager, size_t size) {
    void* ptr;
    CHECK_CUDA_ERROR(cudaMallocManaged(&ptr, size));
    return ptr;
}

__host__ void prefetch_to_gpu(void* ptr, size_t size, int device_id) {
    // Disabled due to API changes
    (void)ptr; (void)size; (void)device_id;
}

__host__ void set_read_mostly_advice(void* ptr, size_t size, int device_id) {
    // Disabled due to API changes
    (void)ptr; (void)size; (void)device_id;
}

} // namespace quila
