#pragma once
#include <cuda_runtime.h>

namespace quila {

// Set memory advise hints for hot parameters (Req 22.1.2)
cudaError_t advise_hot_parameters(
    void* ptr,
    size_t size,
    int device_id
);

// Mark memory as read-mostly for multi-GPU access
cudaError_t advise_read_mostly(void* ptr, size_t size);

// Set preferred location for frequently accessed data
cudaError_t advise_preferred_location(void* ptr, size_t size, int device_id);

} // namespace quila
