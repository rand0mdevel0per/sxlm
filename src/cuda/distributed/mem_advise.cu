#include "mem_advise.cuh"

namespace quila {

cudaError_t advise_hot_parameters(void* ptr, size_t size, int device_id) {
    // Set preferred location for hot parameters (Req 22.1.2)
    cudaError_t err = cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device_id);
    if (err != cudaSuccess) return err;

    // Mark as accessed by this device
    return cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy, device_id);
}

cudaError_t advise_read_mostly(void* ptr, size_t size) {
    return cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, cudaCpuDeviceId);
}

cudaError_t advise_preferred_location(void* ptr, size_t size, int device_id) {
    return cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, device_id);
}

} // namespace quila
