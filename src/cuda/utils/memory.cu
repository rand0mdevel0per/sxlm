#include "error.cuh"
#include <cuda_runtime.h>

namespace quila {

void* allocate_unified(size_t size) {
    void* ptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    return ptr;
}

void deallocate_unified(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

void prefetch_to_device(void* ptr, size_t size, int device) {
    // Disabled due to API changes in CUDA 13
    (void)ptr; (void)size; (void)device;
}

void advise_read_mostly(void* ptr, size_t size, int device) {
    // Simplified: skip advise for now due to API changes in CUDA 13
    (void)ptr; (void)size; (void)device;
}

} // namespace quila
