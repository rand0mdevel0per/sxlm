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
    CUDA_CHECK(cudaMemPrefetchAsync(ptr, size, device));
}

void advise_read_mostly(void* ptr, size_t size, int device) {
    CUDA_CHECK(cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, device));
}

} // namespace quila
