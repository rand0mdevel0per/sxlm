#pragma once
#include <cuda_runtime.h>

namespace quila {

void* allocate_unified(size_t size);
void deallocate_unified(void* ptr);
void prefetch_to_device(void* ptr, size_t size, int device);
void advise_read_mostly(void* ptr, size_t size, int device);

} // namespace quila
