#pragma once
#include <cuda_runtime.h>

namespace quila {

// Stream manager for multi-stream execution
struct StreamManager {
    cudaStream_t* streams;
    int num_streams;
    int device_id;
};

// Initialize stream manager (8 streams per GPU)
__host__ void init_stream_manager(StreamManager* manager, int device_id, int num_streams = 8);

// Free stream manager
__host__ void free_stream_manager(StreamManager* manager);

// Get stream by index
__host__ cudaStream_t get_stream(const StreamManager* manager, int stream_idx);

// Synchronize all streams
__host__ void sync_all_streams(const StreamManager* manager);

} // namespace quila
