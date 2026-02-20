#include "stream_manager.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_stream_manager(StreamManager* manager, int device_id, int num_streams) {
    manager->device_id = device_id;
    manager->num_streams = num_streams;
    manager->streams = new cudaStream_t[num_streams];

    cudaSetDevice(device_id);
    for (int i = 0; i < num_streams; i++) {
        CHECK_CUDA_ERROR(cudaStreamCreate(&manager->streams[i]));
    }
}

__host__ void free_stream_manager(StreamManager* manager) {
    if (manager->streams) {
        for (int i = 0; i < manager->num_streams; i++) {
            cudaStreamDestroy(manager->streams[i]);
        }
        delete[] manager->streams;
    }
}

__host__ cudaStream_t get_stream(const StreamManager* manager, int stream_idx) {
    return manager->streams[stream_idx % manager->num_streams];
}

__host__ void sync_all_streams(const StreamManager* manager) {
    for (int i = 0; i < manager->num_streams; i++) {
        CHECK_CUDA_ERROR(cudaStreamSynchronize(manager->streams[i]));
    }
}

} // namespace quila
