#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <atomic>

namespace sintellix {

/**
 * Chunk of data for streaming processing
 */
struct DataChunk {
    std::vector<double> data;
    size_t chunk_id;
    size_t total_size;
    bool is_last;

    DataChunk() : chunk_id(0), total_size(0), is_last(false) {}

    DataChunk(const double* src, size_t size, size_t id, bool last = false)
        : data(src, src + size)
        , chunk_id(id)
        , total_size(size)
        , is_last(last) {}
};

/**
 * Thread-safe buffer for streaming input/output
 */
class StreamingBuffer {
public:
    StreamingBuffer(size_t max_chunks = 16);
    ~StreamingBuffer();

    // Push a chunk to the buffer (blocks if buffer is full)
    bool push(const DataChunk& chunk);

    // Try to push without blocking
    bool try_push(const DataChunk& chunk);

    // Pop a chunk from the buffer (blocks if buffer is empty)
    bool pop(DataChunk& chunk);

    // Try to pop without blocking
    bool try_pop(DataChunk& chunk);

    // Check if buffer is empty
    bool empty() const;

    // Check if buffer is full
    bool full() const;

    // Get current buffer size
    size_t size() const;

    // Clear all chunks
    void clear();

    // Signal end of stream
    void signal_end();

    // Check if stream has ended
    bool is_ended() const;

private:
    std::queue<DataChunk> buffer_;
    size_t max_chunks_;
    mutable std::mutex mutex_;
    std::condition_variable cv_not_full_;
    std::condition_variable cv_not_empty_;
    std::atomic<bool> ended_;
};

/**
 * GPU buffer for async processing
 */
class GPUStreamingBuffer {
public:
    GPUStreamingBuffer(size_t chunk_size, size_t num_buffers = 4);
    ~GPUStreamingBuffer();

    // Allocate GPU memory for chunks
    bool allocate();

    // Get next available buffer
    double* get_next_buffer(cudaStream_t stream);

    // Release buffer after processing
    void release_buffer(double* buffer);

    // Copy chunk to GPU asynchronously
    bool copy_to_gpu(const DataChunk& chunk, double* gpu_buffer, cudaStream_t stream);

    // Copy result from GPU asynchronously
    bool copy_from_gpu(double* gpu_buffer, DataChunk& chunk, cudaStream_t stream);

private:
    size_t chunk_size_;
    size_t num_buffers_;
    std::vector<double*> gpu_buffers_;
    std::vector<bool> buffer_available_;
    std::mutex mutex_;
};

} // namespace sintellix
