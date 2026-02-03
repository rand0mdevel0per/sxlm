#include "sintellix/core/streaming_buffer.cuh"
#include <stdexcept>

namespace sintellix {

// ============================================================================
// StreamingBuffer Implementation
// ============================================================================

StreamingBuffer::StreamingBuffer(size_t max_chunks)
    : max_chunks_(max_chunks)
    , ended_(false) {
}

StreamingBuffer::~StreamingBuffer() {
    clear();
}

bool StreamingBuffer::push(const DataChunk& chunk) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait until buffer is not full
    cv_not_full_.wait(lock, [this] {
        return buffer_.size() < max_chunks_ || ended_.load();
    });

    if (ended_.load()) {
        return false;
    }

    buffer_.push(chunk);
    cv_not_empty_.notify_one();
    return true;
}

bool StreamingBuffer::try_push(const DataChunk& chunk) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (buffer_.size() >= max_chunks_ || ended_.load()) {
        return false;
    }

    buffer_.push(chunk);
    cv_not_empty_.notify_one();
    return true;
}

bool StreamingBuffer::pop(DataChunk& chunk) {
    std::unique_lock<std::mutex> lock(mutex_);

    // Wait until buffer is not empty or stream ended
    cv_not_empty_.wait(lock, [this] {
        return !buffer_.empty() || ended_.load();
    });

    if (buffer_.empty()) {
        return false;
    }

    chunk = buffer_.front();
    buffer_.pop();
    cv_not_full_.notify_one();
    return true;
}

bool StreamingBuffer::try_pop(DataChunk& chunk) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (buffer_.empty()) {
        return false;
    }

    chunk = buffer_.front();
    buffer_.pop();
    cv_not_full_.notify_one();
    return true;
}

bool StreamingBuffer::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
}

bool StreamingBuffer::full() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size() >= max_chunks_;
}

size_t StreamingBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

void StreamingBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    while (!buffer_.empty()) {
        buffer_.pop();
    }
}

void StreamingBuffer::signal_end() {
    ended_.store(true);
    cv_not_empty_.notify_all();
    cv_not_full_.notify_all();
}

bool StreamingBuffer::is_ended() const {
    return ended_.load();
}

// ============================================================================
// GPUStreamingBuffer Implementation
// ============================================================================

GPUStreamingBuffer::GPUStreamingBuffer(size_t chunk_size, size_t num_buffers)
    : chunk_size_(chunk_size)
    , num_buffers_(num_buffers)
    , buffer_available_(num_buffers, true) {
}

GPUStreamingBuffer::~GPUStreamingBuffer() {
    for (auto* buffer : gpu_buffers_) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
}

bool GPUStreamingBuffer::allocate() {
    gpu_buffers_.resize(num_buffers_);

    for (size_t i = 0; i < num_buffers_; i++) {
        cudaError_t err = cudaMalloc(&gpu_buffers_[i], chunk_size_ * sizeof(double));
        if (err != cudaSuccess) {
            return false;
        }
    }

    return true;
}

double* GPUStreamingBuffer::get_next_buffer(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < num_buffers_; i++) {
        if (buffer_available_[i]) {
            buffer_available_[i] = false;
            return gpu_buffers_[i];
        }
    }

    return nullptr;
}

void GPUStreamingBuffer::release_buffer(double* buffer) {
    std::lock_guard<std::mutex> lock(mutex_);

    for (size_t i = 0; i < num_buffers_; i++) {
        if (gpu_buffers_[i] == buffer) {
            buffer_available_[i] = true;
            return;
        }
    }
}

bool GPUStreamingBuffer::copy_to_gpu(const DataChunk& chunk, double* gpu_buffer, cudaStream_t stream) {
    if (!gpu_buffer || chunk.data.empty()) {
        return false;
    }

    size_t copy_size = std::min(chunk.total_size, chunk_size_);
    cudaError_t err = cudaMemcpyAsync(
        gpu_buffer,
        chunk.data.data(),
        copy_size * sizeof(double),
        cudaMemcpyHostToDevice,
        stream
    );

    return err == cudaSuccess;
}

bool GPUStreamingBuffer::copy_from_gpu(double* gpu_buffer, DataChunk& chunk, cudaStream_t stream) {
    if (!gpu_buffer || chunk.data.empty()) {
        return false;
    }

    size_t copy_size = std::min(chunk.total_size, chunk_size_);
    cudaError_t err = cudaMemcpyAsync(
        chunk.data.data(),
        gpu_buffer,
        copy_size * sizeof(double),
        cudaMemcpyDeviceToHost,
        stream
    );

    return err == cudaSuccess;
}

} // namespace sintellix
