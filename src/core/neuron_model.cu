#include "sintellix/core/neuron_model.cuh"
#include <cuda_runtime.h>
#include <fstream>
#include <zstd.h>
#include <algorithm>
#include <numeric>
#include "model_state.pb.h"

namespace sintellix {

// Forward declaration of global aggregation kernel from neuron.cu
__global__ void global_aggregation_kernel(
    const double* local_output,
    const double** all_neurons_output,
    double* aggregated_output,
    const int* top_k_indices,
    int dim,
    int top_k
);

// ============================================================================
// KFEManager Implementation
// ============================================================================

KFEManager::KFEManager(size_t max_slots)
    : max_slots_(max_slots)
{
}

KFEManager::~KFEManager() {
    // Free all GPU memory
    for (auto& pair : kfe_storage_) {
        if (pair.second.gpu_ptr) {
            cudaFree(pair.second.gpu_ptr);
        }
    }
}

bool KFEManager::store_kfe(const std::string& key, const double* kfe_matrix, size_t dim) {
    // Check if already exists
    auto it = kfe_storage_.find(key);
    if (it != kfe_storage_.end()) {
        // Update existing KFE
        cudaMemcpy(it->second.gpu_ptr, kfe_matrix, dim * dim * sizeof(double), cudaMemcpyDeviceToDevice);
        it->second.access_count++;
        it->second.last_access = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        return true;
    }

    // Check if we have space
    if (kfe_storage_.size() >= max_slots_) {
        // Evict least recently used
        std::string lru_key;
        uint64_t min_access = UINT64_MAX;
        for (const auto& p : kfe_storage_) {
            if (p.second.last_access < min_access) {
                min_access = p.second.last_access;
                lru_key = p.first;
            }
        }

        // Remove LRU
        cudaFree(kfe_storage_[lru_key].gpu_ptr);
        kfe_storage_.erase(lru_key);
    }

    // Create new slot
    KFESlot slot;
    slot.dim = dim;
    slot.access_count = 1;
    slot.last_access = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    cudaMalloc(&slot.gpu_ptr, dim * dim * sizeof(double));
    cudaMemcpy(slot.gpu_ptr, kfe_matrix, dim * dim * sizeof(double), cudaMemcpyDeviceToDevice);

    kfe_storage_[key] = slot;
    return true;
}

bool KFEManager::retrieve_kfe(const std::string& key, double* kfe_matrix, size_t dim) {
    auto it = kfe_storage_.find(key);
    if (it == kfe_storage_.end()) {
        return false;
    }

    cudaMemcpy(kfe_matrix, it->second.gpu_ptr, dim * dim * sizeof(double), cudaMemcpyDeviceToDevice);
    it->second.access_count++;
    it->second.last_access = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();

    return true;
}

bool KFEManager::has_kfe(const std::string& key) const {
    return kfe_storage_.find(key) != kfe_storage_.end();
}

// ============================================================================
// NeuronModel Implementation
// ============================================================================

NeuronModel::NeuronModel(const NeuronConfig& config)
    : config_(config)
    , kfe_manager_(10000)
    , grid_x_(config.grid_size().x())
    , grid_y_(config.grid_size().y())
    , grid_z_(config.grid_size().z())
    , total_neurons_(grid_x_ * grid_y_ * grid_z_)
    , dim_(config.dim())
    , multi_gpu_enabled_(false)
    , streaming_enabled_(false)
    , streaming_active_(false)
    , streaming_chunk_size_(0)
    , streaming_chunk_counter_(0)
{
    // Create tiered storage manager
    storage_manager_ = std::make_unique<TieredStorageManager>(config);

    // Create CUDA streams for parallel execution
    int num_streams = 8;
    streams_.resize(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams_[i]);
    }
}

NeuronModel::~NeuronModel() {
    // Destroy CUDA streams
    for (auto stream : streams_) {
        cudaStreamDestroy(stream);
    }
}

bool NeuronModel::initialize() {
    neurons_.reserve(total_neurons_);

    // Create all neurons
    for (uint32_t x = 0; x < grid_x_; x++) {
        for (uint32_t y = 0; y < grid_y_; y++) {
            for (uint32_t z = 0; z < grid_z_; z++) {
                int neuron_id = get_neuron_index(x, y, z);
                auto neuron = std::make_unique<Neuron>(config_, neuron_id, x, y, z);

                if (!neuron->initialize()) {
                    return false;
                }

                neurons_.push_back(std::move(neuron));
            }
        }
    }

    return true;
}

void NeuronModel::forward(const double* input, double* output) {
    // Parallel forward pass through all neurons
    int stream_idx = 0;

    for (size_t i = 0; i < neurons_.size(); i++) {
        cudaStream_t stream = streams_[stream_idx % streams_.size()];

        // Each neuron processes the input
        neurons_[i]->forward(input, output, stream);

        stream_idx++;
    }

    // Wait for all streams to complete
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }

    // Global aggregation (if enabled)
    if (config_.modules().enable_global_aggregation()) {
        global_aggregation();
    }
}

void NeuronModel::backward(const double* grad_output, double* grad_input) {
    // Parallel backward pass through all neurons
    int stream_idx = 0;

    for (size_t i = 0; i < neurons_.size(); i++) {
        cudaStream_t stream = streams_[stream_idx % streams_.size()];

        neurons_[i]->backward(grad_output, grad_input, stream);

        stream_idx++;
    }

    // Wait for all streams to complete
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
}

void NeuronModel::update_parameters(float learning_rate) {
    // Parallel parameter update for all neurons
    int stream_idx = 0;

    for (size_t i = 0; i < neurons_.size(); i++) {
        cudaStream_t stream = streams_[stream_idx % streams_.size()];

        neurons_[i]->update_parameters(learning_rate, stream);

        stream_idx++;
    }

    // Wait for all streams to complete
    for (auto stream : streams_) {
        cudaStreamSynchronize(stream);
    }
}

void NeuronModel::global_aggregation() {
    // Get global aggregation config
    uint32_t top_k = config_.global_aggregation().top_k();
    if (top_k == 0 || top_k > total_neurons_) {
        top_k = std::min(256u, total_neurons_);  // Default: top 256 neurons
    }

    // Allocate temporary buffers for neuron outputs and importance scores
    std::vector<double*> neuron_outputs_cpu(total_neurons_);
    std::vector<double> importance_scores(total_neurons_);

    // Calculate importance scores for each neuron (using L2 norm of output)
    // Note: This is a simplified version. In production, you'd want to store
    // neuron outputs during forward pass for efficiency
    for (size_t i = 0; i < total_neurons_; i++) {
        // For now, use neuron position as a proxy for importance
        // In full implementation, this would use actual output norms
        int x = i / (grid_y_ * grid_z_);
        int y = (i / grid_z_) % grid_y_;
        int z = i % grid_z_;

        // Simple importance metric: distance from center
        double cx = grid_x_ / 2.0;
        double cy = grid_y_ / 2.0;
        double cz = grid_z_ / 2.0;
        double dist = std::sqrt((x - cx) * (x - cx) +
                               (y - cy) * (y - cy) +
                               (z - cz) * (z - cz));
        importance_scores[i] = 1.0 / (1.0 + dist);  // Higher score for center neurons
    }

    // Select top-K neurons based on importance scores
    std::vector<int> indices(total_neurons_);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                     [&importance_scores](int a, int b) {
                         return importance_scores[a] > importance_scores[b];
                     });

    // Copy top-K indices to GPU
    int* top_k_indices_gpu;
    cudaMalloc(&top_k_indices_gpu, top_k * sizeof(int));
    cudaMemcpy(top_k_indices_gpu, indices.data(), top_k * sizeof(int), cudaMemcpyHostToDevice);

    // Note: Full implementation would call global_aggregation_kernel here
    // For now, we've implemented the selection logic
    // The kernel call would require storing individual neuron outputs

    // Cleanup
    cudaFree(top_k_indices_gpu);
}

void NeuronModel::replay_context(const std::vector<const double*>& history, bool fast_mode) {
    // Fast context replay for state injection
    // This allows quick restoration of model state from historical inputs

    double* temp_output;
    cudaMalloc(&temp_output, dim_ * dim_ * sizeof(double));

    for (size_t i = 0; i < history.size(); i++) {
        const double* input = history[i];

        if (fast_mode) {
            // Fast mode: Skip output generation, only update internal state
            int stream_idx = 0;

            for (size_t j = 0; j < neurons_.size(); j++) {
                cudaStream_t stream = streams_[stream_idx % streams_.size()];

                // Forward pass (internal state update only)
                neurons_[j]->forward(input, temp_output, stream);

                stream_idx++;
            }

            // Wait for completion
            for (auto stream : streams_) {
                cudaStreamSynchronize(stream);
            }
        } else {
            // Normal mode: Full forward pass
            forward(input, temp_output);
        }

        // Store KFE for this input
        std::string kfe_key = "context_" + std::to_string(i);
        kfe_manager_.store_kfe(kfe_key, temp_output, dim_);
    }

    cudaFree(temp_output);
}

bool NeuronModel::save_state(const std::string& path) {
    // Create ModelState protobuf
    ModelState model_state;

    // Set configuration
    *model_state.mutable_config() = config_;

    // Set metadata
    model_state.set_version("0.1.0");
    model_state.set_save_timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count());

    // Serialize all neurons (simplified - full implementation would serialize all state)
    // For now, just save basic statistics
    auto* stats = model_state.mutable_stats();
    stats->set_total_steps(0);
    stats->set_total_tokens(0);
    stats->set_average_loss(0.0f);

    // Serialize to string
    std::string serialized;
    if (!model_state.SerializeToString(&serialized)) {
        return false;
    }

    // Compress with zstd
    size_t compressed_bound = ZSTD_compressBound(serialized.size());
    std::vector<char> compressed(compressed_bound);

    size_t compressed_size = ZSTD_compress(
        compressed.data(), compressed_bound,
        serialized.data(), serialized.size(),
        3  // Compression level
    );

    if (ZSTD_isError(compressed_size)) {
        return false;
    }

    // Write to file
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.write(compressed.data(), compressed_size);
    file.close();

    return true;
}

bool NeuronModel::load_state(const std::string& path) {
    // Read compressed file
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> compressed(file_size);
    file.read(compressed.data(), file_size);
    file.close();

    // Decompress
    unsigned long long decompressed_size = ZSTD_getFrameContentSize(compressed.data(), file_size);
    if (decompressed_size == ZSTD_CONTENTSIZE_ERROR || decompressed_size == ZSTD_CONTENTSIZE_UNKNOWN) {
        return false;
    }

    std::vector<char> decompressed(decompressed_size);
    size_t actual_size = ZSTD_decompress(
        decompressed.data(), decompressed_size,
        compressed.data(), file_size
    );

    if (ZSTD_isError(actual_size)) {
        return false;
    }

    // Deserialize protobuf
    ModelState model_state;
    if (!model_state.ParseFromArray(decompressed.data(), actual_size)) {
        return false;
    }

    // Load configuration
    config_ = model_state.config();

    // Reinitialize model with loaded config
    // (Full implementation would restore all neuron states)

    return true;
}

// Multi-GPU support methods
bool NeuronModel::enable_multi_gpu(const std::vector<int>& device_ids) {
    try {
        // Create multi-GPU manager
        multi_gpu_manager_ = std::make_unique<MultiGPUManager>(device_ids);

        // Distribute neurons across devices
        int num_devices = multi_gpu_manager_->get_device_count();
        neuron_distribution_ = distribute_neurons(total_neurons_, num_devices);

        multi_gpu_enabled_ = true;

        std::cout << "Multi-GPU enabled with " << num_devices << " devices" << std::endl;
        for (size_t i = 0; i < neuron_distribution_.size(); i++) {
            std::cout << "  Device " << i << ": "
                      << neuron_distribution_[i].second << " neurons" << std::endl;
        }

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to enable multi-GPU: " << e.what() << std::endl;
        multi_gpu_enabled_ = false;
        return false;
    }
}

int NeuronModel::get_gpu_count() const {
    if (multi_gpu_manager_) {
        return multi_gpu_manager_->get_device_count();
    }
    return 1;  // Single GPU mode
}

// ============================================================================
// Streaming Processing Implementation
// ============================================================================

bool NeuronModel::enable_streaming(size_t chunk_size, size_t max_chunks) {
    if (streaming_enabled_) {
        return false;  // Already enabled
    }

    try {
        streaming_chunk_size_ = chunk_size;
        streaming_chunk_counter_ = 0;

        // Create buffers
        input_buffer_ = std::make_unique<StreamingBuffer>(max_chunks);
        output_buffer_ = std::make_unique<StreamingBuffer>(max_chunks);
        gpu_buffer_ = std::make_unique<GPUStreamingBuffer>(chunk_size, 4);

        // Allocate GPU buffers
        if (!gpu_buffer_->allocate()) {
            return false;
        }

        // Start streaming worker thread
        streaming_active_.store(true);
        streaming_thread_ = std::thread(&NeuronModel::streaming_worker, this);

        streaming_enabled_ = true;
        std::cout << "Streaming mode enabled (chunk_size=" << chunk_size
                  << ", max_chunks=" << max_chunks << ")" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to enable streaming: " << e.what() << std::endl;
        return false;
    }
}

bool NeuronModel::disable_streaming() {
    if (!streaming_enabled_) {
        return false;
    }

    // Signal end of stream
    input_buffer_->signal_end();
    streaming_active_.store(false);

    // Wait for worker thread to finish
    if (streaming_thread_.joinable()) {
        streaming_thread_.join();
    }

    // Clear buffers
    input_buffer_.reset();
    output_buffer_.reset();
    gpu_buffer_.reset();

    streaming_enabled_ = false;
    std::cout << "Streaming mode disabled" << std::endl;
    return true;
}

bool NeuronModel::push_input_chunk(const double* data, size_t size, bool is_last) {
    if (!streaming_enabled_) {
        return false;
    }

    DataChunk chunk(data, size, streaming_chunk_counter_++, is_last);
    return input_buffer_->push(chunk);
}

bool NeuronModel::try_get_output_chunk(double* data, size_t& size, bool& is_last) {
    if (!streaming_enabled_) {
        return false;
    }

    DataChunk chunk;
    if (output_buffer_->try_pop(chunk)) {
        size = std::min(chunk.total_size, streaming_chunk_size_);
        std::copy(chunk.data.begin(), chunk.data.begin() + size, data);
        is_last = chunk.is_last;
        return true;
    }

    return false;
}

bool NeuronModel::get_output_chunk(double* data, size_t& size, bool& is_last) {
    if (!streaming_enabled_) {
        return false;
    }

    DataChunk chunk;
    if (output_buffer_->pop(chunk)) {
        size = std::min(chunk.total_size, streaming_chunk_size_);
        std::copy(chunk.data.begin(), chunk.data.begin() + size, data);
        is_last = chunk.is_last;
        return true;
    }

    return false;
}

void NeuronModel::streaming_worker() {
    while (streaming_active_.load()) {
        DataChunk input_chunk;

        // Try to get input chunk
        if (!input_buffer_->pop(input_chunk)) {
            if (input_buffer_->is_ended()) {
                break;
            }
            continue;
        }

        // Get GPU buffer
        double* gpu_input = gpu_buffer_->get_next_buffer(streams_[0]);
        if (!gpu_input) {
            std::cerr << "Failed to get GPU buffer" << std::endl;
            continue;
        }

        // Allocate output buffer
        double* gpu_output = nullptr;
        cudaMalloc(&gpu_output, input_chunk.total_size * sizeof(double));

        // Copy input to GPU
        if (!gpu_buffer_->copy_to_gpu(input_chunk, gpu_input, streams_[0])) {
            std::cerr << "Failed to copy input to GPU" << std::endl;
            gpu_buffer_->release_buffer(gpu_input);
            cudaFree(gpu_output);
            continue;
        }

        // Process on GPU
        forward(gpu_input, gpu_output);

        // Create output chunk
        DataChunk output_chunk;
        output_chunk.data.resize(input_chunk.total_size);
        output_chunk.chunk_id = input_chunk.chunk_id;
        output_chunk.total_size = input_chunk.total_size;
        output_chunk.is_last = input_chunk.is_last;

        // Copy output from GPU
        cudaMemcpy(output_chunk.data.data(), gpu_output,
                   input_chunk.total_size * sizeof(double),
                   cudaMemcpyDeviceToHost);

        // Push to output buffer
        output_buffer_->push(output_chunk);

        // Cleanup
        gpu_buffer_->release_buffer(gpu_input);
        cudaFree(gpu_output);

        // Signal end if this was the last chunk
        if (input_chunk.is_last) {
            output_buffer_->signal_end();
            break;
        }
    }
}

} // namespace sintellix
