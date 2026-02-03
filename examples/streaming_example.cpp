#include "sintellix/core/neuron_model.cuh"
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

using namespace sintellix;

/**
 * Example: Streaming Processing with NeuronModel
 *
 * This example demonstrates how to use streaming mode for processing
 * large amounts of data in chunks without memory limitations.
 */

void producer_thread(NeuronModel& model, size_t total_chunks) {
    std::cout << "Producer: Starting to send " << total_chunks << " chunks..." << std::endl;

    for (size_t i = 0; i < total_chunks; i++) {
        // Generate input data (4096 elements per chunk)
        std::vector<double> input_data(4096);
        for (size_t j = 0; j < 4096; j++) {
            input_data[j] = static_cast<double>(i * 4096 + j) / 1000.0;
        }

        // Push chunk to model
        bool is_last = (i == total_chunks - 1);
        if (model.push_input_chunk(input_data.data(), input_data.size(), is_last)) {
            std::cout << "Producer: Sent chunk " << i << "/" << total_chunks
                      << (is_last ? " (LAST)" : "") << std::endl;
        } else {
            std::cerr << "Producer: Failed to send chunk " << i << std::endl;
        }

        // Simulate data generation delay
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "Producer: Finished sending all chunks" << std::endl;
}

void consumer_thread(NeuronModel& model) {
    std::cout << "Consumer: Starting to receive chunks..." << std::endl;

    size_t chunk_count = 0;
    while (true) {
        std::vector<double> output_data(4096);
        size_t output_size = 0;
        bool is_last = false;

        // Try to get output chunk (non-blocking)
        if (model.try_get_output_chunk(output_data.data(), output_size, is_last)) {
            chunk_count++;
            std::cout << "Consumer: Received chunk " << chunk_count
                      << " (size=" << output_size << ")"
                      << (is_last ? " (LAST)" : "") << std::endl;

            // Process output data here
            // ...

            if (is_last) {
                break;
            }
        } else {
            // No output available yet, wait a bit
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    std::cout << "Consumer: Finished receiving all chunks (total: "
              << chunk_count << ")" << std::endl;
}

int main() {
    std::cout << "=== Sintellix Streaming Processing Example ===" << std::endl;

    // Create neuron configuration
    NeuronConfig config;
    config.set_dim(512);
    auto* grid = config.mutable_grid_size();
    grid->set_x(8);
    grid->set_y(8);
    grid->set_z(8);

    // Create and initialize model
    NeuronModel model(config);
    if (!model.initialize()) {
        std::cerr << "Failed to initialize model" << std::endl;
        return 1;
    }

    std::cout << "Model initialized successfully" << std::endl;

    // Enable streaming mode
    // chunk_size: 4096 elements per chunk
    // max_chunks: 16 chunks in buffer
    if (!model.enable_streaming(4096, 16)) {
        std::cerr << "Failed to enable streaming mode" << std::endl;
        return 1;
    }

    std::cout << "Streaming mode enabled" << std::endl;

    // Start producer and consumer threads
    size_t total_chunks = 100;  // Process 100 chunks (409,600 elements total)

    std::thread producer(producer_thread, std::ref(model), total_chunks);
    std::thread consumer(consumer_thread, std::ref(model));

    // Wait for both threads to complete
    producer.join();
    consumer.join();

    // Disable streaming mode
    model.disable_streaming();

    std::cout << "=== Streaming processing completed ===" << std::endl;
    std::cout << "Total data processed: " << (total_chunks * 4096) << " elements" << std::endl;

    return 0;
}
