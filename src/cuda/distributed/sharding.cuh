#pragma once
#include <cuda_runtime.h>

namespace quila {

// Neuron sharding for multi-GPU distribution
struct NeuronSharding {
    int* neuron_to_gpu;      // Maps neuron ID to GPU ID
    int* gpu_neuron_counts;  // Number of neurons per GPU
    int num_neurons;
    int num_gpus;
};

// Initialize sharding
__host__ void init_neuron_sharding(NeuronSharding* sharding, int num_neurons, int num_gpus);

// Free sharding
__host__ void free_neuron_sharding(NeuronSharding* sharding);

// Assign neurons to GPUs (simplified round-robin)
__host__ void assign_neurons_to_gpus(NeuronSharding* sharding);

// Get GPU ID for neuron
__host__ int get_neuron_gpu(const NeuronSharding* sharding, int neuron_id);

} // namespace quila
