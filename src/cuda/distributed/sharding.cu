#include "sharding.cuh"
#include "../utils/memory.cuh"

namespace quila {

__host__ void init_neuron_sharding(NeuronSharding* sharding, int num_neurons, int num_gpus) {
    sharding->num_neurons = num_neurons;
    sharding->num_gpus = num_gpus;

    sharding->neuron_to_gpu = (int*)allocate_unified(num_neurons * sizeof(int));
    sharding->gpu_neuron_counts = (int*)allocate_unified(num_gpus * sizeof(int));

    cudaMemset(sharding->gpu_neuron_counts, 0, num_gpus * sizeof(int));
}

__host__ void free_neuron_sharding(NeuronSharding* sharding) {
    if (sharding->neuron_to_gpu) deallocate_unified(sharding->neuron_to_gpu);
    if (sharding->gpu_neuron_counts) deallocate_unified(sharding->gpu_neuron_counts);
}

__host__ void assign_neurons_to_gpus(NeuronSharding* sharding) {
    // Simplified: round-robin assignment
    for (int i = 0; i < sharding->num_neurons; i++) {
        int gpu_id = i % sharding->num_gpus;
        sharding->neuron_to_gpu[i] = gpu_id;
        sharding->gpu_neuron_counts[gpu_id]++;
    }
}

__host__ int get_neuron_gpu(const NeuronSharding* sharding, int neuron_id) {
    return sharding->neuron_to_gpu[neuron_id];
}

} // namespace quila
