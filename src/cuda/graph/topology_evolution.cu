#include "topology_evolution.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_graph_topology(GraphTopology* graph, int num_neurons, int max_edges_per_neuron) {
    graph->num_neurons = num_neurons;
    graph->max_edges_per_neuron = max_edges_per_neuron;

    graph->adjacency_list = (int*)allocate_unified(num_neurons * max_edges_per_neuron * sizeof(int));
    graph->edge_counts = (int*)allocate_unified(num_neurons * sizeof(int));

    cudaMemset(graph->adjacency_list, -1, num_neurons * max_edges_per_neuron * sizeof(int));
    cudaMemset(graph->edge_counts, 0, num_neurons * sizeof(int));
}

__host__ void free_graph_topology(GraphTopology* graph) {
    if (graph->adjacency_list) deallocate_unified(graph->adjacency_list);
    if (graph->edge_counts) deallocate_unified(graph->edge_counts);
}

__global__ void grow_edges_kernel(
    int* adjacency_list,
    int* edge_counts,
    const float* utility_scores,
    int num_neurons,
    int max_edges,
    float threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_neurons) return;

    if (utility_scores[tid] > threshold && edge_counts[tid] < max_edges) {
        int target = (tid + 1) % num_neurons;
        adjacency_list[tid * max_edges + edge_counts[tid]] = target;
        edge_counts[tid]++;
    }
}

__host__ void grow_edges(GraphTopology* graph, const float* utility_scores, float threshold) {
    int threads = 256;
    int blocks = (graph->num_neurons + threads - 1) / threads;

    grow_edges_kernel<<<blocks, threads>>>(
        graph->adjacency_list, graph->edge_counts, utility_scores,
        graph->num_neurons, graph->max_edges_per_neuron, threshold
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

__global__ void prune_edges_kernel(
    int* adjacency_list,
    int* edge_counts,
    const float* utility_scores,
    int num_neurons,
    int max_edges,
    float threshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_neurons) return;

    if (utility_scores[tid] < threshold && edge_counts[tid] > 0) {
        edge_counts[tid]--;
        adjacency_list[tid * max_edges + edge_counts[tid]] = -1;
    }
}

__host__ void prune_edges(GraphTopology* graph, const float* utility_scores, float threshold) {
    int threads = 256;
    int blocks = (graph->num_neurons + threads - 1) / threads;

    prune_edges_kernel<<<blocks, threads>>>(
        graph->adjacency_list, graph->edge_counts, utility_scores,
        graph->num_neurons, graph->max_edges_per_neuron, threshold
    );
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
