#pragma once
#include <cuda_runtime.h>

namespace quila {

// Graph topology for neuron connections
struct GraphTopology {
    int* adjacency_list;  // Flattened adjacency list
    int* edge_counts;     // Number of edges per neuron
    int num_neurons;
    int max_edges_per_neuron;
};

// Initialize graph topology
__host__ void init_graph_topology(GraphTopology* graph, int num_neurons, int max_edges_per_neuron);

// Free graph topology
__host__ void free_graph_topology(GraphTopology* graph);

// Grow edges based on utility
__host__ void grow_edges(GraphTopology* graph, const float* utility_scores, float threshold);

// Prune edges based on low utility
__host__ void prune_edges(GraphTopology* graph, const float* utility_scores, float threshold);

} // namespace quila
