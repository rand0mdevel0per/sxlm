#pragma once
#include <cuda_runtime.h>
#include "port.cuh"
#include "linear_scan.cuh"
#include "utility.cuh"
#include "nsa_macro.cuh"
#include "intercept.cuh"

namespace quila {

// Planner-Port: Strategic planning and goal setting
__global__ void planner_port_forward(
    PortState* port,
    const float* neuron_outputs,
    const float* wkv_weights,
    const float* session_el_trace,
    const float* persistent_el_trace,
    float* output,
    const PortConfig& config,
    int num_neurons
) {
    // Step 1: Linear attention scan
    linear_attention_scan(neuron_outputs, port, num_neurons, port->hidden_dim);
    __syncthreads();

    // Step 2: Compute utility scores
    float* utility_scores = (float*)malloc(num_neurons * sizeof(float));
    int* nsa_mask = (int*)malloc(config.nsa_top_k * sizeof(int));

    compute_utility(wkv_weights, session_el_trace, persistent_el_trace,
                    utility_scores, config, num_neurons);
    __syncthreads();

    // Step 3: Select top-k neurons
    select_top_k(utility_scores, nsa_mask, num_neurons, config.nsa_top_k);
    __syncthreads();

    // Step 4: NSA macro attention
    float* out_nsa = (float*)malloc(port->hidden_dim * sizeof(float));
    nsa_macro_attention(port->current_state, neuron_outputs, nsa_mask,
                        out_nsa, port->hidden_dim, config.nsa_top_k);
    __syncthreads();

    // Step 5: Intercept layer
    intercept_layer(port->aggr, out_nsa, neuron_outputs, nsa_mask,
                    output, port->hidden_dim, config.nsa_top_k);

    free(utility_scores);
    free(nsa_mask);
    free(out_nsa);
}

} // namespace quila
