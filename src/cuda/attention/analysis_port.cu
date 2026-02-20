#pragma once
#include <cuda_runtime.h>
#include "port.cuh"
#include "linear_scan.cu"
#include "utility.cu"
#include "nsa_macro.cu"
#include "intercept.cu"

namespace quila {

// Analysis-Port: Verification and quality assessment
__global__ void analysis_port_forward(
    PortState* port,
    const float* neuron_outputs,
    const float* wkv_weights,
    const float* session_el_trace,
    const float* persistent_el_trace,
    float* output,
    const PortConfig& config,
    int num_neurons
) {
    linear_attention_scan(neuron_outputs, port, num_neurons, port->hidden_dim);
    __syncthreads();

    float* utility_scores = (float*)malloc(num_neurons * sizeof(float));
    int* nsa_mask = (int*)malloc(config.nsa_top_k * sizeof(int));

    compute_utility(wkv_weights, session_el_trace, persistent_el_trace,
                    utility_scores, config, num_neurons);
    __syncthreads();

    select_top_k(utility_scores, nsa_mask, num_neurons, config.nsa_top_k);
    __syncthreads();

    float* out_nsa = (float*)malloc(port->hidden_dim * sizeof(float));
    nsa_macro_attention(port->current_state, neuron_outputs, nsa_mask,
                        out_nsa, port->hidden_dim, config.nsa_top_k);
    __syncthreads();

    intercept_layer(port->aggr, out_nsa, neuron_outputs, nsa_mask,
                    output, port->hidden_dim, config.nsa_top_k);

    free(utility_scores);
    free(nsa_mask);
    free(out_nsa);
}

} // namespace quila
