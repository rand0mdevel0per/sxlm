#pragma once
#include <cuda_runtime.h>

namespace quila {

// Gradient buffers
struct GradientBuffers {
    float* grad_neuron_states;
    float* grad_port_states;
    float* grad_weights;
    int hidden_dim;
    int num_neurons;
    int num_ports;
};

// Initialize gradient buffers
__host__ void init_gradient_buffers(GradientBuffers* grads, int hidden_dim, int num_neurons, int num_ports);

// Free gradient buffers
__host__ void free_gradient_buffers(GradientBuffers* grads);

// Zero gradients
__host__ void zero_gradients(GradientBuffers* grads);

} // namespace quila
