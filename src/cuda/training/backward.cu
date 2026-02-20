#include "backward.cuh"
#include "../utils/memory.cuh"
#include "../utils/error.cuh"

namespace quila {

__host__ void init_gradient_buffers(GradientBuffers* grads, int hidden_dim, int num_neurons, int num_ports) {
    grads->hidden_dim = hidden_dim;
    grads->num_neurons = num_neurons;
    grads->num_ports = num_ports;

    grads->grad_neuron_states = (float*)allocate_unified(num_neurons * hidden_dim * sizeof(float));
    grads->grad_port_states = (float*)allocate_unified(num_ports * hidden_dim * sizeof(float));
    grads->grad_weights = (float*)allocate_unified(hidden_dim * hidden_dim * sizeof(float));
}

__host__ void free_gradient_buffers(GradientBuffers* grads) {
    if (grads->grad_neuron_states) deallocate_unified(grads->grad_neuron_states);
    if (grads->grad_port_states) deallocate_unified(grads->grad_port_states);
    if (grads->grad_weights) deallocate_unified(grads->grad_weights);
}

__global__ void zero_gradients_kernel(float* grad, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    grad[tid] = 0.0f;
}

__host__ void zero_gradients(GradientBuffers* grads) {
    int threads = 256;
    int blocks;

    blocks = (grads->num_neurons * grads->hidden_dim + threads - 1) / threads;
    zero_gradients_kernel<<<blocks, threads>>>(grads->grad_neuron_states, grads->num_neurons * grads->hidden_dim);

    blocks = (grads->num_ports * grads->hidden_dim + threads - 1) / threads;
    zero_gradients_kernel<<<blocks, threads>>>(grads->grad_port_states, grads->num_ports * grads->hidden_dim);

    blocks = (grads->hidden_dim * grads->hidden_dim + threads - 1) / threads;
    zero_gradients_kernel<<<blocks, threads>>>(grads->grad_weights, grads->hidden_dim * grads->hidden_dim);

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

} // namespace quila
