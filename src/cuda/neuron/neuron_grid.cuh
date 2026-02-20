#pragma once
#include "forward.cu"
#include "init.cu"

namespace quila {

// Host-side wrapper for neuron forward pass
class NeuronGrid {
private:
    NeuronState* d_states;
    float* d_input_messages;
    float* d_output_messages;
    int num_neurons;
    int hidden_dim;

public:
    NeuronGrid(int num_neurons, int hidden_dim)
        : num_neurons(num_neurons), hidden_dim(hidden_dim) {

        // Allocate unified memory for states
        d_states = (NeuronState*)allocate_unified(num_neurons * sizeof(NeuronState));

        // Initialize each neuron state
        for (int i = 0; i < num_neurons; i++) {
            init_neuron_state(&d_states[i], hidden_dim);
        }

        // Allocate message buffers
        d_input_messages = (float*)allocate_unified(num_neurons * hidden_dim * sizeof(float));
        d_output_messages = (float*)allocate_unified(num_neurons * hidden_dim * sizeof(float));
    }

    ~NeuronGrid() {
        for (int i = 0; i < num_neurons; i++) {
            free_neuron_state(&d_states[i]);
        }
        deallocate_unified(d_states);
        deallocate_unified(d_input_messages);
        deallocate_unified(d_output_messages);
    }

    void forward(const float* input, float* output) {
        // Copy input to device
        CUDA_CHECK(cudaMemcpy(d_input_messages, input,
                              num_neurons * hidden_dim * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Launch kernel
        int shared_mem_size = hidden_dim * 7 * sizeof(float);
        neuron_forward_kernel<<<num_neurons, hidden_dim, shared_mem_size>>>(
            d_states, d_input_messages, d_output_messages, num_neurons, hidden_dim
        );
        CUDA_CHECK_LAST();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy output to host
        CUDA_CHECK(cudaMemcpy(output, d_output_messages,
                              num_neurons * hidden_dim * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
};

} // namespace quila
