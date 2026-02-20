#include "../cuda/neuron/neuron_grid.cuh"
#include <iostream>

int main() {
    const int num_neurons = 4;
    const int hidden_dim = 64;

    // Create neuron grid
    quila::NeuronGrid grid(num_neurons, hidden_dim);

    // Prepare input
    float* input = new float[num_neurons * hidden_dim];
    float* output = new float[num_neurons * hidden_dim];

    for (int i = 0; i < num_neurons * hidden_dim; i++) {
        input[i] = 0.1f * (i % 10);
    }

    // Run forward pass
    grid.forward(input, output);

    // Print sample output
    std::cout << "Neuron forward pass test:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "Neuron " << i << " output[0]: " << output[i * hidden_dim] << "\n";
    }

    delete[] input;
    delete[] output;

    std::cout << "Test completed successfully!\n";
    return 0;
}
