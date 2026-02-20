#pragma once
#include <string>
#include <vector>

namespace quila {

// .sxlm model format
struct SXLMFormat {
    int version;
    int num_neurons;
    int hidden_dim;
    std::vector<float> neuron_weights;
};

// Save model to .sxlm format
bool save_sxlm(const char* path, const SXLMFormat& model);

// Load model from .sxlm format
bool load_sxlm(const char* path, SXLMFormat& model);

} // namespace quila
