#include "sxlm_format.h"
#include <fstream>

namespace quila {

bool save_sxlm(const char* path, const SXLMFormat& model) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    file.write(reinterpret_cast<const char*>(&model.version), sizeof(int));
    file.write(reinterpret_cast<const char*>(&model.num_neurons), sizeof(int));
    file.write(reinterpret_cast<const char*>(&model.hidden_dim), sizeof(int));

    size_t size = model.neuron_weights.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(model.neuron_weights.data()), size * sizeof(float));

    return true;
}

bool load_sxlm(const char* path, SXLMFormat& model) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    file.read(reinterpret_cast<char*>(&model.version), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.num_neurons), sizeof(int));
    file.read(reinterpret_cast<char*>(&model.hidden_dim), sizeof(int));

    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    model.neuron_weights.resize(size);
    file.read(reinterpret_cast<char*>(model.neuron_weights.data()), size * sizeof(float));

    return true;
}

} // namespace quila
