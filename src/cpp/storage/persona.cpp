#include "persona.h"
#include <fstream>

namespace quila {

Persona::Persona(int dim) : vector(dim, 0.0f), last_update(0) {}

void Persona::update(const std::vector<float>& new_vector) {
    vector = new_vector;
    last_update++;
}

const std::vector<float>& Persona::get() const {
    return vector;
}

void Persona::save(const char* path) {
    std::ofstream file(path, std::ios::binary);
    size_t size = vector.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(vector.data()), size * sizeof(float));
}

void Persona::load(const char* path) {
    std::ifstream file(path, std::ios::binary);
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    vector.resize(size);
    file.read(reinterpret_cast<char*>(vector.data()), size * sizeof(float));
}

} // namespace quila
