#include "persona.h"
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

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
    // NVRAM (Intel Optane) storage for Persona vector (Req 9.1.2)
    // Path should point to NVRAM-mounted filesystem (e.g., /mnt/pmem)
    std::ofstream file(path, std::ios::binary);
    size_t size = vector.size();
    file.write(reinterpret_cast<const char*>(&size), sizeof(size));
    file.write(reinterpret_cast<const char*>(vector.data()), size * sizeof(float));
    file.flush();  // Ensure data is persisted to NVRAM
}

void Persona::load(const char* path) {
    // Load Persona vector from NVRAM storage (Req 9.1.2)
    std::ifstream file(path, std::ios::binary);
    size_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size));
    vector.resize(size);
    file.read(reinterpret_cast<char*>(vector.data()), size * sizeof(float));
}

} // namespace quila
