#pragma once
#include <vector>
#include <cstdint>

namespace quila {

// Persona vector stored in NVRAM
class Persona {
private:
    std::vector<float> vector;
    uint64_t last_update;

public:
    Persona(int dim);

    void update(const std::vector<float>& new_vector);
    const std::vector<float>& get() const;
    void save(const char* path);
    void load(const char* path);
};

} // namespace quila
