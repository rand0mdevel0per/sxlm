#include "sxlm_format.h"
#include "../utils/sha256.h"
#include <fstream>
#include <sstream>
#include <zstd.h>

namespace quila {

bool save_sxlm(const char* path, const SXLMFormat& model) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;

    // Write model metadata
    std::stringstream buffer;
    buffer.write(reinterpret_cast<const char*>(&model.version), sizeof(int));
    buffer.write(reinterpret_cast<const char*>(&model.num_neurons), sizeof(int));
    buffer.write(reinterpret_cast<const char*>(&model.hidden_dim), sizeof(int));

    // Compress neuron weights with zstd level 3 (Req 21.4.1)
    size_t size = model.neuron_weights.size();
    size_t src_size = size * sizeof(float);
    size_t compressed_bound = ZSTD_compressBound(src_size);
    std::vector<char> compressed(compressed_bound);

    size_t compressed_size = ZSTD_compress(
        compressed.data(), compressed_bound,
        model.neuron_weights.data(), src_size,
        3  // zstd level 3
    );

    buffer.write(reinterpret_cast<const char*>(&size), sizeof(size));
    buffer.write(reinterpret_cast<const char*>(&compressed_size), sizeof(size_t));
    buffer.write(compressed.data(), compressed_size);

    // Compute SHA-256 hash (Req 21.4.2)
    std::string data = buffer.str();
    std::string hash = compute_sha256(data.data(), data.size());

    // Write data and hash
    file.write(data.data(), data.size());
    file.write(hash.data(), hash.size());

    return true;
}

bool load_sxlm(const char* path, SXLMFormat& model) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;

    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Read data (file_size - 64 bytes for SHA-256 hash)
    size_t data_size = file_size - 64;
    std::vector<char> data(data_size);
    file.read(data.data(), data_size);

    // Read stored hash
    std::string stored_hash(64, '\0');
    file.read(&stored_hash[0], 64);

    // Verify SHA-256 hash (Req 21.4.2)
    if (!verify_sha256(data.data(), data_size, stored_hash)) {
        return false;
    }

    // Parse model metadata
    std::stringstream buffer;
    buffer.write(data.data(), data_size);
    buffer.read(reinterpret_cast<char*>(&model.version), sizeof(int));
    buffer.read(reinterpret_cast<char*>(&model.num_neurons), sizeof(int));
    buffer.read(reinterpret_cast<char*>(&model.hidden_dim), sizeof(int));

    // Decompress neuron weights with zstd (Req 21.4.1)
    size_t size, compressed_size;
    buffer.read(reinterpret_cast<char*>(&size), sizeof(size));
    buffer.read(reinterpret_cast<char*>(&compressed_size), sizeof(size_t));

    std::vector<char> compressed(compressed_size);
    buffer.read(compressed.data(), compressed_size);

    model.neuron_weights.resize(size);
    ZSTD_decompress(
        model.neuron_weights.data(), size * sizeof(float),
        compressed.data(), compressed_size
    );

    return true;
}

} // namespace quila
