#include "sxlm/core/toml_config.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace sxlm {

// Simple TOML parser (minimal implementation)
static int parse_int(const std::string& line, const std::string& key) {
    size_t pos = line.find(key);
    if (pos == std::string::npos) return -1;
    pos = line.find('=', pos);
    if (pos == std::string::npos) return -1;
    return std::stoi(line.substr(pos + 1));
}

static float parse_float(const std::string& line, const std::string& key) {
    size_t pos = line.find(key);
    if (pos == std::string::npos) return -1.0f;
    pos = line.find('=', pos);
    if (pos == std::string::npos) return -1.0f;
    return std::stof(line.substr(pos + 1));
}

QuilaConfig QuilaConfig::load(const std::string& path) {
    QuilaConfig config;
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open config: " + path);
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        if (line.find("dim") != std::string::npos) config.dim = parse_int(line, "dim");
        else if (line.find("num_heads") != std::string::npos) config.num_heads = parse_int(line, "num_heads");
        else if (line.find("num_layers") != std::string::npos) config.num_layers = parse_int(line, "num_layers");
        else if (line.find("max_seq_len") != std::string::npos) config.max_seq_len = parse_int(line, "max_seq_len");
        else if (line.find("learning_rate") != std::string::npos) config.learning_rate = parse_float(line, "learning_rate");
    }

    return config;
}

bool QuilaConfig::save(const std::string& path) const {
    std::ofstream file(path);
    if (!file.is_open()) return false;

    file << "# SXLM Quila Configuration\n\n";
    file << "[model]\n";
    file << "dim = " << dim << "\n";
    file << "num_heads = " << num_heads << "\n";
    file << "num_layers = " << num_layers << "\n";
    file << "max_seq_len = " << max_seq_len << "\n\n";

    file << "[training]\n";
    file << "learning_rate = " << learning_rate << "\n";
    file << "el_trace_decay = " << el_trace_decay << "\n";

    return true;
}

} // namespace sxlm
