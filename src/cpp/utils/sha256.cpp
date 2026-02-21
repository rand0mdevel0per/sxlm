#include "sha256.h"
#include <openssl/sha.h>
#include <sstream>
#include <iomanip>

namespace quila {

std::string compute_sha256(const void* data, size_t size) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256(static_cast<const unsigned char*>(data), size, hash);

    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

bool verify_sha256(const void* data, size_t size, const std::string& expected_hash) {
    return compute_sha256(data, size) == expected_hash;
}

} // namespace quila
