#pragma once
#include <string>
#include <vector>

namespace quila {

// Compute SHA-256 hash of data (Req 21.4.2)
std::string compute_sha256(const void* data, size_t size);

// Verify SHA-256 hash matches expected
bool verify_sha256(const void* data, size_t size, const std::string& expected_hash);

} // namespace quila
