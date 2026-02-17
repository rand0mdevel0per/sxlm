#pragma once

#include <string>
#include <memory>
#ifdef USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif
#ifdef USE_PROTOBUF
#include "neuron_config.pb.h"
#else
// Minimal dummy NeuronConfig when Protobuf is not available
struct GridSize {
    int x() const { return x_; }
    int y() const { return y_; }
    int z() const { return z_; }
    void set_x(int v) { x_ = v; }
    void set_y(int v) { y_ = v; }
    void set_z(int v) { z_ = v; }
    int x_ = 32, y_ = 32, z_ = 32;
};

struct ModuleConfig {
    bool enable_multi_head() const { return enable_multi_head_; }
    bool enable_global_aggregation() const { return enable_global_aggregation_; }
    bool enable_noise_filter() const { return enable_noise_filter_; }
    bool enable_temporal_attention() const { return enable_temporal_attention_; }
    bool enable_fxaa_layer() const { return enable_fxaa_layer_; }
    bool enable_kfe_memory() const { return enable_kfe_memory_; }
    bool enable_ssm() const { return enable_ssm_; }
    bool enable_rwkv() const { return enable_rwkv_; }
    bool enable_ddpm() const { return enable_ddpm_; }
    void set_enable_multi_head(bool v) { enable_multi_head_ = v; }
    void set_enable_global_aggregation(bool v) { enable_global_aggregation_ = v; }
    void set_enable_noise_filter(bool v) { enable_noise_filter_ = v; }
    void set_enable_temporal_attention(bool v) { enable_temporal_attention_ = v; }
    void set_enable_fxaa_layer(bool v) { enable_fxaa_layer_ = v; }
    void set_enable_kfe_memory(bool v) { enable_kfe_memory_ = v; }
    void set_enable_ssm(bool v) { enable_ssm_ = v; }
    void set_enable_rwkv(bool v) { enable_rwkv_ = v; }
    void set_enable_ddpm(bool v) { enable_ddpm_ = v; }
    bool enable_multi_head_ = true, enable_global_aggregation_ = true, enable_noise_filter_ = true,
         enable_temporal_attention_ = true, enable_fxaa_layer_ = true, enable_kfe_memory_ = true,
         enable_ssm_ = true, enable_rwkv_ = true, enable_ddpm_ = true;
};

struct StorageConfig {
    int gpu_cache_size_mb() const { return gpu_cache_size_mb_; }
    int ram_cache_size_mb() const { return ram_cache_size_mb_; }
    std::string disk_cache_path() const { return disk_cache_path_; }
    int eviction_threshold() const { return eviction_threshold_; }
    void set_gpu_cache_size_mb(int v) { gpu_cache_size_mb_ = v; }
    void set_ram_cache_size_mb(int v) { ram_cache_size_mb_ = v; }
    void set_disk_cache_path(const std::string& v) { disk_cache_path_ = v; }
    void set_eviction_threshold(int v) { eviction_threshold_ = v; }
    int gpu_cache_size_mb_ = 8192;
    int ram_cache_size_mb_ = 32768;
    std::string disk_cache_path_ = "/tmp/sintellix_cache";
    int eviction_threshold_ = 100;
};

struct OptimizationConfig {
    bool use_tensor_cores() const { return use_tensor_cores_; }
    bool use_fused_kernels() const { return use_fused_kernels_; }
    bool use_kv_cache() const { return use_kv_cache_; }
    int kv_cache_size() const { return kv_cache_size_; }
    void set_use_tensor_cores(bool v) { use_tensor_cores_ = v; }
    void set_use_fused_kernels(bool v) { use_fused_kernels_ = v; }
    void set_use_kv_cache(bool v) { use_kv_cache_ = v; }
    void set_kv_cache_size(int v) { kv_cache_size_ = v; }
    bool use_tensor_cores_ = true, use_fused_kernels_ = true, use_kv_cache_ = true;
    int kv_cache_size_ = 512;
};

struct NoiseFilterConfig {
    float ema_alpha() const { return ema_alpha_; }
    float threshold_multiplier() const { return threshold_multiplier_; }
    void set_ema_alpha(float v) { ema_alpha_ = v; }
    void set_threshold_multiplier(float v) { threshold_multiplier_ = v; }
    float ema_alpha_ = 0.1f, threshold_multiplier_ = 2.0f;
};

struct GlobalAggregationConfig {
    int top_k() const { return top_k_; }
    float attention_dropout() const { return attention_dropout_; }
    void set_top_k(int v) { top_k_ = v; }
    void set_attention_dropout(float v) { attention_dropout_ = v; }
    int top_k_ = 64;
    float attention_dropout_ = 0.1f;
};

struct NeuronConfig {
    int dim() const { return dim_; }
    int num_heads() const { return num_heads_; }
    int temporal_frames() const { return temporal_frames_; }
    const GridSize& grid_size() const { return grid_size_; }
    const ModuleConfig& modules() const { return modules_; }
    const StorageConfig& storage() const { return storage_; }
    const OptimizationConfig& optimization() const { return optimization_; }
    const NoiseFilterConfig& noise_filter() const { return noise_filter_; }
    const GlobalAggregationConfig& global_aggregation() const { return global_aggregation_; }

    void set_dim(int v) { dim_ = v; }
    void set_num_heads(int v) { num_heads_ = v; }
    void set_temporal_frames(int v) { temporal_frames_ = v; }
    GridSize* mutable_grid_size() { return &grid_size_; }
    ModuleConfig* mutable_modules() { return &modules_; }
    StorageConfig* mutable_storage() { return &storage_; }
    OptimizationConfig* mutable_optimization() { return &optimization_; }
    NoiseFilterConfig* mutable_noise_filter() { return &noise_filter_; }
    GlobalAggregationConfig* mutable_global_aggregation() { return &global_aggregation_; }

private:
    int dim_ = 256, num_heads_ = 8, temporal_frames_ = 8;
    GridSize grid_size_{32, 32, 32};
    ModuleConfig modules_{true, true, true, true, true, true, true, true, true};
    StorageConfig storage_{8192, 32768, "/tmp/sintellix_cache", 100};
    OptimizationConfig optimization_{true, true, true, 512};
    NoiseFilterConfig noise_filter_{0.1f, 2.0f};
    GlobalAggregationConfig global_aggregation_{64, 0.1f};
};
#endif

namespace sintellix {

#ifdef USE_PROTOBUF
/**
 * Configuration manager for Sintellix
 * Supports loading from JSON/TOML and Protobuf
 */
class ConfigManager {
public:
    ConfigManager() = default;
    ~ConfigManager() = default;

#ifdef USE_NLOHMANN_JSON
    // Load configuration from JSON file
    bool loadFromJson(const std::string& path);

    // Save configuration to JSON file
    bool saveToJson(const std::string& path) const;
#endif

    // Load configuration from Protobuf
    bool loadFromProto(const std::string& path);

    // Save configuration to Protobuf
    bool saveToProto(const std::string& path) const;

    // Get configuration
    const NeuronConfig& getConfig() const { return config_; }
    NeuronConfig& getConfig() { return config_; }

    // Create default configuration
    static NeuronConfig createDefault();

private:
    NeuronConfig config_;

#ifdef USE_NLOHMANN_JSON
    // Convert JSON to Protobuf
    void jsonToProto(const nlohmann::json& j, NeuronConfig& config);

    // Convert Protobuf to JSON
    nlohmann::json protoToJson(const NeuronConfig& config) const;
#endif
};
#else
// Stub ConfigManager when Protobuf is not available
class ConfigManager {
public:
    ConfigManager() = default;
    ~ConfigManager() = default;
};
#endif

} // namespace sintellix
