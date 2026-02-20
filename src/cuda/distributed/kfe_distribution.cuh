#pragma once
#include <cuda_runtime.h>

namespace quila {

// KFE distribution across GPUs
struct KFEDistribution {
    int* kfe_to_gpu;      // Maps KFE ID to GPU ID
    int* gpu_kfe_counts;  // Number of KFEs per GPU
    int num_kfes;
    int num_gpus;
};

// Initialize KFE distribution
__host__ void init_kfe_distribution(KFEDistribution* dist, int num_kfes, int num_gpus);

// Free KFE distribution
__host__ void free_kfe_distribution(KFEDistribution* dist);

// Assign KFEs to GPUs
__host__ void assign_kfes_to_gpus(KFEDistribution* dist);

// Get GPU ID for KFE
__host__ int get_kfe_gpu(const KFEDistribution* dist, int kfe_id);

} // namespace quila
